"""FlashInfer paged-decode adapter for HiSparse sparse decode.

The adapter receives physical hot-buffer token locations returned by
``HiSparseCoordinator.swap_in_selected_pages`` and turns them into the
``kv_indptr / kv_indices / kv_last_page_len`` inputs expected by FlashInfer's
``BatchDecodeWithPagedKVCacheWrapper``.  In this path FlashInfer page size is
forced to 1, so each KV page is exactly one token row in the HiSparse device
buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import SparseDecodeAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class FlashInferHiSparseAdapter(SparseDecodeAdapter):
    """Prepare FlashInfer paged-decode indices from HiSparse hot-buffer locs."""

    def adapt_dense_for_attn_metadata(
        self,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        layer_id: int,
        **kwargs,
    ) -> Any:
        bs = int(forward_batch.batch_size)
        empty_selected = torch.empty((bs, 0), dtype=torch.int32, device=self.device)
        valid_lengths = torch.zeros(bs, dtype=torch.int32, device=self.device)
        sparse_mask = torch.zeros(bs, dtype=torch.bool, device=self.device)
        return self.adapt_for_attn_metadata(
            selected_indices=empty_selected,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=current_metadata,
            forward_batch=forward_batch,
            req_to_token=kwargs.get("req_to_token"),
            page_size=1,
            layer_id=layer_id,
        )

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        if page_size != 1:
            raise ValueError(
                "FlashInfer HiSparse decode requires page_size=1 so token locs "
                f"can be used directly as paged KV indices (got {page_size})."
            )

        wrapper = current_metadata.decode_wrappers[
            current_metadata.get_wrapper_idx(layer_id)
        ]
        bs = int(forward_batch.batch_size)
        kv_indptr = current_metadata.kv_indptr[: bs + 1]
        kv_last_page_len = current_metadata.kv_last_page_len[:bs]

        row_indices = self._build_row_indices(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
        )

        total_len = sum(int(row.numel()) for row in row_indices)
        kv_indices = torch.empty(total_len, dtype=torch.int32, device=self.device)

        kv_indptr[0] = 0
        offset = 0
        for i, row in enumerate(row_indices):
            row_len = int(row.numel())
            if row_len > 0:
                kv_indices[offset : offset + row_len] = row.to(torch.int32)
            offset += row_len
            kv_indptr[i + 1] = offset

        kv_last_page_len.fill_(1)
        # Keep the dynamically built tensor alive until wrapper.forward() has
        # consumed the plan.
        current_metadata.kv_indices = kv_indices
        current_metadata.kv_indptr_active = kv_indptr

        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            current_metadata.num_qo_heads,
            current_metadata.num_kv_heads,
            current_metadata.head_dim,
            1,  # page_size: HiSparse sparse decode is token-granular.
            data_type=current_metadata.data_type,
            q_data_type=current_metadata.q_data_type,
            non_blocking=True,
            fixed_split_size=current_metadata.fixed_split_size,
            disable_split_kv=current_metadata.disable_split_kv,
        )
        return current_metadata

    def _build_row_indices(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> List[torch.Tensor]:
        coord = getattr(forward_batch, "hisparse_coordinator", None)
        if coord is None:
            raise RuntimeError(
                "FlashInfer HiSparse adapter requires forward_batch.hisparse_coordinator"
            )

        row_indices: List[torch.Tensor] = []
        req_pool_indices_cpu = forward_batch.req_pool_indices.detach().cpu().tolist()
        seq_lens_cpu = forward_batch.seq_lens.detach().cpu().tolist()
        sparse_mask_cpu = sparse_mask.detach().cpu().tolist()
        valid_lengths_cpu = valid_lengths.detach().cpu().tolist()

        for row_id, (req_idx, seq_len, is_sparse, valid_len) in enumerate(
            zip(req_pool_indices_cpu, seq_lens_cpu, sparse_mask_cpu, valid_lengths_cpu)
        ):
            if is_sparse:
                length = max(int(valid_len), 0)
                locs = selected_indices[row_id, :length]
                locs = locs[locs >= 0]
                if locs.numel() == 0:
                    raise RuntimeError(
                        "Quest sparse row produced no valid FlashInfer KV indices "
                        f"(row={row_id}, req_pool_idx={req_idx}, layer sparse)."
                    )
                row_indices.append(locs)
                continue

            # Dense fallback rows can be represented directly from the
            # coordinator-owned hot buffer while the full sequence still fits in
            # the per-request device buffer.
            dense_len = int(seq_len)
            if dense_len > coord.device_buffer_size:
                raise RuntimeError(
                    "FlashInfer HiSparse dense fallback only supports rows that "
                    "fit in the device buffer. Increase sparse coverage or split "
                    f"the batch (seq_len={dense_len}, device_buffer_size="
                    f"{coord.device_buffer_size})."
                )
            row_locs = coord.req_to_device_buffer[int(req_idx), :dense_len].to(
                torch.int32
            )
            row_indices.append(row_locs)

        return row_indices
