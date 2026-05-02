"""FlashInfer paged-decode adapter for HiSparse sparse decode.

The adapter receives physical hot-buffer token locations returned by
``HiSparseCoordinator.swap_in_selected_pages`` and turns them into the
``kv_indptr / kv_indices / kv_last_page_len`` inputs expected by FlashInfer's
``BatchDecodeWithPagedKVCacheWrapper``.  In this path FlashInfer page size is
forced to 1, so each KV page is exactly one token row in the HiSparse device
buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

        kv_indices = self._build_kv_indices(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
            kv_indptr=kv_indptr,
        )

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

    def _build_kv_indices(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        forward_batch: "ForwardBatch",
        kv_indptr: torch.Tensor,
    ) -> torch.Tensor:
        coord = getattr(forward_batch, "hisparse_coordinator", None)
        if coord is None:
            raise RuntimeError(
                "FlashInfer HiSparse adapter requires forward_batch.hisparse_coordinator"
            )

        device = selected_indices.device
        bs = int(forward_batch.batch_size)
        top_k = selected_indices.shape[1]
        dense_width = coord.device_buffer_size
        max_width = max(top_k, dense_width)

        cols = torch.arange(max_width, device=device).view(1, -1)
        sparse_mask = sparse_mask.to(device=device, dtype=torch.bool)
        sparse_lengths = valid_lengths.to(device=device, dtype=torch.int64).clamp(
            min=0, max=top_k
        )
        dense_lengths = forward_batch.seq_lens.to(device=device, dtype=torch.int64).clamp(
            min=0, max=dense_width
        )
        lengths = torch.where(sparse_mask, sparse_lengths, dense_lengths)
        lengths = lengths.clamp(min=1)

        kv_indptr[0] = 0
        kv_indptr[1 : bs + 1].copy_(
            torch.cumsum(lengths.to(torch.int32), dim=0)
        )

        sparse_locs = torch.zeros((bs, max_width), dtype=torch.int32, device=device)
        if top_k > 0:
            sparse_locs[:, :top_k] = torch.clamp(selected_indices, min=0).to(
                torch.int32
            )

        dense_cols = cols.clamp(max=dense_width - 1).expand(bs, max_width)
        dense_locs = coord.req_to_device_buffer[
            forward_batch.req_pool_indices.to(coord.req_to_device_buffer.device).view(-1, 1),
            dense_cols.to(coord.req_to_device_buffer.device),
        ].to(device=device, dtype=torch.int32)

        row_locs = torch.where(sparse_mask.view(-1, 1), sparse_locs, dense_locs)
        valid = cols < lengths.view(-1, 1)
        return row_locs[valid].contiguous()
