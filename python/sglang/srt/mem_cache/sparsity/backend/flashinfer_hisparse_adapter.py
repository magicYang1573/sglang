"""FlashInfer paged-decode adapter for HiSparse sparse decode.

The adapter receives physical hot-buffer token locations returned by
``HiSparseCoordinator.swap_in_selected_pages`` and turns them into the
``kv_indptr / kv_indices / kv_last_page_len`` inputs expected by FlashInfer's
``BatchDecodeWithPagedKVCacheWrapper``.  In this path FlashInfer page size is
forced to 1, so each KV page is exactly one token row in the HiSparse device
buffer.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import SparseDecodeAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


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
        self._maybe_debug_log_indices(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=current_metadata,
            forward_batch=forward_batch,
            layer_id=layer_id,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            top_k_tokens=kwargs.get("top_k_tokens"),
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
        too_long_dense = (~sparse_mask) & (
            forward_batch.seq_lens.to(device=device, dtype=torch.int64) > dense_width
        )
        if torch.any(too_long_dense):
            rows = too_long_dense.nonzero(as_tuple=False).flatten()
            max_report = min(int(rows.numel()), 8)
            rows_report = rows[:max_report]
            logger.error(
                "FlashInfer HiSparse dense fallback row exceeds hot buffer. "
                "These rows cannot be represented as dense FlashInfer kv_indices. "
                "examples=%s",
                [
                    {
                        "row": int(r.item()),
                        "req_pool_idx": int(
                            forward_batch.req_pool_indices[r].detach().cpu().item()
                        ),
                        "seq_len": int(
                            forward_batch.seq_lens[r].detach().cpu().item()
                        ),
                        "device_buffer_size": int(dense_width),
                    }
                    for r in rows_report
                ],
            )
            raise RuntimeError(
                "FlashInfer HiSparse dense fallback row exceeds device_buffer_size."
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

    def _maybe_debug_log_indices(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        layer_id: int,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        top_k_tokens: torch.Tensor | None,
    ) -> None:
        if os.environ.get("SGLANG_HISPARSE_FLASHINFER_DEBUG", "0") != "1":
            return
        if getattr(current_metadata, "_debug_logged", False) and layer_id != 0:
            return

        report_rows = min(int(forward_batch.batch_size), 4)
        prefix = 16
        data = {
            "layer_id": int(layer_id),
            "batch_size": int(forward_batch.batch_size),
            "seq_lens": forward_batch.seq_lens[:report_rows].detach().cpu().tolist(),
            "req_pool_indices": forward_batch.req_pool_indices[:report_rows]
            .detach()
            .cpu()
            .tolist(),
            "sparse_mask": sparse_mask[:report_rows].detach().cpu().tolist(),
            "valid_lengths": valid_lengths[:report_rows].detach().cpu().tolist(),
            "kv_indptr": kv_indptr[: report_rows + 1].detach().cpu().tolist(),
            "kv_indices_head": kv_indices[: min(prefix, kv_indices.numel())]
            .detach()
            .cpu()
            .tolist(),
            "kv_indices_min": int(kv_indices.min().detach().cpu().item())
            if kv_indices.numel() > 0
            else None,
            "kv_indices_max": int(kv_indices.max().detach().cpu().item())
            if kv_indices.numel() > 0
            else None,
            "selected_locs_head": selected_indices[:report_rows, :prefix]
            .detach()
            .cpu()
            .tolist(),
        }
        if top_k_tokens is not None:
            data["top_k_tokens_head"] = (
                top_k_tokens[:report_rows, :prefix].detach().cpu().tolist()
            )
        logger.warning("FlashInfer HiSparse debug indices: %s", data)
        current_metadata._debug_logged = True
