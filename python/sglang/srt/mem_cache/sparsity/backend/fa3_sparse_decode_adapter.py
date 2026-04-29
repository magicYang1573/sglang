"""Generic FA3 metadata adapter for the sparse-decode path.

This adapter is algorithm-agnostic: it receives top-k *physical* device
buffer slots (already populated by
:meth:`HiSparseCoordinator.swap_in_selected_pages`) and rewrites FA3's
:class:`FlashAttentionMetadata` so that the FA3 kernel only looks at those
slots.  It knows nothing about Quest / H2O / SnapKV etc.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import SparseDecodeAdapter

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class Fa3SparseDecodeAdapter(SparseDecodeAdapter):
    """Rewrite FA3 metadata so FA3 only attends to the top-k swapped-in rows.

    Usage contract
    --------------
    * :meth:`save_original_metadata` is called **once per decode step** on the
      first layer, snapshotting the dense metadata so that we can fall back
      for ``sparse_mask == False`` rows on subsequent layers.
    * :meth:`adapt_for_attn_metadata` is called on **every layer**; it always
      resets the metadata to the dense snapshot and then overwrites the
      sparse rows.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._snapshot: Optional[dict] = None
        self._debug_enabled = _env_int("SGLANG_SPARSE_DECODE_DEBUG", 0) > 0
        self._debug_max_logs = _env_int("SGLANG_SPARSE_DECODE_DEBUG_MAX_LOGS", 20)
        self._debug_log_count = 0

    def save_original_metadata(self, metadata: "FlashAttentionMetadata") -> None:
        self._snapshot = {
            "page_table": metadata.page_table.clone(),
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),
            "cu_seqlens_k": (
                metadata.cu_seqlens_k.clone()
                if metadata.cu_seqlens_k is not None
                else None
            ),
            "max_seq_len_k": int(metadata.max_seq_len_k),
        }

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: "FlashAttentionMetadata",
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> "FlashAttentionMetadata":
        """Overwrite ``page_table`` prefix with physical hot-buffer slots."""
        if self._snapshot is None:
            # save_original_metadata was not called (controller contract error
            # or adapter reused across steps) — return metadata unchanged so
            # the dense FA3 path still works.
            return current_metadata

        meta = current_metadata
        # Reset from snapshot every layer: FA3 re-uses the same metadata
        # object across layers, so sparse changes from the previous layer
        # must not leak.
        meta.page_table.copy_(self._snapshot["page_table"])
        meta.cache_seqlens_int32.copy_(self._snapshot["cache_seqlens_int32"])

        if not bool(sparse_mask.any()):
            # Every request stayed dense this layer; snapshot already
            # restored, nothing more to do.
            meta.max_seq_len_k = int(self._snapshot["max_seq_len_k"])
            if self._snapshot["cu_seqlens_k"] is not None:
                meta.cu_seqlens_k = self._snapshot["cu_seqlens_k"].clone()
            return meta

        # Dense FA3 scheduler metadata was computed from the original full
        # context. Once page_table/cache_seqlens are sparsified it is no
        # longer valid; let FA3 take its regular no-scheduler path.
        meta.scheduler_metadata = None

        top_k_device_locs = selected_indices  # [bs, top_k] int32
        bs, top_k = top_k_device_locs.shape
        device = meta.page_table.device

        # Align dtype with page_table (FA3 page_table is int32 in the MHA/GQA path).
        page_table_dtype = meta.page_table.dtype
        locs_aligned = top_k_device_locs.to(page_table_dtype)

        # Build a [bs, top_k] valid-slot mask so invalid (-1) entries are
        # replaced by the first reserved pad row (0) and counted out of
        # cache_seqlens.
        top_k_range = torch.arange(top_k, device=device).unsqueeze(0)
        valid_mask = top_k_range < valid_lengths.to(device).unsqueeze(1)
        invalid_loc_mask = valid_mask & (locs_aligned < 0)
        # If a sparse row contains invalid device locations in its valid
        # prefix, fall back that row to dense for safety.
        row_has_invalid = invalid_loc_mask.any(dim=1)
        sparse_mask_device = sparse_mask.to(device)
        effective_sparse_mask = sparse_mask_device & (~row_has_invalid)
        effective_valid_mask = valid_mask & (~row_has_invalid.unsqueeze(1))
        invalid_sparse_rows = row_has_invalid & sparse_mask_device
        if bool(invalid_sparse_rows.any()):
            logger.warning(
                "FA3 sparse adapter found invalid swapped-in locations; "
                "falling back %d row(s) to dense at layer_id=%d.",
                int(invalid_sparse_rows.sum().item()),
                layer_id,
            )

        # For sparse rows overwrite the ENTIRE top-k prefix of the page
        # table; invalid slots use pad row 0 (FA3 cache_seqlens mask will
        # truncate reads at valid_lengths so those slots don't actually
        # contribute).  Dense rows remain untouched (already reset from
        # snapshot).
        sparse_rowmask = effective_sparse_mask.unsqueeze(1).expand(bs, top_k)
        safe_locs = torch.where(
            effective_valid_mask, locs_aligned, torch.zeros_like(locs_aligned)
        )
        target_page_table = meta.page_table[:, :top_k]
        meta.page_table[:, :top_k] = torch.where(
            sparse_rowmask, safe_locs, target_page_table
        )

        # Rebuild cache_seqlens: sparse rows truncated to valid_lengths,
        # dense rows keep the snapshot's value.
        effective_lengths = effective_valid_mask.sum(dim=1).to(torch.int32)
        new_seqlens = torch.where(
            effective_sparse_mask.to(meta.cache_seqlens_int32.device),
            effective_lengths.to(meta.cache_seqlens_int32.dtype),
            meta.cache_seqlens_int32,
        )
        meta.cache_seqlens_int32 = new_seqlens

        # Recompute cu_seqlens_k and max_seq_len_k from the new cache_seqlens.
        meta.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(meta.cache_seqlens_int32, dim=0, dtype=torch.int32),
            (1, 0),
        )
        meta.max_seq_len_k = int(meta.cache_seqlens_int32.max().item())

        self._maybe_log_adapter_debug(
            layer_id=layer_id,
            meta=meta,
            top_k_device_locs=top_k_device_locs,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask_device,
            effective_sparse_mask=effective_sparse_mask,
            effective_lengths=effective_lengths,
        )

        return meta

    def _maybe_log_adapter_debug(
        self,
        *,
        layer_id: int,
        meta: "FlashAttentionMetadata",
        top_k_device_locs: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        effective_sparse_mask: torch.Tensor,
        effective_lengths: torch.Tensor,
    ) -> None:
        if (
            not self._debug_enabled
            or self._debug_log_count >= self._debug_max_logs
            or top_k_device_locs.shape[0] == 0
        ):
            return

        row = 0
        valid_len = int(valid_lengths[row].item())
        prefix_len = min(valid_len, 16)
        logger.warning(
            "SparseDecodeDebug FA3Adapter #%d: layer=%d bs=%d top_k=%d "
            "row0(sparse=%s effective_sparse=%s valid_len=%d effective_len=%d "
            "meta_cache_seqlen=%d meta_max_seq_len_k=%d page_table_prefix=%s)",
            self._debug_log_count,
            layer_id,
            top_k_device_locs.shape[0],
            top_k_device_locs.shape[1],
            bool(sparse_mask[row].item()),
            bool(effective_sparse_mask[row].item()),
            valid_len,
            int(effective_lengths[row].item()),
            int(meta.cache_seqlens_int32[row].item()),
            int(meta.max_seq_len_k),
            meta.page_table[row, :prefix_len].detach().cpu().tolist()
            if prefix_len > 0
            else [],
        )
        self._debug_log_count += 1
