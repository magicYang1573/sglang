import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class SparseFa3CallArgs:
    """Per-layer FA3 sparse-decode call arguments.

    This is **NOT** a ``FlashAttentionMetadata`` — it is a self-contained
    parameter pack that the backend feeds directly into
    ``flash_attn_with_kvcache`` (see
    ``Fa3SparseDecodeBackend.forward_decode``).

    Tensors live on GPU.  In the eager-mode first release the tensors are
    freshly produced each layer; when CUDA Graph is enabled later the same
    fields will be backed by pre-allocated buffers + in-place writes
    (signature stays unchanged).

    Fields:
        page_table:    [bs, top_k] int32, physical locs in HiSparse device buffer
        cache_seqlens: [bs] int32, ``min(seq_len, top_k)``
        cu_seqlens_q:  [bs+1] int32, arange (decode: each row q_len = 1)
        cu_seqlens_k:  [bs+1] int32, ``cumsum(cache_seqlens)``
        max_seqlen_k:  Python int — static upper bound = top_k
    """

    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_k: int


class BackendAdaptor(ABC):
    """Base class for attention backend adaptors."""

    def __init__(self, device: torch.device):
        self.device = device


class SparseDecodeAdapter(BackendAdaptor):
    """Common base for adapters that operate on **physical device-buffer
    slots** (as returned by ``HiSparseCoordinator.swap_in_selected_pages``)
    rather than on logical pages.

    Subclasses implement :meth:`build_sparse_call_args` to package the
    per-layer FA-style sparse call arguments.  The shared dense
    ``FlashAttentionMetadata`` is treated as **read-only** — adapters only
    borrow fields like ``cu_seqlens_q`` from it and never mutate it.
    """

    @abstractmethod
    def build_sparse_call_args(
        self,
        top_k_device_locs: torch.Tensor,
        valid_lengths: torch.Tensor,
        dense_metadata: Any,
        forward_batch: "ForwardBatch",
        layer_id: int,
        **kwargs,
    ) -> SparseFa3CallArgs:
        """Package per-layer sparse-decode call arguments.

        Args:
            top_k_device_locs: ``[bs, top_k]`` int32, physical hot-buffer row
                indices already populated by
                ``HiSparseCoordinator.swap_in_selected_pages``.
            valid_lengths: ``[bs]`` int32, ``min(seq_len[i], top_k)``.
            dense_metadata: The backend's per-step dense metadata (read-only;
                used only to borrow already-built arange tensors such as
                ``cu_seqlens_q``).
        """
        ...


class NSABackendAdaptor(BackendAdaptor):
    """Adaptor for NSA (Native Sparse Attention) backend."""

    def __init__(
        self,
        device: torch.device,
        req_to_token_pool,
    ):
        super().__init__(device)
        self.req_to_token_pool = req_to_token_pool

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
    ) -> Optional[torch.Tensor]:
        """
        Transform logical page indices to physical device indices for NSA backend.
        """
        # TODO: Implement NSA backend adaptor logic
        pass


class FlashAttentionAdaptor(BackendAdaptor):
    """Adaptor for FlashAttention backend (LEGACY).

    NOTE: this adapter is part of the older "shared metadata rewrite" design
    and is no longer used by the sparse-decode path
    (which now uses :class:`Fa3SparseDecodeAdapter` and the
    :class:`SparseFa3CallArgs`-based call surface).  It is kept for
    compatibility with existing tests and the optional ``backend="fa"``
    config; do not use for new code paths.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._original_metadata: Optional[dict] = None

    def save_original_metadata(self, metadata: Any) -> None:
        self._original_metadata = {
            "page_table": metadata.page_table.clone(),
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),
            "cu_seqlens_k": metadata.cu_seqlens_k.clone(),
            "max_seq_len_k": metadata.max_seq_len_k,
        }

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
        """
        Adapt FlashAttention metadata for sparse KVCache access.

        Modifies page_table, cache_seqlens, and related metadata to redirect
        FlashAttention to only process selected sparse pages.

        # TODO: Optimize performance
        """
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            return current_metadata

        current_metadata.page_table.copy_(self._original_metadata["page_table"])
        current_metadata.cache_seqlens_int32.copy_(
            self._original_metadata["cache_seqlens_int32"]
        )

        physical_pages = self._logical_to_physical_pages_batch(
            selected_indices,
            forward_batch.req_pool_indices,
            req_to_token,
            page_size,
        )

        max_selected = physical_pages.shape[1]
        valid_mask = torch.arange(max_selected, device=physical_pages.device).unsqueeze(
            0
        ) < valid_lengths.unsqueeze(1)
        update_mask = sparse_mask.unsqueeze(1) & valid_mask

        current_metadata.page_table[:, :max_selected] = torch.where(
            update_mask, physical_pages, current_metadata.page_table[:, :max_selected]
        )

        seq_lens = forward_batch.seq_lens
        positions_in_page = (seq_lens - 1) % page_size
        diff = page_size - positions_in_page - 1
        sparse_seq_lens = (valid_lengths * page_size - diff).to(torch.int32)

        current_metadata.cache_seqlens_int32 = torch.where(
            sparse_mask, sparse_seq_lens, self._original_metadata["cache_seqlens_int32"]
        )

        current_metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(
                current_metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
            ),
            (1, 0),
        )
        current_metadata.max_seq_len_k = int(current_metadata.cache_seqlens_int32.max())
        return current_metadata

    def _logical_to_physical_pages_batch(
        self,
        logical_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        bs, max_pages = logical_pages.shape

        page_starts = logical_pages * page_size
        page_starts_clamped = page_starts.clamp(min=0)

        req_indices_expanded = req_pool_indices.unsqueeze(1).expand(-1, max_pages)
        first_tokens = req_to_token[req_indices_expanded, page_starts_clamped]

        physical_pages = first_tokens // page_size
        physical_pages = torch.where(
            logical_pages >= 0, physical_pages, torch.zeros_like(physical_pages)
        )

        return physical_pages.to(torch.int32)
