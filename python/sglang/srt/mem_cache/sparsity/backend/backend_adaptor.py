import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class BackendAdaptor(ABC):
    """Base class for attention backend adaptors."""

    def __init__(self, device: torch.device):
        self.device = device
        self._original_metadata = None

    def save_original_metadata(self, metadata: Any) -> None:
        """Save original metadata in the beginning of the forward pass."""
        pass

    @abstractmethod
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
        Adapt attention metadata for sparse KVCache access.

        Transforms sparse retrieval results (logical indices of important KV pages/tokens)
        into backend-specific attention metadata format.

        Returns:
            Modified attention metadata compatible with the backend
        """
        pass


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
    """Adaptor for FlashAttention backend."""

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

        Supports page-granularity sparse algorithms where
        ``sparse_page_size = backend_page_size * N`` with ``N >= 1``:

        * ``N == 1``: zero-conversion mapping (one selected sparse page
          becomes one backend page).
        * ``N > 1``: each selected sparse page expands into ``N`` consecutive
          backend pages; ``page_table`` is filled with ``N × valid_lengths``
          entries per request.

        When a HiSparse IO subsystem is attached (``io_subsystem`` kwarg)
        the logical backend-page index is additionally translated to the
        compact hisparse device buffer via
        ``HiSparseMHATokenToKVPool.translate_loc_to_hisparse_device``.
        """
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            return current_metadata

        # ``page_size`` arg is the *sparse* page size (passed by
        # SparseCoordinator); backend page size comes from kwargs.
        sparse_page_size = page_size
        backend_page_size = int(
            kwargs.get("backend_page_size", sparse_page_size)
        )
        n = sparse_page_size // backend_page_size
        assert (
            n >= 1 and sparse_page_size == n * backend_page_size
        ), f"sparse_page_size={sparse_page_size} must be backend_page_size={backend_page_size} × integer"

        current_metadata.page_table.copy_(self._original_metadata["page_table"])
        current_metadata.cache_seqlens_int32.copy_(
            self._original_metadata["cache_seqlens_int32"]
        )

        # 1) Sparse logical pages -> backend-page indices.
        #    Result shape: [bs, max_selected_sparse, N], -1 padded.
        physical_backend_pages = self._sparse_to_backend_pages(
            selected_indices,
            forward_batch.req_pool_indices,
            req_to_token,
            sparse_page_size=sparse_page_size,
            backend_page_size=backend_page_size,
        )

        # 2) Optional: translate logical → hisparse device backend pages.
        io_subsystem = kwargs.get("io_subsystem")
        if io_subsystem is not None:
            translate = (
                forward_batch.token_to_kv_pool.translate_loc_to_hisparse_device
            )
            first_tok = (
                physical_backend_pages.to(torch.int64) * backend_page_size
            )
            translated_tok = translate(first_tok).to(torch.int64)
            physical_backend_pages = torch.where(
                physical_backend_pages >= 0,
                (translated_tok // backend_page_size).to(torch.int32),
                physical_backend_pages,
            )

        # 3) Flatten [bs, max_selected_sparse, N] -> [bs, max_selected_backend].
        bs = physical_backend_pages.shape[0]
        max_selected_backend = physical_backend_pages.shape[1] * n
        physical_backend_pages_flat = physical_backend_pages.reshape(
            bs, max_selected_backend
        )

        # 4) Write into page_table.
        valid_lengths_backend = valid_lengths * n
        valid_mask = torch.arange(
            max_selected_backend, device=physical_backend_pages_flat.device
        ).unsqueeze(0) < valid_lengths_backend.unsqueeze(1)
        update_mask = sparse_mask.unsqueeze(1) & valid_mask

        current_metadata.page_table[:, :max_selected_backend] = torch.where(
            update_mask,
            physical_backend_pages_flat,
            current_metadata.page_table[:, :max_selected_backend],
        )

        # 5) Fix up cache seq lens: FA3 reads "cache_seqlens_int32" tokens
        #    starting from page_table[0].  Use sparse_page_size for the
        #    last-page tail correction to match what Quest actually covered.
        seq_lens = forward_batch.seq_lens
        positions_in_sparse_page = (seq_lens - 1) % sparse_page_size
        diff = sparse_page_size - positions_in_sparse_page - 1
        sparse_seq_lens = (
            valid_lengths * sparse_page_size - diff
        ).to(torch.int32)

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

    def _sparse_to_backend_pages(
        self,
        logical_sparse_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        sparse_page_size: int,
        backend_page_size: int,
    ) -> torch.Tensor:
        """Expand ``logical_sparse_pages`` to ``[bs, max_selected, N]`` of
        logical backend-page indices, preserving -1 padding."""
        bs, max_pages = logical_sparse_pages.shape
        n = sparse_page_size // backend_page_size

        first_token_in_sparse_page = (
            logical_sparse_pages.to(torch.int64) * sparse_page_size
        )
        first_token_clamped = first_token_in_sparse_page.clamp(min=0)

        # Index into the per-request token table to get the physical token.
        req_idx_expand = req_pool_indices.unsqueeze(1).expand(-1, max_pages)
        first_physical_tok = req_to_token[req_idx_expand, first_token_clamped]

        # Physical backend page index for the first backend page inside the
        # sparse page.
        first_backend_page = first_physical_tok // backend_page_size  # [bs, P]

        # Add offsets [0..N) to cover the N consecutive backend pages.
        offsets = torch.arange(
            n, device=first_backend_page.device, dtype=first_backend_page.dtype
        )
        backend_pages = first_backend_page.unsqueeze(-1) + offsets.view(1, 1, n)

        # Preserve -1 padding.
        valid_mask = logical_sparse_pages.unsqueeze(-1) >= 0
        backend_pages = torch.where(
            valid_mask, backend_pages, torch.full_like(backend_pages, -1)
        )
        return backend_pages.to(torch.int32)
