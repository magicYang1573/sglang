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
        """Adapt FA3 metadata for sparse KV access.

        SGLang's FA3 path stores ``metadata.page_table`` as **token-level**
        physical KV slot indices (``req_to_token[req, :seq_len]``), because
        the KV buffer is laid out as ``[num_slots, num_heads, head_dim]``
        (flat, no explicit page dim).  FA3 sees this as "page_size=1 paged
        KV" at kernel level, regardless of the server-side
        ``--page-size`` (which only controls allocation alignment).

        Therefore this adaptor:

        1. Expands the selected *sparse* pages ``[bs, num_topk_pages]``
           into token-level logical positions
           ``[bs, num_topk_pages * sparse_page_size]``.
        2. Maps through ``req_to_token`` to get physical token slots.
        3. (Optional) Translates logical slots → hisparse device buffer
           slots via ``HiSparseMHATokenToKVPool.translate_loc_to_hisparse_device``.
        4. Writes the contiguous-compacted slot list into
           ``metadata.page_table[:, :valid_tokens]``; pads the rest with 0.
        5. Fixes ``cache_seqlens_int32 / cu_seqlens_k / max_seq_len_k`` to
           reflect only the tokens we actually selected.
        """
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            return current_metadata

        sparse_page_size = page_size  # from SparseCoordinator
        device = forward_batch.seq_lens.device

        # Reset metadata to its captured baseline before we touch it.
        current_metadata.page_table.copy_(self._original_metadata["page_table"])
        current_metadata.cache_seqlens_int32.copy_(
            self._original_metadata["cache_seqlens_int32"]
        )

        bs, num_topk_pages = selected_indices.shape
        seq_lens = forward_batch.seq_lens.to(torch.int32)

        # 1) Page idx -> token positions per request.
        #    logical_token_pos[b, i*page_size + j] = page[b,i] * page_size + j
        #                                          or -1 if page[b,i] == -1
        offsets = torch.arange(
            sparse_page_size, device=device, dtype=torch.int32
        )
        token_pos = (
            selected_indices.unsqueeze(-1).to(torch.int32) * sparse_page_size
            + offsets.view(1, 1, sparse_page_size)
        ).view(bs, num_topk_pages * sparse_page_size)  # [bs, T]
        page_valid = (
            (selected_indices >= 0)
            .unsqueeze(-1)
            .expand(-1, -1, sparse_page_size)
            .reshape(bs, num_topk_pages * sparse_page_size)
        )
        in_seq = token_pos < seq_lens.unsqueeze(1)
        tok_valid = page_valid & in_seq  # [bs, T]

        # Compact valid token positions to the front of each row via a
        # stable sort on the inverted mask (0 = valid first, 1 = invalid).
        sort_key = (~tok_valid).to(torch.int8)
        order = torch.argsort(sort_key, dim=1, stable=True)
        token_pos_sorted = torch.gather(token_pos, 1, order)
        tok_valid_sorted = torch.gather(tok_valid, 1, order)

        # 2) Logical token pos -> physical KV slot via req_to_token.
        req_idx = forward_batch.req_pool_indices.unsqueeze(1)
        # Clamp out-of-bounds positions (the invalid tail) before indexing.
        token_pos_clamped = token_pos_sorted.clamp(min=0).to(torch.int64)
        max_req_tokens = req_to_token.shape[1]
        token_pos_clamped = token_pos_clamped.clamp(max=max_req_tokens - 1)
        physical_slots = req_to_token[req_idx, token_pos_clamped]  # [bs, T]

        # 3) Optional: translate logical physical slots to the compact
        #    hisparse device buffer slots.  This mirrors the NSA backend's
        #    ``translate_loc_to_hisparse_device(page_table_1)`` call site.
        io_subsystem = kwargs.get("io_subsystem")
        if io_subsystem is not None:
            translate = (
                forward_batch.token_to_kv_pool.translate_loc_to_hisparse_device
            )
            physical_slots = translate(physical_slots.to(torch.int64))

        physical_slots = physical_slots.to(
            current_metadata.page_table.dtype
        )

        # Number of valid tokens per request (for page_table write & seqlen).
        valid_tokens = tok_valid_sorted.sum(dim=1).to(torch.int32)  # [bs]

        # 4) Write token-level slots into page_table[:, :T], zero-pad tail.
        T = token_pos_sorted.shape[1]
        pt_width = current_metadata.page_table.shape[1]
        if T > pt_width:
            # If caller asked for more tokens than page_table can hold,
            # truncate — page_table is sized for the un-sparsified max
            # context, so this should not normally happen.
            T = pt_width
            physical_slots = physical_slots[:, :T]
            tok_valid_sorted = tok_valid_sorted[:, :T]

        tok_valid_sorted_ = tok_valid_sorted & sparse_mask.unsqueeze(1)
        current_metadata.page_table[:, :T] = torch.where(
            tok_valid_sorted_,
            physical_slots,
            torch.zeros_like(physical_slots),
        )
        # Any pre-existing entries past the compacted valid tail are
        # logically "unused" — zero them so FA3 never reads stale slots.
        if T < pt_width:
            current_metadata.page_table[:, T:] = 0

        # 5) Fix up cache seqlens to reflect only the selected tokens.
        current_metadata.cache_seqlens_int32 = torch.where(
            sparse_mask,
            valid_tokens,
            self._original_metadata["cache_seqlens_int32"],
        )
        current_metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(
                current_metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
            ),
            (1, 0),
        )
        current_metadata.max_seq_len_k = int(
            current_metadata.cache_seqlens_int32.max()
        )
        return current_metadata
