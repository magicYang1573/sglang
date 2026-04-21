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
        # ``scheduler_metadata`` is precomputed in ``init_forward_metadata``
        # from the *original* ``cache_seqlens_int32`` / ``max_seq_len_k``.
        # Every adapter rewrite below mutates those fields, which means the
        # precomputed scheduler tile plan is stale: FA3 will dispatch
        # workers for the original (longer) sequence length and read past
        # the compacted ``page_table`` into the zero-padded tail, which
        # contaminates the softmax with slot-0 KV and destroys accuracy
        # (observed as "...\n.\n.\n..." degeneration on 20-shot GSM8K).
        #
        # Clearing ``metadata.scheduler_metadata`` forces FA3's
        # ``flash_attn_with_kvcache`` to recompute the work plan on every
        # layer from the *current* ``cache_seqlens_int32`` we just wrote.
        # This is strictly correct and costs an extra per-layer kernel
        # launch; acceptable for the eager-mode (``--disable-cuda-graph``)
        # Quest + FA3 MVP path.
        had_sched = (
            hasattr(metadata, "scheduler_metadata")
            and metadata.scheduler_metadata is not None
        )
        if hasattr(metadata, "scheduler_metadata"):
            metadata.scheduler_metadata = None

        left = getattr(self, "_debug_save_left", 6)
        if left > 0:
            self._debug_save_left = left - 1
            import logging as _logging
            _logging.getLogger("hisparse.debug").warning(
                "[HS-DBG save_meta] cleared_scheduler_metadata=%s "
                "original_max_seq_len_k=%s original_cache_seqlens[:4]=%s",
                bool(had_sched),
                int(self._original_metadata["max_seq_len_k"]),
                self._original_metadata["cache_seqlens_int32"][:4].tolist()
                if self._original_metadata["cache_seqlens_int32"].numel()
                else [],
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
        """Adapt FA3 metadata for sparse KV access.

        **Interface contract (MVP, see docs/advanced_features/
        hisparse_quest_qwen_design.md §1.4):** when FA3 serves as the
        attention backend for a non-native-sparse algorithm (Quest /
        SnapKV / ...), the ``server_args.page_size`` **must be 1** so that
        ``metadata.page_table`` carries **token-level** physical KV slot
        indices.

        Under HiSparse, K/V is physically stored at the **hisparse device
        slots** (see ``HiSparseMHATokenToKVPool.set_kv_buffer`` which
        translates the logical slot before writing); the vanilla
        ``metadata.page_table`` set up by the FA3 backend however still
        contains **logical** slots (straight from ``req_to_token``).  If
        we leave that as-is, FA3 will read from a slot that was never
        written and produce garbage (the " the the the ..." failure mode
        you saw).

        Therefore this adapter always rewrites ``metadata.page_table``
        into hisparse-device slots:

        * **Quest selected something**: expand the selected sparse pages
          into per-token logical positions, translate through
          ``req_to_token`` and ``translate_loc_to_hisparse_device``, and
          write a compact list of hisparse slots (``cache_seqlens`` is
          shrunk accordingly).
        * **Quest selected nothing** (warmup / no bbox yet): fall back to
          **dense** -- translate the full ``req_to_token[req, :seq_len]``
          into hisparse slots; ``cache_seqlens`` keeps the original
          sequence length.  This preserves correctness at the cost of
          attending to every resident token (which is fine because the
          hot buffer holds everything when the prompt fits).
        """
        if self._original_metadata is None:
            return current_metadata

        if not sparse_mask.any():
            # Without HiSparse the baseline is already correct; with
            # HiSparse, every attention backend forward that hits this
            # sparse-coordinator path has to translate.  ``sparse_mask``
            # being all-False means nothing to do for *sparse*, but we
            # still must ensure ``page_table`` addresses hisparse slots.
            io_subsystem = kwargs.get("io_subsystem")
            if io_subsystem is not None:
                self._write_dense_translated(
                    current_metadata=current_metadata,
                    forward_batch=forward_batch,
                    req_to_token=req_to_token,
                )
            return current_metadata

        sparse_page_size = page_size  # from SparseCoordinator
        backend_page_size = int(kwargs.get("backend_page_size", 1) or 1)
        if backend_page_size != 1:
            raise ValueError(
                "FlashAttentionAdaptor requires --page-size 1 when used as "
                "the attention backend for HiSparse non-native sparse "
                "algorithms (got backend_page_size="
                f"{backend_page_size}).  See "
                "docs/advanced_features/hisparse_quest_qwen_design.md §1.4."
            )
        device = forward_batch.seq_lens.device
        io_subsystem = kwargs.get("io_subsystem")

        # Reset metadata to its captured baseline before we touch it.
        current_metadata.page_table.copy_(self._original_metadata["page_table"])
        current_metadata.cache_seqlens_int32.copy_(
            self._original_metadata["cache_seqlens_int32"]
        )

        empty_selection = (
            selected_indices is None
            or selected_indices.numel() == 0
            or bool((selected_indices < 0).all())
        )
        if empty_selection:
            # Sparse algorithm has nothing to select yet (e.g. warmup).
            # Under HiSparse, the captured baseline page_table holds
            # *logical* slots; FA3 needs hisparse slots.  Translate the
            # full sequence for each request so FA3 reads resident KV.
            if io_subsystem is not None:
                self._write_dense_translated(
                    current_metadata=current_metadata,
                    forward_batch=forward_batch,
                    req_to_token=req_to_token,
                )
            return current_metadata

        self._write_token_level(
            selected_indices=selected_indices,
            sparse_mask=sparse_mask,
            current_metadata=current_metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            sparse_page_size=sparse_page_size,
            io_subsystem=io_subsystem,
            device=device,
        )
        return current_metadata

    def _write_dense_translated(
        self,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
    ) -> None:
        """Write a dense, hot-buffer addressed ``page_table``.

        Rewrites ``metadata.page_table[b, i]`` from the baseline *logical*
        slot to the matching **hot-buffer** (hisparse-device) slot for
        **token position ``i`` in the request**.  The source of truth is
        ``io_subsystem.req_to_device_buffer[req, i]`` which the HiSparse
        coordinator maintains for every resident token (prefill block set
        by ``alloc_device_buffer``; decode tokens appended by
        ``_grow_device_buffers``).  Unlike
        ``translate_loc_to_hisparse_device(logical)`` -- which returns 0
        for *prefill* slots after they have been "reclaimed" into the hot
        buffer and their mapping cleared -- ``req_to_device_buffer`` stays
        valid for the whole request lifetime (fast-path assumption:
        ``seq_len <= device_buffer_size``).

        ``cache_seqlens`` / ``cu_seqlens_k`` remain the baseline
        per-request true sequence lengths.
        """
        current_metadata.page_table.copy_(self._original_metadata["page_table"])
        current_metadata.cache_seqlens_int32.copy_(
            self._original_metadata["cache_seqlens_int32"]
        )
        # Restore cu_seqlens_k / max_seq_len_k too, in case a previous
        # decode step (or another adapter call) left them shrunk.  FA3
        # doesn't consult them for `flash_attn_with_kvcache` but keeping
        # the metadata self-consistent avoids surprises for downstream
        # consumers (e.g. profilers / debuggers).
        current_metadata.cu_seqlens_k = self._original_metadata["cu_seqlens_k"].clone()
        current_metadata.max_seq_len_k = self._original_metadata["max_seq_len_k"]

        io_subsystem = getattr(forward_batch, "hisparse_coordinator", None)
        if io_subsystem is None:
            return
        r2db = getattr(io_subsystem, "req_to_device_buffer", None)
        if r2db is None:
            return

        pt = current_metadata.page_table
        if pt.numel() == 0:
            return
        bs, pt_width = pt.shape
        r2db_width = r2db.shape[1]
        copy_w = min(pt_width, r2db_width)
        req_idx = forward_batch.req_pool_indices
        pt[:, :copy_w] = r2db[req_idx, :copy_w].to(pt.dtype)
        if copy_w < pt_width:
            pt[:, copy_w:] = 0

    # ------------------------------------------------------------------
    # Token-level write strategy (the only supported path for HiSparse +
    # non-native sparse algorithms on FA3; see docstring above).
    # ------------------------------------------------------------------
    def _write_token_level(
        self,
        selected_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        sparse_page_size: int,
        io_subsystem: Optional[Any],
        device: torch.device,
    ) -> None:
        """Token-level write into FA3 ``page_table`` (backend_page_size == 1).

        Expands the Quest-selected sparse pages to per-token logical
        positions, resolves them through ``req_to_token`` (and, when
        HiSparse is attached, ``translate_loc_to_hisparse_device``), and
        writes the compact list of physical KV slots into
        ``page_table[:, :valid_tokens]``.
        """
        bs, num_topk_pages = selected_indices.shape
        seq_lens = forward_batch.seq_lens.to(torch.int32)

        offsets = torch.arange(
            sparse_page_size, device=device, dtype=torch.int32
        )
        token_pos = (
            selected_indices.unsqueeze(-1).to(torch.int32) * sparse_page_size
            + offsets.view(1, 1, sparse_page_size)
        ).view(bs, num_topk_pages * sparse_page_size)
        page_valid = (
            (selected_indices >= 0)
            .unsqueeze(-1)
            .expand(-1, -1, sparse_page_size)
            .reshape(bs, num_topk_pages * sparse_page_size)
        )
        in_seq = token_pos < seq_lens.unsqueeze(1)
        tok_valid = page_valid & in_seq

        sort_key = (~tok_valid).to(torch.int8)
        order = torch.argsort(sort_key, dim=1, stable=True)
        token_pos_sorted = torch.gather(token_pos, 1, order)
        tok_valid_sorted = torch.gather(tok_valid, 1, order)

        req_idx = forward_batch.req_pool_indices.unsqueeze(1)
        token_pos_clamped = token_pos_sorted.clamp(min=0).to(torch.int64)

        if io_subsystem is not None:
            # Under HiSparse, look up the hot-buffer (hisparse-device)
            # slot of each selected token directly via the coordinator's
            # ``req_to_device_buffer[req, token_pos_in_req]`` table.  This
            # is the only authoritative mapping **after**
            # ``alloc_device_buffer`` clears the logical->hisparse
            # mapping; going through ``req_to_token`` +
            # ``translate_loc_to_hisparse_device`` would return 0 for
            # prefill tokens and FA3 would attend to an empty slot.
            r2db = io_subsystem.req_to_device_buffer  # [max_reqs, dbs+page]
            r2db_width = r2db.shape[1]
            token_pos_clamped = token_pos_clamped.clamp(max=r2db_width - 1)
            physical_slots = r2db[req_idx, token_pos_clamped]
        else:
            max_req_tokens = req_to_token.shape[1]
            token_pos_clamped = token_pos_clamped.clamp(max=max_req_tokens - 1)
            physical_slots = req_to_token[req_idx, token_pos_clamped]

        physical_slots = physical_slots.to(current_metadata.page_table.dtype)
        valid_tokens = tok_valid_sorted.sum(dim=1).to(torch.int32)

        T = token_pos_sorted.shape[1]
        pt_width = current_metadata.page_table.shape[1]
        if T > pt_width:
            T = pt_width
            physical_slots = physical_slots[:, :T]
            tok_valid_sorted = tok_valid_sorted[:, :T]

        tok_valid_sorted_ = tok_valid_sorted & sparse_mask.unsqueeze(1)
        current_metadata.page_table[:, :T] = torch.where(
            tok_valid_sorted_,
            physical_slots,
            torch.zeros_like(physical_slots),
        )
        if T < pt_width:
            current_metadata.page_table[:, T:] = 0

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

        left = getattr(self, "_debug_adapter_left", 6)
        if left > 0:
            self._debug_adapter_left = left - 1
            import logging as _logging
            _logging.getLogger("hisparse.debug").warning(
                "[HS-DBG Adapter] sparse_page_size=%d T=%d "
                "valid_tokens=%s cache_seqlens=%s max_k=%d "
                "page_table[0, :16]=%s physical_slots[0, :8]=%s "
                "token_pos_sorted[0, :8]=%s",
                int(sparse_page_size),
                int(T),
                valid_tokens.tolist(),
                current_metadata.cache_seqlens_int32.tolist(),
                int(current_metadata.max_seq_len_k),
                current_metadata.page_table[0, :16].tolist()
                if current_metadata.page_table.numel()
                else [],
                physical_slots[0, :8].tolist()
                if physical_slots.numel()
                else [],
                token_pos_sorted[0, :8].tolist()
                if token_pos_sorted.numel()
                else [],
            )
