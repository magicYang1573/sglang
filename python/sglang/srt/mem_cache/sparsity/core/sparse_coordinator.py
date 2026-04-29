"""Inner controller that couples a sparse algorithm with a backend adapter.

Historical note
---------------
The file used to host a ``SparseCoordinator`` class that was designed to run
side-by-side with ``HiSparseCoordinator`` as a second, parallel protocol.
The new sparse-decode path collapses that split: ``HiSparseCoordinator``
remains the single external entry point for hierarchical sparse attention,
and the algorithm / adapter logic is injected through a
:class:`SparseAlgorithmController` instance held by the coordinator.

For backward compatibility the module keeps the ``SparseConfig`` and
``RequestTrackers`` dataclasses (they are reused by the factory parser).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class RequestTrackers:
    """State tracker for sparse attention requests."""

    def __init__(
        self,
        max_pool_size: int,
        device: torch.device,
        num_layers: int,
        min_sparse_prompt_len: int,
        max_context_len: int,
    ):
        self.device = device
        self.num_layers = num_layers

        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )
        # CPU mirror of ``prompt_lens`` keyed by req_pool_idx.  Used by the
        # decode hot path to decide sparse-vs-dense without syncing GPU state
        # (see ``SparseAlgorithmController.compute_all_dense_flag``).  Kept in
        # sync by :meth:`register` / :meth:`clear`.
        self.prompt_lens_cpu: dict = {}

    def register(self, idx: int, prompt_len: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = prompt_len
        self.last_constructed_page[idx] = 0
        self.prompt_lens_cpu[int(idx)] = int(prompt_len)

    def clear(self, idx: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.last_constructed_page[idx] = 0
        self.prompt_lens_cpu.pop(int(idx), None)


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    top_k: int = 2048
    device_buffer_size: int = 4096
    host_to_device_ratio: int = 2
    algorithm: Optional[str] = None
    backend: Optional[str] = None
    page_size: Optional[int] = None
    min_sparse_prompt_len: Optional[int] = None
    sparse_extra_config: dict = field(default_factory=dict)


class SparseAlgorithmController:
    """Inner module of :class:`HiSparseCoordinator` that composes
    ``(sparse algorithm, backend adapter)`` for non-native sparse decode.

    The controller does NOT own the host pool, the device hot buffer, or the
    swap-in kernel; those live on the parent ``HiSparseCoordinator`` and are
    reached through :attr:`io`.

    Public hooks are called in three groups:

    * forward lifecycle: :meth:`before_decode_step`
    * per-layer lifecycle: :meth:`begin_layer_decode` / :meth:`end_layer_decode`
    * request lifecycle: :meth:`on_request_begin` / :meth:`on_prefill_finished`
      / :meth:`on_request_end`
    """

    def __init__(
        self,
        config: SparseConfig,
        algorithm: BaseSparseAlgorithm,
        adapter: BackendAdaptor,
        req_to_token_pool: ReqToTokenPool,
        device: torch.device,
        start_layer: int,
        end_layer: int,
    ):
        self.config = config
        self.algorithm = algorithm
        self.adapter = adapter
        self.req_to_token_pool = req_to_token_pool
        self.device = device
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._debug_enabled = _env_int("SGLANG_SPARSE_DECODE_DEBUG", 0) > 0
        self._debug_max_logs = _env_int("SGLANG_SPARSE_DECODE_DEBUG_MAX_LOGS", 20)
        self._debug_log_count = 0

        # Will be bound later by HiSparseCoordinator.__init__.
        self.io: Optional["HiSparseCoordinator"] = None

        max_pool_size = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        self.states = RequestTrackers(
            max_pool_size=max_pool_size,
            device=device,
            num_layers=end_layer - start_layer,
            min_sparse_prompt_len=(config.min_sparse_prompt_len or 0),
            max_context_len=max_context_len,
        )

    def bind_io(self, io: "HiSparseCoordinator") -> None:
        """Called once by :class:`HiSparseCoordinator` after construction."""
        self.io = io
        # The algorithm's representation pool is sized against the device KV
        # pool held by the coordinator (same physical rows that FA3 sees
        # during prefill).
        self.algorithm.initialize_representation_pool(
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            token_to_kv_pool=io.mem_pool_device,
            req_to_token_pool=self.req_to_token_pool,
            states=self.states,
        )

    def on_request_begin(self, req: "Req") -> None:
        if req.req_pool_idx is None:
            return
        self.states.register(req.req_pool_idx, len(req.origin_input_ids))

    def on_prefill_finished(self, req: "Req") -> None:
        """Invoked by :meth:`HiSparseCoordinator.admit_request_from_gpu` after
        the coordinator has migrated full KV to the host pool.

        The algorithm representations for this request should already be
        populated by :meth:`construct_representations` (triggered inside
        :meth:`HiSparseCoordinator.admit_request_from_gpu`); extra bookkeeping
        by subclasses (e.g. caching min/max in GPU) can happen here.
        """
        return

    def on_request_end(self, req: "Req") -> None:
        if req.req_pool_idx is None:
            return
        self.states.clear(req.req_pool_idx)

    def before_decode_step(self, forward_batch: "ForwardBatch") -> None:
        """Per-decode-step pre-forward hook.  Currently a no-op."""
        return

    def compute_all_dense_flag(self, req_pool_indices_cpu) -> bool:
        """Return True if no request in the batch needs sparse attention.

        Purely CPU-side (reads ``RequestTrackers.prompt_lens_cpu``) so that
        the decode hot path can short-circuit the sparse pipeline without
        triggering a GPU sync.  Called once per decode step by
        :meth:`HiSparseCoordinator.map_last_loc_to_buffer` and cached on the
        coordinator for every attention layer's
        :meth:`begin_layer_decode`.
        """
        min_len = self.config.min_sparse_prompt_len
        if min_len is None:
            # No threshold configured → every request goes through sparse.
            return False
        prompt_lens_cpu = self.states.prompt_lens_cpu
        for req_idx in req_pool_indices_cpu:
            prompt_len = prompt_lens_cpu.get(int(req_idx), 0)
            if prompt_len >= min_len:
                return False
        return True

    def begin_layer_decode(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Any,
    ) -> Any:
        """Called by :class:`Fa3SparseDecodeBackend.forward_decode` at the
        beginning of every decode layer.

        1. On the first layer of the step, saves a snapshot of the dense
           metadata so later layers can revert.
        2. Asks the algorithm for token-level top-k indices.
        3. Delegates to :meth:`HiSparseCoordinator.swap_in_selected_pages` to
           fetch those tokens into the device hot buffer.
        4. Asks the adapter to rewrite the attention metadata so that FA3
           reads only the newly swapped-in slots.
        """
        if self.io is None:
            raise RuntimeError(
                "SparseAlgorithmController is not bound to a HiSparseCoordinator"
            )

        if layer.layer_id == self.start_layer:
            self.adapter.save_original_metadata(attn_metadata)

        # Dense fallback: every request attends to all tokens, but the KV rows
        # still live in the HiSparse hot buffer after prefill admission /
        # decode writes.  Therefore FA3 must see translated physical rows, not
        # the original logical ``req_to_token`` rows.
        if getattr(self.io, "_all_dense_this_step", False):
            attn_metadata = self._translate_dense_metadata_for_hisparse(
                attn_metadata, forward_batch
            )
            self._maybe_log_all_dense_debug(
                layer_id=layer.layer_id,
                forward_batch=forward_batch,
                attn_metadata=attn_metadata,
            )
            return attn_metadata

        sparse_mask = self._compute_sparse_mask(forward_batch.req_pool_indices)

        top_k_tokens, valid_lengths = self.algorithm.retrieve_topk(
            queries=q,
            layer_id=layer.layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
            attn_metadata=attn_metadata,
        )

        top_k_device_locs = self.io.swap_in_selected_pages(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            top_k_result=top_k_tokens,
            layer_id=layer.layer_id,
        )

        self._maybe_log_decode_debug(
            layer_id=layer.layer_id,
            forward_batch=forward_batch,
            sparse_mask=sparse_mask,
            top_k_tokens=top_k_tokens,
            valid_lengths=valid_lengths,
            top_k_device_locs=top_k_device_locs,
        )

        return self.adapter.adapt_for_attn_metadata(
            selected_indices=top_k_device_locs,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=attn_metadata,
            forward_batch=forward_batch,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_size=(self.config.page_size or 1),
            layer_id=layer.layer_id,
        )

    def _translate_dense_metadata_for_hisparse(
        self,
        attn_metadata: Any,
        forward_batch: "ForwardBatch",
    ) -> Any:
        page_table = getattr(attn_metadata, "page_table", None)
        if page_table is None or self.io is None:
            return attn_metadata

        mapping = self.io.mem_pool_device.full_to_hisparse_device_index_mapping
        max_seq_len = page_table.shape[1]
        translated = mapping[page_table.to(mapping.device)].to(page_table.device)
        translated = torch.where(translated > 0, translated, page_table)

        cols = torch.arange(max_seq_len, device=page_table.device).view(1, -1)
        valid = cols < forward_batch.seq_lens.to(page_table.device).view(-1, 1)
        page_table.copy_(torch.where(valid, translated.to(page_table.dtype), page_table))
        return attn_metadata

    def _maybe_log_decode_debug(
        self,
        *,
        layer_id: int,
        forward_batch: "ForwardBatch",
        sparse_mask: torch.Tensor,
        top_k_tokens: torch.Tensor,
        valid_lengths: torch.Tensor,
        top_k_device_locs: torch.Tensor,
    ) -> None:
        if (
            not self._debug_enabled
            or self._debug_log_count >= self._debug_max_logs
            or top_k_tokens.shape[0] == 0
        ):
            return

        row = 0
        valid_len = int(valid_lengths[row].item())
        prefix_len = min(valid_len, 16)
        token_prefix = (
            top_k_tokens[row, :prefix_len].detach().cpu().tolist()
            if prefix_len > 0
            else []
        )
        loc_prefix = (
            top_k_device_locs[row, :prefix_len].detach().cpu().tolist()
            if prefix_len > 0
            else []
        )
        invalid_in_valid = int((top_k_device_locs[row, :valid_len] < 0).sum().item())
        logger.warning(
            "SparseDecodeDebug Controller #%d: layer=%d bs=%d "
            "row0(req_pool=%d seq_len=%d sparse=%s valid_len=%d "
            "invalid_device_locs_in_valid=%d token_prefix=%s device_loc_prefix=%s)",
            self._debug_log_count,
            layer_id,
            top_k_tokens.shape[0],
            int(forward_batch.req_pool_indices[row].item()),
            int(forward_batch.seq_lens[row].item()),
            bool(sparse_mask[row].item()),
            valid_len,
            invalid_in_valid,
            token_prefix,
            loc_prefix,
        )
        self._debug_log_count += 1

    def _maybe_log_all_dense_debug(
        self,
        *,
        layer_id: int,
        forward_batch: "ForwardBatch",
        attn_metadata: Any,
    ) -> None:
        if (
            not self._debug_enabled
            or self._debug_log_count >= self._debug_max_logs
            or forward_batch.req_pool_indices.shape[0] == 0
            or self.io is None
        ):
            return

        row = 0
        req_idx = int(forward_batch.req_pool_indices[row].item())
        seq_len = int(forward_batch.seq_lens[row].item())
        prefix_len = min(seq_len, 16)
        suffix_len = min(seq_len, 16)

        req_to_token = self.req_to_token_pool.req_to_token
        logical_prefix = (
            req_to_token[req_idx, :prefix_len].detach().cpu().tolist()
            if prefix_len > 0
            else []
        )
        logical_suffix = (
            req_to_token[req_idx, seq_len - suffix_len : seq_len]
            .detach()
            .cpu()
            .tolist()
            if suffix_len > 0
            else []
        )
        hot_prefix = (
            self.io.req_to_device_buffer[req_idx, :prefix_len].detach().cpu().tolist()
            if prefix_len > 0
            else []
        )
        hot_suffix = (
            self.io.req_to_device_buffer[req_idx, seq_len - suffix_len : seq_len]
            .detach()
            .cpu()
            .tolist()
            if suffix_len > 0 and seq_len <= self.io.device_buffer_size
            else []
        )

        page_table = getattr(attn_metadata, "page_table", None)
        page_table_prefix = (
            page_table[row, :prefix_len].detach().cpu().tolist()
            if page_table is not None and prefix_len > 0
            else []
        )
        page_table_suffix = (
            page_table[row, seq_len - suffix_len : seq_len].detach().cpu().tolist()
            if page_table is not None and suffix_len > 0
            else []
        )
        cache_seqlen = getattr(attn_metadata, "cache_seqlens_int32", None)
        cache_seqlen_value = (
            int(cache_seqlen[row].item()) if cache_seqlen is not None else -1
        )

        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        out_loc = int(out_cache_loc[row].item()) if out_cache_loc is not None else -1
        mapped_out_loc = -1
        if out_loc >= 0:
            mapped_out_loc = int(
                self.io.mem_pool_device.full_to_hisparse_device_index_mapping[
                    out_loc
                ].item()
            )

        logger.warning(
            "SparseDecodeDebug AllDense #%d: layer=%d bs=%d "
            "row0(req_pool=%d seq_len=%d prompt_len=%d cache_seqlen=%d "
            "out_cache_loc=%d mapped_out_cache_loc=%d device_buffer_size=%d "
            "logical_prefix=%s hot_prefix=%s page_table_prefix=%s "
            "logical_suffix=%s hot_suffix=%s page_table_suffix=%s)",
            self._debug_log_count,
            layer_id,
            forward_batch.req_pool_indices.shape[0],
            req_idx,
            seq_len,
            self.states.prompt_lens_cpu.get(req_idx, -1),
            cache_seqlen_value,
            out_loc,
            mapped_out_loc,
            self.io.device_buffer_size,
            logical_prefix,
            hot_prefix,
            page_table_prefix,
            logical_suffix,
            hot_suffix,
            page_table_suffix,
        )
        self._debug_log_count += 1

    def end_layer_decode(
        self,
        o: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> None:
        """Called after ``super().forward_decode`` for each decode layer.

        The algorithm updates its representation pool with the new token
        (decoded in the previous step) if necessary.  The K buffer used here
        must cover the logical rows of the active request; see
        :meth:`_k_buffer_for_update` for the resolution.
        """
        if self.io is None:
            return
        # Paired with the short-circuit in :meth:`begin_layer_decode`: when
        # the whole batch stayed dense we also skip representation updates
        # (nothing selected this step, nothing new to track).
        if getattr(self.io, "_all_dense_this_step", False):
            return
        layer_id = layer.layer_id
        k_buffer = self._k_buffer_for_update(layer_id)
        self.algorithm.update_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=k_buffer,
            forward_batch=forward_batch,
        )

    def construct_prefill_representations(
        self,
        req: "Req",
        prefill_seq_lens: torch.Tensor,
        prefill_req_pool_indices: torch.Tensor,
    ) -> None:
        """Trigger one-shot, per-layer representation construction for a
        request that just finished prefill.

        Called by :meth:`HiSparseCoordinator.admit_request_from_gpu` BEFORE
        the logical rows are migrated to host.
        """
        if self.io is None:
            return

        class _FakeFwdBatch:
            """Minimal stand-in for ForwardBatch used by algorithm code."""

            def __init__(self, seq_lens: torch.Tensor):
                from sglang.srt.model_executor.forward_batch_info import ForwardMode

                self.forward_mode = ForwardMode.EXTEND
                self.seq_lens = seq_lens

        fake_fb = _FakeFwdBatch(prefill_seq_lens)
        for layer_id in range(self.start_layer, self.end_layer):
            k_buffer = self.io.mem_pool_device.get_key_buffer(layer_id)
            self.algorithm.construct_representations(
                layer_id=layer_id,
                req_pool_indices=prefill_req_pool_indices,
                seq_lens=prefill_seq_lens,
                k_buffer=k_buffer,
                forward_batch=fake_fb,
            )

    def _compute_sparse_mask(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        if self.config.min_sparse_prompt_len is None:
            # All requests get sparse treatment.
            return torch.ones(
                req_pool_indices.shape,
                dtype=torch.bool,
                device=req_pool_indices.device,
            )
        return (
            self.states.prompt_lens[req_pool_indices]
            >= self.config.min_sparse_prompt_len
        )

    def _k_buffer_for_update(self, layer_id: int) -> torch.Tensor:
        """Resolve where the K buffer lives for incremental representation
        updates during decode.

        In sparse-decode mode the *full* KV is on host after admit; but for
        the update step the algorithm only cares about pages that still have
        valid keys on GPU (recent tail).  HiSparseCoordinator keeps those in
        the hot buffer, whose backing tensor is the same physical
        ``HiSparseMHATokenToKVPool`` object.  We therefore return the full
        pool's K buffer and let the algorithm skip out-of-range pages via
        its own ``last_constructed_page`` state.
        """
        if self.io is None:
            raise RuntimeError("controller.io not bound")
        return self.io.mem_pool_device.get_key_buffer(layer_id)


__all__ = ["RequestTrackers", "SparseConfig", "SparseAlgorithmController"]
