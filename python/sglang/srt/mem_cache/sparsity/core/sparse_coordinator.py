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

# Set SGLANG_SPARSE_DEBUG=1 in the env to turn on verbose per-step logs.
_SPARSE_DEBUG = os.environ.get("SGLANG_SPARSE_DEBUG", "0") == "1"

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class RequestTrackers:
    """State tracker for sparse attention requests."""

    def __init__(
        self,
        max_pool_size: int,
        device: torch.device,
        num_layers: int,
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

    def register(self, idx: int, prompt_len: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = prompt_len
        self.last_constructed_page[idx] = 0

    def clear(self, idx: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.last_constructed_page[idx] = 0


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    top_k: int = 2048
    device_buffer_size: int = 4096
    host_to_device_ratio: int = 2
    algorithm: Optional[str] = None
    backend: Optional[str] = None
    page_size: Optional[int] = None
    # NOTE: kept for backward-compat parsing only; runtime no longer consumes
    # this field.  Short requests auto-degenerate to dense via
    # ``valid_lengths = clamp(seq_lens, top_k)`` inside the algorithm.
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

        # Will be bound later by HiSparseCoordinator.__init__.
        self.io: Optional["HiSparseCoordinator"] = None

        max_pool_size = req_to_token_pool.req_to_token.shape[0]
        self.states = RequestTrackers(
            max_pool_size=max_pool_size,
            device=device,
            num_layers=end_layer - start_layer,
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

    def begin_layer_decode(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Any,
    ) -> Any:
        """Called by :class:`Fa3SparseDecodeBackend.forward_decode` at the
        beginning of every decode layer.

        1. Asks the algorithm for token-level top-k indices for the WHOLE
           batch (no per-request short-circuit; short requests degenerate
           to dense via ``valid_lengths = clamp(seq_lens, top_k)``).
        2. Delegates to :meth:`HiSparseCoordinator.swap_in_selected_pages`
           to fetch those tokens into the device hot buffer.
        3. Asks the adapter to package the per-layer FA3 sparse-call
           arguments (returned to the backend as ``SparseFa3CallArgs``).
        """
        if self.io is None:
            raise RuntimeError(
                "SparseAlgorithmController is not bound to a HiSparseCoordinator"
            )

        debug = _SPARSE_DEBUG and layer.layer_id == 0
        if debug:
            logger.info(
                "[SPARSE/ctl] L0 begin_layer_decode: bs=%d req_pool_indices=%s "
                "seq_lens=%s q.shape=%s q.dtype=%s",
                forward_batch.batch_size,
                forward_batch.req_pool_indices.cpu().tolist(),
                forward_batch.seq_lens.cpu().tolist(),
                tuple(q.shape),
                q.dtype,
            )

        top_k_tokens, valid_lengths = self.algorithm.retrieve_topk(
            queries=q,
            layer_id=layer.layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            forward_batch=forward_batch,
            attn_metadata=attn_metadata,
        )
        if debug:
            torch.cuda.synchronize()
            logger.info(
                "[SPARSE/ctl] L0 retrieve_topk OK: top_k_tokens.shape=%s "
                "dtype=%s valid_lengths=%s top_k_tokens.min=%d top_k_tokens.max=%d "
                "row0[:8]=%s row0[length:length+4]=%s",
                tuple(top_k_tokens.shape),
                top_k_tokens.dtype,
                valid_lengths.cpu().tolist(),
                int(top_k_tokens.min().item()),
                int(top_k_tokens.max().item()),
                top_k_tokens[0, :8].cpu().tolist(),
                top_k_tokens[
                    0,
                    int(valid_lengths[0].item()) : int(valid_lengths[0].item()) + 4,
                ].cpu().tolist(),
            )

        top_k_device_locs = self.io.swap_in_selected_pages(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            top_k_result=top_k_tokens,
            layer_id=layer.layer_id,
        )
        if debug:
            torch.cuda.synchronize()
            logger.info(
                "[SPARSE/ctl] L0 swap_in_selected_pages OK: "
                "top_k_device_locs.shape=%s min=%d max=%d row0[:8]=%s",
                tuple(top_k_device_locs.shape),
                int(top_k_device_locs.min().item()),
                int(top_k_device_locs.max().item()),
                top_k_device_locs[0, :8].cpu().tolist(),
            )

        sparse_args = self.adapter.build_sparse_call_args(
            top_k_device_locs=top_k_device_locs,
            valid_lengths=valid_lengths,
            dense_metadata=attn_metadata,
            forward_batch=forward_batch,
            layer_id=layer.layer_id,
        )
        if debug:
            torch.cuda.synchronize()
            logger.info("[SPARSE/ctl] L0 build_sparse_call_args OK")
        return sparse_args

    def end_layer_decode(
        self,
        o: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> None:
        """Called after the FA3 sparse kernel for each decode layer.

        The algorithm updates its representation pool with the new token
        (decoded in the previous step) if necessary.  The K buffer used here
        must cover the logical rows of the active request; see
        :meth:`_k_buffer_for_update` for the resolution.
        """
        if self.io is None:
            return
        layer_id = layer.layer_id
        debug = _SPARSE_DEBUG and layer_id == 0
        k_buffer = self._k_buffer_for_update(layer_id)
        if debug:
            torch.cuda.synchronize()
            logger.info(
                "[SPARSE/ctl] L0 end_layer_decode begin: o.shape=%s "
                "k_buffer.shape=%s req_pool_indices=%s seq_lens=%s "
                "states.last_constructed_page[req_pool_indices]=%s "
                "states.repr_constructed[req_pool_indices]=%s",
                tuple(o.shape),
                tuple(k_buffer.shape),
                forward_batch.req_pool_indices.cpu().tolist(),
                forward_batch.seq_lens.cpu().tolist(),
                self.states.last_constructed_page[
                    forward_batch.req_pool_indices
                ].cpu().tolist(),
                self.states.repr_constructed[
                    forward_batch.req_pool_indices
                ].cpu().tolist(),
            )
        self.algorithm.update_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=k_buffer,
            forward_batch=forward_batch,
        )
        if debug:
            torch.cuda.synchronize()
            logger.info("[SPARSE/ctl] L0 update_representations OK")

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
