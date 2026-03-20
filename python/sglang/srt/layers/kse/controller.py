"""Central coordinator for the KVCache Sparsity Engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.kse.base_adapter import MetadataAdapter
from sglang.srt.layers.kse.base_policy import EvictionPolicy, SparsityPolicy
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_BACKEND_GRANULARITY = {
    "flashinfer": Granularity.TOKEN,
    "triton": Granularity.TOKEN,
    "flashattention": Granularity.PAGE,
}

_BACKEND_PAGE_SIZE = {
    "flashattention": 256,
}


def _get_backend_granularity(backend_name: str) -> Granularity:
    return _BACKEND_GRANULARITY.get(backend_name, Granularity.TOKEN)


def _get_backend_page_size(backend_name: str) -> int:
    return _BACKEND_PAGE_SIZE.get(backend_name, 1)


class KSEController:
    """Orchestrates sparsity policy, metadata adapter, and eviction.

    Integration points (4 call-sites in sglang):
        1. After extend forward in ``ModelRunner._forward_raw``
           → ``controller.after_prefill()``
        2. Start of ``ModelRunner.forward_decode()``
           → ``controller.before_forward()``
        3. ``AttentionBackend.forward()`` before decode kernel
           → ``controller.before_attention()``
        4. ``AttentionBackend.forward()`` after decode kernel
           → ``controller.after_attention()``
    """

    def __init__(
        self,
        policy: SparsityPolicy,
        adapter: MetadataAdapter,
        eviction: Optional[EvictionPolicy],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        config: KSEConfig,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        self.policy = policy
        self.adapter = adapter
        self.eviction = eviction
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.config = config

        self._validate_granularity_compatibility(
            policy, config.backend_name, config
        )

        self._cached_result: Optional[SelectionResult] = None
        self._cached_step: int = -1
        self._metadata_saved: bool = False

    # ---- Hook: request lifecycle ----------------------------------------

    def on_request_begin(self, req) -> None:
        self.policy.on_request_begin(req)

    def on_request_end(self, req) -> None:
        self.policy.on_request_end(req)

    # ---- Hook: after prefill --------------------------------------------

    def after_prefill(self, forward_batch: ForwardBatch) -> None:
        """Build representations and optionally evict after prefill."""
        end_layer = (
            self.config.end_layer
            if self.config.end_layer > 0
            else self.token_to_kv_pool.layer_num
        )
        for layer_id in range(self.config.start_layer, end_layer):
            k_buf = self.token_to_kv_pool.get_key_buffer(layer_id)
            v_buf = self.token_to_kv_pool.get_value_buffer(layer_id)
            self.policy.on_prefill_complete(
                layer_id,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                k_buf,
                v_buf,
                forward_batch,
            )

        # Initial eviction after prefill (e.g. StreamingLLM trims the middle)
        if self.eviction is not None:
            self._do_eviction(forward_batch)

        if self.policy.frequency() == Frequency.PER_REQUEST:
            self._cached_result = self.policy.select(
                query=None,
                layer_id=-1,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_batch=forward_batch,
            )

    # ---- Hook: before decode forward ------------------------------------

    def before_forward(self, forward_batch: ForwardBatch) -> None:
        """Reset per-step caches and perform periodic eviction."""
        if self.policy.frequency() == Frequency.PER_STEP:
            self._cached_result = None
        self._metadata_saved = False

        # Notify the policy of a new step (for step counting)
        if hasattr(self.policy, "on_step"):
            self.policy.on_step()

        # Periodic eviction: policies with should_evict() control timing
        if self.eviction is not None and hasattr(self.eviction, "should_evict"):
            if self.eviction.should_evict():
                self._do_eviction(forward_batch)

    # ---- Hook: before each layer's attention ----------------------------

    def before_attention(
        self,
        query: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
        forward_metadata: Any,
    ) -> Any:
        """Rewrite metadata for sparse attention; return (possibly modified) metadata."""
        if not self._should_apply(layer_id, forward_batch):
            return forward_metadata

        if not self._metadata_saved:
            self.adapter.save_dense_metadata(forward_metadata)
            self._metadata_saved = True

        result = self._get_selection(query, layer_id, forward_batch)
        return self.adapter.apply(result, forward_metadata, forward_batch, layer_id)

    # ---- Hook: after each layer's attention -----------------------------

    def after_attention(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> None:
        """Incrementally update policy state after attention."""
        if not forward_batch.forward_mode.is_decode():
            return
        k_buf = self.token_to_kv_pool.get_key_buffer(layer_id)
        v_buf = self.token_to_kv_pool.get_value_buffer(layer_id)
        self.policy.on_attention_complete(
            layer_id,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            k_buf,
            v_buf,
            forward_batch,
        )

    # ---- internal -------------------------------------------------------

    def _get_selection(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> SelectionResult:
        freq = self.policy.frequency()
        if freq == Frequency.PER_REQUEST:
            assert self._cached_result is not None
            return self._cached_result
        if freq == Frequency.PER_STEP:
            if self._cached_result is None:
                self._cached_result = self.policy.select(
                    query=query,
                    layer_id=layer_id,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    forward_batch=forward_batch,
                )
            return self._cached_result
        # PER_LAYER — always recompute
        return self.policy.select(
            query=query,
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_batch=forward_batch,
        )

    def _should_apply(self, layer_id: int, forward_batch: ForwardBatch) -> bool:
        if not forward_batch.forward_mode.is_decode():
            return False
        end_layer = (
            self.config.end_layer
            if self.config.end_layer > 0
            else self.token_to_kv_pool.layer_num
        )
        if layer_id < self.config.start_layer or layer_id >= end_layer:
            return False
        return True

    def _do_eviction(self, forward_batch: ForwardBatch) -> None:
        """Evict tokens: compact req_to_token and free physical KV slots."""
        assert self.eviction is not None
        evict_indices = self.eviction.compute_eviction(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch,
        )
        batch_size = evict_indices.shape[0]
        req_to_token = self.req_to_token_pool.req_to_token

        all_freed_phys = []

        for i in range(batch_size):
            mask = evict_indices[i] >= 0
            if not mask.any():
                continue
            logical_pos = evict_indices[i][mask]
            req_idx = forward_batch.req_pool_indices[i]
            phys_indices = req_to_token[req_idx, logical_pos.long()]

            # Collect physical indices to free
            all_freed_phys.append(phys_indices)

            # Compact remaining tokens in req_to_token
            seq_len = forward_batch.seq_lens[i].item()
            all_positions = torch.arange(seq_len, device=evict_indices.device)
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=evict_indices.device)
            keep_mask[logical_pos.long()] = False
            kept_positions = all_positions[keep_mask]
            kept_phys = req_to_token[req_idx, kept_positions.long()]
            new_len = kept_phys.shape[0]
            req_to_token[req_idx, :new_len] = kept_phys
            forward_batch.seq_lens[i] = new_len

        # Free physical KV-cache slots via the allocator
        if all_freed_phys and self.token_to_kv_pool_allocator is not None:
            freed = torch.cat(all_freed_phys)
            self.token_to_kv_pool_allocator.free(freed)

    # ---- validation -----------------------------------------------------

    def _validate_granularity_compatibility(
        self,
        policy: SparsityPolicy,
        backend_name: str,
        config: KSEConfig,
    ) -> None:
        backend_gran = _get_backend_granularity(backend_name)
        policy_gran = policy.granularity()

        if (
            policy_gran == Granularity.TOKEN
            and backend_gran == Granularity.PAGE
        ):
            raise ValueError(
                f"Token-granularity sparse policy '{config.policy_name}' is "
                f"incompatible with page-granularity backend '{backend_name}'. "
                f"Use a token-granularity backend (FlashInfer, Triton) instead."
            )

        if (
            policy_gran == Granularity.PAGE
            and backend_gran == Granularity.PAGE
        ):
            backend_ps = _get_backend_page_size(backend_name)
            if config.page_size < backend_ps:
                raise ValueError(
                    f"Sparse page_size ({config.page_size}) < backend "
                    f"page_size ({backend_ps}). Must be >= and an integer multiple."
                )
            if config.page_size % backend_ps != 0:
                raise ValueError(
                    f"Sparse page_size ({config.page_size}) is not an integer "
                    f"multiple of backend page_size ({backend_ps})."
                )
