"""Prefill-query ChunkKV selection for visibility-only sparse attention."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.contracts import (
    DecisionScope,
    Granularity,
    KVStateAction,
    SelectionContext,
    SelectionEvidence,
    SelectionResult,
    SparsityCapabilities,
)
from sglang.srt.mem_cache.sparsity.policies.base import SparsityPolicy

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


class ChunkKVPolicy(SparsityPolicy):
    """Select semantic pages once from final-prefill observation queries.

    Scores are accumulated across the configured sparse layers. Decode reuses
    the cached semantic pages and appends the current recent-page window, so it
    performs no query/key scoring. Excluded pages stay resident in HBM.
    """

    _CAPABILITIES = SparsityCapabilities(
        state_action=KVStateAction.VISIBILITY_ONLY,
        granularity=Granularity.PAGE,
        evidence=SelectionEvidence.CURRENT_QUERY,
        scope=DecisionScope.STEP,
        # Slot state is invalidated on every extend before it can be read by a
        # later decode. Avoiding per-token GPU->CPU identity synchronization is
        # important for the zero-rescore decode path.
        requires_request_state=False,
        supports_cuda_graph=True,
    )

    def __init__(
        self,
        config: KVSparsityConfig,
        device: torch.device,
        *,
        token_to_kv_pool: KVCache,
        req_to_token_pool: ReqToTokenPool,
        start_layer: int = 0,
        end_layer: int = 1,
    ):
        self.config = config
        self.device = device
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.page_size = config.page_size
        self.start_layer = start_layer
        self.end_layer = end_layer

        known_fields = {"page_budget", "recent_pages", "obs_window"}
        unknown = sorted(set(config.policy_config) - known_fields)
        if unknown:
            raise ValueError(f"Unknown ChunkKV policy field(s): {unknown}")
        self.page_budget = self._positive_int("page_budget", 128)
        self.recent_pages = self._positive_int("recent_pages", 4)
        self.obs_window = self._positive_int("obs_window", 32)
        if self.recent_pages >= self.page_budget:
            raise ValueError("ChunkKV recent_pages must be smaller than page_budget")
        if self.page_size < 2:
            raise ValueError("ChunkKV requires page_size >= 2")
        if self.page_budget > 1024:
            raise ValueError("ChunkKV page_budget must not exceed 1024")

        req_to_token = req_to_token_pool.req_to_token
        self.max_pages = (req_to_token.shape[1] + self.page_size - 1) // self.page_size
        self.history_budget = self.page_budget - self.recent_pages
        if self.max_pages < self.page_budget:
            raise ValueError("ChunkKV page_budget exceeds the model context capacity")
        num_request_slots = req_to_token.shape[0]
        self._score_pool = torch.empty(
            (num_request_slots, self.max_pages),
            dtype=torch.float32,
            device=device,
        )
        self._base_pages = torch.full(
            (num_request_slots, self.history_budget),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._state_valid = torch.zeros(
            num_request_slots, dtype=torch.bool, device=device
        )
        self._query_summary: torch.Tensor | None = None

    def _positive_int(self, name: str, default: int) -> int:
        value = self.config.policy_config.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"ChunkKV {name} must be a positive integer, got {value!r}"
            )
        return value

    @property
    def capabilities(self) -> SparsityCapabilities:
        return self._CAPABILITIES

    def _reshape_query(
        self, context: SelectionContext, num_kv_heads: int, head_dim: int
    ) -> torch.Tensor:
        query = context.query
        if query is None:
            raise ValueError("ChunkKV requires prefill observation queries")
        if query.ndim == 2:
            if query.shape[1] % head_dim:
                raise ValueError("ChunkKV query width is not divisible by head_dim")
            query = query.view(query.shape[0], -1, head_dim)
        if query.ndim != 3 or query.shape[2] != head_dim:
            raise ValueError("ChunkKV query must have shape [tokens, heads, dim]")
        if query.shape[1] % num_kv_heads:
            raise ValueError("ChunkKV query heads must be divisible by KV heads")
        return query

    def _extend_geometry(
        self, context: SelectionContext, query_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_batch = context.forward_batch
        extend_lens = getattr(forward_batch, "extend_seq_lens", None)
        extend_start = getattr(forward_batch, "extend_start_loc", None)
        if extend_lens is not None and extend_start is not None:
            return (
                extend_start.to(device=self.device, dtype=torch.int32),
                extend_lens.to(device=self.device, dtype=torch.int32),
            )
        if context.seq_lens.shape[0] != 1:
            raise ValueError("ChunkKV requires extend geometry for batched prefill")
        return (
            torch.zeros(1, dtype=torch.int32, device=self.device),
            torch.tensor([query_tokens], dtype=torch.int32, device=self.device),
        )

    def _active_final_prefill(self, context: SelectionContext) -> torch.Tensor:
        forward_batch = context.forward_batch
        original_lens = getattr(forward_batch, "orig_seq_lens", None)
        if original_lens is None:
            final_chunk = torch.ones_like(context.seq_lens, dtype=torch.bool)
        else:
            final_chunk = context.seq_lens == original_lens[: context.seq_lens.shape[0]]
        total_pages = torch.div(
            context.seq_lens + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        return (
            final_chunk
            & (context.seq_lens >= self.config.min_sparse_tokens)
            & (total_pages > self.page_budget)
        )

    def _score_portable(
        self,
        *,
        context: SelectionContext,
        query: torch.Tensor,
        key_buffer: torch.Tensor,
        extend_start: torch.Tensor,
        extend_lens: torch.Tensor,
        active: torch.Tensor,
        reset: bool,
    ) -> None:
        req_to_token = self.req_to_token_pool.req_to_token
        num_kv_heads, head_dim = key_buffer.shape[1:]
        group_size = query.shape[1] // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)
        for batch_idx in range(context.seq_lens.shape[0]):
            if not bool(active[batch_idx]):
                continue
            req_idx = int(context.req_pool_indices[batch_idx])
            seq_len = int(context.seq_lens[batch_idx])
            start = int(extend_start[batch_idx])
            extend_len = int(extend_lens[batch_idx])
            obs_len = min(self.obs_window, extend_len)
            q_obs = query[start + extend_len - obs_len : start + extend_len]
            q_summary = (
                q_obs.float()
                .view(obs_len, num_kv_heads, group_size, head_dim)
                .mean(dim=(0, 2))
            )
            total_pages = (seq_len + self.page_size - 1) // self.page_size
            history_end = total_pages - self.recent_pages
            if reset:
                self._score_pool[req_idx].fill_(-float("inf"))
            for page_idx in range(history_end):
                start_token = page_idx * self.page_size
                end_token = min(start_token + self.page_size, seq_len)
                physical = req_to_token[req_idx, start_token:end_token].long()
                key_sum = key_buffer[physical].float().sum(dim=0)
                page_score = (q_summary * key_sum).sum() * scale
                if reset:
                    self._score_pool[req_idx, page_idx] = page_score
                else:
                    self._score_pool[req_idx, page_idx] += page_score

    def _finalize_prefill_state(
        self, context: SelectionContext, active: torch.Tensor
    ) -> None:
        req_indices = context.req_pool_indices.long()
        score_rows = self._score_pool[req_indices]
        _, selected = torch.topk(
            score_rows,
            k=self.history_budget,
            dim=1,
            largest=True,
            sorted=False,
        )
        selected = selected.sort(dim=1).values.to(torch.int32)
        previous = self._base_pages[req_indices]
        self._base_pages[req_indices] = torch.where(
            active.unsqueeze(1), selected, previous
        )
        self._state_valid[req_indices] = active

    def on_attention_complete(self, context: SelectionContext) -> None:
        forward_batch = context.forward_batch
        if forward_batch is None:
            raise ValueError("ChunkKV requires ForwardBatch lifecycle context")
        if forward_batch.forward_mode.is_decode():
            return

        req_indices = context.req_pool_indices.long()
        if context.layer_id == self.start_layer:
            # This also prevents a reused request-pool slot from observing the
            # previous generation's selection before final prefill finishes.
            self._state_valid[req_indices] = False

        key_buffer = self.token_to_kv_pool.get_key_buffer(context.layer_id)
        if key_buffer.ndim != 3:
            raise ValueError(
                "ChunkKV requires an NHD key buffer with shape [slots, heads, dim]"
            )
        query = self._reshape_query(context, key_buffer.shape[1], key_buffer.shape[2])
        extend_start, extend_lens = self._extend_geometry(context, query.shape[0])
        active = self._active_final_prefill(context)
        reset = context.layer_id == self.start_layer

        use_fused = (
            query.is_cuda
            and torch.version.hip is None
            and self.page_size <= 128
            and self.obs_window <= 128
            and key_buffer.shape[1] * key_buffer.shape[2] <= 4096
        )
        if use_fused:
            from sglang.srt.mem_cache.sparsity.kernels.chunkkv_score import (
                chunkkv_accumulate_page_scores_,
            )

            summary_shape = (
                context.seq_lens.shape[0],
                key_buffer.shape[1],
                key_buffer.shape[2],
            )
            if (
                self._query_summary is None
                or self._query_summary.shape != summary_shape
            ):
                self._query_summary = torch.empty(
                    summary_shape, dtype=torch.float32, device=self.device
                )
            chunkkv_accumulate_page_scores_(
                query=query,
                extend_start=extend_start,
                extend_lens=extend_lens,
                active_mask=active,
                req_pool_indices=context.req_pool_indices,
                seq_lens=context.seq_lens,
                req_to_token=self.req_to_token_pool.req_to_token,
                key_buffer=key_buffer,
                score_pool=self._score_pool,
                page_size=self.page_size,
                recent_pages=self.recent_pages,
                obs_window=self.obs_window,
                reset=reset,
                query_summary=self._query_summary,
            )
        else:
            self._score_portable(
                context=context,
                query=query,
                key_buffer=key_buffer,
                extend_start=extend_start,
                extend_lens=extend_lens,
                active=active,
                reset=reset,
            )

        if context.layer_id == self.end_layer - 1:
            self._finalize_prefill_state(context, active)

    def select(self, context: SelectionContext) -> SelectionResult:
        if context.seq_lens.ndim != 1:
            raise ValueError("seq_lens must have shape [batch]")
        if context.seq_lens.is_cuda and torch.version.hip is None:
            from sglang.srt.mem_cache.sparsity.kernels.chunkkv_select import (
                chunkkv_select,
            )

            selected, valid_lengths, visible_lens, sparse_mask = chunkkv_select(
                context.seq_lens,
                context.req_pool_indices,
                self._base_pages,
                self._state_valid,
                page_size=self.page_size,
                page_budget=self.page_budget,
                recent_pages=self.recent_pages,
                min_sparse_tokens=self.config.min_sparse_tokens,
            )
        else:
            req_indices = context.req_pool_indices.long()
            total_pages = torch.div(
                context.seq_lens + self.page_size - 1,
                self.page_size,
                rounding_mode="floor",
            )
            sparse_mask = (
                self._state_valid[req_indices]
                & (context.seq_lens >= self.config.min_sparse_tokens)
                & (total_pages > self.page_budget)
            )
            recent_offsets = torch.arange(self.recent_pages, device=self.device)
            recent = (
                total_pages.unsqueeze(1)
                - self.recent_pages
                + recent_offsets.unsqueeze(0)
            )
            selected = torch.cat((self._base_pages[req_indices], recent), dim=1)
            selected = torch.where(
                sparse_mask.unsqueeze(1), selected, torch.full_like(selected, -1)
            ).to(torch.int32)
            valid_lengths = torch.where(
                sparse_mask,
                torch.full_like(total_pages, self.page_budget, dtype=torch.int32),
                torch.zeros_like(total_pages, dtype=torch.int32),
            )
            padding = total_pages * self.page_size - context.seq_lens
            visible_lens = torch.where(
                sparse_mask,
                (self.page_budget * self.page_size - padding).to(torch.int32),
                torch.zeros_like(total_pages, dtype=torch.int32),
            )

        return SelectionResult(
            granularity=Granularity.PAGE,
            logical_indices=selected,
            valid_lengths=valid_lengths,
            visible_kv_lens=visible_lens,
            sparse_mask=sparse_mask,
            layer_id=None,
            max_visible_kv_len=max(
                self.page_budget * self.page_size,
                max(self.config.min_sparse_tokens - 1, 1),
            ),
        )
