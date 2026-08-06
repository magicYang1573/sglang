"""Quest query-aware page selection for the visibility-only runtime."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _QuestForwardPlan:
    """Layer-independent page geometry prepared once per model forward."""

    forward_batch: object
    max_pages: int
    total_pages: torch.Tensor
    sparse_mask: torch.Tensor
    physical_pages: torch.Tensor
    history_end: torch.Tensor
    recent_pages: torch.Tensor
    recent_valid: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class _QuestFA3SelectionResult(SelectionResult):
    """Deferred exact Quest selection consumed by the fused FA3 adaptor path.

    The public ``SelectionResult`` tensors keep their original shapes and
    meanings.  They are filled on the same CUDA stream when the adaptor maps
    the exact top-k/recent union directly into the FA3 page table.
    """

    topk_scores: torch.Tensor
    topk_indices: torch.Tensor
    k_per_req: torch.Tensor
    recent_indices: torch.Tensor
    recent_valid: torch.Tensor


class QuestPolicy(SparsityPolicy):
    """Select pages using Quest key bounding-box upper bounds.

    The policy keeps backend-independent logical page output.  On NVIDIA CUDA,
    scoring, complete-page updates, selection finalization, and the downstream
    FA3 metadata rewrite use optional Triton kernels derived from the efficient
    implementation in sgl-project/sglang#30277's four-part series.
    """

    _CAPABILITIES = SparsityCapabilities(
        state_action=KVStateAction.VISIBILITY_ONLY,
        granularity=Granularity.PAGE,
        evidence=SelectionEvidence.CURRENT_QUERY,
        scope=DecisionScope.LAYER,
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

        known_fields = {
            "page_budget",
            "recent_pages",
            "cuda_graph_context_buckets",
        }
        unknown = sorted(set(config.policy_config) - known_fields)
        if unknown:
            raise ValueError(f"Unknown Quest policy field(s): {unknown}")
        self.page_budget = self._positive_int("page_budget", 128)
        self.recent_pages = self._positive_int("recent_pages", 4)
        if self.recent_pages >= self.page_budget:
            raise ValueError("Quest recent_pages must be smaller than page_budget")
        if self.page_size < 2:
            raise ValueError("Quest requires page_size >= 2")

        self._page_k_min: dict[int, torch.Tensor] = {}
        self._page_k_max: dict[int, torch.Tensor] = {}
        self._page_valid: dict[int, torch.Tensor] = {}
        self._last_completed_page = torch.zeros(
            req_to_token_pool.req_to_token.shape[0],
            dtype=torch.int32,
            device=device,
        )
        self._forward_plan: _QuestForwardPlan | None = None

        # Allocate representation pools before CUDA Graph capture so their
        # addresses and memory cost are stable for the server lifetime.
        for layer_id in range(start_layer, end_layer):
            self._ensure_page_store(layer_id, token_to_kv_pool.get_key_buffer(layer_id))

    def _positive_int(self, name: str, default: int) -> int:
        value = self.config.policy_config.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Quest {name} must be a positive integer, got {value!r}")
        return value

    @property
    def capabilities(self) -> SparsityCapabilities:
        return self._CAPABILITIES

    def _ensure_page_store(self, layer_id: int, key_buffer: torch.Tensor) -> None:
        if layer_id in self._page_k_min:
            return
        if key_buffer.ndim != 3:
            raise ValueError(
                "Quest requires an NHD key buffer with shape [slots, heads, dim]"
            )
        num_physical_pages = (
            key_buffer.shape[0] + self.page_size - 1
        ) // self.page_size
        shape = (num_physical_pages, key_buffer.shape[1], key_buffer.shape[2])
        self._page_k_min[layer_id] = torch.empty(
            shape, dtype=key_buffer.dtype, device=key_buffer.device
        )
        self._page_k_max[layer_id] = torch.empty_like(self._page_k_min[layer_id])
        self._page_valid[layer_id] = torch.zeros(
            num_physical_pages, dtype=torch.bool, device=key_buffer.device
        )

    def _dense_page_width(self, context: SelectionContext) -> int:
        forward_batch = context.forward_batch
        fixed_capacity = getattr(forward_batch, "kv_sparsity_page_capacity", None)
        if fixed_capacity is not None:
            return int(fixed_capacity)
        metadata = context.metadata
        if metadata is not None and hasattr(metadata, "page_table"):
            return metadata.page_table.shape[1]
        max_tokens = int(context.seq_lens.max().item())
        return (max_tokens + self.page_size - 1) // self.page_size

    def _build_forward_plan(self, context: SelectionContext) -> _QuestForwardPlan:
        max_pages = self._dense_page_width(context)
        total_pages = torch.div(
            context.seq_lens + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        sparse_mask = (context.seq_lens >= self.config.min_sparse_tokens) & (
            total_pages > self.page_budget
        )
        logical_pages = torch.arange(max_pages, device=self.device)
        page_valid = logical_pages.unsqueeze(0) < total_pages.unsqueeze(1)
        req_to_token = self.req_to_token_pool.req_to_token
        page_starts = (logical_pages * self.page_size).clamp(
            max=req_to_token.shape[1] - 1
        )
        physical_tokens = req_to_token[
            context.req_pool_indices.long().unsqueeze(1), page_starts.unsqueeze(0)
        ].long()
        physical_pages = torch.div(
            physical_tokens, self.page_size, rounding_mode="floor"
        )
        physical_pages = torch.where(
            page_valid, physical_pages, torch.full_like(physical_pages, -1)
        )
        history_end = (total_pages - self.recent_pages).clamp(min=0)
        recent_offsets = torch.arange(self.recent_pages, device=self.device)
        recent_pages = history_end.unsqueeze(1) + recent_offsets.unsqueeze(0)
        recent_valid = sparse_mask.unsqueeze(1) & (
            recent_pages < total_pages.unsqueeze(1)
        )
        return _QuestForwardPlan(
            forward_batch=context.forward_batch,
            max_pages=max_pages,
            total_pages=total_pages,
            sparse_mask=sparse_mask,
            physical_pages=physical_pages,
            history_end=history_end,
            recent_pages=recent_pages,
            recent_valid=recent_valid,
        )

    def begin_forward(self, context: SelectionContext) -> None:
        self._forward_plan = self._build_forward_plan(context)

    def _rebuild_request_pages(
        self, context: SelectionContext, key_buffer: torch.Tensor
    ) -> None:
        """Build every currently visible page after an extend/prefill layer."""
        batch_size = context.seq_lens.shape[0]
        max_pages = self._dense_page_width(context)
        if batch_size == 0 or max_pages == 0:
            return

        req_to_token = self.req_to_token_pool.req_to_token
        page_ids = torch.arange(max_pages, device=self.device)
        offsets = torch.arange(self.page_size, device=self.device)
        positions = page_ids.view(1, -1, 1) * self.page_size + offsets.view(1, 1, -1)
        positions = positions.expand(batch_size, -1, -1)
        token_valid = positions < context.seq_lens.view(-1, 1, 1)
        safe_positions = positions.clamp(max=req_to_token.shape[1] - 1).long()
        rows = context.req_pool_indices.view(-1, 1, 1).expand_as(safe_positions)
        physical_tokens = req_to_token[rows, safe_positions].long()
        safe_physical_tokens = physical_tokens.clamp(0, key_buffer.shape[0] - 1)
        keys = key_buffer[safe_physical_tokens]
        extrema_mask = token_valid.unsqueeze(-1).unsqueeze(-1)
        page_min = torch.where(
            extrema_mask, keys, torch.full_like(keys, float("inf"))
        ).amin(dim=2)
        page_max = torch.where(
            extrema_mask, keys, torch.full_like(keys, float("-inf"))
        ).amax(dim=2)
        page_valid = page_ids.unsqueeze(0) < torch.div(
            context.seq_lens.view(-1, 1) + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        physical_pages = torch.div(
            physical_tokens[:, :, 0], self.page_size, rounding_mode="floor"
        ).clamp(0, self._page_k_min[context.layer_id].shape[0] - 1)
        valid_physical_pages = physical_pages[page_valid]
        layer_id = context.layer_id
        self._page_k_min[layer_id][valid_physical_pages] = page_min[page_valid]
        self._page_k_max[layer_id][valid_physical_pages] = page_max[page_valid]
        self._page_valid[layer_id][valid_physical_pages] = True
        if layer_id == self.end_layer - 1:
            complete_pages = torch.div(
                context.seq_lens, self.page_size, rounding_mode="floor"
            ).to(self._last_completed_page.dtype)
            self._last_completed_page[context.req_pool_indices.long()] = complete_pages

    def _update_last_page_portable(
        self, context: SelectionContext, key_buffer: torch.Tensor
    ) -> None:
        """Portable fallback; the CUDA path only updates newly complete pages."""
        req_to_token = self.req_to_token_pool.req_to_token
        last_pages = torch.div(
            context.seq_lens - 1, self.page_size, rounding_mode="floor"
        )
        offsets = torch.arange(self.page_size, device=self.device).unsqueeze(0)
        positions = last_pages.unsqueeze(1) * self.page_size + offsets
        token_valid = positions < context.seq_lens.unsqueeze(1)
        safe_positions = positions.clamp(max=req_to_token.shape[1] - 1).long()
        rows = context.req_pool_indices.unsqueeze(1).expand_as(safe_positions)
        physical_tokens = req_to_token[rows, safe_positions].long()
        safe_physical_tokens = physical_tokens.clamp(0, key_buffer.shape[0] - 1)
        keys = key_buffer[safe_physical_tokens]
        extrema_mask = token_valid.unsqueeze(-1).unsqueeze(-1)
        page_min = torch.where(
            extrema_mask, keys, torch.full_like(keys, float("inf"))
        ).amin(dim=1)
        page_max = torch.where(
            extrema_mask, keys, torch.full_like(keys, float("-inf"))
        ).amax(dim=1)
        physical_pages = torch.div(
            physical_tokens[:, 0], self.page_size, rounding_mode="floor"
        ).clamp(0, self._page_k_min[context.layer_id].shape[0] - 1)
        layer_id = context.layer_id
        self._page_k_min[layer_id][physical_pages] = page_min
        self._page_k_max[layer_id][physical_pages] = page_max
        self._page_valid[layer_id][physical_pages] = True

    def on_attention_complete(self, context: SelectionContext) -> None:
        if context.forward_batch is None:
            raise ValueError("Quest requires ForwardBatch lifecycle context")
        key_buffer = self.token_to_kv_pool.get_key_buffer(context.layer_id)
        self._ensure_page_store(context.layer_id, key_buffer)
        is_decode = context.forward_batch.forward_mode.is_decode()
        if (
            key_buffer.is_cuda
            and torch.version.hip is None
            and self.page_size <= 128
            and key_buffer.shape[-1] <= 256
        ):
            from sglang.srt.mem_cache.sparsity.kernels.quest_page_update import (
                quest_update_complete_pages_,
            )

            quest_update_complete_pages_(
                context.req_pool_indices,
                context.seq_lens,
                self.req_to_token_pool.req_to_token,
                key_buffer,
                self._last_completed_page,
                self._page_k_min[context.layer_id],
                self._page_k_max[context.layer_id],
                self._page_valid[context.layer_id],
                self.page_size,
                advance_trackers=context.layer_id == self.end_layer - 1,
                rebuild_all=not is_decode,
            )
        elif not is_decode:
            self._rebuild_request_pages(context, key_buffer)
        else:
            self._update_last_page_portable(context, key_buffer)

    def _reshape_query(
        self, context: SelectionContext, num_kv_heads: int, head_dim: int
    ) -> torch.Tensor:
        query = context.query
        if query is None:
            raise ValueError("Quest requires the current query")
        if query.ndim == 2:
            if query.shape[1] % head_dim:
                raise ValueError("Quest query hidden size is not divisible by head_dim")
            query = query.view(query.shape[0], -1, head_dim)
        if query.ndim != 3 or query.shape[0] != context.seq_lens.shape[0]:
            raise ValueError("Quest decode query must have shape [batch, heads, dim]")
        if query.shape[2] != head_dim or query.shape[1] % num_kv_heads:
            raise ValueError("Quest query heads must be divisible by KV heads")
        return query

    def _page_scores(
        self, context: SelectionContext, plan: _QuestForwardPlan
    ) -> torch.Tensor:
        layer_id = context.layer_id
        k_min_pool = self._page_k_min[layer_id]
        k_max_pool = self._page_k_max[layer_id]
        query = self._reshape_query(context, k_min_pool.shape[1], k_min_pool.shape[2])
        if query.is_cuda and torch.version.hip is None and query.shape[-1] <= 256:
            from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
                quest_page_scores,
            )

            return quest_page_scores(
                query,
                k_min_pool,
                k_max_pool,
                self._page_valid[layer_id],
                plan.physical_pages,
                active_mask=plan.sparse_mask,
                history_page_counts=plan.history_end,
            )

        physical_pages = plan.physical_pages
        safe_pages = physical_pages.clamp(0, k_min_pool.shape[0] - 1)
        k_min = k_min_pool[safe_pages]
        k_max = k_max_pool[safe_pages]
        bbox_valid = self._page_valid[layer_id][safe_pages] & (physical_pages >= 0)
        num_kv_heads, head_dim = k_min.shape[-2:]
        query = query.view(
            query.shape[0], num_kv_heads, query.shape[1] // num_kv_heads, head_dim
        ).float()
        q = query.unsqueeze(1)
        bounds = torch.where(
            q >= 0,
            q * k_max.float().unsqueeze(3),
            q * k_min.float().unsqueeze(3),
        ).sum(dim=-1)
        scores = bounds.amax(dim=(-1, -2))
        logical_pages = torch.arange(plan.max_pages, device=self.device)
        candidate_valid = (
            plan.sparse_mask.unsqueeze(1)
            & bbox_valid
            & (logical_pages.unsqueeze(0) < plan.history_end.unsqueeze(1))
        )
        return scores.masked_fill(~candidate_valid, float("-inf"))

    def _dense_result(
        self, context: SelectionContext, max_pages: int
    ) -> SelectionResult:
        batch_size = context.seq_lens.shape[0]
        return SelectionResult(
            granularity=Granularity.PAGE,
            logical_indices=torch.full(
                (batch_size, 1), -1, dtype=torch.int32, device=self.device
            ),
            valid_lengths=torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            ),
            visible_kv_lens=torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            ),
            sparse_mask=torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            layer_id=context.layer_id,
            max_visible_kv_len=max(
                self.page_budget * self.page_size,
                max(self.config.min_sparse_tokens - 1, 1),
            ),
        )

    def select(self, context: SelectionContext) -> SelectionResult:
        layer_id = context.layer_id
        if layer_id not in self._page_k_min:
            raise RuntimeError(
                f"Quest page representations are unavailable for layer {layer_id}"
            )
        plan = self._forward_plan
        if plan is None or plan.forward_batch is not context.forward_batch:
            plan = self._build_forward_plan(context)
            self._forward_plan = plan
        if (
            plan.max_pages <= self.page_budget
            or plan.max_pages * self.page_size < self.config.min_sparse_tokens
        ):
            return self._dense_result(context, plan.max_pages)

        history_budget = self.page_budget - self.recent_pages
        topk_count = min(history_budget, plan.max_pages)
        scores = self._page_scores(context, plan)
        topk_scores, topk_pages = torch.topk(scores, k=topk_count, dim=1, sorted=False)
        k_per_req = torch.where(
            plan.sparse_mask,
            torch.full_like(plan.total_pages, topk_count, dtype=torch.int32),
            torch.zeros_like(plan.total_pages, dtype=torch.int32),
        )
        direct_fa3 = (
            self.config.backend == "fa3"
            and scores.is_cuda
            and torch.version.hip is None
            and context.metadata is not None
            and all(
                hasattr(context.metadata, name)
                for name in (
                    "page_table",
                    "cache_seqlens_int32",
                    "cu_seqlens_k",
                )
            )
        )
        if direct_fa3:
            output_width = topk_count + plan.recent_pages.shape[1]
            return _QuestFA3SelectionResult(
                granularity=Granularity.PAGE,
                logical_indices=torch.empty(
                    (scores.shape[0], output_width),
                    dtype=torch.int32,
                    device=self.device,
                ),
                valid_lengths=torch.empty(
                    scores.shape[0], dtype=torch.int32, device=self.device
                ),
                visible_kv_lens=torch.empty(
                    scores.shape[0], dtype=torch.int32, device=self.device
                ),
                sparse_mask=plan.sparse_mask,
                layer_id=layer_id,
                max_visible_kv_len=max(
                    self.page_budget * self.page_size,
                    max(self.config.min_sparse_tokens - 1, 1),
                ),
                topk_scores=topk_scores.contiguous(),
                topk_indices=topk_pages.contiguous(),
                k_per_req=k_per_req.contiguous(),
                recent_indices=plan.recent_pages.contiguous(),
                recent_valid=plan.recent_valid.contiguous(),
            )
        if scores.is_cuda and torch.version.hip is None:
            from sglang.srt.mem_cache.sparsity.kernels.quest_finalize import (
                quest_finalize_selected_pages,
            )

            selected, valid_lengths = quest_finalize_selected_pages(
                topk_scores.contiguous(),
                topk_pages.contiguous(),
                k_per_req.contiguous(),
                plan.recent_pages.contiguous(),
                plan.recent_valid.contiguous(),
            )
        else:
            selected = torch.cat((topk_pages, plan.recent_pages), dim=1)
            valid = torch.cat(
                (
                    plan.sparse_mask.unsqueeze(1).expand(-1, topk_count),
                    plan.recent_valid,
                ),
                dim=1,
            )
            sentinel = plan.max_pages
            selected = (
                torch.where(valid, selected, torch.full_like(selected, sentinel))
                .sort(dim=1)
                .values
            )
            selected = torch.where(
                selected == sentinel,
                torch.full_like(selected, -1),
                selected,
            ).to(torch.int32)
            valid_lengths = valid.sum(dim=1).to(torch.int32)

        final_page_padding = plan.total_pages * self.page_size - context.seq_lens
        visible_kv_lens = torch.where(
            plan.sparse_mask,
            (valid_lengths * self.page_size - final_page_padding).to(torch.int32),
            torch.zeros_like(plan.total_pages, dtype=torch.int32),
        )
        return SelectionResult(
            granularity=Granularity.PAGE,
            logical_indices=selected,
            valid_lengths=valid_lengths,
            visible_kv_lens=visible_kv_lens,
            sparse_mask=plan.sparse_mask,
            layer_id=layer_id,
            max_visible_kv_len=max(
                self.page_budget * self.page_size,
                max(self.config.min_sparse_tokens - 1, 1),
            ),
        )
