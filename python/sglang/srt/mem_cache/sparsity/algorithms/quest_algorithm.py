"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.

Output contract
---------------
``retrieve_topk`` always returns **token-level** positions padded with ``-1``
(shape ``[bs, top_k]`` int32), so it can be fed directly to the HiSparse
``load_cache_to_device_buffer_mha`` kernel.  Internally scoring still happens
at page granularity; the selected pages are expanded to their member tokens
before returning.
"""

import logging

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class QuestAlgorithm(BaseSparseAlgorithmImpl):
    """Quest page-wise sparse attention using bounding-box criticality."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.quest_page_size = int(
            config.sparse_extra_config.get("quest_page_size", self.page_size)
        )
        if self.quest_page_size <= 0:
            raise ValueError(
                f"Quest quest_page_size must be positive, got {self.quest_page_size}"
            )
        # Quest's page controls only the algorithm's score / representation
        # granularity. The FA3/KV backend page size remains config.page_size.
        self.page_size = self.quest_page_size
        self.page_k_min = {}
        self.page_k_max = {}
        self.page_valid = {}
        self.max_context_pages = None
        self.topk_pages_budget = None
        self._page_ids = None
        self._page_token_offsets = None

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]
        max_context_len = self.req_to_token_pool.max_context_len
        self.max_context_pages = (max_context_len + self.page_size - 1) // self.page_size
        max_history_pages = max(self.max_context_pages - self.num_recent_pages, 0)
        self.topk_pages_budget = (
            min(max(int(max_history_pages * self.sparsity_ratio), 1), max_history_pages)
            if max_history_pages > 0
            else 0
        )
        self._page_ids = torch.arange(
            self.max_context_pages, dtype=torch.int64, device=self.device
        )
        self._page_token_offsets = torch.arange(
            self.page_size, dtype=torch.int64, device=self.device
        )

        for layer_id in range(start_layer, end_layer):
            self.page_k_min[layer_id] = torch.zeros(
                (total_num_pages, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_k_max[layer_id] = torch.zeros_like(self.page_k_min[layer_id])
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        logger.info(
            "Initialized Quest page reps: %d pages, %d layers, head_num=%d, head_dim=%d",
            total_num_pages,
            end_layer - start_layer,
            head_num,
            head_dim,
        )

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * self.page_size
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off
        tok_mask = (
            tok_pos
            < (tok_start + self.page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)
        ) & pg_mask.unsqueeze(2)

        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        keys = k_buffer[phys_tok].to(torch.float32)
        mask = tok_mask.unsqueeze(-1).unsqueeze(-1)

        page_min = torch.where(mask, keys, torch.full_like(keys, float("inf"))).amin(
            dim=2
        )
        page_max = torch.where(mask, keys, torch.full_like(keys, float("-inf"))).amax(
            dim=2
        )

        phys_pg = (
            req_to_token[
                reqs.unsqueeze(1).expand(n, max_pages),
                tok_start.clamp(0, req_to_token.shape[1] - 1),
            ]
            // self.page_size
        )

        idx = pg_mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return

        target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
            0, self.page_k_min[layer_id].shape[0] - 1
        )
        self.page_k_min[layer_id][target_pages] = page_min[idx[:, 0], idx[:, 1]]
        self.page_k_max[layer_id][target_pages] = page_max[idx[:, 0], idx[:, 1]]
        self.page_valid[layer_id][target_pages] = True

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """Return ``[bs, top_k]`` int32 token positions (padded with ``-1``).

        Pipeline
        --------
        1. Score pages via query-page bounding-box criticality (reuses
           :meth:`_retrieve_page_scores`).
        2. Pick top-k pages (respecting ``num_recent_pages`` / sparsity ratio
           / ``SparseConfig.top_k`` as a token budget).
        3. Expand each picked page into its member tokens, then truncate /
           pad to ``SparseConfig.top_k``.

        Requests with ``sparse_mask == False`` OR those whose sequence is
        shorter than ``num_recent_pages * page_size`` fall back to dense, and
        the output row is filled with ``-1`` (FA3 adapter will honor the
        original dense metadata for such rows).
        """
        bs = queries.shape[0]
        device = queries.device
        top_k_budget = int(self.config.top_k)

        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for Quest TopK retrieval"
            )
        seq_lens = seq_lens_source.seq_lens.to(device)

        req_to_token = self.req_to_token_pool.req_to_token
        max_req_tokens = req_to_token.shape[1]
        page_ids = self._page_ids.to(device=device)
        page_offsets = self._page_token_offsets.to(device=device)
        max_context_pages = page_ids.numel()

        num_pages = (seq_lens + self.page_size - 1) // self.page_size
        num_pages = num_pages.clamp(min=0, max=max_context_pages)
        recent_count = torch.minimum(
            num_pages, torch.full_like(num_pages, self.num_recent_pages)
        )
        recent_start = num_pages - recent_count

        logical_pages = page_ids.view(1, -1).expand(bs, max_context_pages)
        page_valid = logical_pages < num_pages.unsqueeze(1)
        sparse_rows = sparse_mask.to(device=device, dtype=torch.bool).unsqueeze(1)

        page_start_positions = (logical_pages * self.page_size).clamp(
            0, max_req_tokens - 1
        )
        page_start_tokens = req_to_token[
            req_pool_indices.to(req_to_token.device).view(-1, 1),
            page_start_positions.to(req_to_token.device),
        ].to(device)
        phys_pages = page_start_tokens // self.page_size

        scores = self._retrieve_page_scores(
            layer_id,
            phys_pages,
            req_pool_indices,
            queries,
        )

        history_mask = logical_pages < recent_start.unsqueeze(1)
        eligible_history = sparse_rows & page_valid & history_mask
        masked_scores = torch.where(
            eligible_history, scores, torch.full_like(scores, float("-inf"))
        )

        history_pages = recent_start
        k_pages = torch.floor(history_pages.to(torch.float32) * self.sparsity_ratio).to(
            torch.int64
        )
        k_pages = torch.where(history_pages > 0, torch.clamp(k_pages, min=1), k_pages)
        k_pages = torch.minimum(k_pages, history_pages)

        if self.topk_pages_budget > 0:
            k_select = self.topk_pages_budget
            topk_scores, topk_pages = torch.topk(
                masked_scores, k=k_select, dim=1, sorted=True
            )
            topk_rank = torch.arange(k_select, dtype=torch.int64, device=device).view(
                1, -1
            )
            topk_valid = (topk_rank < k_pages.unsqueeze(1)) & torch.isfinite(
                topk_scores
            )
            history_pages_for_sort = torch.where(
                topk_valid,
                topk_pages.to(torch.int64),
                torch.full_like(topk_pages, max_context_pages, dtype=torch.int64),
            )
            selected_history_pages = torch.sort(history_pages_for_sort, dim=1)[0]
            selected_history_valid = selected_history_pages < max_context_pages
        else:
            selected_history_pages = torch.empty(
                (bs, 0), dtype=torch.int64, device=device
            )
            selected_history_valid = torch.empty(
                (bs, 0), dtype=torch.bool, device=device
            )

        recent_slots = min(self.num_recent_pages, max_context_pages)
        if recent_slots > 0:
            recent_offsets = torch.arange(
                recent_slots, dtype=torch.int64, device=device
            ).view(1, -1)
            selected_recent_pages = recent_start.unsqueeze(1) + recent_offsets
            selected_recent_valid = (
                (recent_offsets < recent_count.unsqueeze(1))
                & (selected_recent_pages < num_pages.unsqueeze(1))
                & sparse_rows
            )
        else:
            selected_recent_pages = torch.empty(
                (bs, 0), dtype=torch.int64, device=device
            )
            selected_recent_valid = torch.empty(
                (bs, 0), dtype=torch.bool, device=device
            )

        selected_pages = torch.cat(
            [selected_history_pages, selected_recent_pages], dim=1
        )
        selected_page_valid = torch.cat(
            [selected_history_valid, selected_recent_valid], dim=1
        )

        token_positions = (
            selected_pages.unsqueeze(2) * self.page_size
            + page_offsets.view(1, 1, -1)
        ).reshape(bs, -1)
        token_valid = (
            selected_page_valid.unsqueeze(2)
            & (token_positions.view(bs, -1, self.page_size) < seq_lens.view(bs, 1, 1))
        ).reshape(bs, -1)

        total_token_slots = token_positions.shape[1]
        token_order = torch.arange(total_token_slots, dtype=torch.int64, device=device)
        sort_key = torch.where(
            token_valid,
            token_order.view(1, -1),
            token_order.view(1, -1) + total_token_slots,
        )
        compact_indices = torch.argsort(sort_key, dim=1)
        compact_width = min(top_k_budget, total_token_slots)
        compact_tokens = token_positions.gather(
            1, compact_indices[:, :compact_width]
        ).to(torch.int32)

        valid_lengths = torch.minimum(
            token_valid.sum(dim=1),
            torch.full((bs,), top_k_budget, dtype=torch.int64, device=device),
        ).to(torch.int32)
        out_tokens = torch.full(
            (bs, top_k_budget), -1, dtype=torch.int32, device=device
        )
        if compact_width > 0:
            out_range = torch.arange(compact_width, dtype=torch.int32, device=device)
            out_tokens[:, :compact_width] = torch.where(
                out_range.view(1, -1) < valid_lengths.view(-1, 1),
                compact_tokens,
                torch.full_like(compact_tokens, -1),
            )

        return out_tokens, valid_lengths

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp pages to valid storage range
        phys_pages_clamped = phys_pages.clamp(0, self.page_k_min[layer_id].shape[0] - 1)

        k_min = self.page_k_min[layer_id][phys_pages_clamped]
        k_max = self.page_k_max[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]
        # Align query shape to KV heads.
        head_dim = k_min.shape[-1]
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.view(bs, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")

        kv_heads = k_min.shape[-2]
        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            # Average grouped query heads to align with KV heads (approximation for MQA/GQA).
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        q = q.to(k_min.dtype).unsqueeze(1)  # [bs, 1, kv_heads, head_dim]

        criticality = torch.where(q >= 0, q * k_max, q * k_min).sum(dim=(2, 3))
        criticality = torch.where(
            valid_mask, criticality, torch.full_like(criticality, float("-inf"))
        )

        return criticality
