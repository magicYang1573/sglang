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
        self.page_k_min = {}
        self.page_k_max = {}
        self.page_valid = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

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
        **kwargs,
    ) -> tuple:
        """Return ``[bs, top_k]`` int32 token positions (padded with ``-1``).

        Contract (see :class:`BaseSparseAlgorithm`):
        - Output shape is ``(bs, top_k)`` int32; slots beyond
          ``valid_lengths[i]`` are padded with ``-1``.
        - ``valid_lengths[i] = min(seq_len[i], top_k)``.
        - **No** ``sparse_mask`` input: short requests (seq_len <= top_k)
          auto-degenerate to dense by selecting every valid token (the
          algorithm's recent window + top-k pages naturally cover the full
          range when ``num_pages <= top_k_budget``).

        Pipeline (per request, eager-mode reference impl)
        -------------------------------------------------
        1. Score pages via query-page bounding-box criticality.
        2. Pick top-k pages (recent window + history top scoring), trying to
           saturate ``top_k_budget`` whenever possible.
        3. Expand each picked page into its member tokens, then truncate to
           the budget if needed.

        NOTE: this implementation contains a Python ``for`` loop and is
        therefore eager-only.  CUDA-Graph compatibility requires rewriting
        this method as a fixed-shape broadcast (see design doc §10.4).
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

        out_tokens = torch.full(
            (bs, top_k_budget), -1, dtype=torch.int32, device=device
        )
        # valid_lengths = min(seq_len, top_k_budget) — short requests will
        # report "all tokens are selected" so attention sees the full key
        # set (equivalent to dense).
        out_lengths = seq_lens.clamp(max=top_k_budget).to(torch.int32)

        req_to_token = self.req_to_token_pool.req_to_token
        max_req_tokens = req_to_token.shape[1]

        for i in range(bs):
            seq_len = int(seq_lens[i].item())
            if seq_len <= 0:
                out_lengths[i] = 0
                continue

            num_pages = int((seq_len + self.page_size - 1) // self.page_size)
            if num_pages <= 0:
                out_lengths[i] = 0
                continue

            page_idx = torch.arange(num_pages, device=device)
            page_start_token = req_to_token[
                req_pool_indices[i],
                (page_idx * self.page_size).clamp(0, max_req_tokens - 1),
            ]
            phys_pages = (page_start_token // self.page_size).unsqueeze(0)

            scores = self._retrieve_page_scores(
                layer_id,
                phys_pages,
                req_pool_indices[i : i + 1],
                queries[i : i + 1],
            )  # [1, num_pages]

            # Recent window: clamp to num_pages so short requests don't
            # over-claim recent pages.
            recent_count = min(self.num_recent_pages, num_pages)
            recent_start = num_pages - recent_count

            masked_scores = scores.clone()
            if recent_count > 0:
                masked_scores[:, recent_start:] = float("-inf")

            history_pages = recent_start
            # Budget top-k pages: respect sparsity_ratio but cap at history.
            # When history_pages == 0 (i.e. num_pages <= num_recent_pages),
            # k_pages stays 0 and we fall back to recent-only — which itself
            # already covers every valid page → equivalent to dense.
            if history_pages > 0:
                k_pages = min(
                    max(int(history_pages * self.sparsity_ratio), 1),
                    history_pages,
                )
            else:
                k_pages = 0

            if k_pages > 0:
                topk_idx = torch.topk(masked_scores, k=k_pages, dim=1, sorted=False)[1]
                topk_idx = topk_idx.squeeze(0)
            else:
                topk_idx = torch.empty(0, dtype=torch.int64, device=device)

            recent_idx = torch.arange(
                recent_start, num_pages, device=device, dtype=torch.int64
            )
            combined_pages = torch.cat([topk_idx.to(torch.int64), recent_idx], dim=0)
            # Deduplicate in ascending order (recent window may overlap).
            combined_pages = torch.unique(combined_pages)

            # Expand page indices to token positions
            if self.page_size == 1:
                token_positions = combined_pages.to(torch.int32)
            else:
                page_starts = combined_pages * self.page_size
                offsets = torch.arange(self.page_size, device=device, dtype=torch.int64)
                token_positions = (
                    page_starts.unsqueeze(1) + offsets.unsqueeze(0)
                ).reshape(-1)
                token_positions = token_positions[token_positions < seq_len]
                token_positions = token_positions.to(torch.int32)

            # Truncate to budget (keep the most recent tokens first, then older).
            if token_positions.numel() > top_k_budget:
                # Prioritize recent tokens: keep the last ``num_recent_pages``
                # worth of tokens, then top scoring older ones.
                recent_token_start = int(recent_start * self.page_size)
                recent_mask = token_positions >= recent_token_start
                recent_tokens = token_positions[recent_mask]
                history_tokens = token_positions[~recent_mask]
                if recent_tokens.numel() >= top_k_budget:
                    token_positions = recent_tokens[:top_k_budget]
                else:
                    remainder = top_k_budget - recent_tokens.numel()
                    token_positions = torch.cat(
                        [history_tokens[:remainder], recent_tokens], dim=0
                    )

            length = int(token_positions.numel())
            if length == 0:
                out_lengths[i] = 0
                continue
            out_tokens[i, :length] = token_positions
            # Reflect actual filled length (may be < min(seq_len, top_k_budget)
            # in pathological cases where dedup shrank the set).
            out_lengths[i] = length

        return out_tokens, out_lengths

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
