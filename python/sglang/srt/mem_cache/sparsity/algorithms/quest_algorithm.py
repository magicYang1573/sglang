"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.
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

    # ------------------------------------------------------------------
    # Batched top-k retrieval (CUDA-graph friendly)
    # ------------------------------------------------------------------
    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """Batched Quest top-k.

        Returns ``(selected_pages, valid_lengths)`` where
        ``selected_pages`` is ``[bs, max_k]`` int32 with -1 padding and
        ``valid_lengths`` is ``[bs]`` int32 giving the real number of
        selected *pages* per request.  When ``sparse_mask[i]`` is False we
        emit an empty row.
        """
        forward_batch = kwargs.get("forward_batch", None)
        if forward_batch is None or not hasattr(forward_batch, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for Quest retrieve_topk"
            )

        device = queries.device
        seq_lens = forward_batch.seq_lens.to(device)
        bs = queries.shape[0]
        if bs == 0:
            return (
                torch.empty(0, 0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )

        req_to_token = self.req_to_token_pool.req_to_token
        max_req_tokens = req_to_token.shape[1]

        num_pages_per_req = (seq_lens + self.page_size - 1) // self.page_size
        max_num_pages = int(num_pages_per_req.max().item()) if bs > 0 else 0
        if max_num_pages == 0:
            return (
                torch.empty(bs, 0, dtype=torch.int32, device=device),
                torch.zeros(bs, dtype=torch.int32, device=device),
            )

        # ----- logical -> physical page map per request ----------------
        page_idx = torch.arange(max_num_pages, device=device)
        page_start_tok = (page_idx * self.page_size).unsqueeze(0).expand(bs, -1)
        page_start_tok = page_start_tok.clamp(max=max_req_tokens - 1)
        first_tokens = req_to_token[
            req_pool_indices.unsqueeze(1).expand(-1, max_num_pages),
            page_start_tok,
        ]
        phys_pages = first_tokens // self.page_size  # [bs, max_num_pages]

        # ----- bounding-box scores -------------------------------------
        scores = self._retrieve_page_scores(
            layer_id,
            phys_pages,
            req_pool_indices,
            queries,
        )  # [bs, max_num_pages]

        # Mask out out-of-range pages for each request.
        valid_page_mask = (
            page_idx.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
        )  # [bs, max_num_pages]
        scores = torch.where(
            valid_page_mask,
            scores,
            torch.full_like(scores, float("-inf")),
        )

        # Mask the tail ``num_recent_pages`` so they are always selected
        # via the "recent" append path below.
        recent_start = (num_pages_per_req - self.num_recent_pages).clamp(min=0)
        recent_mask = page_idx.unsqueeze(0) >= recent_start.unsqueeze(1)
        scores = torch.where(
            recent_mask, torch.full_like(scores, float("-inf")), scores
        )

        # Per-request history page count (positions before the recent tail).
        history_pages = recent_start.clamp(min=1)
        k_per_req = (history_pages.float() * self.sparsity_ratio).clamp(min=1).long()
        k_per_req = torch.minimum(k_per_req, history_pages)
        k_global = int(k_per_req.max().item())

        # Global top-k (sorted by score desc) then mask to per-req k.
        topk_scores, topk_idx = torch.topk(
            scores, k=k_global, dim=1, sorted=True
        )  # [bs, k_global]
        arange_k = torch.arange(k_global, device=device).unsqueeze(0)
        keep_topk = (arange_k < k_per_req.unsqueeze(1)) & torch.isfinite(topk_scores)

        # ----- append recent pages ------------------------------------
        recent_idx = recent_start.unsqueeze(1) + torch.arange(
            self.num_recent_pages, device=device
        ).unsqueeze(0)
        keep_recent = (recent_idx < num_pages_per_req.unsqueeze(1)) & (
            recent_idx >= 0
        )

        # Concatenate top-k + recent into a single padded tensor.
        max_combined = k_global + self.num_recent_pages
        selected_pages = torch.full(
            (bs, max_combined), -1, dtype=torch.int32, device=device
        )
        selected_pages[:, :k_global] = torch.where(
            keep_topk, topk_idx.to(torch.int32), torch.full_like(topk_idx, -1).to(torch.int32)
        )
        selected_pages[:, k_global:] = torch.where(
            keep_recent,
            recent_idx.to(torch.int32),
            torch.full_like(recent_idx, -1).to(torch.int32),
        )

        # Zero-out rows that don't need sparse attention.
        sparse_mask_row = sparse_mask.unsqueeze(1)
        selected_pages = torch.where(
            sparse_mask_row,
            selected_pages,
            torch.full_like(selected_pages, -1),
        )

        valid_lengths = (selected_pages >= 0).sum(dim=1).to(torch.int32)
        return selected_pages, valid_lengths

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
