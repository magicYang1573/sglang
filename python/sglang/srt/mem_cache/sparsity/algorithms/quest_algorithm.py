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
        """Quest page-wise top-k retrieval.

        Semantics are aligned with the DSA HiSparse path: the caller only
        configures ``top_k`` (target number of *tokens*); Quest selects the
        best ``ceil(top_k / page_size)`` pages by bounding-box criticality,
        which covers ``top_k`` tokens (plus up to ``page_size - 1`` tail
        tokens from the last selected page).

        Returns ``(selected_pages, valid_lengths)`` where
        ``selected_pages`` is ``[bs, num_topk_pages]`` int32 with -1 padding
        and ``valid_lengths`` is ``[bs]`` int32 giving the real number of
        selected *pages* per request.  When ``sparse_mask[i]`` is False we
        emit an all-(-1) row.
        """
        forward_batch = kwargs.get("forward_batch", None)
        if forward_batch is None or not hasattr(forward_batch, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for Quest retrieve_topk"
            )

        device = queries.device
        seq_lens = forward_batch.seq_lens.to(device)
        bs = queries.shape[0]

        top_k_tokens = int(self.config.top_k)
        page_size = int(self.page_size)
        num_topk_pages = (top_k_tokens + page_size - 1) // page_size

        if bs == 0:
            return (
                torch.empty(0, num_topk_pages, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )

        req_to_token = self.req_to_token_pool.req_to_token
        max_req_tokens = req_to_token.shape[1]

        num_pages_per_req = (seq_lens + page_size - 1) // page_size
        max_num_pages = int(num_pages_per_req.max().item())
        if max_num_pages == 0:
            return (
                torch.full(
                    (bs, num_topk_pages), -1, dtype=torch.int32, device=device
                ),
                torch.zeros(bs, dtype=torch.int32, device=device),
            )

        # ----- logical -> physical page map per request ----------------
        page_idx = torch.arange(max_num_pages, device=device)
        page_start_tok = (page_idx * page_size).unsqueeze(0).expand(bs, -1)
        page_start_tok = page_start_tok.clamp(max=max_req_tokens - 1)
        first_tokens = req_to_token[
            req_pool_indices.unsqueeze(1).expand(-1, max_num_pages),
            page_start_tok,
        ]
        phys_pages = first_tokens // page_size  # [bs, max_num_pages]

        # ----- bounding-box scores -------------------------------------
        scores = self._retrieve_page_scores(
            layer_id,
            phys_pages,
            req_pool_indices,
            queries,
        )  # [bs, max_num_pages]

        # Mask out-of-range (padded) pages per request.
        valid_page_mask = (
            page_idx.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
        )  # [bs, max_num_pages]
        scores = torch.where(
            valid_page_mask,
            scores,
            torch.full_like(scores, float("-inf")),
        )

        # Per-request number of pages to actually return (capped by what the
        # request has).  We always try to return ``num_topk_pages`` pages
        # (so the expander below produces exactly ``top_k`` tokens on long
        # sequences); shorter sequences naturally produce fewer.
        k_per_req = torch.minimum(
            num_pages_per_req,
            torch.full_like(num_pages_per_req, num_topk_pages),
        )
        k_global = num_topk_pages  # fixed width, CUDA-graph safe

        # Take global top-k by score; pad rows with fewer valid pages.
        # ``k`` passed to torch.topk must be <= scores.shape[1]; if the
        # longest request has fewer pages than ``num_topk_pages`` we simply
        # ask for ``min(k_global, max_num_pages)`` and pad the rest with -1.
        k_take = min(k_global, max_num_pages)
        topk_scores, topk_idx = torch.topk(
            scores, k=k_take, dim=1, sorted=False
        )  # [bs, k_take]

        arange_k = torch.arange(k_take, device=device).unsqueeze(0)
        keep = (arange_k < k_per_req.unsqueeze(1)) & torch.isfinite(topk_scores)

        selected_pages = torch.full(
            (bs, k_global), -1, dtype=torch.int32, device=device
        )
        selected_pages[:, :k_take] = torch.where(
            keep, topk_idx.to(torch.int32), torch.full_like(topk_idx, -1, dtype=torch.int32)
        )

        # Zero-out rows that don't need sparse attention.
        sparse_mask_row = sparse_mask.unsqueeze(1)
        selected_pages = torch.where(
            sparse_mask_row,
            selected_pages,
            torch.full_like(selected_pages, -1),
        )

        valid_lengths = (selected_pages >= 0).sum(dim=1).to(torch.int32)

        if not getattr(self, "_debug_retrieve_left", 6):
            pass
        else:
            left = getattr(self, "_debug_retrieve_left", 6)
            if left > 0 and layer_id == self.start_layer:
                self._debug_retrieve_left = left - 1
                import logging as _logging
                _logging.getLogger("hisparse.debug").warning(
                    "[HS-DBG Quest.retrieve L%d] bs=%d top_k=%d page_size=%d "
                    "num_topk_pages=%d k_take=%d max_num_pages=%d "
                    "selected_pages[0, :8]=%s valid_lengths=%s "
                    "scores.max=%.2f scores.min_finite=%.2f "
                    "num_finite_scores=%d",
                    layer_id,
                    int(bs),
                    int(top_k_tokens),
                    int(page_size),
                    int(num_topk_pages),
                    int(k_take),
                    int(max_num_pages),
                    selected_pages[0, :8].tolist() if bs > 0 else [],
                    valid_lengths.tolist(),
                    float(scores.max().item()) if scores.numel() else 0.0,
                    float(
                        torch.where(
                            torch.isfinite(scores),
                            scores,
                            torch.full_like(scores, float("inf")),
                        )
                        .min()
                        .item()
                    )
                    if scores.numel()
                    else 0.0,
                    int(torch.isfinite(scores).sum().item()),
                )

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

        # Resolve the physical K slot for each (req, token_pos_in_req) cell.
        #
        # Under HiSparse we have TWO sources of truth for the mapping
        # "token position in request" -> "physical k_buffer slot":
        #
        # 1. ``req_to_token[req, pos] -> logical_slot``, followed by
        #    ``full_to_hisparse_device_index_mapping[logical_slot] ->
        #    hisparse_slot`` (i.e. ``_translate_loc_to_hisparse_device``).
        #    This path is **only valid before ``alloc_device_buffer``**
        #    runs: after prefill transitions into the decode hot buffer,
        #    ``alloc_device_buffer`` *clears* the logical->hisparse
        #    mapping and ``translate`` returns 0 for prefill tokens.
        #
        # 2. ``io_subsystem.req_to_device_buffer[req, pos_in_req] ->
        #    hot_buffer_slot``.  The HiSparse MHA coordinator maintains
        #    this table for the full request lifetime: ``alloc_device_
        #    buffer`` seeds it with the reclaimed prefill hisparse slots,
        #    and ``_grow_device_buffers`` / ``map_last_loc_to_buffer``
        #    append fresh slots for every new decode token.
        #
        # During the *prefill* ``attention_end`` call we still have to
        # use path 1 because ``req_to_device_buffer`` hasn't been
        # populated yet (``alloc_device_buffer`` runs on the next
        # scheduler tick, after staging DMA acks).  During *decode*
        # ``update_representations`` we MUST use path 2, otherwise we
        # would read ``k_buffer[0]`` (garbage) for every historical
        # token in the tail page and overwrite the correct bounding
        # box computed during prefill with random values -- which in
        # turn makes Quest's criticality scores meaningless and causes
        # the model to read zero / wrong KV through the adapter.
        #
        # Heuristic: prefer the coordinator table when it is attached
        # AND the request has a live device buffer for this position
        # (``req_to_device_buffer[req, pos] > 0``).  Fall back to the
        # legacy translate path otherwise (prefill-time bbox build and
        # non-HiSparse standalone tests).
        io_subsystem = getattr(self, "_sparse_io_subsystem", None)
        r2db = (
            io_subsystem.req_to_device_buffer
            if io_subsystem is not None
            else None
        )
        if r2db is not None:
            r2db_width = r2db.shape[1]
            tok_pos_in_req = tok_pos.clamp(0, r2db_width - 1).to(torch.int64)
            reqs_expand = reqs.view(n, 1, 1).expand(n, max_pages, self.page_size)
            hot_slot = r2db[reqs_expand, tok_pos_in_req]  # int64
            # For positions beyond the currently-resident prefix the
            # coordinator stores 0 (never written).  Fall back to the
            # translate path for those cells to preserve the original
            # prefill semantics (where mapping is still valid).
            phys_tok_legacy = req_to_token[
                reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
                tok_pos.clamp(0, req_to_token.shape[1] - 1),
            ]
            translate = getattr(
                self.token_to_kv_pool, "_translate_loc_to_hisparse_device", None
            )
            if translate is not None:
                phys_tok_legacy = translate(phys_tok_legacy.to(torch.int64))
            phys_tok_legacy = phys_tok_legacy.to(torch.int64)
            phys_tok = torch.where(hot_slot > 0, hot_slot, phys_tok_legacy)
        else:
            phys_tok = req_to_token[
                reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
                tok_pos.clamp(0, req_to_token.shape[1] - 1),
            ]
            translate = getattr(
                self.token_to_kv_pool, "_translate_loc_to_hisparse_device", None
            )
            if translate is not None:
                phys_tok = translate(phys_tok.to(torch.int64))

        phys_tok = phys_tok.to(torch.int64).clamp(0, k_buffer.shape[0] - 1)

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

        if layer_id == self.start_layer and not getattr(
            self, "_debug_compute_logged", False
        ):
            import logging as _logging
            self._debug_compute_logged = True
            _logging.getLogger("hisparse.debug").warning(
                "[HS-DBG Quest._compute L%d] n=%d max_pages=%d page_size=%d "
                "phys_pg_unique=%d target_pages[:8]=%s "
                "page_min.abs.mean=%.4f page_max.abs.mean=%.4f "
                "k_buffer_idx_range=(%d,%d) logical_first=%s hisparse_first=%s",
                layer_id,
                int(n),
                int(max_pages),
                int(self.page_size),
                int(target_pages.unique().numel()),
                target_pages[:8].tolist(),
                float(page_min.abs().mean().item()),
                float(page_max.abs().mean().item()),
                int(phys_tok.min().item()),
                int(phys_tok.max().item()),
                req_to_token[reqs[:1].view(1, 1, 1).expand(1, 1, self.page_size),
                             tok_pos[:1, :1].clamp(0, req_to_token.shape[1] - 1)][
                    0, 0
                ].tolist(),
                phys_tok[0, 0].tolist(),
            )

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
