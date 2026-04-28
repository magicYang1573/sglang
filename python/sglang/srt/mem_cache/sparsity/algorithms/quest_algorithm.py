"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.

Indexing model
--------------
``page_k_min/max/valid`` are stored **per request** with shape
``[num_layers, num_pages, kv_heads, head_dim]`` (and ``[num_layers, num_pages]``
for ``valid``).  Allocation happens lazily at prefill time once ``kv_len`` is
known, and is released when the request ends.  This keeps memory usage
proportional to the active request footprint:

    bytes_per_req = num_layers * num_pages * kv_heads * head_dim * dtype_size

For a typical Qwen2-7B (28 layers, 8 KV heads, 128 head_dim) with bf16 reps
and ``page_size=16``, a 4K-token request costs ~17 MB; a 32K-token request
costs ~138 MB.

Output contract
---------------
``retrieve_topk`` returns **token-level** positions padded with ``-1``
(shape ``[bs, top_k]`` int32), so the values feed directly into the HiSparse
``load_cache_to_device_buffer_mha`` kernel.  Internally scoring still happens
at page granularity; the selected pages are expanded to their member tokens
before returning.

Note that the algorithm picks pages purely based on
``num_history_pages * sparsity_ratio + num_recent_pages``; it does NOT pad
back up to ``top_k`` tokens.  The adapter then sets
``cache_seqlens = valid_lengths``, so FA3 only attends over those tokens
(this matches the standard Quest paper semantics).
"""

import logging
import os
from typing import Dict, Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)
_SPARSE_ACCURACY_DEBUG = os.environ.get("SGLANG_SPARSE_ACCURACY_DEBUG", "0") == "1"


class QuestAlgorithm(BaseSparseAlgorithmImpl):
    """Quest page-wise sparse attention using bounding-box criticality.

    Per-request representation tensors are stored in dicts keyed by
    ``req_pool_idx``:

    * ``self.req_page_k_min[idx]``: ``[num_layers, num_pages, kv_heads, head_dim]``
    * ``self.req_page_k_max[idx]``: same shape
    * ``self.req_page_valid[idx]``: ``[num_layers, num_pages]`` bool
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.req_page_k_min: Dict[int, torch.Tensor] = {}
        self.req_page_k_max: Dict[int, torch.Tensor] = {}
        self.req_page_valid: Dict[int, torch.Tensor] = {}

        # Filled in :meth:`_initialize_representation_pools`.
        self._kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._num_layers: Optional[int] = None
        self._start_layer: Optional[int] = None
        self._end_layer: Optional[int] = None
        # Storage dtype for the (k_min, k_max) tensors.  Quest's bounding-box
        # estimation does not need fp32 precision; bf16 cuts the per-request
        # memory cost in half versus fp32 at no observable accuracy loss.
        self._repr_dtype: torch.dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # Pool init / per-request lifecycle
    # ------------------------------------------------------------------

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ) -> None:
        """Cache shape metadata only; per-request tensors are allocated lazily.

        ``total_num_pages`` is unused — kept for compatibility with the
        :class:`BaseSparseAlgorithmImpl` contract.
        """
        del total_num_pages
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        self._kv_heads = int(key_buf.shape[1])
        self._head_dim = int(key_buf.shape[2])
        self._start_layer = int(start_layer)
        self._end_layer = int(end_layer)
        self._num_layers = int(end_layer - start_layer)

        logger.info(
            "Initialized Quest reps (lazy per-req): num_layers=%d kv_heads=%d "
            "head_dim=%d page_size=%d repr_dtype=%s",
            self._num_layers,
            self._kv_heads,
            self._head_dim,
            self.page_size,
            self._repr_dtype,
        )

    def _ensure_request_pool(self, req_pool_idx: int, num_pages: int) -> None:
        """Allocate ``page_k_min/max/valid`` for one request when its prefill
        finishes.  Resizes (with copy) if a larger ``num_pages`` is requested.
        """
        if num_pages <= 0:
            return
        existing = self.req_page_k_min.get(req_pool_idx)
        if existing is not None and existing.shape[1] >= num_pages:
            return

        new_shape = (self._num_layers, num_pages, self._kv_heads, self._head_dim)
        new_min = torch.zeros(new_shape, dtype=self._repr_dtype, device=self.device)
        new_max = torch.zeros_like(new_min)
        new_valid = torch.zeros(
            (self._num_layers, num_pages), dtype=torch.bool, device=self.device
        )
        if existing is not None:
            old_pages = existing.shape[1]
            new_min[:, :old_pages] = existing
            new_max[:, :old_pages] = self.req_page_k_max[req_pool_idx]
            new_valid[:, :old_pages] = self.req_page_valid[req_pool_idx]

        self.req_page_k_min[req_pool_idx] = new_min
        self.req_page_k_max[req_pool_idx] = new_max
        self.req_page_valid[req_pool_idx] = new_valid

    def release_request_pool(self, req_pool_idx: int) -> None:
        """Free the per-request representation tensors.

        Called by :class:`SparseAlgorithmController.on_request_end`.  Safe to
        call multiple times.
        """
        self.req_page_k_min.pop(req_pool_idx, None)
        self.req_page_k_max.pop(req_pool_idx, None)
        self.req_page_valid.pop(req_pool_idx, None)

    # ------------------------------------------------------------------
    # Per-layer representation construction
    # ------------------------------------------------------------------

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ) -> None:
        """Compute per-page (min, max) of ``k`` over ``[start_page, end_page)``
        for each request and scatter into the per-request rep tensors.

        ``k_buffer`` is the layer-local KV pool slice.  Indexing into
        ``k_buffer`` uses the *physical row id* obtained from
        ``req_to_token`` (sparse-decode KV pool stores 1 row per token, so
        ``physical_row == token_position`` for the request's logical view).
        """
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        layer_offset = layer_id - self._start_layer

        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off  # [n, max_pages]
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * self.page_size  # [n, max_pages]
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off  # [n, max_pages, page_size]
        tok_mask = (
            tok_pos
            < (tok_start + self.page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)
        ) & pg_mask.unsqueeze(2)

        # Map (req_pool_idx, token_pos) -> physical KV row.
        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        # Gather K and reduce per page (min / max over tokens, retaining
        # kv_heads and head_dim).  Float32 internally for stable reductions.
        keys = k_buffer[phys_tok].to(torch.float32)  # [n, max_pages, page_size, kv_heads, head_dim]
        mask = tok_mask.unsqueeze(-1).unsqueeze(-1)
        page_min = torch.where(mask, keys, torch.full_like(keys, float("inf"))).amin(
            dim=2
        )
        page_max = torch.where(mask, keys, torch.full_like(keys, float("-inf"))).amax(
            dim=2
        )

        # Scatter into per-request representation tensors.  The Python loop
        # over ``n`` is acceptable because (a) ``n`` is the active extend
        # batch size (typically 1 in sparse-decode admit, small for chunked
        # prefill), and (b) sparse-decode currently disables CUDA Graph.
        reqs_cpu = reqs.cpu().tolist()
        end_page_cpu = end_page.cpu().tolist()
        start_page_cpu = start_page.cpu().tolist()
        for i in range(n):
            req_idx = int(reqs_cpu[i])
            sp = int(start_page_cpu[i])
            ep = int(end_page_cpu[i])
            if ep <= sp:
                continue

            self._ensure_request_pool(req_idx, ep)
            local_pages = ep - sp
            pmin = page_min[i, :local_pages].to(self._repr_dtype)
            pmax = page_max[i, :local_pages].to(self._repr_dtype)

            self.req_page_k_min[req_idx][layer_offset, sp:ep] = pmin
            self.req_page_k_max[req_idx][layer_offset, sp:ep] = pmax
            self.req_page_valid[req_idx][layer_offset, sp:ep] = True

    # ------------------------------------------------------------------
    # Top-k retrieval
    # ------------------------------------------------------------------

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """Return ``[bs, top_k]`` int32 token positions (padded with ``-1``).

        Algorithm (per request, eager-mode reference impl):

        1. Score every page via Quest's query-page bounding-box criticality.
        2. Pick ``int(history_pages * sparsity_ratio)`` highest-scoring history
           pages plus the ``num_recent_pages`` most recent ones.
        3. Expand each picked page into its member tokens, truncate to the
           ``top_k`` budget if the expansion exceeds it.

        The number of selected tokens may be **less than** ``top_k`` — this
        is the standard Quest behaviour.  ``valid_lengths[i]`` reflects the
        actual fill, and the FA3 adapter passes it as ``cache_seqlens`` so
        attention only sees those tokens (no spurious padding).

        NOTE: this implementation contains a Python ``for`` loop and is
        therefore eager-only.  CUDA-Graph compatibility requires rewriting
        as a fixed-shape broadcast.
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
        out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

        for i in range(bs):
            seq_len = int(seq_lens[i].item())
            if seq_len <= 0:
                continue

            num_pages = int((seq_len + self.page_size - 1) // self.page_size)
            if num_pages <= 0:
                continue

            req_idx = int(req_pool_indices[i].item())

            # Build query-page criticality scores indexed by ``page_id``.
            scores = self._retrieve_page_scores(
                layer_id=layer_id,
                req_pool_idx=req_idx,
                num_pages=num_pages,
                queries=queries[i : i + 1],
            )  # [1, num_pages]

            # Recent window: clamp to num_pages so short requests don't
            # over-claim recent pages.
            recent_count = min(self.num_recent_pages, num_pages)
            recent_start = num_pages - recent_count

            masked_scores = scores.clone()
            if recent_count > 0:
                masked_scores[:, recent_start:] = float("-inf")

            history_pages = recent_start
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
            combined_pages = torch.unique(combined_pages)

            # Expand page indices to token positions inside the request.
            if self.page_size == 1:
                token_positions = combined_pages.to(torch.int32)
            else:
                page_starts = combined_pages * self.page_size
                offsets = torch.arange(
                    self.page_size, device=device, dtype=torch.int64
                )
                token_positions = (
                    page_starts.unsqueeze(1) + offsets.unsqueeze(0)
                ).reshape(-1)
                token_positions = token_positions[token_positions < seq_len]
                token_positions = token_positions.to(torch.int32)

            # Truncate to budget (priority: keep recent tokens, then add the
            # highest-scoring history tokens).
            if token_positions.numel() > top_k_budget:
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
                continue
            out_tokens[i, :length] = token_positions
            out_lengths[i] = length

            if _SPARSE_ACCURACY_DEBUG and layer_id == 0 and i == 0:
                pool = self.req_page_valid.get(req_idx)
                valid_count = (
                    int(pool[layer_id - self._start_layer, :num_pages].sum().item())
                    if pool is not None
                    else 0
                )
                topk_idx_unique = (
                    int(torch.unique(topk_idx).numel())
                    if topk_idx.numel() > 0
                    else 0
                )
                logger.info(
                    "[SPARSE/diag] quest req=%d seq_len=%d page_size=%d num_pages=%d "
                    "page_valid_true=%d history_pages=%d num_recent_pages=%d k_pages=%d "
                    "topk_unique=%d combined_unique=%d tokens_after_truncate=%d "
                    "top_k_budget=%d sparsity_ratio=%s",
                    req_idx,
                    seq_len,
                    self.page_size,
                    num_pages,
                    valid_count,
                    history_pages,
                    self.num_recent_pages,
                    k_pages,
                    topk_idx_unique,
                    int(combined_pages.numel()),
                    length,
                    top_k_budget,
                    self.sparsity_ratio,
                )

        return out_tokens, out_lengths

    def _retrieve_page_scores(
        self,
        layer_id: int,
        req_pool_idx: int,
        num_pages: int,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Compute query-page bounding-box criticality scores.

        Returns ``[1, num_pages]`` fp32 scores; pages without a constructed
        representation get ``-inf`` so ``torch.topk`` ignores them.
        """
        layer_offset = layer_id - self._start_layer
        pool_min = self.req_page_k_min.get(req_pool_idx)
        pool_max = self.req_page_k_max.get(req_pool_idx)
        pool_valid = self.req_page_valid.get(req_pool_idx)

        if pool_min is None or pool_min.shape[1] < num_pages:
            # Representation missing or partial: report all -inf so the
            # caller falls back to recent pages only.  This should not
            # happen in normal flow because admit_request_from_gpu builds
            # the full representation before the first decode step.
            return torch.full(
                (1, num_pages),
                float("-inf"),
                dtype=torch.float32,
                device=queries.device,
            )

        k_min = pool_min[layer_offset, :num_pages]  # [num_pages, kv_heads, head_dim]
        k_max = pool_max[layer_offset, :num_pages]
        valid_mask = pool_valid[layer_offset, :num_pages]  # [num_pages]

        head_dim = self._head_dim
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

        kv_heads = self._kv_heads
        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            # Average grouped query heads to align with KV heads (approximation
            # for MQA/GQA — Quest paper algorithm 1).
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        # Broadcast q to [bs, 1, kv_heads, head_dim] vs k_min/max
        # [num_pages, kv_heads, head_dim] -> result [bs, num_pages]
        q = q.to(torch.float32).unsqueeze(1)
        k_min_f = k_min.to(torch.float32).unsqueeze(0)
        k_max_f = k_max.to(torch.float32).unsqueeze(0)

        criticality = torch.where(q >= 0, q * k_max_f, q * k_min_f).sum(dim=(2, 3))
        criticality = torch.where(
            valid_mask.unsqueeze(0),
            criticality,
            torch.full_like(criticality, float("-inf")),
        )

        return criticality
