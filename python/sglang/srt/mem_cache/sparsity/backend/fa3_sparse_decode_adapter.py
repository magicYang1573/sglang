"""Generic FA3 sparse-decode adapter.

This adapter packages the per-layer FA3 *sparse* call arguments
(``page_table`` / ``cache_seqlens`` / ``cu_seqlens_q`` / ``cu_seqlens_k`` /
``max_seqlen_k``) into a single :class:`SparseFa3CallArgs` object that
``Fa3SparseDecodeBackend.forward_decode`` feeds **directly** into
``flash_attn_with_kvcache``.  The adapter never touches the shared
``FlashAttentionMetadata`` of the dense path.

This design choice mirrors the NSA path's ``_forward_fa3`` (see
``nsa_backend.py``): sparse and dense pages live on disjoint tensors, which
keeps the dense ``self.forward_metadata`` graph-stable and is the basis of
the sparse-decode CUDA-Graph compatibility plan documented in
``docs/advanced_features/hisparse_sparse_decode_design.md`` §10.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    SparseDecodeAdapter,
    SparseFa3CallArgs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)


class Fa3SparseDecodeAdapter(SparseDecodeAdapter):
    """Adapter that wraps swapped-in top-k physical slots into the minimal
    set of arguments FA3's ``flash_attn_with_kvcache`` needs.

    Algorithm-agnostic: it knows nothing about Quest / H2O / etc.  It only
    knows about the FA3 sparse-decode call surface.

    Eager-mode reference impl: each layer freshly computes
    ``cu_seqlens_k = pad(cumsum(cache_seqlens))``.  When CUDA Graph is
    later enabled this becomes an in-place ``cumsum(out=...)`` into a
    pre-allocated buffer (the ``page_table`` / ``cache_seqlens`` tensors
    coming from the controller are already pre-allocated by
    ``HiSparseCoordinator``).
    """

    def build_sparse_call_args(
        self,
        top_k_device_locs: torch.Tensor,
        valid_lengths: torch.Tensor,
        dense_metadata: "FlashAttentionMetadata",
        forward_batch: "ForwardBatch",
        layer_id: int,
        **kwargs,
    ) -> SparseFa3CallArgs:
        """Package FA3 sparse-decode call arguments.

        Args:
            top_k_device_locs: ``[bs, top_k]`` int32. Physical device-buffer
                row indices already populated by
                :meth:`HiSparseCoordinator.swap_in_selected_pages`.
            valid_lengths: ``[bs]`` int32. ``min(seq_len[i], top_k)``.
            dense_metadata: The FA3 dense per-step ``FlashAttentionMetadata``.
                Read-only here — we only borrow ``cu_seqlens_q`` (which is
                always ``arange(0, bs+1)`` for decode).
        """
        # FA3's sparse path only needs cache_seqlens to be int32; valid_lengths
        # already has that dtype but make it explicit to be defensive.
        cache_seqlens = valid_lengths.to(torch.int32)

        # cu_seqlens_q: reuse FA3's pre-built arange (decode → q_len = 1 per row).
        cu_seqlens_q = dense_metadata.cu_seqlens_q

        # cu_seqlens_k: built fresh per layer in eager mode.  When graph mode
        # is enabled later this becomes an out= cumsum into a pre-allocated
        # buffer.
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32),
            (1, 0),
        )

        # ``max_seqlen_k`` is a static upper bound (FA3 uses cache_seqlens
        # for masking, so a constant is correct and graph-friendly).
        top_k = top_k_device_locs.shape[1]

        return SparseFa3CallArgs(
            page_table=top_k_device_locs,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=top_k,
        )
