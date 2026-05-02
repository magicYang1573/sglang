"""Thin FA3 subclass that lets :class:`HiSparseCoordinator` inject sparse
per-layer metadata without touching the FA3 kernel call sites.

The class is NEVER imported or held by :class:`ModelRunner` directly.  It is
registered through :mod:`attention_registry` and selected via
``ServerArgs.get_attention_backends()`` which rewrites
``decode_attention_backend`` to ``"fa3_sparse_decode"`` when
``--enable-sparse-decode`` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Fa3SparseDecodeBackend(FlashAttentionBackend):
    """FA3 backend that re-writes ``forward_metadata`` per decode layer via
    :class:`SparseAlgorithmController`.

    The wrapper is deliberately minimal; everything except
    :meth:`forward_decode` is inherited from :class:`FlashAttentionBackend`.
    """

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coord = getattr(forward_batch, "hisparse_coordinator", None)
        controller = getattr(coord, "algorithm_controller", None) if coord else None

        if controller is None:
            # No sparse controller attached → behave exactly like plain FA3.
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        original_metadata = self.forward_metadata
        try:
            self.forward_metadata = controller.begin_layer_decode(
                q=q,
                layer=layer,
                forward_batch=forward_batch,
                attn_metadata=original_metadata,
            )
            out = super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )
        finally:
            # Always restore the per-step (dense) metadata so downstream
            # callers (e.g. next layer in the same step) see the pristine
            # object before the next begin_layer_decode call.
            self.forward_metadata = original_metadata

        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if not get_is_capture_mode():
            controller.end_layer_decode(o=out, layer=layer, forward_batch=forward_batch)
        return out
