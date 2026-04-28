"""FA3 attention backend subclass for the non-native sparse-decode path.

Inherits :class:`FlashAttentionBackend` for everything except
``forward_decode``.  For decode, this class:

  1. writes the new token's K/V using the standard pool ``set_kv_buffer``;
  2. asks the controller for a :class:`SparseFa3CallArgs` package
     (``page_table`` / ``cache_seqlens`` / ``cu_seqlens_q`` /
     ``cu_seqlens_k`` / ``max_seqlen_k``);
  3. directly calls ``flash_attn_with_kvcache`` with those args.

The shared FA3 ``self.forward_metadata`` is **never read or mutated** on
sparse rows — dense and sparse paths run on disjoint tensors (see design
doc :file:`docs/advanced_features/hisparse_sparse_decode_design.md` §6
and §10).

Not imported / held by :class:`ModelRunner` directly.  Registered through
:mod:`attention_registry` and selected via
``ServerArgs.get_attention_backends()`` rewriting
``decode_attention_backend`` to ``"fa3_sparse_decode"`` when
``--enable-sparse-decode`` is set.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)
# Set SGLANG_SPARSE_ACCURACY_DEBUG=1 to diagnose sparse decode correctness.
_SPARSE_ACCURACY_DEBUG = os.environ.get("SGLANG_SPARSE_ACCURACY_DEBUG", "0") == "1"
_SPARSE_ACCURACY_DEBUG_MAX_REQS = int(
    os.environ.get("SGLANG_SPARSE_ACCURACY_DEBUG_MAX_REQS", "4")
)


class Fa3SparseDecodeBackend(FlashAttentionBackend):
    """FA3 backend that delegates per-layer decode to the sparse path.

    First release scope (mirrors §3.4 / §6.3 of the design doc):
      * Plain self-attention only — no cross attention, sliding window, MLA,
        cascade attention, or fp8 KV.  These cases fall back to the parent
        class' dense ``forward_decode`` for safety.
      * ``page_size = 1`` (validated by ServerArgs guard).
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

        # Fall back to dense FA3 when:
        # 1. there's no sparse controller (e.g. the hisparse coordinator runs
        #    in the legacy DSA mode, or sparse_decode is disabled);
        # 2. the layer / KV layout is not in the supported subset (cross
        #    attention, SWA, MLA, or cascade attention).
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        use_cascade_attn = (
            forward_batch.spec_info is not None and self.topk > 1
        )
        if (
            controller is None
            or layer.is_cross_attention
            or is_swa_layer
            or self.use_mla
            or use_cascade_attn
        ):
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

        # 1. Standard KV write of the *new* token (matches parent class).
        if k is not None and save_kv_cache:
            cache_loc = forward_batch.out_cache_loc
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        # 2. Controller produces a sparse FA3 call-arg pack for this layer.
        sparse_args = controller.begin_layer_decode(
            q=q,
            layer=layer,
            forward_batch=forward_batch,
            attn_metadata=self.forward_metadata,
        )
        # 3. K/V buffers (page_size=1 makes the view nearly a no-op).
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        # 4. Direct FA3 sparse call (bypasses the parent's dense metadata path).
        #
        # IMPORTANT signal/mask conventions:
        #   * ``page_table`` MUST contain valid (non-negative) row indices in
        #     every slot, even for padded slots beyond ``cache_seqlens[i]``.
        #     FA3 may speculatively prefetch ``page_table[i, j]`` regardless
        #     of whether ``j < cache_seqlens[i]`` (cache_seqlens is only used
        #     for masking attention scores).  A ``-1`` entry would be
        #     interpreted as an enormous unsigned offset and cause
        #     ``cudaErrorIllegalAddress``.  We therefore replace ``-1`` with
        #     ``0`` (slot 0 is always valid in HiSparseMHATokenToKVPool
        #     because the pool is sized for the full logical space).  The
        #     attention output for those padded slots is then ignored by the
        #     ``cache_seqlens`` mask.
        safe_page_table = torch.where(
            sparse_args.page_table >= 0,
            sparse_args.page_table,
            torch.zeros_like(sparse_args.page_table),
        )
        if _SPARSE_ACCURACY_DEBUG and layer.layer_id == 0:
            self._log_sparse_call_diagnostics_inputs(
                forward_batch=forward_batch,
                sparse_args=sparse_args,
                safe_page_table=safe_page_table,
                key_cache_rows=key_cache.shape[0],
            )

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        # Note: we deliberately do NOT pass ``cu_seqlens_k_new`` — that
        # parameter is only needed when FA3 itself is appending new K/V to
        # the cache (incremental decode with k/v args).  We already wrote the
        # new token via ``set_kv_buffer`` above, so the cache is up-to-date
        # and FA3 just reads it.  The plain decode path of the parent class
        # (``FlashAttentionBackend.forward_decode``, line 1162) follows the
        # same convention.
        out = flash_attn_with_kvcache(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=safe_page_table,
            cache_seqlens=sparse_args.cache_seqlens,
            cu_seqlens_q=sparse_args.cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=layer.scaling,
            causal=True,
            window_size=(-1, -1),
            softcap=layer.logit_cap,
            num_splits=self.num_splits,
            ver=self.fa_impl_ver,
            **kwargs,
        )

        # Reshape output back to ``(bs, num_heads * v_head_dim)`` so the
        # downstream ``o_proj`` linear layer receives a 2-D tensor (matches
        # the parent class' final reshape; see ``flashattention_backend.py``
        # line 1292).
        out = out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        if _SPARSE_ACCURACY_DEBUG and layer.layer_id == 0:
            self._log_sparse_call_diagnostics_output(out)

        controller.end_layer_decode(o=out, layer=layer, forward_batch=forward_batch)
        return out

    def _log_sparse_call_diagnostics_inputs(
        self,
        forward_batch: "ForwardBatch",
        sparse_args,
        safe_page_table: torch.Tensor,
        key_cache_rows: int,
    ) -> None:
        sample_n = min(forward_batch.batch_size, _SPARSE_ACCURACY_DEBUG_MAX_REQS)
        if sample_n <= 0:
            return

        seq_sample = forward_batch.seq_lens[:sample_n]
        cache_sample = sparse_args.cache_seqlens[:sample_n]
        expected = seq_sample.clamp(max=safe_page_table.shape[1]).to(cache_sample.dtype)
        mismatch = int((cache_sample != expected).sum().item())

        logger.info(
            "[SPARSE/diag] fa3 input summary: sample=%d top_k=%d cache_len_mismatch=%d "
            "seq_lens=%s cache_seqlens=%s expected=%s cu_seqlens_q=%s",
            sample_n,
            safe_page_table.shape[1],
            mismatch,
            seq_sample.cpu().tolist(),
            cache_sample.cpu().tolist(),
            expected.cpu().tolist(),
            sparse_args.cu_seqlens_q[: sample_n + 1].cpu().tolist(),
        )

        page_table_cpu = safe_page_table[:sample_n].cpu()
        cache_cpu = cache_sample.cpu().tolist()
        for i in range(sample_n):
            cache_len = max(0, min(int(cache_cpu[i]), safe_page_table.shape[1]))
            row = page_table_cpu[i, :cache_len]
            unique_i = int(torch.unique(row).numel()) if cache_len > 0 else 0
            oob_i = int(((row < 0) | (row >= key_cache_rows)).sum().item())
            logger.info(
                "[SPARSE/diag] fa3 input req_idx=%d cache_len=%d unique_pages=%d "
                "oob_pages=%d page_head=%s page_tail=%s",
                int(forward_batch.req_pool_indices[i].item()),
                cache_len,
                unique_i,
                oob_i,
                row[:8].tolist(),
                row[max(0, cache_len - 8) : cache_len].tolist(),
            )

    def _log_sparse_call_diagnostics_output(self, out: torch.Tensor) -> None:
        finite_mask = torch.isfinite(out)
        non_finite = int((~finite_mask).sum().item())
        max_abs = float(out.abs().max().item()) if out.numel() > 0 else 0.0
        mean = float(out.mean().item()) if out.numel() > 0 else 0.0
        std = float(out.std().item()) if out.numel() > 1 else 0.0
        logger.info(
            "[SPARSE/diag] fa3 output: shape=%s non_finite=%d max_abs=%.6f mean=%.6f std=%.6f",
            tuple(out.shape),
            non_finite,
            max_abs,
            mean,
            std,
        )
