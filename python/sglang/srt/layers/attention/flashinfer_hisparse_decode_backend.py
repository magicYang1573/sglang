"""FlashInfer decode backend for HiSparse sparse decode.

This backend is the FlashInfer counterpart of ``fa3_sparse_decode``.  It keeps
prefill on the normal dense backend (via ``HybridAttnBackend``) and uses
FlashInfer paged decode for sparse decode by feeding the HiSparse hot-buffer
locations into FlashInfer's paged KV indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, List

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    BatchDecodeWithPagedKVCacheWrapper,
    DecodeMetadata,
    FlashInferAttnBackend,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class FlashInferHiSparseDecodeMetadata(DecodeMetadata):
    kv_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    data_type: torch.dtype
    q_data_type: torch.dtype
    get_wrapper_idx: Callable[[int], int]
    fixed_split_size: Any = None
    disable_split_kv: bool = False
    kv_indices: torch.Tensor = None
    kv_indptr_active: torch.Tensor = None


class FlashInferHiSparseDecodeBackend(FlashInferAttnBackend):
    """HiSparse-specific FlashInfer paged decode backend.

    FlashInfer page size is forced to 1 in ``begin_forward`` so token-level
    HiSparse hot-buffer locations can be used directly as ``kv_indices``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, skip_prefill=True, **kwargs)
        if self.num_wrappers != 1:
            raise ValueError(
                "flashinfer_hisparse_decode currently supports only one decode "
                "wrapper (no sliding-window or cross-attention dispatch)."
            )
        self._decode_updater = self.indices_updater_decode

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if not forward_batch.forward_mode.is_decode_or_idle():
            return super().init_forward_metadata(forward_batch)

        coord = getattr(forward_batch, "hisparse_coordinator", None)
        controller = getattr(coord, "algorithm_controller", None) if coord else None
        if controller is None or getattr(coord, "_all_dense_this_step", False):
            self._hisparse_sparse_active = False
            return super().init_forward_metadata(forward_batch)

        self._hisparse_sparse_active = True
        self.forward_metadata = FlashInferHiSparseDecodeMetadata(
            decode_wrappers=self.decode_wrappers,
            kv_indptr=self.kv_indptr[0],
            kv_last_page_len=self.kv_last_page_len,
            num_qo_heads=self._decode_updater.num_qo_heads,
            num_kv_heads=self._decode_updater.num_kv_heads,
            head_dim=self._decode_updater.head_dim,
            data_type=self._decode_updater.data_type,
            q_data_type=self._decode_updater.q_data_type,
            get_wrapper_idx=lambda _layer_id: 0,
            fixed_split_size=self.decode_split_tile_size,
            disable_split_kv=False,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        coord = getattr(forward_batch, "hisparse_coordinator", None)
        controller = getattr(coord, "algorithm_controller", None) if coord else None
        if controller is None or not getattr(self, "_hisparse_sparse_active", False):
            return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)

        controller.begin_layer_decode(
            q=q,
            layer=layer,
            forward_batch=forward_batch,
            attn_metadata=self.forward_metadata,
        )

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        decode_wrapper = self.forward_metadata.decode_wrappers[0]
        raw_k, raw_v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        page_size = forward_batch.token_to_kv_pool.page_size
        kv_buffer = (
            raw_k.view(-1, page_size, layer.tp_k_head_num, layer.head_dim),
            raw_v.view(-1, page_size, layer.tp_v_head_num, layer.v_head_dim),
        )
        if (
            os.environ.get("SGLANG_HISPARSE_FLASHINFER_DEBUG", "0") == "1"
            and layer.layer_id
            < int(os.environ.get("SGLANG_HISPARSE_FLASHINFER_DEBUG_LAYERS", "4"))
        ):
            logged_backend_layers = getattr(
                self.forward_metadata, "_debug_backend_logged_layers", set()
            )
            if layer.layer_id not in logged_backend_layers:
                logger.warning(
                    "FlashInfer HiSparse backend debug: layer_id=%s q_shape=%s q_dtype=%s "
                    "k_shape=%s v_shape=%s k_stride=%s v_stride=%s page_size=%s",
                    layer.layer_id,
                    list(q.shape),
                    str(q.dtype),
                    list(kv_buffer[0].shape),
                    list(kv_buffer[1].shape),
                    list(kv_buffer[0].stride()),
                    list(kv_buffer[1].stride()),
                    page_size,
                )
                logged_backend_layers.add(layer.layer_id)
                self.forward_metadata._debug_backend_logged_layers = logged_backend_layers
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_buffer,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        self._maybe_compare_flashinfer_with_torch_ref(
            q=q,
            o=o,
            kv_buffer=kv_buffer,
            layer=layer,
        )

        controller.end_layer_decode(o=o, layer=layer, forward_batch=forward_batch)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _maybe_compare_flashinfer_with_torch_ref(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        kv_buffer,
        layer: RadixAttention,
    ) -> None:
        if os.environ.get("SGLANG_HISPARSE_FLASHINFER_COMPARE", "0") != "1":
            return
        max_layers = int(os.environ.get("SGLANG_HISPARSE_FLASHINFER_COMPARE_LAYERS", "4"))
        if layer.layer_id >= max_layers:
            return
        compared_layers = getattr(self.forward_metadata, "_debug_compare_layers", set())
        if layer.layer_id in compared_layers:
            return
        kv_indices = getattr(self.forward_metadata, "kv_indices", None)
        kv_indptr = getattr(self.forward_metadata, "kv_indptr_active", None)
        if kv_indices is None or kv_indptr is None:
            return

        row = 0
        start = int(kv_indptr[row].detach().cpu().item())
        end = int(kv_indptr[row + 1].detach().cpu().item())
        idx = kv_indices[start:end].to(torch.long)
        if idx.numel() == 0:
            return

        q_heads = layer.tp_q_head_num
        kv_heads = layer.tp_k_head_num
        group = q_heads // kv_heads
        q_row = q.contiguous().view(-1, q_heads, layer.head_dim)[row].to(torch.float32)
        k_row = kv_buffer[0][idx, 0].to(torch.float32)
        v_row = kv_buffer[1][idx, 0].to(torch.float32)

        if group > 1:
            k_row = k_row.repeat_interleave(group, dim=1)
            v_row = v_row.repeat_interleave(group, dim=1)

        scores = torch.einsum("hd,thd->ht", q_row, k_row) * layer.scaling
        if layer.logit_cap is not None and layer.logit_cap > 0:
            scores = layer.logit_cap * torch.tanh(scores / layer.logit_cap)
        probs = torch.softmax(scores, dim=-1)
        ref = torch.einsum("ht,thd->hd", probs, v_row)
        flash = o.view(-1, q_heads, layer.v_head_dim)[row].to(torch.float32)
        diff = (flash - ref).abs()
        logger.warning(
            "FlashInfer HiSparse torch-ref compare: layer_id=%d row=%d len=%d "
            "max_abs_diff=%s mean_abs_diff=%s flash_norm=%s ref_norm=%s "
            "idx_head=%s idx_tail=%s",
            layer.layer_id,
            row,
            int(idx.numel()),
            float(diff.max().detach().cpu().item()),
            float(diff.mean().detach().cpu().item()),
            float(flash.norm().detach().cpu().item()),
            float(ref.norm().detach().cpu().item()),
            idx[:16].detach().cpu().tolist(),
            idx[-16:].detach().cpu().tolist(),
        )
        compared_layers.add(layer.layer_id)
        self.forward_metadata._debug_compare_layers = compared_layers
