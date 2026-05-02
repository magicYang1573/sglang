"""FlashInfer decode backend for HiSparse sparse decode.

This backend is the FlashInfer counterpart of ``fa3_sparse_decode``.  It keeps
prefill on the normal dense backend (via ``HybridAttnBackend``) and uses
FlashInfer paged decode for sparse decode by feeding the HiSparse hot-buffer
locations into FlashInfer's paged KV indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    BatchDecodeWithPagedKVCacheWrapper,
    DecodeMetadata,
    FlashInferAttnBackend,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


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

        coord = getattr(self, "model_runner", None)
        coord = getattr(coord, "hisparse_coordinator", None)
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
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_buffer,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        controller.end_layer_decode(o=o, layer=layer, forward_batch=forward_batch)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
