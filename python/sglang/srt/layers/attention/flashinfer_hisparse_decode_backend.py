"""FlashInfer decode backend for HiSparse sparse decode.

This backend is the FlashInfer counterpart of ``fa3_sparse_decode``.  It keeps
prefill on the normal dense backend (via ``HybridAttnBackend``) and uses
FlashInfer paged decode for sparse decode by feeding the HiSparse hot-buffer
locations into FlashInfer's paged KV indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    BatchDecodeWithPagedKVCacheWrapper,
    DecodeMetadata,
    FlashInferAttnBackend,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


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
    use_cuda_graph: bool = False


class FlashInferHiSparseDecodeBackend(FlashInferAttnBackend):
    """HiSparse-specific FlashInfer paged decode backend.

    FlashInfer page size is forced to 1 in ``begin_forward`` so token-level
    HiSparse hot-buffer locations can be used directly as ``kv_indices``.
    """

    def __init__(self, model_runner: "ModelRunner", **kwargs):
        # FlashInferAttnBackend does not keep ``model_runner``; this backend needs
        # it for HiSparse coordinator / graph buffers (decode path + CUDA graph buffers).
        self.model_runner = model_runner
        super().__init__(model_runner, skip_prefill=True, **kwargs)
        if self.num_wrappers != 1:
            raise ValueError(
                "flashinfer_hisparse_decode currently supports only one decode "
                "wrapper (no sliding-window or cross-attention dispatch)."
            )
        self._decode_updater = self.indices_updater_decode
        self._cuda_graph_sparse_wrappers = {}
        self._cuda_graph_sparse_kv_indices = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # Sparse decode uses exactly top_k one-token pages per request in the
        # captured graph.  Dense batches are gated out of CUDA Graph for the
        # first graph-compatible implementation.
        coord = getattr(self.model_runner, "hisparse_coordinator", None)
        top_k = coord.top_k if coord is not None else self.max_context_len
        self._cuda_graph_sparse_kv_indices = torch.zeros(
            max_bs * top_k, dtype=torch.int32, device=self.model_runner.device
        )
        super().init_cuda_graph_state(
            max_bs,
            max_num_tokens,
            kv_indices_buf=self._cuda_graph_sparse_kv_indices,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if not forward_batch.forward_mode.is_decode_or_idle():
            return super().init_forward_metadata(forward_batch)

        # ``self.model_runner`` is set in :meth:`__init__`.  Use direct access
        # so a misconfigured coordinator surfaces immediately instead of
        # silently falling through to dense FlashInfer.
        coord = self.model_runner.hisparse_coordinator
        controller = coord.algorithm_controller if coord is not None else None
        if controller is None:
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

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info,
    ):
        if not forward_mode.is_decode_or_idle():
            return super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

        coord = self.model_runner.hisparse_coordinator
        top_k = coord.top_k
        kv_indptr = self.kv_indptr[0][: bs + 1]
        kv_indptr.copy_(
            torch.arange(
                0,
                (bs + 1) * top_k,
                step=top_k,
                dtype=torch.int32,
                device=kv_indptr.device,
            )
        )
        kv_last_page_len = self.kv_last_page_len[:bs]
        kv_last_page_len.fill_(1)
        kv_indices = self._cuda_graph_sparse_kv_indices[: bs * top_k]
        kv_indices.zero_()

        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.decode_backend,
            use_cuda_graph=True,
            use_tensor_cores=self.decode_use_tensor_cores,
            paged_kv_indptr_buffer=kv_indptr,
            paged_kv_indices_buffer=kv_indices,
            paged_kv_last_page_len_buffer=kv_last_page_len,
        )
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self._decode_updater.num_qo_heads,
            self._decode_updater.num_kv_heads,
            self._decode_updater.head_dim,
            1,
            data_type=self._decode_updater.data_type,
            q_data_type=self._decode_updater.q_data_type,
            non_blocking=True,
            fixed_split_size=None,
            disable_split_kv=self.disable_cuda_graph_kv_split,
        )
        self._cuda_graph_sparse_wrappers[bs] = [wrapper]
        self.forward_metadata = self._make_sparse_metadata(
            decode_wrappers=[wrapper],
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_indices=kv_indices,
            use_cuda_graph=True,
            disable_split_kv=self.disable_cuda_graph_kv_split,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: torch.Tensor,
    ):
        if not forward_mode.is_decode_or_idle():
            return super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

        coord = self.model_runner.hisparse_coordinator
        top_k = coord.top_k
        kv_indptr = self.kv_indptr[0][: bs + 1]
        kv_indptr.copy_(
            torch.arange(
                0,
                (bs + 1) * top_k,
                step=top_k,
                dtype=torch.int32,
                device=kv_indptr.device,
            )
        )
        kv_last_page_len = self.kv_last_page_len[:bs]
        kv_last_page_len.fill_(1)
        kv_indices = self._cuda_graph_sparse_kv_indices[: bs * top_k]
        self.forward_metadata = self._make_sparse_metadata(
            decode_wrappers=self._cuda_graph_sparse_wrappers[bs],
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_indices=kv_indices,
            use_cuda_graph=True,
            disable_split_kv=self.disable_cuda_graph_kv_split,
        )

    def _make_sparse_metadata(
        self,
        decode_wrappers,
        kv_indptr,
        kv_last_page_len,
        kv_indices=None,
        use_cuda_graph: bool = False,
        disable_split_kv: bool = False,
    ):
        return FlashInferHiSparseDecodeMetadata(
            decode_wrappers=decode_wrappers,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            num_qo_heads=self._decode_updater.num_qo_heads,
            num_kv_heads=self._decode_updater.num_kv_heads,
            head_dim=self._decode_updater.head_dim,
            data_type=self._decode_updater.data_type,
            q_data_type=self._decode_updater.q_data_type,
            get_wrapper_idx=lambda _layer_id: 0,
            fixed_split_size=None if use_cuda_graph else self.decode_split_tile_size,
            disable_split_kv=disable_split_kv,
            kv_indices=kv_indices,
            kv_indptr_active=kv_indptr,
            use_cuda_graph=use_cuda_graph,
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

        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if not get_is_capture_mode():
            controller.end_layer_decode(o=o, layer=layer, forward_batch=forward_batch)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
