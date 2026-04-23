# mapping on device memory, host memory and memory allocator

import weakref
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, NSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import get_num_new_pages

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer, transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )

    def transfer_kv_all_layer(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseNSATokenToKVPool(NSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=size * host_to_device_ratio,
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")


class HiSparseMHATokenToKVPool(MHATokenToKVPool):
    """MHA counterpart of :class:`HiSparseNSATokenToKVPool`.

    The underlying physical storage is allocated big enough to hold BOTH:
      * the prefill full-logical rows (managed by
        ``HiSparseTokenToKVPoolAllocator.logical_attn_allocator``), and
      * the decode-time device hot buffer (managed by
        ``HiSparseTokenToKVPoolAllocator.hisparse_attn_allocator``).

    ``set_kv_buffer`` translates ``loc`` through the
    ``full_to_hisparse_device_index_mapping`` whenever the mapping entry is
    non-zero.  That entry is non-zero exactly when
    ``HiSparseCoordinator.map_last_loc_to_buffer`` has reserved a hot-buffer
    slot for "the next decode token", mirroring the semantics used by
    :class:`HiSparseNSATokenToKVPool` for MLA.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        host_to_device_ratio: int = 2,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        # Physical storage must hold BOTH the logical (prefill) rows
        # (``size * host_to_device_ratio``) and the decode hot buffer rows
        # (``size``).  The allocator carves non-overlapping sub-ranges, so we
        # size the underlying MHA tensor to their sum.
        physical_size = size * host_to_device_ratio + size
        super().__init__(
            size=physical_size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )
        self.head_num_for_host = head_num
        self.head_dim_for_host = head_dim
        # bytes per token per layer (K or V side only; host pool has separate K/V)
        self.bytes_per_token_one_side = head_num * head_dim * self.dtype.itemsize

        # Populated by the HiSparse allocator (see register_mapping).
        self.full_to_hisparse_device_index_mapping: Optional[torch.Tensor] = None

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def _translate_loc_to_hisparse_device(self, loc: torch.Tensor) -> torch.Tensor:
        """Return raw mapping entries (0 = no hot-buffer slot reserved).

        Matches the semantics of :class:`HiSparseNSATokenToKVPool` so that the
        allocator's ``free_hisparse`` can filter positive entries to release
        hot-buffer rows.
        """
        if self.full_to_hisparse_device_index_mapping is None:
            # No mapping attached yet (e.g. during construction) — fall back
            # to identity so that callers see "no redirection".
            return loc
        return self.full_to_hisparse_device_index_mapping[loc]

    def translate_loc_to_hisparse_device(self, loc: torch.Tensor) -> torch.Tensor:
        return self._translate_loc_to_hisparse_device(loc).to(torch.int32)

    def _resolve_write_loc(self, loc: torch.Tensor) -> torch.Tensor:
        """Decide the physical row that ``set_kv_buffer`` should write to.

        * mapping entry > 0  → write to the reserved hot-buffer slot
          (decode: the coordinator has pre-reserved this row).
        * mapping entry == 0 → write to the original loc (prefill: logical
          rows are directly mapped to the physical tensor).
        * mapping entry  < 0 → sentinel tail of the mapping table, route to
          slot 0 which is the pad row in :class:`MHATokenToKVPool`.
        """
        if self.full_to_hisparse_device_index_mapping is None:
            return loc
        mapped = self.full_to_hisparse_device_index_mapping[loc]
        loc_to_write = torch.where(mapped > 0, mapped, loc)
        loc_to_write = torch.where(
            mapped < 0, torch.zeros_like(loc_to_write), loc_to_write
        )
        return loc_to_write

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc = self._resolve_write_loc(loc)
        super().set_kv_buffer(
            layer,
            loc,
            cache_k,
            cache_v,
            k_scale=k_scale,
            v_scale=v_scale,
            layer_id_override=layer_id_override,
        )

    def transfer_values_on_device(self, dst_indices, src_indices):
        """Device->device transfer used by host-pool backup fallback."""
        if not (_is_cuda or _is_hip):
            raise RuntimeError(
                "HiSparseMHATokenToKVPool.transfer_values_on_device requires "
                "sgl_kernel.kvcacheio (CUDA/ROCm)."
            )
        transfer_kv_all_layer(
            src_k_layers=self.k_data_ptrs,
            dst_k_layers=self.k_data_ptrs,
            src_v_layers=self.v_data_ptrs,
            dst_v_layers=self.v_data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token_one_side,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError(
            "HiSparseMHATokenToKVPool does not support get_cpu_copy"
        )

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError(
            "HiSparseMHATokenToKVPool does not support load_cpu_copy"
        )


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache,
        need_sort: bool,
        host_to_device_ratio: int = 2,
        sparse_decode_mode: bool = False,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort
        # When True the allocator acts as a "logical only during prefill" pool
        # (MHA sparse-decode path): alloc/alloc_extend/alloc_decode allocate
        # rows from the logical pool only; the hisparse hot buffer is grown
        # by HiSparseCoordinator once the request transitions to decode.
        self.sparse_decode_mode = sparse_decode_mode

        self.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_full,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )

        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )
        # In sparse_decode mode both sub-allocators carve distinct RANGES of
        # the same underlying MHA storage tensor (physically sized
        # ``size_full + size`` via ``HiSparseMHATokenToKVPool.__init__``).
        # We shift the hisparse sub-allocator's free pages into the high range
        # ``[size_full_pages, size_full_pages + size_hisparse_pages)``; the
        # shift is re-applied on every :meth:`clear` via
        # :meth:`_reinit_hisparse_free_pages`.
        self._hisparse_page_shift = (
            self._size_full // self.page_size if self.sparse_decode_mode else 0
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_full + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self._kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    def _reinit_hisparse_free_pages(self) -> None:
        if not getattr(self, "_hisparse_page_shift", 0):
            return
        size_hisparse_pages = self._size_hisparse // self.page_size
        self.hisparse_attn_allocator.free_pages = torch.arange(
            self._hisparse_page_shift,
            self._hisparse_page_shift + size_hisparse_pages,
            dtype=torch.int64,
            device=self.device,
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    def available_size(self) -> int:
        if self.sparse_decode_mode:
            # In sparse_decode mode only the logical pool is consumed during
            # prefill; the hot buffer is managed implicitly by
            # HiSparseCoordinator (and is strictly smaller than the logical
            # pool), so reporting the logical pool's availability is the
            # correct bound.
            return self.logical_attn_allocator.available_size()
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def alloc(self, need_size: int):
        if self.sparse_decode_mode:
            # Allocate logical rows only; hisparse hot buffer slots are
            # reserved later by HiSparseCoordinator.map_last_loc_to_buffer
            # (per decode token) or admit_request_from_gpu (prefill done).
            return self.logical_attn_allocator.alloc(need_size)
        raise NotImplementedError(
            "Page size = 1 is not supported in HiSparse allocator"
        )

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate only logical indices without hisparse device indices.

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging.
        """
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(hisparse_indices[need_size:])
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor):
        # disable free group mechanism for device buffer free
        self.hisparse_attn_allocator.is_not_in_free_group = True
        self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0])

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        hisparse_last_locs = self._kvcache._translate_loc_to_hisparse_device(last_locs)
        return hisparse_last_locs

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        if self.sparse_decode_mode:
            # Logical-only allocation for sparse_decode prefill.  Hot buffer
            # slots are reserved later by HiSparseCoordinator.
            return self.logical_attn_allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
            )

        assert self.page_size > 1

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if (
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
            len(logical_indices),
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        return logical_indices

    def alloc_decode_debug(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_decode(
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
        )

        if logical_indices is None or hisparse_indices is None:
            return None

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        return logical_indices

    def free_hisparse(self, free_indices: torch.Tensor):
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_indices)
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[free_indices] = 0

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()
        self._reinit_hisparse_free_pages()

        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.sparse_decode_mode:
                # Only the logical side owns these rows in sparse_decode mode
                # (hot-buffer rows are released via ``free_hisparse_indices``
                # by HiSparseCoordinator).  Still call free_hisparse to clear
                # any residual mapping entries set for reserved slots.
                self.logical_attn_allocator.free(free_index)
                self.free_hisparse(free_index)
            else:
                self.logical_attn_allocator.free(free_index)
                self.free_hisparse(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert (
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )
