"""HiSparse memory pool specialization for MHA / GQA models (e.g. Qwen).

This mirrors :mod:`sglang.srt.mem_cache.hisparse_memory_pool` (which specializes
for MLA / DSA), but targets ``MHATokenToKVPool`` where K and V live in separate
buffers.

The HiSparse addressing scheme is unchanged:

* ``logical_attn_allocator``  manages a large "logical" page pool backed by
  host memory capacity (size = ``device_size * host_to_device_ratio``).
* ``hisparse_attn_allocator``  manages the small per-layer device KV buffer.
* ``full_to_hisparse_device_index_mapping`` maps logical indices to device
  buffer indices (zero means "currently not resident on device").
"""

from __future__ import annotations

import weakref
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.utils.common import get_num_new_pages


class HiSparseMHATokenToKVPool(MHATokenToKVPool):
    """Device-side KV pool for HiSparse + MHA/GQA models.

    Identical to ``MHATokenToKVPool`` on the storage side, but exposes the
    logical→hisparse address translation helpers that the sparse coordinator
    and the FlashAttention backend adapter rely on.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.host_to_device_ratio = host_to_device_ratio
        # Per-token stride in bytes (K and V share the same stride for Qwen).
        self.bytes_per_token = self.head_num * self.head_dim * self.dtype.itemsize
        self.v_bytes_per_token = (
            self.head_num * self.v_head_dim * self.dtype.itemsize
        )
        assert self.bytes_per_token == self.v_bytes_per_token, (
            "HiSparseMHATokenToKVPool assumes K and V share per-token stride; "
            "got K={} bytes, V={} bytes".format(
                self.bytes_per_token, self.v_bytes_per_token
            )
        )

        self.full_to_hisparse_device_index_mapping: Optional[torch.Tensor] = None

    def register_mapping(
        self, full_to_hisparse_device_index_mapping: torch.Tensor
    ) -> None:
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)

    # Some utility methods that HiCache implements are intentionally
    # not supported in the HiSparse flow (see MLA variant for parity).
    def get_cpu_copy(self, indices):
        raise NotImplementedError(
            "HiSparseMHATokenToKVPool does not support get_cpu_copy"
        )

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError(
            "HiSparseMHATokenToKVPool does not support load_cpu_copy"
        )


class HiSparseMHATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator mirroring :class:`HiSparseTokenToKVPoolAllocator` for MHA.

    It keeps a logical pool (host-backed) and a hisparse pool (device-backed).
    The mapping ``logical → hisparse`` is populated on ``alloc_extend`` /
    ``alloc_decode`` and consulted whenever we need to translate a
    ``out_cache_loc`` back to the small device KV buffer slot.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseMHATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort

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

        # Layout mirrors the MLA variant exactly: last slot is -1 so that
        # stray ``-1`` indices resolve to -1 after translation.
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

    @property
    def size_full(self) -> int:
        return self._size_full

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def alloc(self, need_size: int):
        # With ``--page-size 1`` the scheduler's ``alloc_for_extend`` /
        # ``alloc_for_decode`` both collapse to
        # ``alloc_token_slots -> allocator.alloc(need_size)`` (see
        # ``sglang.srt.mem_cache.common``).  We must therefore service
        # **both** the prefill/extend and the decode call paths here.
        #
        # Semantics (page_size == 1):
        # * Always allocate ``need_size`` logical slots -- these are what
        #   the scheduler writes into ``req_to_token_pool.req_to_token``.
        # * During prefill/extend, also allocate matching hisparse slots
        #   and populate ``full_to_hisparse_device_index_mapping`` so that
        #   ``HiSparseMHATokenToKVPool.set_kv_buffer`` writes KV at the
        #   correct hisparse-device location.  ``alloc_device_buffer``
        #   will later "reclaim" these hisparse slots into the per-request
        #   hot buffer and clear the mapping.
        # * During decode, the coordinator's ``map_last_loc_to_buffer``
        #   overwrites the mapping with a hot-buffer slot *after* this
        #   function returns.  It is cheap and harmless to still pair a
        #   fresh hisparse slot here: (a) decode K/V also flows through
        #   ``set_kv_buffer`` *before* ``map_last_loc_to_buffer`` on the
        #   very first step, so the mapping needs a temporary value; (b)
        #   the soon-to-be-unused hisparse slot is freed by the
        #   coordinator via ``alloc_device_buffer`` / grow-path or by
        #   ``free_hisparse`` on request finish.
        # * For ``page_size > 1`` the scheduler routes extend/decode
        #   through ``alloc_extend`` / ``alloc_decode`` directly; this
        #   ``alloc(need_size)`` entry point is not used (aside from
        #   ``alloc_device_buffer`` top-ups, which hit
        #   ``hisparse_attn_allocator.alloc`` directly).
        if self.page_size != 1:
            raise NotImplementedError(
                "HiSparseMHATokenToKVPoolAllocator.alloc(need_size) is only "
                "defined for page_size==1 (the Quest + FA3 MVP path); for "
                "page_size>1 use alloc_extend / alloc_decode."
            )

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None
        hisparse_indices = self.hisparse_attn_allocator.alloc(need_size)
        if hisparse_indices is None:
            # Rollback the logical reservation so the caller's OOM path
            # observes a consistent state.
            self.logical_attn_allocator.free(logical_indices)
            return None
        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
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
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(hisparse_indices[need_size:])
        else:
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
        self.hisparse_attn_allocator.is_not_in_free_group = True
        self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0])

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self._kvcache._translate_loc_to_hisparse_device(last_locs)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        # ``page_size == 1`` is the Quest + FA3 MVP (see
        # docs/advanced_features/hisparse_quest_qwen_design.md §1.4).  For
        # that case ``get_num_new_pages`` degenerates to
        # ``extend_num_tokens`` and the underlying
        # ``PagedTokenToKVPoolAllocator.alloc_extend`` still works (page ==
        # token).  No special branching needed here.

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
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
        last_loc: torch.Tensor,
    ):
        # Under HiSparse, new decode tokens land into the hot buffer managed by
        # ``HiSparseCoordinator.map_last_loc_to_buffer``.  We only need a
        # logical slot here; the hisparse slot for the fresh token is patched
        # via ``full_to_hisparse_device_index_mapping[out_cache_loc] = ...`` in
        # ``HiSparseCoordinator.map_last_loc_to_buffer``.
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_hisparse(self, free_indices: torch.Tensor):
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(
            free_indices
        )
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[free_indices] = 0

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()
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
            self.logical_attn_allocator.free(free_index)
            self.free_hisparse(free_index)
        else:
            self.free_group.append(free_index)
