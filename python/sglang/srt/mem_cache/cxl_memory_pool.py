"""CXL Memory Pool for HiSparse KV Cache.

Manages CXL DAX device mmap and provides torch.Tensor backed by CXL memory.
Does NOT use sgl-kernel/cxl_utils for data transfer — all operations through
standard OS APIs (mmap) and CUDA APIs (cudaHostRegister, torch.frombuffer).

After cudaHostRegister, GPU kernels (including HiSparse's ld.global.nc.b64
swap-in kernel) can directly access CXL memory with ~1.0-1.7x DRAM latency.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

logger = logging.getLogger(__name__)

HUGE_PAGE_2MB = 2 * 1024 * 1024


@dataclass
class CXLConfig:
    """CXL device configuration parsed from hisparse_config JSON."""

    dev_path: str = "/dev/dax0.0"
    map_bytes: int = 64 * 1024 * 1024 * 1024  # 64 GB default
    gpu_id: int = 0


class CXLMemoryRegion:
    """Manages a single CXL DAX device mmap region with CUDA registration.

    Lifecycle:
        1. open DAX device -> mmap (2MB-aligned, MAP_SHARED)
        2. cudaHostRegister (Portable | Mapped)
        3. Allocate tensors via make_tensor (torch.frombuffer, zero-copy)
        4. All GPU DMA and kernel access through registered pointers
        5. Close: cudaHostUnregister -> munmap -> close fd
    """

    def __init__(self, config: CXLConfig):
        self.config = config
        self._fd: int = -1
        self._mmap: Optional[mmap.mmap] = None
        self._base_ptr: int = 0
        self._size: int = 0
        self._cuda_registered: bool = False
        self._alloc_offset: int = 0

    def init(self) -> None:
        """Open the DAX device, mmap it, and register with CUDA."""
        dev_path = self.config.dev_path
        raw_size = self.config.map_bytes

        # DAX devices require mmap size aligned to huge page boundary (2MB)
        aligned_size = (
            (raw_size + HUGE_PAGE_2MB - 1) // HUGE_PAGE_2MB
        ) * HUGE_PAGE_2MB

        self._fd = os.open(dev_path, os.O_RDWR)
        try:
            self._mmap = mmap.mmap(
                self._fd,
                aligned_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except OSError as e:
            os.close(self._fd)
            self._fd = -1
            raise RuntimeError(
                f"CXL mmap failed for {dev_path} (size={aligned_size}): {e}. "
                f"Check DAX device permissions and ensure size is 2MB-aligned."
            ) from e

        buf = ctypes.c_char.from_buffer(self._mmap)
        self._base_ptr = ctypes.addressof(buf)
        self._size = aligned_size

        # cudaHostRegisterPortable (0x01) | cudaHostRegisterMapped (0x02)
        flags = 0x01 | 0x02
        torch.cuda.set_device(self.config.gpu_id)
        ret = torch.cuda.cudart().cudaHostRegister(
            ctypes.c_void_p(self._base_ptr), self._size, flags
        )
        if ret[0] != 0:
            self._cleanup_mmap()
            raise RuntimeError(
                f"cudaHostRegister failed for CXL region (ret={ret[0]}). "
                f"Check CUDA driver compatibility with CXL DAX devices."
            )
        self._cuda_registered = True

        logger.info(
            "CXL region initialized: %s, size=%.2f GB, gpu_id=%d, base_ptr=0x%x",
            dev_path,
            aligned_size / 1e9,
            self.config.gpu_id,
            self._base_ptr,
        )

    @property
    def base_ptr(self) -> int:
        return self._base_ptr

    @property
    def size(self) -> int:
        return self._size

    def allocate(self, nbytes: int, alignment: int = 64) -> int:
        """Bump-allocate from the CXL region. Returns byte offset.

        Thread safety: this is only called during initialization,
        not on the hot path.
        """
        offset = (self._alloc_offset + alignment - 1) & ~(alignment - 1)
        if offset + nbytes > self._size:
            raise RuntimeError(
                f"CXL region exhausted: need {nbytes} bytes at offset {offset}, "
                f"but only {self._size - offset} available"
            )
        self._alloc_offset = offset + nbytes
        return offset

    def make_tensor(self, dims: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Allocate a torch.Tensor backed by CXL memory (zero-copy).

        The returned tensor's data_ptr() points directly into the CXL mmap
        region. Since the region is cudaHostRegister'd, GPU kernels and
        cudaMemcpy can access it through this pointer — identical to pinned
        DRAM semantics.
        """
        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * dtype.itemsize

        offset = self.allocate(nbytes)
        ptr = self._base_ptr + offset

        c_array = (ctypes.c_byte * nbytes).from_address(ptr)
        tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(dims)

        # Zero-initialize. GPU DMA bypasses CPU cache, so no clflush needed
        # for subsequent GPU access.
        tensor.zero_()

        return tensor

    def close(self) -> None:
        """Release CUDA registration and unmap."""
        if self._cuda_registered:
            torch.cuda.cudart().cudaHostUnregister(ctypes.c_void_p(self._base_ptr))
            self._cuda_registered = False
        self._cleanup_mmap()
        logger.info("CXL region closed")

    def _cleanup_mmap(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self):
        self.close()


class CXLMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool backed by CXL memory.

    Inherits ALL swap-in / backup / load logic from MLATokenToKVPoolHost.
    Only overrides memory allocation to use CXL mmap instead of pinned DRAM.

    After init:
        - self.kv_buffer.data_ptr() points to CXL memory
        - self.data_ptrs contains per-layer CXL pointers on GPU
        - All existing transfer_kv_* functions and JIT kernels work unchanged
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        cxl_region: CXLMemoryRegion,
        override_kv_cache_dim: Optional[int] = None,
    ):
        self._cxl_region = cxl_region
        # pin_memory=False: the CXL region is already cudaHostRegister'd in
        # CXLMemoryRegion.init(), so HostKVCache.__init__ should NOT call
        # cudaHostRegister again via alloc_with_host_register.
        #
        # NOTE: we set _skip_host_mem_check = True to bypass psutil check
        # in HostKVCache.__init__, because CXL memory is not visible to
        # psutil.virtual_memory().
        self._skip_host_mem_check = True
        super().__init__(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=False,
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=override_kv_cache_dim,
        )

    def init_kv_buffer(self):
        """Override: allocate KV buffer on CXL memory instead of DRAM.

        Uses CXLMemoryRegion.make_tensor() which returns a zero-copy
        torch.Tensor backed by the CXL mmap region.
        """
        if self.layout == "layer_first":
            dims = (self.layer_num, self.size, 1, self.kv_cache_dim)
        elif self.layout == "page_first":
            dims = (self.size, self.layer_num, 1, self.kv_cache_dim)
        else:
            raise ValueError(f"Unsupported layout for CXL: {self.layout}")

        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        buffer = self._cxl_region.make_tensor(dims, self.dtype)

        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * self.dtype.itemsize
        logger.info(
            "CXL KV buffer allocated: dims=%s, %.2f GB, layout=%s, "
            "data_ptr=0x%x (CXL mmap)",
            dims,
            nbytes / 1e9,
            self.layout,
            buffer.data_ptr(),
        )
        return buffer
