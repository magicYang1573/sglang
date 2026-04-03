"""CXL Memory Pool for HiSparse KV Cache.

Uses sgl-kernel/cxl_utils (cxl_mem_ext) for CXL device initialization
(mmap + cudaHostRegister), then constructs torch.Tensor on top of the
registered CXL base pointer via torch.frombuffer (zero-copy).

Data transfer does NOT use cxl_mem_ext's copy functions — all reads go
through HiSparse's existing JIT kernel (ld.global.nc.b64) and cudaMemcpy,
which achieve ~1.0-1.7x DRAM latency on CXL.
"""

from __future__ import annotations

import ctypes
import logging
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


def _load_cxl_ext():
    """Load the cxl_mem_ext C++ extension (built from sgl-kernel/cxl_utils)."""
    try:
        import cxl_mem_ext

        return cxl_mem_ext
    except ImportError:
        raise ImportError(
            "cxl_mem_ext not found. Build it from sgl-kernel/cxl_utils: "
            "cd sgl-kernel/cxl_utils && pip install -e ."
        )


class CXLMemoryRegion:
    """Manages a single CXL DAX device mmap region with CUDA registration.

    Uses the proven cxl_mem_ext C++ library for the low-level init:
        mmap -> cudaSetDevice -> cudaHostRegister -> cudaHostGetDevicePointer

    Then provides make_tensor() to allocate torch.Tensor views on top of
    the registered CXL memory via torch.frombuffer (zero-copy).

    Lifecycle:
        1. cxl_ext.cxl_init() — mmap + optional cudaHostRegister
        2. make_tensor() — bump-allocate + torch.frombuffer
        3. close() — cxl_ext.cxl_close() (unregister + munmap)
    """

    def __init__(self, config: CXLConfig):
        self.config = config
        self._cxl_ext = None
        self._base_ptr: int = 0
        self._size: int = 0
        self._initialized: bool = False
        self._alloc_offset: int = 0

    def init(self) -> None:
        """Initialize CXL region via cxl_mem_ext C++ library."""
        self._cxl_ext = _load_cxl_ext()
        dev_path = self.config.dev_path
        raw_size = self.config.map_bytes

        # DAX devices require mmap size aligned to huge page boundary (2MB)
        aligned_size = (
            (raw_size + HUGE_PAGE_2MB - 1) // HUGE_PAGE_2MB
        ) * HUGE_PAGE_2MB

        ok = self._cxl_ext.cxl_init(
            dev_path=dev_path,
            map_bytes=aligned_size,
            register_cuda=True,
            gpu_id=self.config.gpu_id,
        )
        if not ok:
            raise RuntimeError(
                f"cxl_init failed for {dev_path} (size={aligned_size}). "
                f"Check DAX device permissions and CUDA driver compatibility."
            )

        self._base_ptr = self._cxl_ext.get_cxl_base_ptr()
        if self._base_ptr == 0:
            self._cxl_ext.cxl_close()
            raise RuntimeError("CXL base_ptr is null after cxl_init")

        device_ptr = self._cxl_ext.get_cxl_device_ptr()
        if device_ptr == 0:
            self._cxl_ext.cxl_close()
            raise RuntimeError(
                "CXL device_ptr is null; cudaHostRegister may have failed"
            )

        self._size = aligned_size
        self._initialized = True

        logger.info(
            "CXL region initialized: %s, size=%.2f GB, gpu_id=%d, "
            "base_ptr=0x%x, device_ptr=0x%x",
            dev_path,
            aligned_size / 1e9,
            self.config.gpu_id,
            self._base_ptr,
            device_ptr,
        )

    @property
    def base_ptr(self) -> int:
        return self._base_ptr

    @property
    def size(self) -> int:
        return self._size

    def allocate(self, nbytes: int, alignment: int = 64) -> int:
        """Bump-allocate from the CXL region. Returns byte offset."""
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
        cudaMemcpy can access it through this pointer.
        """
        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * dtype.itemsize

        offset = self.allocate(nbytes)
        ptr = self._base_ptr + offset

        c_array = (ctypes.c_byte * nbytes).from_address(ptr)
        tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(dims)
        tensor.zero_()

        return tensor

    def close(self) -> None:
        """Release CXL region (cudaHostUnregister + munmap)."""
        if self._initialized and self._cxl_ext is not None:
            self._cxl_ext.cxl_close()
            self._initialized = False
            logger.info("CXL region closed")

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
        # Bypass psutil host memory check — CXL memory is not visible to it.
        self._skip_host_mem_check = True
        super().__init__(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=False,  # CXL already cudaHostRegister'd
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=override_kv_cache_dim,
        )

    def init_kv_buffer(self):
        """Override: allocate KV buffer on CXL memory instead of DRAM."""
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
