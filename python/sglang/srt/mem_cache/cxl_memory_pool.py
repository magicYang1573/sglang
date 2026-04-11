"""CXL Memory Pool for HiSparse KV Cache.

Uses sgl-kernel/cxl_utils (cxl_mem_ext) for CXL device initialization
(mmap + cudaHostRegister), then constructs torch.Tensor on top of the
registered CXL base pointer via torch.frombuffer (zero-copy).

Data transfer does NOT use cxl_mem_ext's copy functions — all reads go
through HiSparse's existing JIT kernel (ld.global.nc.b64) and cudaMemcpy,
which achieve ~1.0-1.7x DRAM latency on CXL.

Multi-device interleave: when multiple CXL DAX devices are configured,
ranks are distributed across devices via ``tp_rank % num_devices``.  This
spreads concurrent GPU reads across independent PCIe/CXL links, providing
linear aggregate bandwidth scaling.  See design doc §8 for details.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

logger = logging.getLogger(__name__)

HUGE_PAGE_2MB = 2 * 1024 * 1024


@dataclass
class CXLConfig:
    """CXL device configuration parsed from hisparse_config JSON.

    Single-device (backward compatible):
        dev_path + map_bytes

    Multi-device interleave:
        dev_paths + map_bytes_per_device
        Ranks are assigned to devices via ``tp_rank % len(dev_paths)``.
    """

    dev_path: str = "/dev/dax0.0"
    map_bytes: int = 64 * 1024 * 1024 * 1024  # 64 GB default
    gpu_id: int = 0
    # Multi-device interleave fields
    dev_paths: List[str] = field(default_factory=list)
    map_bytes_per_device: int = 0

    @property
    def interleave_enabled(self) -> bool:
        return len(self.dev_paths) > 1

    @property
    def num_devices(self) -> int:
        return max(len(self.dev_paths), 1)


_cxl_ext_cache = None


def _get_cxl_source_paths():
    """Return (cxl_root, build_dir) for cxl_mem_ext source and build."""
    from pathlib import Path

    sglang_root = Path(__file__).resolve().parents[4]
    cxl_root = sglang_root / "sgl-kernel" / "cxl_utils"
    build_dir = sglang_root / "build" / "cxl_mem_ext"
    return cxl_root, build_dir


def precompile_cxl_ext():
    """Pre-compile cxl_mem_ext in the main process before workers are spawned.

    Called from engine.py when CXL is enabled. This ensures the .so is
    ready before any worker process starts, so workers just load it.
    """
    try:
        import cxl_mem_ext  # noqa: F401

        logger.info("cxl_mem_ext already installed, skipping compilation")
        return
    except ImportError:
        pass

    from torch.utils.cpp_extension import load

    cxl_root, build_dir = _get_cxl_source_paths()
    if not (cxl_root / "cxl_mem.cpp").exists():
        raise ImportError(
            f"CXL source not found at {cxl_root}. "
            f"Ensure sgl-kernel/cxl_utils exists in the repository."
        )

    build_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Pre-compiling cxl_mem_ext from %s ...", cxl_root)
    load(
        name="cxl_mem_ext",
        sources=[
            str(cxl_root / "cxl_mem.cpp"),
            str(cxl_root / "cxl_mem_cuda.cu"),
            str(cxl_root / "cxl_mem_pybind.cpp"),
        ],
        extra_cflags=[
            "-O3", "-std=c++17",
            "-mclflushopt", "-mclwb", "-msse4.1", "-fopenmp",
        ],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lgomp"],
        build_directory=str(build_dir),
        verbose=False,
        with_cuda=True,
    )
    logger.info("cxl_mem_ext pre-compiled successfully")


def _load_cxl_ext():
    """Load the already-compiled cxl_mem_ext extension.

    Expects precompile_cxl_ext() to have been called in the main process
    before workers are spawned. Falls back to JIT compilation if .so is
    not found (e.g. standalone testing).
    """
    global _cxl_ext_cache
    if _cxl_ext_cache is not None:
        return _cxl_ext_cache

    try:
        import cxl_mem_ext

        _cxl_ext_cache = cxl_mem_ext
        return _cxl_ext_cache
    except ImportError:
        pass

    # Try loading from build directory (pre-compiled by main process)
    import glob as glob_mod
    import importlib.util
    import sys

    _, build_dir = _get_cxl_source_paths()
    so_files = glob_mod.glob(str(build_dir / "cxl_mem_ext*.so"))
    if so_files:
        spec = importlib.util.spec_from_file_location("cxl_mem_ext", so_files[0])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cxl_mem_ext"] = mod
        spec.loader.exec_module(mod)
        _cxl_ext_cache = mod
        return _cxl_ext_cache

    # Last resort: compile in-process (for standalone scripts / tests)
    logger.warning("cxl_mem_ext not pre-compiled, compiling in worker process...")
    precompile_cxl_ext()
    import cxl_mem_ext

    _cxl_ext_cache = cxl_mem_ext
    return _cxl_ext_cache


class CXLMemoryRegion:
    """Manages a single CXL DAX device mmap region with CUDA registration.

    Uses the proven cxl_mem_ext C++ library for the low-level init:
        mmap -> cudaSetDevice -> cudaHostRegister -> cudaHostGetDevicePointer

    Then provides make_tensor() to allocate torch.Tensor views on top of
    the registered CXL memory via torch.frombuffer (zero-copy).

    Lifecycle:
        1. cxl_ext.cxl_init() — mmap + optional cudaHostRegister
        2. set_partition() — restrict allocator to a rank-local slice
        3. make_tensor() — bump-allocate + torch.frombuffer
        4. close() — cxl_ext.cxl_close() (unregister + munmap)
    """

    def __init__(self, config: CXLConfig):
        self.config = config
        self._cxl_ext = None
        self._base_ptr: int = 0
        self._size: int = 0
        self._initialized: bool = False
        self._alloc_offset: int = 0
        self._partition_end: int = 0
        self._partition_size: int = 0

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
        self._partition_end = aligned_size
        self._partition_size = aligned_size
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

    def set_partition(self, tp_rank: int, tp_size: int) -> None:
        """Restrict this region's allocator to a disjoint partition.

        All TP processes mmap the entire CXL DAX device, but each rank
        only allocates from its own 1/tp_size slice. Partitions are
        2MB-aligned to respect DAX huge page boundaries.
        """
        if tp_size <= 1:
            return
        per_rank = self._size // tp_size
        # Align down to 2MB boundary
        per_rank = (per_rank // HUGE_PAGE_2MB) * HUGE_PAGE_2MB
        if per_rank == 0:
            raise RuntimeError(
                f"CXL region too small to partition among {tp_size} ranks: "
                f"{self._size} bytes total, need at least {HUGE_PAGE_2MB * tp_size}"
            )
        start = tp_rank * per_rank
        self._alloc_offset = start
        self._partition_end = start + per_rank
        self._partition_size = per_rank
        logger.info(
            "CXL partition for tp_rank=%d: [0x%x, 0x%x), %.2f GB",
            tp_rank, start, self._partition_end, per_rank / 1e9,
        )

    @property
    def base_ptr(self) -> int:
        return self._base_ptr

    @property
    def size(self) -> int:
        return self._size

    def allocate(self, nbytes: int, alignment: int = 64) -> int:
        """Bump-allocate from this rank's partition. Returns byte offset."""
        offset = (self._alloc_offset + alignment - 1) & ~(alignment - 1)
        if offset + nbytes > self._partition_end:
            raise RuntimeError(
                f"CXL partition exhausted: need {nbytes} bytes at offset {offset}, "
                f"but partition ends at {self._partition_end} "
                f"(only {self._partition_end - offset} available)"
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


def compute_interleave_assignment(
    tp_rank: int, tp_size: int, num_devices: int
) -> tuple:
    """Compute which CXL device a rank uses and its local rank within that device.

    With ``num_devices`` CXL devices and ``tp_size`` ranks, rank *i* is
    assigned to device ``i % num_devices``.  The *local_rank* is the rank's
    ordinal among all ranks sharing the same device, and *local_size* is
    the total number of ranks on that device.

    Returns:
        (device_idx, local_rank, local_size)
    """
    device_idx = tp_rank % num_devices
    # Ranks on this device: {r for r in range(tp_size) if r % num_devices == device_idx}
    local_size = sum(1 for r in range(tp_size) if r % num_devices == device_idx)
    local_rank = sum(
        1 for r in range(tp_rank) if r % num_devices == device_idx
    )
    return device_idx, local_rank, local_size


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
