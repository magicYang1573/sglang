# CXL 存储后端 + HiSparse 设计方案

**基于 SGLang HiSparse 框架的 DeepSeek-V3.2 NSA Sparse Attention KV Cache 向 CXL 内存池迁移**

---

## 目录

1. [概述](#1-概述)
2. [现有 HiSparse Sparse Attention Offloading 实现概述](#2-现有-hisparse-sparse-attention-offloading-实现概述)
  - 2.1 [核心思想](#21-核心思想)
  - 2.2 [数据流：请求生命周期](#22-数据流请求生命周期)
  - 2.3 [CXL 替换的目标层](#23-cxl-替换的目标层)
  - 2.4 [关键接口汇总](#24-关键接口汇总)
3. [CXL 直接 GPU 读取方案验证](#3-cxl-直接-gpu-读取方案验证)
4. [CXL 内存池集成设计](#4-cxl-内存池集成设计)
  - 4.1 [设计原则](#41-设计原则)
  - 4.2 [核心方案：mmap + cudaHostRegister + torch.frombuffer](#42-核心方案mmap--cudahostregister--torchfrombuffer)
  - 4.3 [整体架构变更](#43-整体架构变更)
  - 4.4 [模块设计](#44-模块设计)
  - 4.5 [数据流变更](#45-数据流变更)
5. [详细代码设计](#5-详细代码设计)
  - 5.1 [CXLMemoryRegion 轻量管理类](#51-cxlmemoryregion-轻量管理类)
  - 5.2 [CXL Tensor 分配函数](#52-cxl-tensor-分配函数)
  - 5.3 [CXLMLATokenToKVPoolHost 类设计](#53-cxlmlatokentokv-poolhost-类设计)
  - 5.4 [HiSparseCoordinator 适配](#54-hisparsecoordinator-适配)
  - 5.5 [ServerArgs 与配置扩展](#55-serverargs-与配置扩展)
6. [为什么不使用 cxl_utils 库的传输函数](#6-为什么不使用-cxl_utils-库的传输函数)
7. [CXL 内存策略与优化](#7-cxl-内存策略与优化)
8. [分阶段实施计划](#8-分阶段实施计划)
9. [风险与缓解措施](#9-风险与缓解措施)

---

## 1. 概述

### 1.1 背景

DeepSeek-V3.2 采用原生稀疏注意力（NSA），每层 attention 仅选取 top-k（1–10%）的 KV Cache 条目参与计算。SGLang 中已有的 **HiSparse** 框架实现了完整的 NSA sparse KV offloading 栈：将 KV Cache 从 GPU VRAM 备份到 CPU pinned memory（host pool），decode 时根据 NSA indexer 产生的 top-k 结果，通过 JIT CUDA kernel 仅将所需 token 的 KV 数据从 host swap-in 到 device buffer。

本文档设计如何将 HiSparse 的 **host memory backend** 从 CPU pinned DRAM 替换为 **CXL 共享内存池**。

### 1.2 核心设计决策

**经过 micro-benchmark 验证（`test/cxl_utils/sparse_kv_bench.py`），我们确定了以下关键结论：**


| 结论                                                              | 详细说明                                                         |
| --------------------------------------------------------------- | ------------------------------------------------------------ |
| CXL mmap + `cudaHostRegister` 后，GPU 可用 `ld.global.nc.b64` 直接读取  | CXL 延迟仅为 DRAM 的 1.0~1.7x，可接受                                 |
| **不使用 `cxl_utils` 库的传输函数**（`cxl2vram`、`cxl2vram_copy_kernel` 等） | 库的 copy kernel 逐字节读取，延迟为 DRAM 的 3x+，性能不可接受                   |
| CXL mmap 区域经 `cudaHostRegister` 后对 GPU 完全透明                     | HiSparse JIT kernel（`hisparse.cuh`）**无需任何修改**                |
| 使用 `torch.frombuffer` 在 CXL mmap 上直接构建 Tensor                   | GPU DMA（`cudaMemcpy`）和 JIT kernel 均可通过 `data_ptr()` 直接访问 CXL |


**因此，本方案不依赖 `sgl-kernel/cxl_utils` 的数据传输接口，仅需要最底层的 mmap + cudaHostRegister 能力，通过极简的 Python 封装实现。**

### 1.3 设计约束


| 约束       | 说明                                                                     |
| -------- | ---------------------------------------------------------------------- |
| CXL 设备接口 | 通过 DAX 设备 mmap 映射到进程地址空间                                               |
| GPU 可访问性 | mmap 区域经 `cudaHostRegister` 注册后，GPU kernel 可通过 `ld.global.nc.b64` 直接读取 |
| DAX 对齐要求 | mmap size 必须对齐到 huge page 边界（通常 2MB）                                   |
| 现有接口兼容   | HiSparse JIT kernel 通过 host pointer 读取数据，不关心底层是 DRAM 还是 CXL            |


---

## 2. 现有 HiSparse Sparse Attention Offloading 实现概述

### 2.1 核心思想

HiSparse 维护两级存储：

- **Host Pool**（CPU pinned DRAM）：全量 KV Cache 备份，由 `MLATokenToKVPoolHost` 管理
- **Device Buffer**（GPU VRAM 热缓冲）：每请求一个固定大小的小池（如 4096 slot），仅存放近期被 top-k 选中的热数据

Decode 时 NSA indexer 为每层产出 top-k token 位置，JIT kernel 检查 device buffer 命中情况，miss 的从 host pool 通过 `ld.global.nc.b64` 加载到 buffer 中 LRU 最冷的 slot。

### 2.2 数据流：请求生命周期

```
1. Prefill 完成 → Staging（GPU → Host 全量备份）
   admit_request_into_staging(req)
   → 分配 host 槽位，异步 DMA 所有层的 KV 到 host pool
   → staging 完成后分配 device buffer

2. 每步 Decode 开始 → Backup（GPU → Host 单 token）
   _eager_backup_previous_token()
   → 将上一步新产生的 token KV 备份到 host（防止被 swap-in 驱逐后丢失）

3. 每层 Attention forward → Swap-in（Host → GPU 按层逐层触发）
   swap_in_selected_pages(layer_id)
   → JIT kernel: hash 查找 hit/miss → miss 从 host 加载 → LRU 驱逐最冷 slot
   → 返回 device 地址，执行稀疏 attention

4. 请求结束 → request_finished()
   → 释放 device buffer 和 host pool 的所有槽位
```

**关键特性**：

- 每层有**独立的 LRU 管理**，各层 buffer 互不干扰、无跨层驱逐
- Device buffer 从分配到释放始终**全量常驻 GPU**
- 当前 swap-in 与 attention 是**串行**的，无预取

### 2.3 CXL 替换的目标层

`MLATokenToKVPoolHost` 通过 `alloc_with_host_register` 分配 CPU pinned memory，其关键属性：

- `kv_buffer`: 底层存储，shape `(layer_num, size, 1, kv_cache_dim)`
- `data_ptrs`: 每层 host 地址（`uint64` on GPU）— JIT kernel 和 `transfer_kv_*` 函数通过此访问 host 数据

**CXL 集成只需替换 `kv_buffer` 的底层内存**：将 DRAM 替换为 CXL mmap 区域。只要 `data_ptrs` 指向的地址经过 `cudaHostRegister`，所有上层传输函数和 JIT kernel 零修改工作。

### 2.4 关键接口汇总


| 接口                                    | 所在文件                    | CXL 影响                     |
| ------------------------------------- | ----------------------- | -------------------------- |
| `MLATokenToKVPoolHost.__init__`       | memory_pool_host.py     | **需替换**：分配到 CXL            |
| `MLATokenToKVPoolHost.init_kv_buffer` | memory_pool_host.py     | **需替换**：CXL mmap tensor    |
| `alloc_with_host_register`            | memory_pool_host.py     | **不使用**：CXL 自行管理注册         |
| `backup_from_device_all_layer`        | memory_pool_host.py     | **无需修改**：通过 `data_ptrs` 操作 |
| `load_to_device_per_layer`            | memory_pool_host.py     | **无需修改**：通过 `data_ptrs` 操作 |
| `HiSparseCoordinator.__init__`        | hisparse_coordinator.py | **微调**：传入 CXL 配置           |
| `swap_in_selected_pages`              | hisparse_coordinator.py | **无需修改**                   |
| `hisparse.cuh` transfer_item_warp     | jit_kernel/csrc/        | **无需修改**（指针透明）             |
| `transfer_kv_all_layer_mla`           | sgl_kernel.kvcacheio    | **无需修改**（通过 data_ptrs 操作）  |


---

## 3. CXL 直接 GPU 读取方案验证

### 3.1 Micro-benchmark 结果

通过 `test/cxl_utils/sparse_kv_bench.py` 的实验验证了两种 CXL 读取方式的性能差异：

**方案 A（已弃用）：使用 `cxl_utils` 库的 `cxl2vram_copy_kernel`**

```cuda
// 逐字节读取 —— 性能极差
for (size_t i = threadIdx.x; i < bytes_per_segment; i += blockDim.x) {
    out[i] = src[i];
}
```

**方案 B（采用）：使用与 DRAM 相同的 `ld.global.nc.b64` scatter kernel**

```cuda
// 8 字节宽度加载 —— 性能接近 DRAM
for (int j = tid; j < total_u64; j += blockDim.x) {
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src64 + j));
    asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(dst64 + j), "l"(tmp));
}
```

### 3.2 性能对比（seq_len=32K, H20 GPU）


| top-k | DRAM (μs) | CXL 方案A (μs) | CXL 方案B (μs) | A/DRAM | B/DRAM    |
| ----- | --------- | ------------ | ------------ | ------ | --------- |
| 64    | 11.25     | 28.14        | ~11-19       | 2.50x  | ~1.0-1.7x |
| 512   | 17.19     | 60.97        | ~17-29       | 3.55x  | ~1.0-1.7x |
| 2048  | 43.76     | 154.48       | ~44-74       | 3.53x  | ~1.0-1.7x |


**结论**：方案 B（`ld.global.nc.b64`）将 CXL 开销从 ~3.5x 降低到 ~1.0-1.7x，接近硬件极限。这是因为：

1. **加载宽度**：8 bytes vs 1 byte → 同等数据量下 CXL 事务数减少 8 倍
2. **Non-cached 路径**：`ld.global.nc` 绕过 L2 cache，直接通过 PCIe/CXL DMA，减少 TLB 和 cache 层级开销
3. **零额外开销**：无 pybind 调用、无 offset 格式转换、无额外 `cudaMemcpyAsync`

### 3.3 对 HiSparse 的指导意义

HiSparse 的 JIT kernel 已经使用了 `ld.global.nc.b64`（`transfer_item_warp`），**与方案 B 完全一致**。因此：

- CXL mmap 经 `cudaHostRegister` 后，HiSparse swap-in 延迟即为方案 B 的水平
- **不需要任何 kernel 修改**，不需要使用 `cxl_utils` 库的 copy 函数
- 唯一需要做的是让 `MLATokenToKVPoolHost.kv_buffer` 的底层存储指向 CXL mmap 区域

---

## 4. CXL 内存池集成设计

### 4.1 设计原则

1. **指针透明性**：CXL mmap → `cudaHostRegister` → `torch.frombuffer` → `data_ptr()` → 与 DRAM pinned memory 语义完全一致
2. **不依赖 cxl_utils 传输层**：所有数据传输使用已有的 `cudaMemcpy`（staging/backup）和 JIT kernel（swap-in）
3. **最小侵入性**：仅替换 `MLATokenToKVPoolHost` 的内存分配后端，不修改 Coordinator、LRU、JIT kernel
4. **轻量实现**：CXL 管理仅需 ~100 行 Python 代码（mmap + register + frombuffer），不需要额外 C++ 扩展

### 4.2 核心方案：mmap + cudaHostRegister + torch.frombuffer

```
CXL DAX 设备 (/dev/dax0.0)
    │
    ▼  os.open + mmap.mmap (MAP_SHARED)
CXL mmap 虚拟地址 (host_ptr)
    │
    ▼  cudaHostRegister(host_ptr, size, Portable|Mapped)
CUDA 可访问的 host memory
    │
    ├──▷ cudaHostGetDevicePointer → device_ptr
    │    (GPU kernel 通过此指针 ld.global.nc.b64 读取)
    │
    └──▷ torch.frombuffer(mmap_buffer, dtype) → torch.Tensor
         (kv_buffer, data_ptr() == host_ptr)
         │
         ├── transfer_kv_all_layer_mla 通过 data_ptrs 执行 cudaMemcpy D2H → 写入 CXL
         ├── JIT kernel 通过 data_ptrs 执行 ld.global.nc.b64 → 从 CXL 读取
         └── 对上层完全透明
```

**关键点**：`torch.frombuffer` 创建的 tensor 的 `data_ptr()` 就是 CXL mmap 地址。所有通过 `data_ptrs` 操作的传输函数（`transfer_kv_all_layer_mla`、JIT kernel 的 `host_cache_k`）直接访问 CXL，无需任何中间层。

### 4.3 整体架构变更

```
┌────────────────────────────────────────────────────┐
│  HiSparseCoordinator                                │
│    └─ mem_pool_host: CXLMLATokenToKVPoolHost        │
│         └─ kv_buffer: torch.Tensor                  │
│              │  (data_ptr → CXL mmap region)        │
│         ┌────┴─────────────────────┐                │
│         │  内存后端选择 (配置驱动)    │                │
│         ├─────────┬────────────────┤                │
│         │ DRAM    │ CXL            │                │
│         │ (现有)   │ (新增)         │                │
│         │         │                │                │
│         │ torch   │ mmap DAX       │                │
│         │ .empty  │ + cudaHost     │                │
│         │ + pin   │ Register       │                │
│         │         │ + frombuffer   │                │
│         └─────────┴────────────────┘                │
│                                                      │
│  swap-in:  JIT kernel ld.global.nc.b64 ──→ 同一路径  │
│  staging:  cudaMemcpy D2H ──→ 同一路径               │
│  backup:   cudaMemcpy D2H ──→ 同一路径               │
└────────────────────────────────────────────────────┘
```

### 4.4 模块设计

```
python/sglang/srt/mem_cache/
    ├── memory_pool_host.py                # [不变] 基类 HostKVCache、MLATokenToKVPoolHost
    ├── cxl_memory_pool.py                 # [新增] CXLMemoryRegion + CXLMLATokenToKVPoolHost
    └── hisparse_memory_pool.py            # [不变]

python/sglang/srt/managers/
    └── hisparse_coordinator.py            # [微调] 根据配置选择 DRAM 或 CXL host pool

python/sglang/srt/
    └── server_args.py                     # [修改] 增加 CXL 配置参数
```

**注意**：不需要任何新的 C++/CUDA 编译。CXL 操作全部通过 Python 标准库（`os.open`, `mmap`）和 PyTorch（`cudaHostRegister`, `torch.frombuffer`）完成。

### 4.5 数据流变更

#### 4.5.1 Prefill Staging（GPU → CXL）

```
现有:  transfer_kv_all_layer_mla(dst=host.data_ptrs) → cudaMemcpy D2H → pinned DRAM
CXL:   transfer_kv_all_layer_mla(dst=host.data_ptrs) → cudaMemcpy D2H → CXL mmap
```

`data_ptrs` 指向 CXL mmap 区域（已 `cudaHostRegister`）。`cudaMemcpy D2H` 的目标是注册过的 host memory，DMA 引擎直接写入 CXL 物理地址。**无需修改传输代码，无需 clflush**（DMA 不经过 CPU cache）。

#### 4.5.2 Decode Swap-in（CXL → GPU）

```
现有:  JIT kernel ld.global.nc.b64 from pinned DRAM → st.global.cg.b64 to VRAM
CXL:   JIT kernel ld.global.nc.b64 from CXL mmap   → st.global.cg.b64 to VRAM
```

JIT kernel 的 `host_cache_k` 指针来自 `data_ptrs`，指向 CXL mmap。`ld.global.nc.b64` 通过 PCIe/CXL 链路直接加载，延迟为 DRAM 的 1.0~1.7x（已验证）。**JIT kernel 零修改。**

#### 4.5.3 Decode Backup（GPU → CXL）

```
现有:  transfer_kv_all_layer_mla → cudaMemcpy D2H → pinned DRAM
CXL:   transfer_kv_all_layer_mla → cudaMemcpy D2H → CXL mmap
```

同 staging 路径。**无需修改。**

---

## 5. 详细代码设计

### 5.1 CXLMemoryRegion 轻量管理类

**新文件**: `python/sglang/srt/mem_cache/cxl_memory_pool.py`

```python
"""CXL Memory Pool for HiSparse KV Cache.

Manages CXL DAX device mmap and provides torch.Tensor backed by CXL memory.
Does NOT use sgl-kernel/cxl_utils — all operations through standard OS and CUDA APIs.
"""

import ctypes
import logging
import mmap
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)

HUGE_PAGE_2MB = 2 * 1024 * 1024


@dataclass
class CXLConfig:
    """CXL device configuration."""
    dev_path: str = "/dev/dax0.0"
    map_bytes: int = 64 * 1024 * 1024 * 1024   # 64 GB
    gpu_id: int = 0


class CXLMemoryRegion:
    """Manages a single CXL DAX device mmap region with CUDA registration.

    Lifecycle:
        1. open DAX device → mmap (2MB-aligned)
        2. cudaHostRegister (Portable | Mapped)
        3. Allocate tensors via torch.frombuffer (zero-copy)
        4. All GPU DMA and kernel access through registered pointers
        5. Close: cudaHostUnregister → munmap
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
        dev_path = self.config.dev_path
        raw_size = self.config.map_bytes

        # Align to 2MB huge page boundary (required by DAX devices)
        aligned_size = ((raw_size + HUGE_PAGE_2MB - 1) // HUGE_PAGE_2MB) * HUGE_PAGE_2MB

        self._fd = os.open(dev_path, os.O_RDWR)
        try:
            self._mmap = mmap.mmap(
                self._fd, aligned_size,
                mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except OSError as e:
            os.close(self._fd)
            self._fd = -1
            raise RuntimeError(
                f"CXL mmap failed for {dev_path} (size={aligned_size}): {e}. "
                f"Check DAX device permissions and ensure size is 2MB-aligned."
            ) from e

        # Get the base address of the mmap region
        buf = ctypes.c_char.from_buffer(self._mmap)
        self._base_ptr = ctypes.addressof(buf)
        self._size = aligned_size

        # Register with CUDA for GPU DMA access
        flags = 0x01 | 0x02  # cudaHostRegisterPortable | cudaHostRegisterMapped
        ret = torch.cuda.cudart().cudaSetDevice(self.config.gpu_id)
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
            dev_path, aligned_size / 1e9, self.config.gpu_id, self._base_ptr,
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

        The tensor's data_ptr() points directly into the CXL mmap region.
        Since the region is cudaHostRegister'd, GPU kernels and cudaMemcpy
        can access it through this pointer.
        """
        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * dtype.itemsize

        offset = self.allocate(nbytes)
        ptr = self._base_ptr + offset

        # Create a ctypes array viewing the CXL memory, then wrap as tensor
        c_array = (ctypes.c_byte * nbytes).from_address(ptr)
        tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(dims)

        # Zero-initialize (CPU memset to CXL mmap — fast, no clflush needed
        # because subsequent access is by GPU DMA which bypasses CPU cache)
        tensor.zero_()

        return tensor

    def close(self) -> None:
        if self._cuda_registered:
            torch.cuda.cudart().cudaHostUnregister(ctypes.c_void_p(self._base_ptr))
            self._cuda_registered = False
        self._cleanup_mmap()

    def _cleanup_mmap(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self):
        self.close()
```

### 5.2 CXL Tensor 分配函数

不需要独立的分配函数——`CXLMemoryRegion.make_tensor()` 直接完成 CXL 内存上的 tensor 构建。

核心原理：

```python
# CXL mmap base address = 0x7f0000000000 (例)
# cudaHostRegister(0x7f0000000000, size, Portable|Mapped) → 注册成功
# offset = region.allocate(nbytes) → 0
# ptr = base + offset = 0x7f0000000000
# tensor = torch.frombuffer(ctypes_view_at_ptr, dtype, count).reshape(dims)
# tensor.data_ptr() == 0x7f0000000000 → 指向 CXL 物理内存
#
# 此后：
# - cudaMemcpy(tensor.data_ptr(), gpu_src, ..., D2H) → DMA 直接写 CXL
# - JIT kernel: ld.global.nc.b64 from tensor.data_ptr() → GPU 直接读 CXL
```

### 5.3 CXLMLATokenToKVPoolHost 类设计

**新文件**: `python/sglang/srt/mem_cache/cxl_memory_pool.py`（续）

```python
class CXLMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool backed by CXL memory.

    Inherits ALL swap-in/backup/load logic from MLATokenToKVPoolHost.
    Only overrides memory allocation to use CXL mmap instead of pinned DRAM.

    After init, self.kv_buffer.data_ptr() points to CXL memory.
    self.data_ptrs contains per-layer CXL pointers on GPU.
    All existing transfer_kv_* functions and JIT kernels work unchanged.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        cxl_region: CXLMemoryRegion,
        override_kv_cache_dim=None,
    ):
        self._cxl_region = cxl_region
        # Call parent init — this will call init_kv_buffer() which we override
        super().__init__(
            device_pool=device_pool,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=False,  # CXL region is already CUDA-registered
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=override_kv_cache_dim,
        )

    def init_kv_buffer(self):
        """Override: allocate KV buffer on CXL memory instead of DRAM.

        Uses CXLMemoryRegion.make_tensor() which creates a zero-copy
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
            "CXL KV buffer allocated: %.2f GB, layout=%s, "
            "data_ptr=0x%x (CXL mmap)",
            nbytes / 1e9, self.layout, buffer.data_ptr(),
        )
        return buffer

    # load_to_device_per_layer:  inherited — uses data_ptrs → CXL pointer
    # backup_from_device_all_layer: inherited — uses data_ptrs → CXL pointer
    # get_data_page:             inherited — tensor indexing on CXL-backed buffer
    # All transfer_kv_* functions work unchanged because data_ptrs points to CXL.
```

**关键设计点**：

1. `pin_memory=False`：`HostKVCache.__init__` 中，`pin_memory=False` 使得 `alloc_with_host_register` 跳过 `cudaHostRegister`（CXL 已经在 `CXLMemoryRegion.init()` 中完成注册）
2. `init_kv_buffer` override：直接调用 `CXLMemoryRegion.make_tensor()`，返回的 tensor `data_ptr()` 指向 CXL
3. `data_refs` 和 `data_ptrs`：在父类 `MLATokenToKVPoolHost.__init__` 中通过 `kv_buffer[i].data_ptr()` 构建。由于 `kv_buffer` 底层是 CXL，`data_ptrs` 自然指向 CXL 地址
4. 所有 `transfer_kv_`* 和 JIT kernel 通过 `data_ptrs` 操作，**零修改**

### 5.4 HiSparseCoordinator 适配

**修改文件**: `python/sglang/srt/managers/hisparse_coordinator.py`

变更极小——仅在构造 `mem_pool_host` 时根据配置选择 DRAM 或 CXL：

```python
class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: HiSparseTokenToKVPoolAllocator,
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group: torch.distributed.ProcessGroup,
        host_to_device_ratio: int = 2,
        cxl_config: Optional[dict] = None,
    ):
        # ... existing init ...

        self.mem_pool_device = self.token_to_kv_pool_allocator.get_kvcache()

        if cxl_config is not None and cxl_config.get("enabled", False):
            from sglang.srt.mem_cache.cxl_memory_pool import (
                CXLConfig, CXLMemoryRegion, CXLMLATokenToKVPoolHost,
            )
            region = CXLMemoryRegion(CXLConfig(
                dev_path=cxl_config.get("dev_path", "/dev/dax0.0"),
                map_bytes=cxl_config.get("map_bytes", 64 * 1024**3),
                gpu_id=cxl_config.get("gpu_id", 0),
            ))
            region.init()
            self._cxl_region = region  # prevent GC

            self.mem_pool_host = CXLMLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=1,
                layout="layer_first",
                cxl_region=region,
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
        else:
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=1,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )

        # 其余初始化完全不变
```

### 5.5 ServerArgs 与配置扩展

```bash
# 在 hisparse_config JSON 中增加 CXL 字段
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 \
    --tp 8 \
    --enable-hisparse \
    --hisparse-config '{
        "top_k": 2048,
        "device_buffer_size": 4096,
        "host_to_device_ratio": 2,
        "cxl": {
            "enabled": true,
            "dev_path": "/dev/dax0.0",
            "map_bytes": 68719476736,
            "gpu_id": 0
        }
    }'
```

---

## 6. 为什么不使用 cxl_utils 库的传输函数


| 方面                  | cxl_utils 库路径                                | 本方案（mmap + register + ld.global.nc）           |
| ------------------- | -------------------------------------------- | --------------------------------------------- |
| CXL → GPU kernel    | `cxl2vram_copy_kernel`：逐字节 `out[i] = src[i]` | HiSparse JIT kernel：`ld.global.nc.b64`（8B 宽度） |
| **延迟**              | **DRAM 的 ~3.5x**                             | **DRAM 的 ~1.0-1.7x**                          |
| GPU → CXL (staging) | `vram2cxl`：`cudaMemcpy D2H` + `clflush`      | 直接 `cudaMemcpy D2H`（无需 clflush）               |
| 需要额外编译              | 是（C++ extension JIT / build）                 | **否**（纯 Python + 标准 CUDA API）                 |
| 全局状态                | `g_cxl` 全局单例，不支持多 region                     | Python 对象，可多实例                                |
| offset 格式           | 需要 byte offset + H2D copy                    | 直接 tensor index，无额外转换                         |
| clflush 需求          | CPU 写入后需 clflush                             | 不需要（GPU DMA 绕过 CPU cache）                     |


**核心原因**：cxl_utils 的 `cxl2vram_copy_kernel` 使用 1 字节加载宽度，在 CXL 高延迟链路上每次加载都产生独立事务，导致 ~3.5x 开销。而 `ld.global.nc.b64` 使用 8 字节非缓存加载，充分利用了 PCIe/CXL 的批量传输能力。

**cxl_utils 库仍可用于**：CXL 设备初始化（如果需要更灵活的 mmap 选项）、CXL barrier（多 rank TP 同步）等场景。但数据传输路径完全不使用其函数。

---

## 7. CXL 内存策略与优化

### 7.1 DAX 大页对齐

CXL DAX 设备的 mmap 必须 2MB 对齐。`CXLMemoryRegion.init()` 中已自动处理：

```python
aligned_size = ((raw_size + HUGE_PAGE_2MB - 1) // HUGE_PAGE_2MB) * HUGE_PAGE_2MB
```

### 7.2 TLB 优化

sparse top-k 的离散访问模式会产生大量 TLB miss。DAX 设备天然使用大页（2MB），相比 4KB 页可将 TLB 条目覆盖范围扩大 512 倍。如需进一步优化，可通过内核配置启用 1GB 大页。

### 7.3 多 CXL 设备 Interleave

```python
# 按 layer 分布到不同 CXL 设备：
# layer_id % num_cxl_devices → 选择 CXL 设备
# 利用 sparse attention 的层间独立性，不同层的 swap-in 可并行访问不同设备

class InterleavedCXLPool:
    def __init__(self, regions: List[CXLMemoryRegion], layer_num: int):
        self.regions = regions
        self.layer_to_device = [i % len(regions) for i in range(layer_num)]

    def make_layer_tensor(self, layer_id, dims, dtype):
        region = self.regions[self.layer_to_device[layer_id]]
        return region.make_tensor(dims, dtype)
```

### 7.4 写入一致性


| 操作                      | 写入方                       | 是否需要 clflush         |
| ----------------------- | ------------------------- | -------------------- |
| Staging (GPU→CXL)       | GPU DMA                   | 不需要：DMA 直接写 CXL      |
| Decode backup (GPU→CXL) | GPU DMA                   | 不需要                  |
| Swap-in (CXL→GPU)       | GPU kernel `ld.global.nc` | 不需要：不经过 CPU cache    |
| CPU 读 CXL（如有）           | CPU                       | 需要：但 HiSparse 流程无此操作 |


---

## 8. 分阶段实施计划

### Phase 1: 基础功能（3-5 天）

**目标**：单 CXL 设备替换 pinned DRAM，验证正确性

- 实现 `CXLMemoryRegion` 类（mmap + cudaHostRegister + make_tensor）
- 实现 `CXLMLATokenToKVPoolHost`（override init_kv_buffer）
- 修改 `HiSparseCoordinator.__init_`_ 支持 `cxl_config`
- 添加 `hisparse-config` CXL 字段解析
- 单元测试：CXL tensor 分配 → GPU DMA 读写正确性

### Phase 2: 端到端验证（3-5 天）

**目标**：在 DeepSeek-V3.2 上端到端验证

- 在 CXL 设备上运行 HiSparse + DeepSeek-V3.2 推理
- 验证输出与 DRAM 基线完全一致（exact match）
- 验证完整生命周期：staging → swap-in → LRU → backup → request finish
- 检查 CXL region 的 allocation tracking，无内存泄漏

### Phase 3: 性能调优（1 周）

- 对比 CXL vs DRAM 端到端 serving 指标（吞吐、TTFT、TBT）
- Nsight Systems profiling：量化 swap-in kernel 中 CXL 读取延迟占比
- 探索多 CXL 设备 interleave
- 测试 NUMA pinning 策略对 CXL 访问延迟的影响

### Phase 4: 对比实验（1 周）

按 `docs/plan/experiment_plan_deepseek_v32.md` 和 `docs/plan/microbench/sparse_kv_read_latency_bench.md` 执行：

- Micro-benchmark：CXL vs DRAM vs RDMA 离散 KV 读取（已完成）
- 端到端 Serving：吞吐、延迟对比
- 内存效率：最大并发请求数对比
- 消融实验：buffer size、sparse ratio

---

## 9. 风险与缓解措施


| 风险                               | 影响                                  | 缓解措施                                      |
| -------------------------------- | ----------------------------------- | ----------------------------------------- |
| CXL 延迟高于 DRAM                    | swap-in 延迟增加 ~1.0-1.7x              | 已验证可接受；LRU buffer 利用层间相似性减少 miss          |
| `cudaHostRegister` 对 CXL mmap 失败 | GPU 无法 DMA                          | 检查 CUDA driver 版本和 PCIe topology；降级为 DRAM |
| `torch.frombuffer` 生命周期          | tensor 释放后 mmap 仍有效                 | CXLMemoryRegion 绑定到 Coordinator 生命周期      |
| DAX 设备 2MB 对齐要求                  | mmap EINVAL                         | `CXLMemoryRegion.init()` 自动向上对齐           |
| 多 GPU TP 场景                      | 多 GPU 竞争 CXL 带宽                     | 每 GPU 使用独立 CXL region offset 或独立 DAX 设备   |
| HostKVCache 内存检查                 | `psutil.virtual_memory()` 可能不识别 CXL | 在 CXL 子类中 override 检查逻辑                   |


---

## 附录 A：关键文件索引


| 文件                                                    | 角色                            | CXL 变更             |
| ----------------------------------------------------- | ----------------------------- | ------------------ |
| `python/sglang/srt/mem_cache/cxl_memory_pool.py`      | **新增** CXL region + host pool | —                  |
| `python/sglang/srt/managers/hisparse_coordinator.py`  | HiSparse 核心协调器                | 增加 `cxl_config` 参数 |
| `python/sglang/srt/mem_cache/memory_pool_host.py`     | Host KV Pool 基类               | 不变                 |
| `python/sglang/srt/mem_cache/hisparse_memory_pool.py` | Device Pool + Allocator       | 不变                 |
| `python/sglang/jit_kernel/csrc/hisparse.cuh`          | JIT swap-in kernel            | 不变                 |
| `python/sglang/jit_kernel/hisparse.py`                | JIT Python 入口                 | 不变                 |
| `python/sglang/srt/layers/attention/nsa_backend.py`   | NSA attention 后端              | 不变                 |
| `python/sglang/srt/model_executor/model_runner.py`    | 模型初始化                         | 传递 CXL 配置          |
| `python/sglang/srt/server_args.py`                    | 启动参数                          | CXL 配置解析           |


## 附录 B：代码量估算


| 文件                        | 新增/修改行数    | 说明                                        |
| ------------------------- | ---------- | ----------------------------------------- |
| `cxl_memory_pool.py`      | ~200 行     | CXLMemoryRegion + CXLMLATokenToKVPoolHost |
| `hisparse_coordinator.py` | ~20 行      | CXL 分支 + region 初始化                       |
| `server_args.py`          | ~5 行       | CXL config 解析                             |
| **合计**                    | **~225 行** | 无 C++/CUDA 编译需求                           |


## 附录 C：配置示例

```bash
# 单 CXL 设备，64GB
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 \
    --tp 8 \
    --enable-hisparse \
    --hisparse-config '{
        "top_k": 2048,
        "device_buffer_size": 4096,
        "host_to_device_ratio": 2,
        "cxl": {
            "enabled": true,
            "dev_path": "/dev/dax0.0",
            "map_bytes": 68719476736,
            "gpu_id": 0
        }
    }' \
    --disable-radix-cache
```

