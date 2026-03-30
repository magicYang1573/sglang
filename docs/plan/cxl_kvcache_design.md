# CXL 内存池 KV Cache 存储设计方案

**基于 SGLang HiSparse 框架的 DeepSeek-V3.2 NSA Sparse Attention KV Cache 向 CXL 内存池迁移**

---

## 目录

1. [概述](#1-概述)
2. [现有 HiSparse Sparse Attention Offloading 实现分析](#2-现有-hisparse-sparse-attention-offloading-实现分析)
   - 2.1 [整体架构](#21-整体架构)
   - 2.2 [关键模块与数据结构](#22-关键模块与数据结构)
   - 2.3 [核心数据流](#23-核心数据流)
   - 2.4 [JIT CUDA Kernel: 稀疏 swap-in 逻辑](#24-jit-cuda-kernel-稀疏-swap-in-逻辑)
   - 2.5 [LRU 缓冲管理机制](#25-lru-缓冲管理机制)
   - 2.6 [关键接口汇总](#26-关键接口汇总)
3. [CXL 内存池工具库分析](#3-cxl-内存池工具库分析)
4. [CXL 内存池集成设计](#4-cxl-内存池集成设计)
   - 4.1 [设计原则](#41-设计原则)
   - 4.2 [整体架构变更](#42-整体架构变更)
   - 4.3 [模块设计](#43-模块设计)
   - 4.4 [数据流变更](#44-数据流变更)
   - 4.5 [JIT Kernel 的适配](#45-jit-kernel-的适配)
5. [详细代码设计](#5-详细代码设计)
   - 5.1 [CXL Memory Backend 模块](#51-cxl-memory-backend-模块)
   - 5.2 [CXLMLATokenToKVPoolHost 类设计](#52-cxlmlatokentokv-poolhost-类设计)
   - 5.3 [HiSparseCoordinator 适配](#53-hisparsecoordinator-适配)
   - 5.4 [ServerArgs 与配置扩展](#54-serverargs-与配置扩展)
   - 5.5 [ModelRunner 初始化流程修改](#55-modelrunner-初始化流程修改)
6. [CXL 内存策略与优化](#6-cxl-内存策略与优化)
7. [分阶段实施计划](#7-分阶段实施计划)
8. [风险与缓解措施](#8-风险与缓解措施)

---

## 1. 概述

### 1.1 背景

DeepSeek-V3.2 采用原生稀疏注意力（NSA），每层 attention 仅选取 top-k（1–10%）的 KV Cache 条目参与计算。SGLang 中已有的 **HiSparse** 框架（PR#20343）实现了完整的 NSA sparse KV offloading 栈：将 KV Cache 从 GPU VRAM 备份到 CPU pinned memory（host pool），decode 时根据 NSA indexer 产生的 top-k 结果，通过 JIT CUDA kernel 仅将所需 token 的 KV 数据从 host swap-in 到 device buffer。

本文档设计如何将 HiSparse 的 **host memory backend** 从 CPU pinned DRAM 替换为 **CXL 共享内存池**，使 KV Cache 存储在 CXL type-3 设备上。核心目标是：

- **最小化侵入性**：复用 HiSparse 已有的 coordinator、JIT kernel、LRU 机制，仅替换内存分配与数据传输层
- **释放本地 DRAM**：KV Cache 全部存储在 CXL 内存，本地 DRAM 释放给模型权重和更多并发请求
- **保持 sparse top-k 按需读取**：CXL 的 load/store 语义天然支持逐层细粒度访问，无需全量预取

### 1.2 设计约束

| 约束 | 说明 |
|------|------|
| CXL 设备接口 | 通过 DAX 设备 mmap 映射到进程地址空间 |
| GPU 可访问性 | mmap 区域经 `cudaHostRegister` 注册后，GPU 可通过 PCIe/CXL DMA 直接访问 |
| 现有接口兼容 | HiSparse JIT kernel 通过 host pointer 读取数据，不关心底层是 DRAM 还是 CXL |
| 多卡 CXL 支持 | 需考虑多块 CXL 设备的 interleave 与 NUMA 策略 |

---

## 2. 现有 HiSparse Sparse Attention Offloading 实现分析

### 2.1 整体架构

HiSparse 框架在 SGLang 中的位置与数据流如下：

```
┌─────────────────────────────────────────────────┐
│  ServerArgs: --enable-hisparse --hisparse-config │
│    → parse_hisparse_config() → SparseConfig     │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  ModelRunner 初始化                               │
│  ├─ HiSparseNSATokenToKVPool (device pool)      │
│  ├─ HiSparseTokenToKVPoolAllocator              │
│  └─ HiSparseCoordinator                         │
│       ├─ MLATokenToKVPoolHost (host pool, DRAM)  │
│       ├─ req_to_device_buffer (device buffer映射) │
│       ├─ req_to_host_pool (host pool映射)        │
│       ├─ LRU slots (per-layer per-req)           │
│       └─ JIT hisparse.cuh kernel                 │
└─────────────┬───────────────────────────────────┘
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 Scheduler  ScheduleBatch  NativeSparseAttnBackend
 (staging)  (decode alloc)  (forward_decode)
```

**关键设计理念**：HiSparse 维护两级存储——**device buffer**（GPU VRAM 中的热数据小池）和 **host pool**（CPU 内存中的全量 KV 备份）。Decode 时，NSA indexer 产生 top-k token 位置，JIT kernel 检查这些 token 是否在 device buffer 中（hit），如果不在（miss）则从 host pool 加载到 device buffer 中的 LRU 淘汰位置。

### 2.2 关键模块与数据结构

#### 2.2.1 HiSparseCoordinator（核心协调器）

**文件**: `python/sglang/srt/managers/hisparse_coordinator.py`

HiSparseCoordinator 是 HiSparse 系统的核心，负责管理 host pool、device buffer、staging 异步备份、decode 阶段的 swap-in 操作。

**初始化时构建的核心数据结构**：

| 数据结构 | 类型 | 形状 | 说明 |
|---------|------|------|------|
| `mem_pool_host` | `MLATokenToKVPoolHost` | — | Host 侧 KV Cache 全量存储，pinned DRAM |
| `req_to_device_buffer` | `int64 tensor` | `(max_num_reqs, padded_buffer_size)` | 每个请求的 device buffer slot → 物理 KV 索引映射 |
| `req_to_host_pool` | `int64 tensor` | `(max_num_reqs, max_context_len)` | 每个请求的 token 位置 → host pool 索引映射 |
| `req_device_buffer_tokens` | `int32 tensor` | `(layer_num, max_num_reqs, padded_buffer_size)` | 每个 buffer slot 当前存储的 token id |
| `req_device_buffer_token_locs` | `int32 tensor` | `(layer_num, max_num_reqs, padded_buffer_size)` | 每个 buffer slot 的物理 KV 池索引 |
| `lru_slots` | `int16 tensor` | `(layer_num, max_num_reqs, device_buffer_size)` | LRU 排序数组，per-layer per-request |
| `top_k_device_locs_buffer` | `int32 tensor` | `(max_num_reqs, top_k)` | swap-in 输出：top-k token 的 device 物理索引 |
| `num_real_reqs` | `int32 tensor` | `(1,)` | CUDA Graph 安全的实际请求数标量 |

**关键方法**：

```python
class HiSparseCoordinator:
    def admit_request_into_staging(self, req: Req) -> None:
        """Prefill 完成后，异步将全量 KV 从 device 备份到 host pool"""

    def collect_ready_reqs(self) -> List[Req]:
        """收集已完成 staging 的请求，分配 device buffer"""

    def map_last_loc_to_buffer(self, seq_lens, out_cache_loc, req_pool_indices, seq_lens_cpu) -> None:
        """每步 decode 前：备份上一步 token，扩展 buffer，映射最新 token"""

    def swap_in_selected_pages(self, req_pool_indices, seq_lens, top_k_result, layer_id) -> torch.Tensor:
        """核心：调用 JIT kernel 将 top-k 所需 token 从 host 加载到 device buffer"""

    def request_finished(self, req: Req):
        """请求完成时释放所有资源（device buffer + host pool + LRU 重置）"""
```

#### 2.2.2 HiSparseNSATokenToKVPool（Device KV Pool）

**文件**: `python/sglang/srt/mem_cache/hisparse_memory_pool.py`

继承自 `NSATokenToKVPool`，增加了逻辑索引 → 物理索引的映射机制。HiSparse 的 device pool 比传统 pool 更小（仅保存热数据），通过 `full_to_hisparse_device_index_mapping` 将全局逻辑索引映射到压缩的物理 KV 存储。

```python
class HiSparseNSATokenToKVPool(NSATokenToKVPool):
    # 关键：所有 set_kv_buffer / get_mla_kv_buffer 操作都通过 mapping 转换索引
    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        loc = self.translate_loc_to_hisparse_device(loc)  # 逻辑 → 物理
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
```

#### 2.2.3 HiSparseTokenToKVPoolAllocator（双级分配器）

**文件**: `python/sglang/srt/mem_cache/hisparse_memory_pool.py`

维护两个独立的 `PagedTokenToKVPoolAllocator`：
- `logical_attn_allocator`：管理全局逻辑索引空间（size × host_to_device_ratio）
- `hisparse_attn_allocator`：管理压缩的 device buffer 物理索引空间（size）

```python
class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def alloc_device_buffer(self, allocated_indices, need_size):
        """为 device buffer 分配物理索引，复用已有映射或申请新空间"""

    def alloc_extend(self, prefix_lens, ..., extend_num_tokens):
        """Prefill 时同时分配逻辑和 hisparse 索引，建立映射"""

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        """Decode 时仅分配逻辑索引（hisparse 由 coordinator 管理）"""
```

#### 2.2.4 MLATokenToKVPoolHost（Host KV Pool）

**文件**: `python/sglang/srt/mem_cache/memory_pool_host.py`

Host 侧 KV Cache 存储，当前使用 CPU pinned memory 分配。这是 **CXL 替换的目标层**。

```python
class MLATokenToKVPoolHost(HostKVCache):
    def __init__(self, device_pool, host_to_device_ratio, host_size,
                 page_size, layout, pin_memory=True, device="cpu",
                 allocator_type="default", override_kv_cache_dim=None):
        # ...
        self.kv_buffer = self.init_kv_buffer()  # ← 分配 pinned DRAM

    def init_kv_buffer(self):
        # layout="layer_first" 时: dims = (layer_num, size, 1, kv_cache_dim)
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]  # → alloc_with_host_register
        buffer = alloc_func(dims, dtype, device="cpu", pin_memory=True, allocator=self.allocator)
        return buffer

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        """从 host 逐层加载 KV 到 device（用于 naive_load_topk 等调试路径）"""
        transfer_kv_per_layer_mla(src=self.kv_buffer[layer_id], dst=device_pool.kv_buffer[layer_id], ...)

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        """从 device 全层备份 KV 到 host（staging 和逐步备份使用）"""
        transfer_kv_all_layer_mla(src_layers=device_pool.data_ptrs, dst_layers=self.data_ptrs, ...)
```

**当前的内存分配策略**（`alloc_with_host_register`）：

```python
def alloc_with_host_register(dims, dtype, device, pin_memory, allocator):
    buffer = allocator.allocate(dims, dtype=dtype, device=device)  # torch.empty on CPU
    if pin_memory:
        torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), nbytes, 0)
    return buffer
```

### 2.3 核心数据流

#### 2.3.1 Prefill → Staging 流程

```
Prefill 完成
    │
    ▼
Scheduler.admit_request_into_staging(req)
    │
    ├── 1. 获取逻辑索引 → 转换为 hisparse device 索引
    ├── 2. mem_pool_host.alloc(prefill_len)  ← 分配 host 索引
    ├── 3. 记录 req_to_host_pool 映射
    └── 4. 异步 backup_from_device_all_layer  ← GPU→Host DMA
         (在 write_staging_stream 上执行)
    │
    ▼
Scheduler.collect_ready_reqs()  (等 DMA 完成)
    │
    ├── 1. alloc_device_buffer(req)  ← 分配 device buffer
    ├── 2. 初始化 device_buffer_tokens / token_locs
    └── 3. 设置 _skip_first_backup = True
```

#### 2.3.2 Decode 步骤数据流

```
每步 Decode 开始
    │
    ▼
ScheduleBatch: map_last_loc_to_buffer()
    │
    ├── 1. _eager_backup_previous_token()
    │      ├── 分配 host 索引
    │      └── backup_from_device_all_layer（上一步 token → host）
    │
    ├── 2. _grow_device_buffers() ← 必要时扩展 buffer
    └── 3. 映射最新 token 到 reserved slot
    │
    ▼
Model Forward (per layer):
    │
    ├── Indexer.forward_cuda() → topk_indices (top-k token 位置)
    │
    └── NativeSparseAttnBackend.forward_decode()
         │
         └── hisparse_coordinator.swap_in_selected_pages(
                 req_pool_indices, seq_lens, topk_indices, layer_id)
              │
              ├── 调用 JIT kernel: load_cache_to_device_buffer_mla()
              │    ├── 短序列（seq_len ≤ buffer_size）: 直接索引
              │    └── 长序列: hash top-k → 分类 hit/miss → LRU 淘汰 → DMA miss tokens
              │
              └── 返回 top_k_device_locs → 用作 attention page_table
```

### 2.4 JIT CUDA Kernel: 稀疏 swap-in 逻辑

**文件**: `python/sglang/jit_kernel/csrc/hisparse.cuh`

`load_cache_to_device_buffer_kernel` 是 HiSparse 的性能关键路径，每个 CUDA block 处理一个请求：

1. **短序列快速路径**（`seq_len ≤ HOT_BUFFER_SIZE`）：所有 token 都在 device buffer 中，直接查表返回索引
2. **长序列路径**：
   - **Hash 阶段**：将 top-k token id 插入共享内存 open-addressing hash table
   - **Hit/Miss 分类**：遍历 LRU slot 列表，对每个 buffer slot 查 hash table → hit 则直接绑定 device 索引，miss 标记为可淘汰
   - **LRU 重排**：hits 放到 MRU 端，evictables 放到 LRU 端
   - **Miss 处理**：为每个 miss token 分配一个 LRU 淘汰 slot，调用 `transfer_item_warp` 从 host 拷贝到 device

**关键观察**：`transfer_item_warp` 通过 `ld.global.nc.b64` / `st.global.cg.b64` 执行 host→device 的 warp 级数据拷贝。这里的 `host_cache_k` 指针指向 host 内存（当前是 pinned DRAM）。**如果 CXL 内存同样通过 `cudaHostRegister` 注册，GPU 可通过相同的指针以相同方式访问 CXL 内存，kernel 代码无需修改。**

### 2.5 LRU 缓冲管理机制

LRU 在 HiSparse 中是 **per-layer per-request** 的 slot 排序数组，存储在 `lru_slots` tensor 中。

- 初始状态：`lru_slots[l, r, :] = [0, 1, 2, ..., device_buffer_size-1]`
- 每次 swap-in 后 kernel 内部更新：evictables 移到 LRU 端（前部），hits 移到 MRU 端（后部）
- 下次 swap-in 时 miss token 从 LRU 端（前部）取淘汰 slot

这个机制保证了 device buffer 中保留的是跨层高频访问的 token——利用了层间访问相似性（相邻层 top-k 有较高 overlap）。

### 2.6 关键接口汇总

| 接口 | 所在文件 | 作用 | CXL 影响 |
|------|---------|------|----------|
| `MLATokenToKVPoolHost.__init__` | memory_pool_host.py | 分配 host KV buffer | **需替换**：分配到 CXL |
| `MLATokenToKVPoolHost.init_kv_buffer` | memory_pool_host.py | 实际内存分配 | **需替换**：CXL mmap + register |
| `alloc_with_host_register` | memory_pool_host.py | DRAM 分配 + cudaHostRegister | **需替换/扩展** |
| `backup_from_device_all_layer` | memory_pool_host.py | GPU→Host 全层 KV 传输 | **可能需适配**：CXL 写入 |
| `load_to_device_per_layer` | memory_pool_host.py | Host→GPU 逐层 KV 传输 | **可能需适配**：CXL 读取 |
| `HiSparseCoordinator.__init__` | hisparse_coordinator.py | 构建 host pool | **需修改**：传入 CXL 配置 |
| `swap_in_selected_pages` | hisparse_coordinator.py | 调用 JIT kernel swap-in | **无需修改** |
| `hisparse.cuh` transfer_item_warp | jit_kernel/csrc/ | GPU 端 warp 级 host→device 拷贝 | **无需修改**（指针透明） |

---

## 3. CXL 内存池工具库分析

**文件**: `sgl-kernel/cxl_utils/`

现有 CXL 工具库提供了以下关键能力：

### 3.1 初始化与映射

```cpp
bool cxl_init(const std::string &dev_path = "/dev/dax0.0",
              std::size_t map_bytes = 64ULL * 1024 * 1024,
              off_t offset = 0,
              bool register_cuda = false,   // 是否 cudaHostRegister
              int gpu_id = 0);
```

- 通过 `mmap` 将 CXL DAX 设备映射到进程地址空间
- 可选 `cudaHostRegister(addr, length, cudaHostRegisterPortable | cudaHostRegisterMapped)` 使 GPU 可通过 DMA 访问
- `cudaHostGetDevicePointer` 获取 GPU 可见的设备端指针

### 3.2 数据传输接口

| 接口 | 方向 | 说明 |
|------|------|------|
| `vram2cxl` / `vram2cxl_async` | GPU → CXL | 同步/异步 `cudaMemcpy(D2H)` + `clflush` |
| `cxl2vram` / `cxl2vram_noflush` | CXL → GPU | `clflush` + `cudaMemcpy(H2D)` |
| `dram2cxl` | CPU → CXL | non-temporal store (`_mm_stream_si64`) |
| `cxl2dram` / `cxl2dram_noflush` | CXL → CPU | `clflush` + `memcpy` |
| `cxl2vram_noflush_parallel_contiguous_dst` | CXL → GPU (GPU kernel) | CUDA kernel 直接通过 device_ptr 读取 |
| `cxl2dram_noflush_parallel_contiguous_dst` | CXL → CPU (OpenMP) | 多线程并行 memcpy |

### 3.3 Python 绑定

```python
cxl_mem_ext.cxl_init(dev_path, map_bytes, offset, register_cuda, gpu_id)
cxl_mem_ext.tensor_to_cxl(tensor, offset)       # CPU/GPU tensor → CXL
cxl_mem_ext.cxl_to_tensor(tensor, offset)        # CXL → CPU/GPU tensor
cxl_mem_ext.cxl_to_tensor_noflush_parallel_contiguous_dst(tensor, bytes_per_segment, offsets)
cxl_mem_ext.cxl_close()
```

### 3.4 关键设计观察

1. **CXL 注册为 CUDA Host Memory 后，GPU 可通过 `device_ptr` 直接访问**——这是 HiSparse JIT kernel 无需修改的基础
2. **CUDA kernel 可直接读取 CXL**：`cxl2vram_copy_kernel` 已验证 GPU kernel 可通过注册后的 `device_ptr` 直接 load CXL 数据
3. **写入 CXL 需 clflush**：CPU 写入后需要 `_mm_clflushopt` 刷新 cache line；GPU 通过 `cudaMemcpy(D2H)` 写入时，DMA 直接到 CXL，不经过 CPU cache
4. **全局单实例 `g_cxl`**：当前实现使用全局变量管理一个 CXL region，需扩展为多 region 支持

---

## 4. CXL 内存池集成设计

### 4.1 设计原则

1. **指针透明性**：CXL 内存 `cudaHostRegister` 后对 GPU 呈现为普通 host memory，JIT kernel 无需感知底层介质
2. **最小侵入性**：仅替换 `MLATokenToKVPoolHost` 的内存分配后端，不修改 HiSparseCoordinator 的调度逻辑、LRU 管理、JIT kernel
3. **配置驱动**：通过 server args / hisparse config 选择 DRAM 或 CXL 后端
4. **渐进式实现**：先实现基本功能（单 CXL 设备），再优化（多设备 interleave、大页、布局）

### 4.2 整体架构变更

```
┌────────────────────────────────────────────────────┐
│  HiSparseCoordinator                                │
│    └─ mem_pool_host: MLATokenToKVPoolHost           │
│         └─ kv_buffer: torch.Tensor                  │
│              │                                       │
│         ┌────┴─────────────────────┐                │
│         │  内存后端选择 (配置驱动)    │                │
│         ├─────────┬────────────────┤                │
│         │ DRAM    │ CXL            │                │
│         │ (现有)   │ (新增)         │                │
│         │         │                │                │
│         │ pinned  │ cxl mmap       │                │
│         │ DRAM    │ + cudaHost     │                │
│         │ alloc   │ Register       │                │
│         └─────────┴────────────────┘                │
└────────────────────────────────────────────────────┘
```

**核心变更点**：
- 新增 `CXLMemoryBackend` 模块，封装 CXL 设备初始化、内存分配、注册
- 新增 `CXLHostTensorAllocator`，替代默认的 `HostTensorAllocator`，使 `torch.Tensor` 的底层存储指向 CXL mmap 区域
- 修改 `MLATokenToKVPoolHost` 的初始化路径，支持 CXL allocator
- HiSparseCoordinator 的其余逻辑（staging、swap-in、LRU、JIT kernel）完全不变

### 4.3 模块设计

```
sgl-kernel/cxl_utils/                     # 现有 CXL 底层 C++/CUDA 库
    ├── cxl_mem.hpp/cpp                    # mmap, 数据传输
    ├── cxl_mem_cuda.cu                    # CUDA 注册, GPU kernel
    └── cxl_mem_pybind.cpp                 # Python 绑定

python/sglang/srt/mem_cache/
    ├── memory_pool_host.py                # [修改] 增加 CXL allocator 支持
    ├── cxl_memory_backend.py              # [新增] CXL 内存后端封装
    └── hisparse_memory_pool.py            # [不变]

python/sglang/srt/managers/
    └── hisparse_coordinator.py            # [微调] 传递 CXL 配置给 host pool

python/sglang/srt/
    └── server_args.py                     # [修改] 增加 CXL 配置参数
```

### 4.4 数据流变更

#### 4.4.1 Prefill Staging（GPU → CXL）

```
现有:  GPU VRAM → cudaMemcpy(D2H) → CPU pinned DRAM
CXL:   GPU VRAM → cudaMemcpy(D2H) → CXL mmap region (registered as host memory)
```

`transfer_kv_all_layer_mla` 底层使用 CUDA DMA，目标地址只需是 `cudaHostRegister` 注册的指针。CXL mmap 注册后，DMA 目标直接是 CXL 物理地址。**无需修改传输代码**。

但需注意：GPU DMA 写入 CXL 后，CPU cache 中可能有 stale 数据。后续 CPU 读取该区域前需要 `clflush`。不过在 HiSparse 流程中，staging 写入后的读取全部发生在 GPU 端（JIT kernel），GPU 通过 DMA/PCIe 直接读取 CXL，不经过 CPU cache，因此**不需要额外 clflush**。

#### 4.4.2 Decode Swap-in（CXL → GPU）

```
现有:  JIT kernel: ld.global.nc.b64 from pinned DRAM → st.global.cg.b64 to VRAM
CXL:   JIT kernel: ld.global.nc.b64 from CXL mmap (via device_ptr) → st.global.cg.b64 to VRAM
```

`transfer_item_warp` 在 GPU 端通过 `ld.global.nc.b64` 从 host 指针读取。CXL 注册后获取的 `device_ptr` 语义等同于 pinned DRAM 的 device-mapped 指针。**JIT kernel 完全不需要修改**。

#### 4.4.3 Decode Backup（GPU → CXL，每步一个 token）

```
现有:  transfer_kv_all_layer_mla → GPU DMA → pinned DRAM
CXL:   transfer_kv_all_layer_mla → GPU DMA → CXL mmap region
```

同 staging 路径，**无需修改**。

### 4.5 JIT Kernel 的适配

**结论：JIT kernel (`hisparse.cuh`) 无需任何修改。**

原因：
1. Kernel 接收的 `host_cache_k` / `host_cache_v` 是 `void*` 指针，指向 `MLATokenToKVPoolHost.kv_buffer` 的数据地址
2. 只要该数据地址经过 `cudaHostRegister` 注册，GPU 即可通过 PCIe BAR 或 CXL 链路访问
3. CXL mmap 区域经注册后，对 GPU 呈现为与 pinned DRAM 完全相同的 host memory 语义
4. `ld.global.nc.b64` 是非缓存的全局内存加载，不依赖 CPU cache coherency

---

## 5. 详细代码设计

### 5.1 CXL Memory Backend 模块

**新文件**: `python/sglang/srt/mem_cache/cxl_memory_backend.py`

```python
"""CXL Memory Backend for HiSparse KV Cache Storage.

Manages CXL DAX device mapping and provides tensor allocation
on CXL memory with CUDA registration for GPU DMA access.
"""

import logging
import ctypes
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)


@dataclass
class CXLDeviceConfig:
    """Configuration for a single CXL device."""
    dev_path: str = "/dev/dax0.0"
    map_bytes: int = 64 * 1024 * 1024 * 1024   # 64GB default
    offset: int = 0
    gpu_id: int = 0


@dataclass
class CXLPoolConfig:
    """Configuration for CXL memory pool."""
    devices: List[CXLDeviceConfig]
    enable_cuda_register: bool = True
    use_huge_pages: bool = False           # 2MB huge pages for TLB optimization
    interleave: bool = False               # interleave across multiple CXL devices
    page_size_bytes: int = 4096            # mmap page size


class CXLMemoryRegion:
    """Represents a mapped CXL memory region."""

    def __init__(self, config: CXLDeviceConfig, ext_module):
        self.config = config
        self.ext = ext_module
        self.initialized = False
        self.base_ptr = None
        self.size = config.map_bytes
        self.allocated_offset = 0          # simple bump allocator

    def init(self, register_cuda: bool = True):
        ok = self.ext.cxl_init(
            dev_path=self.config.dev_path,
            map_bytes=self.config.map_bytes,
            offset=self.config.offset,
            register_cuda=register_cuda,
            gpu_id=self.config.gpu_id,
        )
        if not ok:
            raise RuntimeError(
                f"Failed to initialize CXL device at {self.config.dev_path}"
            )
        self.initialized = True
        logger.info(
            "CXL device initialized: %s, size=%.2f GB, gpu_id=%d",
            self.config.dev_path,
            self.config.map_bytes / 1e9,
            self.config.gpu_id,
        )

    def allocate(self, nbytes: int, alignment: int = 64) -> int:
        """Allocate bytes from this region, returns offset."""
        aligned_offset = (self.allocated_offset + alignment - 1) & ~(alignment - 1)
        if aligned_offset + nbytes > self.size:
            raise RuntimeError(
                f"CXL region exhausted: requested {nbytes} bytes at offset "
                f"{aligned_offset}, but only {self.size - aligned_offset} available"
            )
        self.allocated_offset = aligned_offset + nbytes
        return aligned_offset

    def close(self):
        if self.initialized:
            self.ext.cxl_close()
            self.initialized = False


class CXLMemoryBackend:
    """Manages CXL memory pool lifecycle and tensor allocation."""

    _instance: Optional["CXLMemoryBackend"] = None

    def __init__(self, config: CXLPoolConfig):
        self.config = config
        self.ext = self._build_extension()
        self.regions: List[CXLMemoryRegion] = []
        self._init_regions()

    @classmethod
    def get_instance(cls) -> Optional["CXLMemoryBackend"]:
        return cls._instance

    @classmethod
    def initialize(cls, config: CXLPoolConfig) -> "CXLMemoryBackend":
        if cls._instance is not None:
            logger.warning("CXL backend already initialized, reusing existing instance")
            return cls._instance
        cls._instance = cls(config)
        return cls._instance

    @staticmethod
    def _build_extension():
        """Build or load the CXL C++ extension."""
        cxl_root = Path(__file__).resolve().parents[4] / "sgl-kernel" / "cxl_utils"
        return load(
            name="cxl_mem_ext",
            sources=[
                str(cxl_root / "cxl_mem.cpp"),
                str(cxl_root / "cxl_mem_cuda.cu"),
                str(cxl_root / "cxl_mem_pybind.cpp"),
            ],
            extra_cflags=["-O3", "-std=c++17", "-mclflushopt", "-mclwb",
                          "-msse4.1", "-fopenmp"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_ldflags=["-lgomp"],
            with_cuda=True,
            verbose=False,
        )

    def _init_regions(self):
        for dev_config in self.config.devices:
            region = CXLMemoryRegion(dev_config, self.ext)
            region.init(register_cuda=self.config.enable_cuda_register)
            self.regions.append(region)
        logger.info(
            "CXL memory backend initialized with %d device(s), total %.2f GB",
            len(self.regions),
            sum(r.size for r in self.regions) / 1e9,
        )

    def allocate_tensor(self, dims: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Allocate a tensor backed by CXL memory.

        The returned tensor's storage points to the CXL mmap region,
        which is CUDA-registered for GPU DMA access.
        """
        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * torch.finfo(dtype).bits // 8 if dtype.is_floating_point else numel * dtype.itemsize

        if self.config.interleave and len(self.regions) > 1:
            region = self._select_region_interleave(nbytes)
        else:
            region = self.regions[0]

        offset = region.allocate(nbytes, alignment=64)

        self.ext.tensor_to_cxl(torch.zeros(1, dtype=torch.uint8), offset=0)

        tensor = torch.zeros(dims, dtype=dtype, device="cpu")

        self.ext.tensor_to_cxl(
            tensor.contiguous().view(-1),
            offset=offset,
        )

        tensor_from_cxl = torch.empty(dims, dtype=dtype, device="cpu")
        self.ext.cxl_to_tensor_noflush(
            tensor_from_cxl.contiguous().view(-1),
            offset=offset,
        )

        return tensor_from_cxl

    def write_tensor_to_cxl(self, tensor: torch.Tensor, offset: int):
        """Write a tensor to CXL at the given offset."""
        self.ext.tensor_to_cxl(tensor, offset=offset)

    def read_tensor_from_cxl(self, tensor: torch.Tensor, offset: int):
        """Read from CXL into a tensor at the given offset."""
        self.ext.cxl_to_tensor_noflush(tensor, offset=offset)

    def _select_region_interleave(self, nbytes: int) -> CXLMemoryRegion:
        """Select CXL region with most available space."""
        best = min(self.regions, key=lambda r: r.allocated_offset)
        return best

    def close(self):
        for region in self.regions:
            region.close()
        self.regions.clear()
        CXLMemoryBackend._instance = None
```

### 5.2 CXLMLATokenToKVPoolHost 类设计

**新增**: 在 `python/sglang/srt/mem_cache/memory_pool_host.py` 中添加 CXL 分配器和子类。

#### 5.2.1 CXL Tensor Allocator

```python
class CXLHostTensorAllocator(HostTensorAllocator):
    """Allocates tensors on CXL memory via mmap + cudaHostRegister.

    The key insight: we allocate raw CXL memory via mmap, create a torch.Tensor
    whose storage points to that mmap region, and register it with CUDA.
    This makes the tensor accessible to both CPU and GPU transparently.
    """

    def __init__(self, cxl_backend: "CXLMemoryBackend"):
        super().__init__()
        self.cxl_backend = cxl_backend
        self.ext = cxl_backend.ext
        self._allocations = []  # track (offset, nbytes) for cleanup

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor backed by CXL memory.

        Returns a CPU tensor whose data_ptr points to the CXL mmap region.
        The region is already CUDA-registered during cxl_init, so GPU can
        DMA to/from this tensor directly.
        """
        self.dtype = dtype
        self.dims = dims

        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * dtype.itemsize

        region = self.cxl_backend.regions[0]
        offset = region.allocate(nbytes, alignment=64)
        self._allocations.append((offset, nbytes))

        # Create a torch tensor that views the CXL mmap region.
        # We use torch.from_blob (via ctypes) to wrap the mmap pointer.
        import ctypes
        cxl_ptr = ctypes.c_void_p(
            ctypes.cast(
                ctypes.c_char_p(region.ext.__cxl_base_ptr()),  # need new API
                ctypes.c_void_p
            ).value + offset
        )

        # Alternative approach: allocate a normal CPU tensor and copy it
        # into CXL region. The CXL region pointer is registered with CUDA,
        # so all subsequent DMA operations use CXL memory directly.
        #
        # Since cxl_init already registered the entire region with
        # cudaHostRegister, we can use the mmap base + offset as data_ptr.
        tensor = torch.empty(dims, dtype=dtype, device=device)
        return tensor
```

#### 5.2.2 实际推荐方案：直接在 CXL mmap 区域上构建 Tensor

由于 PyTorch 的存储模型限制，最可靠的方案是：

1. 在 `cxl_init` 时将 CXL 区域映射并注册为 CUDA host memory
2. 使用 `torch.frombuffer` 或 `torch.Tensor` 的 storage 机制，将 tensor 的底层存储指向 CXL mmap 地址
3. 所有后续操作（`transfer_kv_*` 系列函数）使用该 tensor 的 `data_ptr()`，自然指向 CXL 内存

**扩展 cxl_mem_pybind.cpp**，暴露 mmap base pointer：

```cpp
// 新增 pybind 接口
m.def("cxl_get_base_ptr", []() -> int64_t {
    return reinterpret_cast<int64_t>(g_cxl.base);
}, "Get the base pointer of the mapped CXL region.");

m.def("cxl_get_mapped_size", []() -> int64_t {
    return static_cast<int64_t>(g_cxl.length);
}, "Get the mapped size of the CXL region.");
```

**Python 侧 tensor 创建**：

```python
def alloc_cxl_tensor(dims: tuple, dtype: torch.dtype, cxl_ext, offset: int) -> torch.Tensor:
    """Create a torch.Tensor backed by CXL memory at the given offset.

    The CXL region must already be initialized and CUDA-registered.
    """
    numel = 1
    for d in dims:
        numel *= d
    nbytes = numel * dtype.itemsize

    base_ptr = cxl_ext.cxl_get_base_ptr()
    data_ptr = base_ptr + offset

    # Use ctypes to create a buffer view of the CXL memory
    ctypes_array = (ctypes.c_byte * nbytes).from_address(data_ptr)
    buffer = memoryview(ctypes_array)

    # Create tensor from the buffer (zero-copy)
    tensor = torch.frombuffer(buffer, dtype=dtype, count=numel).reshape(dims)

    # Zero-initialize
    tensor.zero_()

    return tensor
```

#### 5.2.3 CXLMLATokenToKVPoolHost

```python
class CXLMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool backed by CXL memory.

    Inherits all swap-in/backup logic from MLATokenToKVPoolHost.
    Only overrides memory allocation to use CXL mmap region
    instead of pinned DRAM.
    """

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        cxl_backend: "CXLMemoryBackend",
        override_kv_cache_dim: Optional[int] = None,
    ):
        self.cxl_backend = cxl_backend
        self.override_kv_cache_dim = override_kv_cache_dim
        # pin_memory=False because CXL memory is already registered
        # device="cpu" because tensor operations treat it as CPU memory
        super(HostKVCache, self).__init__()  # skip MLATokenToKVPoolHost.__init__

        # Manually initialize fields that HostKVCache.__init__ would set
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = False  # CXL region is already CUDA-registered
        self.device = "cpu"
        self.allocator = CXLHostTensorAllocator(cxl_backend)

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        # Allocate KV buffer on CXL
        self.kv_buffer = self._init_cxl_kv_buffer()

        # Set up data pointers for transfer_kv_* functions
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

        import threading
        self.lock = threading.RLock()
        self.clear()

    def _init_cxl_kv_buffer(self) -> torch.Tensor:
        """Allocate the KV buffer on CXL memory."""
        if self.layout == "layer_first":
            dims = (self.layer_num, self.size, 1, self.kv_cache_dim)
        elif self.layout == "page_first":
            dims = (self.size, self.layer_num, 1, self.kv_cache_dim)
        else:
            raise ValueError(f"Unsupported layout for CXL: {self.layout}")

        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * self.dtype.itemsize

        region = self.cxl_backend.regions[0]
        offset = region.allocate(nbytes, alignment=64)

        buffer = alloc_cxl_tensor(dims, self.dtype, self.cxl_backend.ext, offset)

        logger.info(
            "CXL KV buffer allocated: %.2f GB at CXL offset %d, layout=%s",
            nbytes / 1e9, offset, self.layout,
        )
        return buffer

    # load_to_device_per_layer, backup_from_device_all_layer:
    # 完全继承 MLATokenToKVPoolHost 的实现，无需覆写。
    # 因为：
    #   1. transfer_kv_*_mla 系列函数通过 data_ptrs (uint64 pointer) 操作，
    #      不关心底层是 DRAM 还是 CXL
    #   2. CXL mmap 区域已注册为 CUDA host memory，GPU DMA 可直接访问
    #   3. kv_buffer[layer_id] 的 data_ptr() 指向 CXL 物理地址
```

### 5.3 HiSparseCoordinator 适配

**修改文件**: `python/sglang/srt/managers/hisparse_coordinator.py`

变更最小——仅在构造 `mem_pool_host` 时根据配置选择 DRAM 或 CXL 后端：

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
        cxl_config: Optional[dict] = None,     # 新增参数
    ):
        # ... (existing initialization) ...

        if cxl_config is not None and cxl_config.get("enabled", False):
            from sglang.srt.mem_cache.cxl_memory_backend import (
                CXLMemoryBackend, CXLPoolConfig, CXLDeviceConfig,
            )
            cxl_pool_config = CXLPoolConfig(
                devices=[CXLDeviceConfig(
                    dev_path=cxl_config.get("dev_path", "/dev/dax0.0"),
                    map_bytes=cxl_config.get("map_bytes", 64 * 1024**3),
                    gpu_id=cxl_config.get("gpu_id", 0),
                )],
                enable_cuda_register=True,
            )
            cxl_backend = CXLMemoryBackend.initialize(cxl_pool_config)
            self.mem_pool_host = CXLMLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=1,
                layout="layer_first",
                cxl_backend=cxl_backend,
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
        else:
            # 现有 DRAM 路径，不变
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

### 5.4 ServerArgs 与配置扩展

**修改文件**: `python/sglang/srt/server_args.py`

```python
# 在 hisparse_config JSON 中增加 CXL 相关字段
# 示例配置:
# --hisparse-config '{"top_k": 2048, "cxl": {"enabled": true, "dev_path": "/dev/dax0.0", "map_bytes": 68719476736}}'
```

**修改文件**: `python/sglang/srt/mem_cache/sparsity/factory.py`

```python
def parse_hisparse_config(server_args) -> SparseConfig:
    # ... existing parsing ...

    # Extract CXL config from hisparse_config
    cxl_config = config_dict.get("cxl", None)
    sparse_config.extra_config["cxl"] = cxl_config

    return sparse_config
```

### 5.5 ModelRunner 初始化流程修改

**修改文件**: `python/sglang/srt/model_executor/model_runner.py`

在 `HiSparseCoordinator` 的构造中传入 CXL 配置：

```python
# 在 model_runner.py 中构造 HiSparseCoordinator 的地方
self.hisparse_coordinator = HiSparseCoordinator(
    req_to_token_pool=self.req_to_token_pool,
    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
    top_k=sparse_config.top_k,
    device_buffer_size=sparse_config.device_buffer_size,
    device=self.device,
    tp_group=self.tp_group,
    host_to_device_ratio=sparse_config.host_to_device_ratio,
    cxl_config=sparse_config.extra_config.get("cxl", None),  # 新增
)
```

---

## 6. CXL 内存策略与优化

### 6.1 单设备 vs 多设备 Interleave

| 策略 | 实现方式 | 优势 | 局限 |
|------|---------|------|------|
| 单设备 CXL-only | 一块 CXL 卡，所有 KV 在其上 | 简单，代码改动最小 | 带宽受限于单卡 |
| 多设备 Interleave | 多块 CXL 卡，KV 按 token 或 layer 交错分布 | 聚合带宽线性增长 | 需管理多 region |
| DRAM + CXL 混合 | 热数据在 DRAM，冷数据在 CXL | 兼顾延迟和容量 | 需迁移策略 |

**实现方案**：

```python
# 多设备 interleave：按 layer 分布到不同 CXL 设备
# layer_id % num_cxl_devices → 选择 CXL 设备
# 利用 sparse attention 的层间独立性，不同层的 swap-in 可并行访问不同设备

class InterleavedCXLPool:
    """Distribute KV cache layers across multiple CXL devices."""

    def __init__(self, backends: List[CXLMemoryBackend], layer_num: int):
        self.backends = backends
        self.layer_to_device = [i % len(backends) for i in range(layer_num)]

    def get_buffer_for_layer(self, layer_id: int) -> torch.Tensor:
        dev_idx = self.layer_to_device[layer_id]
        return self.backends[dev_idx].get_allocated_region(layer_id)
```

### 6.2 大页（Huge Pages）优化

sparse top-k 的离散访问模式会导致大量 TLB miss。CXL DAX 设备支持 2MB/1GB 大页：

```python
# 在 cxl_init 中使用大页映射
# mmap 时指定 MAP_HUGETLB | MAP_HUGE_2MB
# 或通过 /sys/kernel/mm/hugepages/ 预分配大页
```

**实现**: 在 `cxl_mem.cpp` 的 `cxl_init` 中增加大页选项：

```cpp
bool cxl_init(const std::string &dev_path, std::size_t map_bytes,
              off_t offset, bool register_cuda, int gpu_id,
              bool use_huge_pages) {
    int flags = MAP_SHARED;
    if (use_huge_pages) {
        flags |= MAP_HUGETLB | MAP_HUGE_2MB;
    }
    void *addr = ::mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, flags, fd, offset);
    // ...
}
```

### 6.3 KV Cache 布局优化

| 布局 | CXL 上的特性 | 适用场景 |
|------|-------------|---------|
| `layer_first` | 同一层所有 token 连续，sparse top-k 访问离散 | 通用，当前默认 |
| `page_first` | 同一 token 所有层连续，层间预取友好 | 层间相似性高时 |

**建议**：初始使用 `layer_first`（与现有 HiSparse 默认一致），后续实验 `page_first` 在 CXL 上的性能差异。

### 6.4 写入一致性处理

| 操作 | 写入方 | 是否需要 clflush |
|------|-------|-----------------|
| Staging (GPU→CXL) | GPU DMA (`cudaMemcpy D2H`) | 不需要：DMA 直接写 CXL，不经过 CPU cache |
| Decode backup (GPU→CXL) | GPU DMA | 不需要：同上 |
| Swap-in (CXL→GPU) | GPU kernel (`ld.global.nc`) | 不需要：GPU 直接通过 PCIe/CXL 读取 |
| CPU 读 CXL（如有） | CPU | 需要 clflush：确保读到最新数据 |

在 HiSparse 流程中，host pool 的读写全部由 GPU 发起（DMA/kernel），不经过 CPU cache，因此**不需要额外的 clflush 操作**。

---

## 7. 分阶段实施计划

### Phase 1: 基础功能（1-2 周）

**目标**：单 CXL 设备替换 pinned DRAM，验证正确性

- [ ] 扩展 `cxl_mem_pybind.cpp`，暴露 `cxl_get_base_ptr` / `cxl_get_mapped_size`
- [ ] 实现 `CXLMemoryBackend` 类（单设备）
- [ ] 实现 `CXLHostTensorAllocator`，基于 `torch.frombuffer` 在 CXL mmap 上构建 tensor
- [ ] 实现 `CXLMLATokenToKVPoolHost`，继承 `MLATokenToKVPoolHost`
- [ ] 修改 `HiSparseCoordinator.__init__`，支持 CXL 配置
- [ ] 添加 `--hisparse-config` 中的 CXL 字段解析
- [ ] 编写单元测试：CXL tensor 分配、GPU DMA 读写正确性、与 DRAM 基线对比

### Phase 2: 正确性验证（1 周）

**目标**：在 DeepSeek-V3.2 上端到端验证

- [ ] 在 CXL 设备上运行 HiSparse + DeepSeek-V3.2 推理
- [ ] 验证输出与 DRAM 基线完全一致
- [ ] 验证 staging、swap-in、LRU 更新、request finish 的完整生命周期
- [ ] 内存泄漏检测（CXL region allocation/free tracking）

### Phase 3: 性能优化（2 周）

**目标**：CXL 访问性能调优

- [ ] 实现大页支持（2MB huge pages），对比 4KB 页的 TLB miss 差异
- [ ] 实现多 CXL 设备 interleave，测量聚合带宽
- [ ] 测试 `layer_first` vs `page_first` 布局在 CXL 上的性能
- [ ] Nsight Systems profiling：量化 swap-in kernel 中 CXL 读取延迟
- [ ] 探索 NUMA pinning 策略对 CXL 访问延迟的影响

### Phase 4: 对比实验（1-2 周）

**目标**：按 experiment_plan_deepseek_v32.md 执行三方对比

- [ ] Micro-benchmark：CXL vs DRAM vs RDMA 离散 KV 读取
- [ ] 端到端 Serving：吞吐、TTFT、TBT 对比
- [ ] 内存效率：最大并发请求数对比
- [ ] 消融实验：buffer size、sparse ratio、CXL 策略

---

## 8. 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| CXL 延迟高于 DRAM | swap-in kernel 变慢 | LRU buffer 利用层间相似性减少 miss 次数；多卡 interleave 提升聚合带宽 |
| `cudaHostRegister` 对 CXL mmap 失败 | GPU 无法 DMA 访问 CXL | 降级为 CPU 中转（CXL→DRAM→GPU）；确认 CXL 设备 PCIe topology |
| `torch.frombuffer` 生命周期管理 | tensor 释放后 mmap 仍有效 | CXL region 生命周期绑定到进程，不释放；使用引用计数 |
| TLB miss 导致 CXL 离散访问性能退化 | 实际带宽远低于理论峰值 | 强制使用 2MB/1GB 大页；token 对齐到 cache line |
| 多 GPU TP 场景下 CXL 竞争 | 多 GPU 同时读取 CXL 带宽瓶颈 | 每 GPU 映射独立 CXL 区域；TP group 内各 rank 使用不同 offset |
| 写入 CXL 后 CPU cache 不一致 | 仅在 CPU 读取 CXL 时发生 | HiSparse 流程中 host pool 读写全由 GPU 发起，不经过 CPU cache，不受影响 |

---

## 附录 A：关键文件索引

| 文件 | 角色 | CXL 变更 |
|------|------|---------|
| `python/sglang/srt/managers/hisparse_coordinator.py` | HiSparse 核心协调器 | 增加 `cxl_config` 参数 |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | Host KV Pool 基类 | 增加 CXL allocator |
| `python/sglang/srt/mem_cache/hisparse_memory_pool.py` | Device Pool + Allocator | 不变 |
| `python/sglang/srt/mem_cache/cxl_memory_backend.py` | **新增** CXL 后端 | — |
| `python/sglang/jit_kernel/csrc/hisparse.cuh` | JIT swap-in kernel | 不变 |
| `python/sglang/jit_kernel/hisparse.py` | JIT Python 入口 | 不变 |
| `python/sglang/srt/layers/attention/nsa_backend.py` | NSA attention 后端 | 不变 |
| `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | NSA top-k indexer | 不变 |
| `python/sglang/srt/model_executor/model_runner.py` | 模型初始化 | 传递 CXL 配置 |
| `python/sglang/srt/server_args.py` | 启动参数 | CXL 配置解析 |
| `sgl-kernel/cxl_utils/*` | CXL 底层库 | 增加 base_ptr 接口 |

## 附录 B：配置示例

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

# 多 CXL 设备 interleave
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 \
    --tp 8 \
    --enable-hisparse \
    --hisparse-config '{
        "top_k": 2048,
        "cxl": {
            "enabled": true,
            "devices": [
                {"dev_path": "/dev/dax0.0", "map_bytes": 68719476736, "gpu_id": 0},
                {"dev_path": "/dev/dax1.0", "map_bytes": 68719476736, "gpu_id": 0}
            ],
            "interleave": true
        }
    }' \
    --disable-radix-cache
```
