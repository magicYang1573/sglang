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
3. [SGLang 并行架构与存储层次](#3-sglang-并行架构与存储层次)
  - 3.1 [进程模型：每个 TP Rank 一个独立进程](#31-进程模型每个-tp-rank-一个独立进程)
  - 3.2 [并行方式概览：TP、DPA、EP 对 KV Cache 的影响](#32-并行方式概览tpdpaep-对-kv-cache-的影响)
  - 3.3 [核心对象所有权](#33-核心对象所有权)
  - 3.4 [存储层次与对象关系图](#34-存储层次与对象关系图)
  - 3.5 [MLA KV Cache 的特性](#35-mla-kv-cache-的特性)
  - 3.6 [CXL 在多进程架构下的设计](#36-cxl-在多进程架构下的设计)
  - 3.7 [架构设计的合理性](#37-架构设计的合理性)
4. [CXL 直接 GPU 读取方案验证](#4-cxl-直接-gpu-读取方案验证)
5. [CXL 内存池集成设计](#5-cxl-内存池集成设计)
  - 5.1 [设计原则](#51-设计原则)
  - 5.2 [核心方案：mmap + cudaHostRegister + torch.frombuffer](#52-核心方案mmap--cudahostregister--torchfrombuffer)
  - 5.3 [整体架构变更](#53-整体架构变更)
  - 5.4 [模块设计](#54-模块设计)
  - 5.5 [CXL 分区与 TP 多进程](#55-cxl-分区与-tp-多进程)
  - 5.6 [CXL 容量约束与 KV Pool 大小自动适配](#56-cxl-容量约束与-kv-pool-大小自动适配)
  - 5.7 [数据流变更](#57-数据流变更)
6. [详细代码设计](#6-详细代码设计)
  - 6.1 [CXLMemoryRegion 轻量管理类](#61-cxlmemoryregion-轻量管理类)
  - 6.2 [CXL Tensor 分配函数](#62-cxl-tensor-分配函数)
  - 6.3 [CXLMLATokenToKVPoolHost 类设计](#63-cxlmlatokentokv-poolhost-类设计)
  - 6.4 [HiSparseCoordinator 适配](#64-hisparsecoordinator-适配)
  - 6.5 [ServerArgs 与配置扩展](#65-serverargs-与配置扩展)
7. [为什么不使用 cxl_utils 库的传输函数](#7-为什么不使用-cxl_utils-库的传输函数)
8. [多 CXL 设备 Interleave 设计方案](#8-多-cxl-设备-interleave-设计方案)
  - 8.1 [目标配置与 KV Cache 特性](#81-目标配置与-kv-cache-特性)
  - 8.2 [单设备瓶颈与多设备机会](#82-单设备瓶颈与多设备机会)
  - 8.3 [Interleave 策略](#83-interleave-策略)
  - 8.4 [实现接口](#84-实现接口)
  - 8.5 [容量规划](#85-容量规划)
  - 8.6 [其他可选优化](#86-其他可选优化)
9. [分阶段实施计划](#9-分阶段实施计划)
10. [风险与缓解措施](#10-风险与缓解措施)

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

## 3. SGLang 并行架构与存储层次

### 3.1 进程模型：每个 TP Rank 一个独立进程

SGLang 的 Tensor Parallel（TP）采用**多进程架构**。以 TP=8 为例：

```
Engine (主进程)
  └── _launch_scheduler_processes()
       ├── mp.Process(tp_rank=0, gpu_id=0) → Scheduler → TpModelWorker → ModelRunner
       ├── mp.Process(tp_rank=1, gpu_id=1) → Scheduler → TpModelWorker → ModelRunner
       ├── ...
       └── mp.Process(tp_rank=7, gpu_id=7) → Scheduler → TpModelWorker → ModelRunner
```

**关键事实**：

- 每个 `tp_rank` 是一个独立的 OS **进程**（`multiprocessing.Process`），不是线程
- 每个进程绑定一个 GPU（`gpu_id`），有独立的 Python 解释器和 CUDA context
- 进程间通过 NCCL 通信（all-reduce / all-gather），不共享 Python 对象

### 3.2 并行方式概览：TP、DPA、EP 对 KV Cache 的影响

SGLang 支持多种并行方式，它们对 KV Cache 存储的影响各不相同：

```
并行层次（由外到内）:
  Attention 路径: Global(TP) → DP → ATTN_CP → ATTN_TP
  MoE 路径:      Global(TP) → MOE_DP → EP → MOE_TP
```

#### 3.2.1 纯 TP（Tensor Parallel）

**配置示例**：`--tp 8`

```
8 个进程，每个处理相同的请求批次
┌──────┬──────┬──────┬───┬──────┐
│rank 0│rank 1│rank 2│...│rank 7│  ← 同一批请求，KV Cache 内容相同
└──────┴──────┴──────┴───┴──────┘
  NCCL all-reduce (attention output)
```

- 所有 rank 处理**同一批请求**（同一个 Scheduler 分发）
- MLA KV Cache 是 latent vector（`kv_lora_rank + qk_rope_head_dim`），**不按 head 分片**
- 因此**每个 rank 的 KV Cache 内容完全相同**，是互为副本
- Host pool 每个 rank 独立存一份 → TP=8 时 host 总内存 = 8 × 单 rank host pool 大小

**为什么不共享 KV？** MLA 的 KV 是压缩后的 latent，维度很小（~656 bytes/token），TP 分片无收益。且各 rank 是独立进程，跨进程共享 tensor 需要 IPC 同步，复杂度远大于各自独立存储。这是**现有 DRAM HiSparse 的行为**，CXL 方案保持一致。

#### 3.2.2 DPA（Data Parallel Attention）

**配置示例**：`--tp 8 --dp 2 --enable-dp-attention`

```
                    DataParallelController (负载均衡)
                   /                              \
          DP group 0                        DP group 1
          (请求集 A)                         (请求集 B)
   ┌──────┬──────┬──────┬──────┐   ┌──────┬──────┬──────┬──────┐
   │rank 0│rank 1│rank 2│rank 3│   │rank 4│rank 5│rank 6│rank 7│
   └──────┴──────┴──────┴──────┘   └──────┴──────┴──────┴──────┘
    attn_tp: 4 ranks 共处理 A        attn_tp: 4 ranks 共处理 B
```

- `DataParallelController` 将请求**负载均衡**到不同的 DP group
- **不同 DP group 处理不同的请求** → **KV Cache 内容不同**
- 同一 DP group 内的 `attn_tp` ranks 仍处理相同请求（通过 broadcast 同步）
- `max_running_requests` 按 `dp_size` 均分

**对 KV Cache 的影响**：不同 DP group 的 KV 数据不同，每个 worker 独立存储自己负责的请求的 KV Cache。**"每个 worker 一份独立 host pool" 的架构完全正确**——因为各 worker 的数据本来就不同。

#### 3.2.3 EP（Expert Parallel）

**配置示例**：`--tp 8 --ep 8`

- EP 仅影响 **MoE FFN 层**的 expert 分布：不同 rank 持有不同的 expert 权重子集
- **EP 完全不影响 KV Cache 存储**：MLA attention 的 KV 数据与 MoE routing 无关
- EP rank 的 KV pool 大小和内容与非 EP 模式完全一致

#### 3.2.4 DPA + EP（推荐的 DeepSeek-V3 大规模部署配置）

**配置示例**：`--tp 16 --dp 2 --enable-dp-attention --ep 16`

```
Global TP = 16 GPUs
├── Attention: dp_size=2, attn_tp_size=8
│   DP group 0 (rank 0-7): 请求集 A → 8 rank 各自独立 KV Cache
│   DP group 1 (rank 8-15): 请求集 B → 8 rank 各自独立 KV Cache
│
└── MoE: ep_size=16
    16 个 rank 各持有不同 expert，对 KV Cache 无影响
```

**在 DPA+EP 下**：

- 不同 DP group 的 KV 内容不同（处理不同请求）
- 同一 DP group 内的 attn_tp ranks 的 KV 内容相同（处理同一批请求）
- EP 对 KV 完全透明
- 每个 worker 的 host pool 存储**只与该 worker 所属 DP group 负责的请求相关**

#### 3.2.5 总结：各并行方式下 host KV Cache 的独立性

| 并行配置 | 同一 group 内 KV 相同? | 不同 group 间 KV 不同? | 每 worker 一份 host pool? |
| --- | --- | --- | --- |
| 纯 TP=8 | 是（全部 8 rank 相同） | N/A（只有 1 个 group） | 是（8 份副本，与现有 DRAM 行为一致） |
| TP=8, DPA dp=2 | 是（同一 DP group 内 4 rank 相同） | **是**（不同 DP group 处理不同请求） | 是（天然正确） |
| TP=8, EP=8 | 是（全部 8 rank 相同） | N/A | 是（EP 不影响 KV） |
| TP=16, DPA dp=2, EP=16 | 是（同一 DP group 内 8 rank 相同） | **是** | 是（天然正确） |

**结论**：**"每个 worker 独立持有一份 host pool"的架构在所有并行配置下都是正确的。** 对于纯 TP 模式，虽然各 rank 存储的 KV 内容相同（有冗余），但这与现有 DRAM HiSparse 行为一致，且避免了跨进程 IPC 同步的复杂性。对于 DPA 模式，不同 DP group 的 KV 内容本身就不同，独立存储是必要的。

### 3.3 核心对象所有权

| 对象 | 创建位置 | 每个 Worker 独有? | 说明 |
| --- | --- | --- | --- |
| `Scheduler` | `run_scheduler_process()` | **是** | 调度器，管理请求队列和前向循环 |
| `TpModelWorker` | `Scheduler.init_tp_model_worker()` | **是** | 封装 ModelRunner，每进程一个 |
| `ModelRunner` | `TpModelWorker._init_model_runner()` | **是** | 模型执行器，持有模型权重、KV pool |
| `HiSparseCoordinator` | `ModelRunner.initialize()` | **是** | HiSparse 核心协调器 |
| `HiSparseNSATokenToKVPool` | `ModelRunner.init_memory_pool()` | **是** | GPU 端 KV cache（Device Buffer） |
| `HiSparseTokenToKVPoolAllocator` | `ModelRunner.init_memory_pool()` | **是** | Device token 分配器 |
| `MLATokenToKVPoolHost` / `CXLMLATokenToKVPoolHost` | `HiSparseCoordinator.__init__()` | **是** | Host 端 KV cache（DRAM 或 CXL） |
| `ReqToTokenPool` | `TpModelWorker.__init__()` | **是** | 请求→token 映射池 |
| NCCL Communicator | ModelRunner distributed init | **共享通信** | 跨 rank 的 all-reduce/broadcast 通道 |
| CXL DAX 物理设备 | `/dev/dax0.0` | **物理共享** | 所有 rank 共用同一物理设备，按 rank 分区 |

### 3.4 存储层次与对象关系图

```
Per Worker Process (tp_rank=i, gpu_id=i)
┌─────────────────────────────────────────────────────────────────┐
│  Scheduler                                                       │
│   └── TpModelWorker                                              │
│        └── ModelRunner (tp_rank=i, gpu_id=i)                     │
│             │                                                    │
│             ├── model weights (GPU VRAM, TP 分片)                │
│             │                                                    │
│             ├── HiSparseNSATokenToKVPool (GPU VRAM)              │
│             │   ├── kv_buffer: (layer_num, size, 1, kv_cache_dim)│
│             │   ├── index_buffer: NSA 索引                       │
│             │   └── 每 rank 独立，大小 = max_total_num_tokens     │
│             │                                                    │
│             ├── HiSparseTokenToKVPoolAllocator                   │
│             │   └── 管理 device token 的分配/释放                 │
│             │                                                    │
│             └── HiSparseCoordinator                              │
│                  │                                               │
│                  ├── mem_pool_device ──→ HiSparseNSATokenToKVPool│
│                  │   (引用，不额外分配)                           │
│                  │                                               │
│                  ├── mem_pool_host ──→ MLATokenToKVPoolHost       │
│                  │   │                 或 CXLMLATokenToKVPoolHost │
│                  │   │                                           │
│                  │   ├── kv_buffer: (layer_num, host_size, 1, d) │
│                  │   │   DRAM 模式: torch.empty + pin_memory     │
│                  │   │   CXL 模式: CXL mmap + frombuffer         │
│                  │   │                                           │
│                  │   └── data_ptrs: per-layer host 指针 (on GPU) │
│                  │       → JIT kernel / cudaMemcpy 的访问入口     │
│                  │                                               │
│                  ├── device_buffer_manager (per-request buffers) │
│                  └── LRU 管理器 (per-layer)                      │
└─────────────────────────────────────────────────────────────────┘

物理存储:
┌──────────────────────────┐ ┌───────────────────────────────────┐
│ GPU i VRAM               │ │ CXL DAX 设备 (/dev/dax0.0)       │
│ ┌──────────────────────┐ │ │ ┌─────────────────────────────┐   │
│ │ model weights (分片)  │ │ │ │ rank 0 分区 [0, N/T)       │   │
│ │ device KV buffer     │ │ │ │ rank 1 分区 [N/T, 2N/T)    │   │
│ │ activations          │ │ │ │ ...                         │   │
│ │ CUDA graphs          │ │ │ │ rank T-1 分区 [(T-1)N/T, N)│   │
│ └──────────────────────┘ │ │ └─────────────────────────────┘   │
└──────────────────────────┘ └───────────────────────────────────┘
                               T = tp_size（总 worker 数）
```

### 3.5 MLA KV Cache 的特性

DeepSeek-V3.2 使用 MLA（Multi-head Latent Attention），其 KV Cache 是一个 **latent vector**（`kv_lora_rank + qk_rope_head_dim`），**不按 attention head 分片**。

- 每个 worker 存储**完整维度**的 KV Cache（所有层、所有 token、全 `kv_cache_dim`）
- TP/DPA 只在 attention 的 Q/K/V projection 计算时分片，KV **存储**不分
- `host_size = device_pool.size × host_to_device_ratio`，其中 `device_pool.size` 是该 rank 根据 GPU 可用 VRAM profiling 得出的 `max_total_num_tokens`

**现有 DRAM HiSparse 行为**：TP=8 时，8 个 rank 各自分配独立的 pinned DRAM host pool，大小相同。这是当前框架的标准行为。CXL 方案保持完全一致，仅将底层内存从 DRAM 替换为 CXL。

### 3.6 CXL 在多进程架构下的设计

由于每个 worker 是独立进程：

1. **mmap 是进程级的**：每个进程独立调用 `cxl_init()` → `mmap(MAP_SHARED)` 映射同一个 DAX 设备。Linux 为每个进程创建独立的虚拟地址映射，指向相同的物理 CXL 内存
2. **cudaHostRegister 是 CUDA context 级的**：每个进程有自己的 CUDA context（绑定到各自的 GPU），各自注册 OK
3. **必须分区**：虽然映射了整个设备，但每个 rank 只能使用自己的分区，否则并发写入会产生数据竞争
4. **分区方案**：`CXLMemoryRegion.set_partition(tp_rank, tp_size)` 将整个 CXL 区域等分为 `tp_size` 份

```
CXL 物理设备 (64 GB, tp_size=8)
├── Rank 0 分区: [0, 8GB)      ← alloc_offset 从 0 开始
├── Rank 1 分区: [8GB, 16GB)   ← alloc_offset 从 8GB 开始
├── ...
└── Rank 7 分区: [56GB, 64GB)  ← alloc_offset 从 56GB 开始
```

每个分区内部通过 `allocate()` + `make_tensor()` 进行 bump allocation，分区边界 2MB 对齐。

### 3.7 架构设计的合理性

**为什么采用 "每个 worker 独立一份 host pool" 而非共享？**

1. **与现有 DRAM 行为一致**：现有 HiSparse 在 DRAM 上就是每 rank 一份独立 host pool，CXL 方案保持这一设计，最小化侵入性
2. **DPA 下天然正确**：不同 DP group 处理不同请求，KV 内容不同，必须独立存储
3. **避免 IPC 复杂性**：跨进程共享 tensor 需要进程间同步（如 POSIX shared memory + semaphore），引入额外延迟和复杂度
4. **MLA latent 很小**：每 token 仅 ~656 bytes，host pool 总内存相对可控
5. **CXL 分区零开销**：`set_partition()` 仅调整 bump allocator 的偏移量，无额外 mmap 或 CUDA 调用

**未来优化方向**：对于纯 TP 模式下同一 attn_tp group 内的 rank，理论上可以共享同一份 CXL KV 数据（因为内容相同），减少 CXL 容量需求。但这需要跨进程的同步机制，作为后续优化考虑。

---

## 4. CXL 直接 GPU 读取方案验证

### 4.1 Micro-benchmark 结果

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

### 4.2 性能对比（seq_len=32K, H20 GPU）


| top-k | DRAM (μs) | CXL 方案A (μs) | CXL 方案B (μs) | A/DRAM | B/DRAM    |
| ----- | --------- | ------------ | ------------ | ------ | --------- |
| 64    | 11.25     | 28.14        | ~11-19       | 2.50x  | ~1.0-1.7x |
| 512   | 17.19     | 60.97        | ~17-29       | 3.55x  | ~1.0-1.7x |
| 2048  | 43.76     | 154.48       | ~44-74       | 3.53x  | ~1.0-1.7x |


**结论**：方案 B（`ld.global.nc.b64`）将 CXL 开销从 ~3.5x 降低到 ~1.0-1.7x，接近硬件极限。这是因为：

1. **加载宽度**：8 bytes vs 1 byte → 同等数据量下 CXL 事务数减少 8 倍
2. **Non-cached 路径**：`ld.global.nc` 绕过 L2 cache，直接通过 PCIe/CXL DMA，减少 TLB 和 cache 层级开销
3. **零额外开销**：无 pybind 调用、无 offset 格式转换、无额外 `cudaMemcpyAsync`

### 4.3 对 HiSparse 的指导意义

HiSparse 的 JIT kernel 已经使用了 `ld.global.nc.b64`（`transfer_item_warp`），**与方案 B 完全一致**。因此：

- CXL mmap 经 `cudaHostRegister` 后，HiSparse swap-in 延迟即为方案 B 的水平
- **不需要任何 kernel 修改**，不需要使用 `cxl_utils` 库的 copy 函数
- 唯一需要做的是让 `MLATokenToKVPoolHost.kv_buffer` 的底层存储指向 CXL mmap 区域

---

## 5. CXL 内存池集成设计

### 5.1 设计原则

1. **指针透明性**：CXL mmap → `cudaHostRegister` → `torch.frombuffer` → `data_ptr()` → 与 DRAM pinned memory 语义完全一致
2. **不依赖 cxl_utils 传输层**：所有数据传输使用已有的 `cudaMemcpy`（staging/backup）和 JIT kernel（swap-in）
3. **最小侵入性**：仅替换 `MLATokenToKVPoolHost` 的内存分配后端，不修改 Coordinator、LRU、JIT kernel
4. **TP 感知分区**：所有 TP rank 共用同一 CXL 物理设备，按 `tp_rank` 等分为不相交的分区（参见第 3 章）
5. **CXL 初始化使用 `cxl_mem_ext`**：底层 mmap + cudaHostRegister 由 `sgl-kernel/cxl_utils` C++ 扩展完成（JIT 编译），确保跨平台可靠性

### 5.2 核心方案：mmap + cudaHostRegister + torch.frombuffer

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

### 5.3 整体架构变更

```
Per TP Rank Process (tp_rank=i)
┌───────────────────────────────────────────────────────────┐
│  HiSparseCoordinator (tp_rank=i, gpu_id=i)                │
│    │                                                       │
│    ├─ mem_pool_host: CXLMLATokenToKVPoolHost               │
│    │    └─ kv_buffer: torch.Tensor                         │
│    │         │  data_ptr → CXL mmap + offset[i]            │
│    │         │                                             │
│    │    ┌────┴─────────────────────┐                       │
│    │    │  内存后端选择 (配置驱动)    │                       │
│    │    ├─────────┬────────────────┤                       │
│    │    │ DRAM    │ CXL            │                       │
│    │    │ (现有)   │ (新增)         │                       │
│    │    │ torch   │ CXLMemoryRegion│                       │
│    │    │ .empty  │ .set_partition │                       │
│    │    │ + pin   │   (tp_rank, 8) │                       │
│    │    │         │ .make_tensor() │                       │
│    │    └─────────┴────────────────┘                       │
│    │                                                       │
│    │  swap-in:  JIT kernel ld.global.nc.b64 ──→ 同一路径    │
│    │  staging:  cudaMemcpy D2H ──→ 同一路径                 │
│    │  backup:   cudaMemcpy D2H ──→ 同一路径                 │
│    │                                                       │
│    └─ _cxl_region: CXLMemoryRegion (引用，防止 GC)          │
│         partition: [i*N/8, (i+1)*N/8)                      │
└───────────────────────────────────────────────────────────┘
              ↓ mmap(MAP_SHARED)
┌───────────────────────────────────────────────────────────┐
│  CXL DAX 设备 /dev/dax0.0 (物理共享)                       │
│  ┌────────┬────────┬────────┬───┬────────┐                 │
│  │ rank 0 │ rank 1 │ rank 2 │...│ rank 7 │                 │
│  └────────┴────────┴────────┴───┴────────┘                 │
└───────────────────────────────────────────────────────────┘
```

### 5.4 模块设计

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

**注意**：CXL 底层初始化使用 `sgl-kernel/cxl_utils` 的 C++ 扩展（`cxl_mem_ext`），支持 JIT 编译。数据传输不使用 `cxl_mem_ext` 的 copy 函数。

### 5.5 CXL 分区与 TP 多进程

由于 TP 架构下每个 rank 是独立进程（见第 3 章），CXL 设备的使用需要分区：

```
初始化流程 (每个 TP rank 进程内):

1. CXLMemoryRegion(config)    ← 创建管理对象
2. region.init()              ← cxl_init: mmap 整个 DAX 设备 + cudaHostRegister
3. region.set_partition(tp_rank, tp_size)  ← 设置本 rank 的分区边界
4. CXLMLATokenToKVPoolHost(cxl_region=region)
      └── init_kv_buffer() → region.make_tensor()  ← 在分区内 bump-allocate
```

分区示意（TP=8, 64GB CXL）：

```
  /dev/dax0.0 — 64 GB, 全量 mmap 到每个进程
  ┌────────┬────────┬────────┬───┬────────┐
  │ Rank 0 │ Rank 1 │ Rank 2 │...│ Rank 7 │  ← 每个 8 GB
  │  8 GB  │  8 GB  │  8 GB  │   │  8 GB  │
  └────────┴────────┴────────┴───┴────────┘
  0        8GB      16GB             56GB    64GB
```

**关键属性**：

- `_alloc_offset`：当前分区内的分配游标，初始值 = `tp_rank * per_rank`
- `_partition_end`：分区结束地址 = `(tp_rank + 1) * per_rank`
- `allocate()` 在 `[_alloc_offset, _partition_end)` 范围内进行 bump allocation
- 分区边界 2MB 对齐，符合 DAX 大页要求
- 进程间不共享 Python 对象，安全无竞争

### 5.6 CXL 容量约束与 KV Pool 大小自动适配

SGLang 通过 GPU VRAM profiling 决定 `max_total_num_tokens`（device pool 大小），进而 host pool 大小 = `max_total_num_tokens × host_to_device_ratio`。如果 CXL 容量不足以容纳 host pool，系统会**自动缩小 `max_total_num_tokens`** 以适配。

**约束公式**：

```
host_pool_bytes = max_total_num_tokens × host_to_device_ratio × bytes_per_token
cxl_partition   = cxl_map_bytes / tp_size  (2MB 对齐)

约束: host_pool_bytes ≤ cxl_partition
=> max_total_num_tokens ≤ cxl_partition / (host_to_device_ratio × bytes_per_token)
```

其中 `bytes_per_token = kv_cache_dim × dtype_size × layer_num`（MLA host pool 每 token 大小）。

**实现位置**：`ModelRunner._resolve_token_capacity()` → `_apply_cxl_capacity_cap()`

```
profiled_tokens (GPU VRAM)
     │
     ▼  min(profiled_tokens, user_limit)
     │
     ▼  min(capacity, cxl_partition / (ratio × bytes_per_token))  ← CXL cap
     │
     ▼  page_size 对齐
     │
     ▼  PP sync (all_reduce MIN)
     │
     = max_total_num_tokens
```

**示例**：TP=8, CXL 64GB, host_to_device_ratio=2, bytes_per_token=656×61=40016

```
cxl_partition  = 64GB / 8 = 8GB
max_capacity   = 8GB / (2 × 40016) ≈ 107,000 tokens
```

如果 GPU profiling 得到 200,000 tokens，会自动缩小到 107,000。日志中会输出 CXL capacity cap 信息。

### 5.7 数据流变更

#### 5.7.1 Prefill Staging（GPU → CXL）

```
现有:  transfer_kv_all_layer_mla(dst=host.data_ptrs) → cudaMemcpy D2H → pinned DRAM
CXL:   transfer_kv_all_layer_mla(dst=host.data_ptrs) → cudaMemcpy D2H → CXL mmap
```

`data_ptrs` 指向 CXL mmap 区域（已 `cudaHostRegister`）。`cudaMemcpy D2H` 的目标是注册过的 host memory，DMA 引擎直接写入 CXL 物理地址。**无需修改传输代码，无需 clflush**（DMA 不经过 CPU cache）。

#### 5.7.2 Decode Swap-in（CXL → GPU）

```
现有:  JIT kernel ld.global.nc.b64 from pinned DRAM → st.global.cg.b64 to VRAM
CXL:   JIT kernel ld.global.nc.b64 from CXL mmap   → st.global.cg.b64 to VRAM
```

JIT kernel 的 `host_cache_k` 指针来自 `data_ptrs`，指向 CXL mmap。`ld.global.nc.b64` 通过 PCIe/CXL 链路直接加载，延迟为 DRAM 的 1.0~1.7x（已验证）。**JIT kernel 零修改。**

#### 5.7.3 Decode Backup（GPU → CXL）

```
现有:  transfer_kv_all_layer_mla → cudaMemcpy D2H → pinned DRAM
CXL:   transfer_kv_all_layer_mla → cudaMemcpy D2H → CXL mmap
```

同 staging 路径。**无需修改。**

---

## 6. 详细代码设计

### 6.1 CXLMemoryRegion 管理类

**文件**: `python/sglang/srt/mem_cache/cxl_memory_pool.py`

底层使用 `cxl_mem_ext` C++ 扩展（JIT 编译自 `sgl-kernel/cxl_utils`）执行 mmap + cudaHostRegister，Python 层负责 TP 分区和 tensor 创建：

```python
class CXLMemoryRegion:
    """Manages a CXL DAX device mmap region with CUDA registration + TP partitioning.

    Lifecycle:
        1. cxl_ext.cxl_init() — mmap + cudaHostRegister (C++ 实现)
        2. set_partition(tp_rank, tp_size) — 设置本 rank 的分区边界
        3. make_tensor() — bump-allocate + torch.frombuffer (zero-copy)
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
        self._cxl_ext = _load_cxl_ext()
        aligned_size = align_to_2mb(self.config.map_bytes)
        ok = self._cxl_ext.cxl_init(
            dev_path=self.config.dev_path,
            map_bytes=aligned_size,
            register_cuda=True,
            gpu_id=self.config.gpu_id,
        )
        if not ok:
            raise RuntimeError(...)
        self._base_ptr = self._cxl_ext.get_cxl_base_ptr()
        self._size = aligned_size
        self._partition_end = aligned_size    # 默认：整个区域
        self._partition_size = aligned_size

    def set_partition(self, tp_rank: int, tp_size: int) -> None:
        """将 CXL 区域按 tp_size 等分，本 rank 仅使用 1/tp_size."""
        if tp_size <= 1:
            return
        per_rank = (self._size // tp_size // HUGE_PAGE_2MB) * HUGE_PAGE_2MB
        start = tp_rank * per_rank
        self._alloc_offset = start
        self._partition_end = start + per_rank
        self._partition_size = per_rank

    def allocate(self, nbytes: int, alignment: int = 64) -> int:
        """Bump-allocate from this rank's partition."""
        offset = (self._alloc_offset + alignment - 1) & ~(alignment - 1)
        if offset + nbytes > self._partition_end:
            raise RuntimeError("CXL partition exhausted")
        self._alloc_offset = offset + nbytes
        return offset

    def make_tensor(self, dims: tuple, dtype: torch.dtype) -> torch.Tensor:
        """在 CXL 分区内分配 tensor (zero-copy)."""
        nbytes = prod(dims) * dtype.itemsize
        offset = self.allocate(nbytes)
        ptr = self._base_ptr + offset
        c_array = (ctypes.c_byte * nbytes).from_address(ptr)
        tensor = torch.frombuffer(c_array, dtype=dtype).reshape(dims)
        tensor.zero_()
        return tensor

    def close(self) -> None:
        if self._initialized:
            self._cxl_ext.cxl_close()
```

**`_load_cxl_ext()`**：先尝试 `import cxl_mem_ext`，失败则自动从 `sgl-kernel/cxl_utils` JIT 编译（使用 `torch.utils.cpp_extension.load()`）。

### 6.2 CXL Tensor 分配函数

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

### 6.3 CXLMLATokenToKVPoolHost 类设计

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

### 6.4 HiSparseCoordinator 适配

**修改文件**: `python/sglang/srt/managers/hisparse_coordinator.py`

Coordinator 接收 `tp_rank`、`tp_size`、`gpu_id`，传递给 CXL 初始化以实现分区：

```python
class HiSparseCoordinator:
    def __init__(
        self,
        # ... existing params ...
        host_to_device_ratio: int = 2,
        cxl_config: Optional[dict] = None,
        tp_rank: int = 0,       # 新增
        tp_size: int = 1,       # 新增
        gpu_id: int = 0,        # 新增：每个 rank 绑定的 GPU
    ):
        # ... existing init ...
        self.mem_pool_device = self.token_to_kv_pool_allocator.get_kvcache()

        self._cxl_region = None
        if cxl_config is not None and cxl_config.get("enabled", False):
            self.mem_pool_host = self._init_cxl_host_pool(
                cxl_config, host_to_device_ratio, tp_rank, tp_size, gpu_id
            )
        else:
            self.mem_pool_host = MLATokenToKVPoolHost(...)

    def _init_cxl_host_pool(self, cxl_config, host_to_device_ratio,
                             tp_rank, tp_size, gpu_id):
        from sglang.srt.mem_cache.cxl_memory_pool import (
            CXLConfig, CXLMemoryRegion, CXLMLATokenToKVPoolHost,
        )
        region = CXLMemoryRegion(CXLConfig(
            dev_path=cxl_config.get("dev_path", "/dev/dax0.0"),
            map_bytes=cxl_config.get("map_bytes", 64 * 1024**3),
            gpu_id=gpu_id,  # 使用当前 rank 的 GPU
        ))
        region.init()               # mmap 整个 DAX 设备 + cudaHostRegister
        region.set_partition(tp_rank, tp_size)  # 设置本 rank 的分区
        self._cxl_region = region
        return CXLMLATokenToKVPoolHost(
            device_pool=self.mem_pool_device,
            host_to_device_ratio=host_to_device_ratio,
            host_size=0, page_size=1, layout="layer_first",
            cxl_region=region,
            override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
        )
```

**ModelRunner 调用侧**（`model_runner.py`）：

```python
self.hisparse_coordinator = HiSparseCoordinator(
    ...,
    cxl_config=hisparse_cfg.cxl_config,
    tp_rank=self.tp_rank,     # 当前进程的 TP rank
    tp_size=self.tp_size,     # TP 总数
    gpu_id=self.gpu_id,       # 当前进程绑定的 GPU
)
```

### 6.5 ServerArgs 与配置扩展

```bash
# 在 hisparse_config JSON 中增加 CXL 字段
# 注意：不需要指定 gpu_id，每个 TP rank 自动使用自己绑定的 GPU
# map_bytes 是整个 CXL 设备大小，会按 tp_size 自动等分
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
            "map_bytes": 68719476736
        }
    }'
```

**配置说明**：

| 字段 | 说明 |
| --- | --- |
| `cxl.enabled` | 启用 CXL 后端替换 DRAM |
| `cxl.dev_path` | DAX 设备路径，所有 TP rank 共用同一设备 |
| `cxl.map_bytes` | CXL 设备总大小（所有 rank 共享，内部按 `tp_size` 自动等分） |
| ~~`cxl.gpu_id`~~ | **已移除**：由 `ModelRunner.gpu_id` 自动提供 |

---

## 7. 为什么不使用 cxl_utils 库的传输函数


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

## 8. 多 CXL 设备 Interleave 设计方案

### 8.1 目标配置与 KV Cache 特性

目标部署配置：

```bash
--tp 8 --dp-size 8 --enable-dp-attention
```

在此配置下，SGLang 的 `compute_dp_attention_world_info()` 计算得：

```
attn_tp_size = tp_size / dp_size = 8 / 8 = 1
attn_dp_rank = tp_rank   (每个 rank 就是一个独立的 DP group)
attn_tp_rank = 0          (无 attention TP 分片)
```

即 **每个 rank 独立处理不同的请求集，KV Cache 内容互不相同，零冗余**：

```
DataParallelController (负载均衡，8 路)
  ├── rank 0 (dp=0): 请求集 A → 独立 KV Cache
  ├── rank 1 (dp=1): 请求集 B → 独立 KV Cache
  ├── ...
  └── rank 7 (dp=7): 请求集 H → 独立 KV Cache
```

这是 MLA KV Cache 存储效率最优的并行方式——8 个 rank 各自存储完全不同的 KV 数据，CXL 容量利用率 100%，无副本浪费。

### 8.2 单设备瓶颈与多设备机会

单 CXL 设备下，8 个 GPU 对**不同数据**的 swap-in 读事务汇聚到同一 CXL 控制器，共享带宽（典型 32~64 GB/s）。多设备 interleave 可将并发读取分散到多条独立 PCIe/CXL 链路，聚合带宽线性增长。

### 8.3 Interleave 策略

#### 8.3.1 按层交织 — 不可行

`layer_id % num_devices` → 同一层所有 8 GPU 仍打同一设备。HiSparse 逐层串行，对并发带宽无改善。

#### 8.3.2 按 Rank 交织 — 推荐方案

由于 `dp_size = tp_size = 8`（即 `attn_dp_rank = tp_rank`），按 rank 交织天然等同于按 DP group 交织：

```
tp_rank % num_devices → 选择 CXL 设备
```

以 2 个 CXL 设备为例：

```
swap_in(layer_id=k), 8 GPU 并发:

  rank 0 ──┐                    rank 1 ──┐
  rank 2 ──┤──→ CXL device 0   rank 3 ──┤──→ CXL device 1
  rank 4 ──┤                    rank 5 ──┤
  rank 6 ──┘                    rank 7 ──┘
  4 GPU / 设备                   4 GPU / 设备
  (各自读不同数据, 无地址级冲突)
```

**核心优点**：

1. 每个 rank 的 KV 数据互不相同，天然无跨设备数据依赖
2. `DataParallelController` 保证各 rank 负载均衡 → 设备间负载均衡
3. 实现极简——仅需 `tp_rank % num_devices` 一行映射
4. N 个设备提供 N 倍聚合带宽

**带宽收益**（单设备带宽上限 B）：

| CXL 设备数 | 每设备并发 GPU | 每 GPU 有效带宽 | 提升 |
| --- | --- | --- | --- |
| 1 | 8 | B/8 | 基线 |
| 2 | 4 | B/4 | **2x** |
| 4 | 2 | B/2 | **4x** |
| 8 | 1 | B | **8x** |

### 8.4 实现接口

多设备 interleave 只影响初始化阶段的设备选择和分区，运行时路径（JIT kernel、`data_ptrs`、`CXLMLATokenToKVPoolHost`）完全不变。

**接入点**：`HiSparseCoordinator._init_cxl_host_pool()`

现有单设备流程：

```
CXLMemoryRegion(config) → init() → set_partition(tp_rank, tp_size) → make_tensor()
```

多设备流程变更：

```
对每个 dev_path 创建 CXLMemoryRegion → init_all()
tp_rank % num_devices → 选出本 rank 的 region
在该 region 上 set_partition(local_rank, local_size) → make_tensor()
```

其中 `local_rank` / `local_size` 是本 rank 在所选设备上的局部排名和总数（如 2 设备 8 rank，设备 0 上有 rank 0/2/4/6，local_size=4）。

**配置扩展**：`hisparse-config` 的 `cxl` 字段增加 `dev_paths`（列表）和 `map_bytes_per_device`。单设备时保持 `dev_path` + `map_bytes` 向后兼容。

```json
"cxl": {
    "enabled": true,
    "dev_paths": ["/dev/dax0.0", "/dev/dax1.0"],
    "map_bytes_per_device": 34359738368
}
```

### 8.5 容量规划

```
per_device_ranks = tp_size / num_devices
per_rank_kv = max_total_num_tokens × host_to_device_ratio × 656 × 61 bytes
per_device_capacity_needed = per_device_ranks × per_rank_kv
```

示例：8 rank, 2 CXL 设备（各 32GB），max_tokens=100K, ratio=2：

```
per_rank_kv = 100,000 × 2 × 656 × 61 ≈ 7.6 GB
per_device = 4 ranks × 7.6 GB = 30.4 GB < 32 GB ✓
```

### 8.6 其他可选优化

以下方案暂不实现，记录为后续方向。

- **TLB 优化**：DAX 设备天然 2MB 大页；可通过 `daxctl` 启用 1GB 大页进一步减少 TLB miss。初始化后顺序遍历 CXL 区域可预热 TLB。
- **按层交织 + 预取流水线**：利用层间 top-k 的高相似性，提前一层发起 prefetch 到不同 CXL 设备上，隐藏 swap-in 延迟。需修改 HiSparse 核心调度逻辑，预估 TBT 可降低 40~50%。
- **写入一致性**：当前所有 CXL 读写均由 GPU DMA / kernel 执行，绕过 CPU cache，无需 `clflush`。
- **NUMA 绑定**：将 GPU 进程绑定到 CXL 设备所在 NUMA node，减少跨 socket PCIe 路由开销。依赖具体硬件拓扑。


---

## 9. 分阶段实施计划

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

## 10. 风险与缓解措施


| 风险                               | 影响                                  | 缓解措施                                      |
| -------------------------------- | ----------------------------------- | ----------------------------------------- |
| CXL 延迟高于 DRAM                    | swap-in 延迟增加 ~1.0-1.7x              | 已验证可接受；LRU buffer 利用层间相似性减少 miss          |
| `cudaHostRegister` 对 CXL mmap 失败 | GPU 无法 DMA                          | 检查 CUDA driver 版本和 PCIe topology；降级为 DRAM |
| `torch.frombuffer` 生命周期          | tensor 释放后 mmap 仍有效                 | CXLMemoryRegion 绑定到 Coordinator 生命周期      |
| DAX 设备 2MB 对齐要求                  | mmap EINVAL                         | `CXLMemoryRegion.init()` 自动向上对齐           |
| TP 多进程 CXL 竞争                    | 8 个 GPU 并发读写同一 CXL 设备竞争带宽          | `set_partition()` 按 rank 等分；考虑多 DAX 设备 interleave |
| CXL 分区不足                         | 每 rank 分区小于 KV cache 需求             | `_apply_cxl_capacity_cap()` 自动缩小 `max_total_num_tokens` |
| HostKVCache 内存检查                 | `psutil.virtual_memory()` 可能不识别 CXL | `_skip_host_mem_check = True` 已实现         |
| `cxl_mem_ext` 全局单例               | C++ 库内 `g_cxl` 是全局变量               | 每 TP rank 是独立进程，各有自己的全局变量副本，无冲突 |


---

## 附录 A：关键文件索引


| 文件                                                    | 角色                            | CXL 变更                     |
| ----------------------------------------------------- | ----------------------------- | -------------------------- |
| `python/sglang/srt/mem_cache/cxl_memory_pool.py`      | **新增** CXL region + host pool | CXLMemoryRegion (含 TP 分区) + CXLMLATokenToKVPoolHost |
| `python/sglang/srt/managers/hisparse_coordinator.py`  | HiSparse 核心协调器                | 增加 `cxl_config` + `tp_rank/tp_size/gpu_id` |
| `python/sglang/srt/mem_cache/memory_pool_host.py`     | Host KV Pool 基类               | 增加 `_skip_host_mem_check` |
| `python/sglang/srt/mem_cache/hisparse_memory_pool.py` | Device Pool + Allocator       | 不变                         |
| `python/sglang/jit_kernel/csrc/hisparse.cuh`          | JIT swap-in kernel            | 不变                         |
| `python/sglang/jit_kernel/hisparse.py`                | JIT Python 入口                 | 不变                         |
| `python/sglang/srt/layers/attention/nsa_backend.py`   | NSA attention 后端              | 不变                         |
| `python/sglang/srt/model_executor/model_runner.py`    | 模型初始化                         | 传递 CXL 配置 + TP 信息         |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | KV Pool 大小计算       | CXL 容量约束 `_apply_cxl_capacity_cap()` |
| `python/sglang/srt/entrypoints/engine.py`             | Engine 启动入口                   | worker 启动前预编译 `cxl_mem_ext` |
| `python/sglang/srt/mem_cache/sparsity/factory.py`     | HiSparse 配置解析                 | 解析 `cxl` JSON 字段          |


## 附录 B：代码量估算


| 文件                        | 新增/修改行数    | 说明                                        |
| ------------------------- | ---------- | ----------------------------------------- |
| `cxl_memory_pool.py`      | ~310 行     | CXLMemoryRegion (含 TP 分区) + CXLMLATokenToKVPoolHost + JIT loader |
| `hisparse_coordinator.py` | ~40 行      | CXL 分支 + region 初始化 + TP 分区               |
| `memory_pool_host.py`     | ~5 行       | psutil 检查跳过                                |
| `model_runner.py`         | ~3 行       | 传递 tp_rank/tp_size/gpu_id                 |
| `factory.py`              | ~5 行       | CXL config 解析                             |
| **合计**                    | **~360 行** | 需 JIT 编译 cxl_mem_ext（自动）                  |


## 附录 C：配置示例

```bash
# 单 CXL 设备 64GB, TP=8 → 每 rank 自动分配 8GB
# gpu_id 不需要在 JSON 中指定，各 rank 自动使用自己的 GPU
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
            "map_bytes": 68719476736
        }
    }' \
    --disable-radix-cache
```

