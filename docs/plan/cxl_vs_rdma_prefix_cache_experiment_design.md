# CXL vs RDMA 内存池对比实验 —— 最小代码开发设计

**目标：以最少的额外代码开发，在单机上完成 CXL vs RDMA 分布式内存池的端到端对比实验。**

---

## 1. 核心论点与验证策略

### 1.1 要验证的论点

> 在 sparse attention（DeepSeek-V3.2 NSA）场景下，CXL 按需读取 top-k KV Cache 显著优于 RDMA 全量预取。
> 具体体现在：decode 吞吐更高、尾延迟更低、本地 DRAM 占用更少、可并发请求数更多。

### 1.2 验证策略：单机 + KV Cache 内存池，无 PD 分离

**所有实验都在单机上完成**，不做 PD 分离。理由：

1. 核心论点在于 **decode 阶段从不同介质读取 KV 的效率差异**，与 KV Cache 是本地 prefill 产生的还是远端传来的无关
2. PD 分离只改变了 KV Cache 的"来源"，不改变 decode 阶段的读取路径——我们的对比变量是"读取介质"而非"KV 来源"
3. 单机实验消除了网络拓扑、跨机同步等干扰因素，让对比更纯粹
4. RDMA 方案使用**本机 NIC loopback** 真实执行全量预取，产生真实硬件延迟（是跨机延迟的下界）

### 1.3 利用已有代码


| 已有组件                | 来源                                  | 本实验复用方式                  |
| ------------------- | ----------------------------------- | ------------------------ |
| HiSparse 框架         | PR#20343 已合并                        | 完整复用 sparse attention 路径 |
| CXL host pool       | `cxl_memory_pool.py` 已实现            | 直接作为方案 C 的 host 后端       |
| DRAM host pool      | `memory_pool_host.py`               | 直接作为方案 A 的基线             |
| RDMA loopback bench | `test/cxl_utils/sparse_kv_bench.py` | 复用 RDMA loopback 基础设施    |


---

## 2. 三种方案的单机架构

```
同一台机器 (8×H20 + CXL + RDMA NIC)
┌─────────────────────────────────────────────────────────────────┐
│                  SGLang + HiSparse 单实例                        │
│                  DeepSeek-V3.2 (TP=8)                            │
│                                                                 │
│  NSA Indexer → top-k indices → swap_in_selected_pages()         │
│                        │                                        │
│         ┌──────────────┼──────────────┐                         │
│         ▼              ▼              ▼                          │
│   ┌──────────┐  ┌──────────────┐  ┌──────────┐                  │
│   │ 方案 A    │  │ 方案 B        │  │ 方案 C    │                  │
│   │ DRAM     │  │ RDMA 内存池    │  │ CXL      │                  │
│   │          │  │ (loopback)   │  │          │                  │
│   │ pinned   │  │              │  │ CXL mmap │                  │
│   │ DRAM     │  │ ┌──────────┐ │  │          │                  │
│   │ (100%    │  │ │host pool │ │  │ 全量 KV   │                  │
│   │  本地)    │  │ │(统一视图) │ │  │ 在 CXL   │                  │
│   │          │  │ ├──────────┤ │  │          │                  │
│   │          │  │ │remote buf│ │  │          │                  │
│   │          │  │ │(RDMA MR) │ │  │          │                  │
│   │          │  │ │ ↕ NIC    │ │  │          │                  │
│   │          │  │ │ loopback │ │  │          │                  │
│   │          │  │ ├──────────┤ │  │          │                  │
│   │          │  │ │staging   │ │  │          │                  │
│   │          │  │ │buffer    │ │  │          │                  │
│   │          │  │ └──────────┘ │  │          │                  │
│   └──────────┘  └──────────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### 方案 B 的内存布局详解

方案 B 模拟"分布式内存池中，X% 的请求 KV 在本地、其余在远端"的场景。`local_ratio` 控制的是**请求级别**的本地命中率——同一请求的全部 KV 要么全在本地，要么全在远端。

#### 远端与本地的存储布局差异

真实分布式 KV 内存池（如 Mooncake、NIXL）中，KV 数据在远端通常以 **page-first（token-major）** 方式组织——同一个 token 的所有层 KV 紧邻存放，便于按 page 粒度做连续大块传输。而 HiSparse 本地 host pool 使用 **layer-first（layer-major）** 布局——同一层的所有 token slot 连续存放，便于 swap-in kernel 按层做 DMA。

因此方案 B 在模拟中显式区分这两种布局：

- **`remote_buffer`（模拟远端）**：**page-first** 布局，形态 `[num_tokens, layer_num, 1, kv_cache_dim]`。同一 token 的全部层 KV 在内存中连续，与真实远端内存池一致。
- **`kv_buffer`（本地 host pool）**：**layer-first** 布局，形态 `[layer_num, num_tokens, 1, kv_cache_dim]`。HiSparse swap-in / backup 等路径不变。

#### 传输模式：bulk RDMA + 布局转换

传输流程为：**remote_buffer（page-first）整块 bulk RDMA 到 landing staging → landing staging 布局转换写入 kv_buffer（layer-first）**。这比旧方案更贴近生产环境：

1. **RDMA 链路上** 搬的是 **远端原生布局的连续大块**，不需要在远端做任何 gather/repack。
2. **本地落盘时** 做 page-first → layer-first 的转换（transpose + scatter），这正是真实 decode 节点收到 KV 后需要做的工作——**该开销是真实系统中不可避免的**。
3. staging buffer 只需 **一块** landing staging（不再需要 gather staging），大小 = 单请求最大 KV 字节数。

```
本机 DRAM 布局（方案 B）:

┌─────────────────────────────┐
│  模型权重 + 其他                │
├─────────────────────────────┤
│  Host Pool (pinned DRAM)     │  ← HiSparse 统一视图，layer-first
│  kv_buffer                   │    形态: [layer_num, num_tokens, 1, kv_cache_dim]
│  (staging DMA 写入 + decode  │    所有请求的 KV（离散 slot 存储）
│   swap-in 读取)              │    local 请求直接可用；remote 请求预取后可用
├─────────────────────────────┤
│  Remote Buffer (RDMA MR)     │  ← page-first 布局（模拟远端节点存储）
│  remote_buffer               │    形态: [num_tokens, layer_num, 1, kv_cache_dim]
│  (ibv_reg_mr, 模拟远端节点)    │    同一 token 的全部层 KV 连续存放
├─────────────────────────────┤
│  Landing Staging (pinned)    │  ← 小型连续缓冲区，page-first 布局
│  landing_staging             │    bulk RDMA READ 的落点
│  (RDMA READ 目标)             │    大小 = max_seq_len × layer_num × kv_dim × dtype
│                              │    转换后写入 kv_buffer
└─────────────────────────────┘

请求进入 Decode 前（按请求粒度）:
  local  请求 → 无操作，KV 已在 host pool
  remote 请求 → 1. bulk RDMA READ: remote_buffer[token_slots]（page-first, 连续）
                   → NIC loopback → landing_staging
                2. transpose + scatter: landing_staging（page-first）
                   → kv_buffer（layer-first）对应 slot 位置
                （一次性，之后与 local 请求行为一致）

Decode 每步（local/remote 一致）:
  HiSparse swap-in: host pool → GPU (仅 top-k miss)
```

#### 为什么 page-first 远端 + layer-first 本地是合理的模拟

1. **RDMA 传输效率**：远端 page-first 布局意味着同一请求的所有 token 可以按地址连续（或大块连续）做 bulk RDMA READ，NIC 看到少量大 WR，充分利用带宽。不需要在远端做 gather——远端数据本身就是连续的。
2. **本地转换开销真实**：收到 page-first 数据后转成 layer-first 写入 host pool，这个 transpose/scatter 是真实系统中不可避免的代价。旧方案远端也用 layer-first + gather/scatter 两遍，反而**多做了一遍 CPU 拷贝**（先从 layer-first remote 逐层 gather 到 staging），不如 page-first 直传更贴近现实。
3. **GPU swap-in 不受影响**：host pool 始终是 layer-first，HiSparse 按层 swap-in 的 kernel 和索引逻辑完全不变。
4. **staging 减半**：不再需要 gather staging（远端已是连续的），只需 landing staging 一块。

---

## 3. 分阶段开发计划

HiSparse + prefix-caching 兼容是整个方案中最关键、风险最高的部分，**优先开发并单独验证**。CXL/RDMA 后端在此基础上叠加。

### 3.1 阶段划分

```
Phase 1: DRAM HiSparse + Prefix-Caching（最高优先级）         ← Week 1
  ├─ 去掉 radix cache 互斥断言 (server_args.py)
  ├─ prefix_host_cache + 引用计数 (hisparse_coordinator.py)
  ├─ admit_request_into_staging 改造
  ├─ alloc_device_buffer 适配
  ├─ request_finished 改造
  ├─ hisparse_memory_pool.py 接口适配
  └─ 端到端验证：DRAM host pool + radix cache + 两轮请求
     ↓
Phase 2: CXL 后端集成                                       ← Week 2 前半
  ├─ CXL host pool 已实现，无需新代码
  └─ 验证 CXL host pool + prefix cache 端到端正确性
     ↓
Phase 3: RDMA 后端 + Benchmark 脚本                         ← Week 2-3
  ├─ rdma_memory_pool.py（新增，page-first 远端 → bulk RDMA → transpose/scatter 模式）
  ├─ rdma_scatter_read.c/.h/_py.cpp bulk read + 多实例扩展
  ├─ coordinator 后端选择逻辑
  └─ e2e_bench.py 两轮实验自动化
     ↓
Phase 4: 完整实验矩阵 + 数据分析                                ← Week 4
```

### 3.2 Phase 1：DRAM HiSparse + Prefix-Caching（详细）

**目标**：在 DRAM host pool 上验证 HiSparse 与 radix cache 正确协作——prefix-hit 请求跳过 prefix prefill，staging 仅备份 extend 部分，decode 从 host pool 正确 swap-in。


| 文件                        | 改动量        | 说明                                                                                                                                                                          |
| ------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `server_args.py`          | ~2 行       | 去掉 `assert self.disable_radix_cache`，改为 warning                                                                                                                             |
| `hisparse_coordinator.py` | ~135 行     | `PrefixHostCacheEntry` 数据结构 + `prefix_host_cache`/`req_prefix_info` + `admit_request_into_staging` prefix-hit/冷启动双路径 + `request_finished` 引用计数释放 + `alloc_device_buffer` 适配 |
| `hisparse_memory_pool.py` | ~10 行      | `alloc_device_buffer` 接口适配（prefix-hit 只传 extend indices）                                                                                                                    |
| **Phase 1 合计**            | **~150 行** |                                                                                                                                                                             |


**验证方式**：

```bash
# 启动 DRAM 模式，不加 --disable-radix-cache
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
    --enable-hisparse \
    --hisparse-config '{"top_k":2048,"device_buffer_size":4096}'

# 发送两个共享 prefix 的请求，检查：
# 1. 第二个请求的 extend_input_len << 第一个（radix cache 命中）
# 2. 两个请求都正确生成 output（decode 路径正确）
# 3. host pool 内存占用低于无 prefix cache 时（prefix 共享生效）
```

### 3.3 Phase 2：CXL 后端集成

CXL host pool（`cxl_memory_pool.py`）已实现且与 HiSparse 集成完毕。Phase 1 的 prefix cache 改动在 coordinator 层，对 host pool 后端透明。**无需额外代码。**

### 3.4 Phase 3：RDMA 后端 + Benchmark


| 文件                              | 改动量        | 说明                                       |
| ------------------------------- | ---------- | ---------------------------------------- |
| `rdma_memory_pool.py`（新增）       | ~300 行     | RDMA host pool（page-first 远端 → bulk RDMA → transpose/scatter + local/remote 分层） |
| `hisparse_coordinator.py`       | ~40 行      | 后端选择 + decode prefetch + cleanup           |
| `factory.py` + `SparseConfig`   | ~15 行      | rdma_pool JSON 解析 + 互斥校验                   |
| `model_runner.py`               | ~2 行       | 传递 rdma_config                            |
| `rdma_scatter_read.c/.h/_py.cpp` | ~60 行      | bulk read + 多实例 RDMAContext 扩展              |
| `e2e_bench.py`（新增）              | ~350 行     | 两轮实验自动化，ShareGPT 多样化请求，TTFT/TBT/TPOT      |
| `run_all_experiments.sh`（新增）    | ~150 行     | 四方案 × 多 prefix 实验矩阵自动化                    |
| **Phase 3 合计**                  | **~920 行** |                                          |


### 3.5 代码总量


| Phase       | 代码量        | 累计      | 关键产出                                            |
| ----------- | ---------- | ------- | ----------------------------------------------- |
| **Phase 1** | **~150 行** | ~150 行  | HiSparse + radix cache 兼容，prefix-hit 跳过 prefill |
| Phase 2     | ~0 行       | ~150 行  | CXL 端到端验证                                       |
| Phase 3     | ~920 行     | ~1070 行 | RDMA 后端（page-first 远端 → bulk → transpose/scatter）+ 多样化 e2e benchmark + 实验自动化 |
| Phase 4     | ~0 行       | ~1070 行 | 完整实验数据                                          |


---

## 4. HiSparse + Prefix-Caching 设计方案

### 4.1 设计目标

让 prefix-hit 请求**跳过 prefix 的 prefill 计算**，使 Round 2 中 GPU 几乎全部用于 decode，充分暴露 RDMA 带宽瓶颈与 CXL 按需读取的优势差异。

利用 SGLang **已有的 radix cache** 实现 prefix 跳过（`match_prefix` → `prefix_indices` → `extend_input_len = N-P`），在 HiSparse coordinator 层改造 host pool 管理实现 prefix KV 的持久化和复用。

### 4.2 关键组件与内存层次交互

```
┌────────────────────────── SGLang Scheduler ─────────────────────────────┐
│                                                                         │
│  waiting_queue → init_next_round_input() → prefill batch → decode batch │
│       ↕                    ↕                     ↕              ↕        │
│  ┌──────────┐   ┌──────────────────┐    ┌────────────┐  ┌───────────┐   │
│  │RadixCache │   │ alloc_for_extend │    │model.forward│  │decode loop│   │
│  │           │   │ (allocator)      │    │(attention)  │  │(per-step) │   │
│  └─────┬─────┘  └────────┬─────────┘    └──────┬──────┘  └─────┬─────┘   │
└────────┼─────────────────┼──────────────────────┼───────────────┼────────┘
         │                 │                      │               │
         │          ┌──────▼──────────────────────▼──────┐        │
         │          │     Device Memory (GPU HBM)        │        │
         │          │                                    │        │
         │          │  ┌──────────────────────────────┐  │        │
         │          │  │ Logical KV Pool (kv_buffer)  │  │        │
         │          │  │ • req_to_token_pool 索引      │  │        │
         │          │  │ • radix cache 管理引用计数      │  │        │
         │          │  │ • prefill 写入 / prefix 复用    │  │        │
         │          │  ├──────────────────────────────┤  │        │
         │          │  │ HiSparse Device Buffer       │  │        │
         │          │  │ • 每请求私有, 固定大小           │  │        │
         │          │  │ • decode 存放 swap-in 的 KV    │  │        │
         │          │  └──────────────────────────────┘  │        │
         │          │  HiSparseNSATokenToKVPool           │        │
         │          │  HiSparseTokenToKVPoolAllocator     │        │
         │          └──────────────────┬─────────────────┘        │
         │                             │                          │
         │                  ┌──────────▼────────────┐             │
         │                  │  HiSparseCoordinator  │◄────────────┘
         │                  │  (核心改造对象)         │
         │                  │                       │
         │                  │  admit_request_       │
         │                  │    into_staging()     │
         │                  │  alloc_device_buffer()│
         │                  │  swap_in_selected_    │
         │                  │    pages()            │
         │                  │  request_finished()   │
         │                  │                       │
         │                  │ ★ 新增:                │
         │                  │  prefix_host_cache    │
         │                  │  req_prefix_info      │
         │                  └──────────┬────────────┘
         │                             │ staging DMA / swap-in
         │                  ┌──────────▼────────────┐
         │                  │   Host Memory Pool     │
         │                  │  (MLATokenToKVPoolHost) │
         │                  │                        │
         │                  │  方案 A: pinned DRAM    │
         │                  │  方案 B: RDMA 分层池     │
         │                  │  方案 C: CXL mmap       │
         │                  └────────────────────────┘
         │
   ┌─────▼───────────────────────────────────┐
   │  Radix Tree                              │
   │  token_ids → logical KV indices 映射     │
   │  • insert: cache_unfinished_req 时       │
   │  • match:  init_next_round_input 时      │
   │  • evict:  内存不足时 LRU 淘汰             │
   └──────────────────────────────────────────┘
```

**三层内存的角色**：


| 层级                         | 位置            | 管理者                                                        | 索引类型                  | 内容                        | 生命周期                         |
| -------------------------- | ------------- | ---------------------------------------------------------- | --------------------- | ------------------------- | ---------------------------- |
| **Logical KV Pool**        | GPU HBM       | Radix Cache + Allocator                                    | logical index         | 所有 token 的 KV（prefill 写入） | radix tree 引用计数，evict 时释放    |
| **HiSparse Device Buffer** | GPU HBM       | Coordinator `req_to_device_buffer`                         | hisparse device index | decode 热数据（top-k + LRU）   | 每请求独立，`request_finished` 释放  |
| **Host Pool**              | DRAM/CXL/RDMA | Coordinator `req_to_host_pool` + **prefix_host_cache（新增）** | host index            | 全量 KV 备份                  | prefix 部分引用计数管理，extend 随请求释放 |


### 4.3 索引体系与数据流

HiSparse 中存在三套独立的索引，分别寻址三层内存中的 KV 数据。理解它们的映射关系是理解 prefix-caching 兼容方案的关键。

```
索引映射链：

  token_ids (语义层)
      │
      │  RadixCache.match_prefix() / cache_finished_req()
      ▼
  logical index ─────────────────────────────────────────────────────┐
  (req_to_token_pool[req_pool_idx, pos])                             │
      │                                                              │
      │ full_to_hisparse_device_index_mapping[logical_index]         │
      ▼                                                              │
  hisparse device index ─────→ kv_buffer[layer][hisparse_idx]        │
  (HiSparseNSATokenToKVPool)   GPU device pool 中的实际 KV 数据       │
      │                                                              │
      │ staging: backup_from_device_all_layer()                      │
      ▼                                                              │
  host index ─────────────────→ kv_buffer[layer][host_idx]           │
  (req_to_host_pool[req_pool_idx, pos])  host pool 中的 KV 备份      │
                                                                     │
                                                                     │
  ★ Radix Tree 存储的是 logical index（非 host index）                │
  ★ 从 token position 到 host index 的映射由 req_to_host_pool 维护 ◄──┘
```

**Radix tree 中保存的不是 host memory 索引**。Radix tree 存储的是 `logical index`——即 `req_to_token_pool` 中的 KV 槽位编号，指向 GPU logical KV pool 中的数据。`match_prefix()` 返回的 `prefix_indices` 是 logical indices。

**Decode 阶段 sparse attention 的 top-k swap-in 不经过 radix tree**。完整的 decode 数据流为：

```
每步 Decode:

  1. NSA Indexer 计算 top-k token positions（序列中的绝对位置 0..N-1）
                     │
  2. 查 req_to_host_pool[req_pool_idx, top_k_positions] → host indices
                     │
  3. 查 device buffer LRU：该 token 是否已在 device buffer 中？
     ├─ 命中 → 直接使用 device buffer 中的 KV
     └─ 未命中 → swap_in: host_pool.kv_buffer[layer][host_idx]
                         → device_buffer.kv_buffer[layer][device_buf_slot]
                     │
  4. 用 device buffer 中的 top-k KV 做 sparse attention
```

关键点：

- **Radix tree** 管理的是 logical KV pool 的索引和引用计数，用于 **prefill 阶段**的 prefix 跳过和 KV 复用
- `**req_to_host_pool`** 管理的是 host pool 的索引，由 staging 环节建立，用于 **decode 阶段**的按需 swap-in
- 两套索引的桥梁是 **staging**：staging 读取 logical indices → 转换为 hisparse device indices → DMA 到 host pool → 在 `req_to_host_pool` 中记录 host indices

**在 prefix-caching 方案中**：

- 冷启动请求 R0 的 staging 建立从 token position 到 host index 的完整映射，并将 prefix 部分的 host indices 注册到 `prefix_host_cache`
- Prefix-hit 请求 R1 的 staging 直接从 `prefix_host_cache` 复用 prefix 部分的 host indices（写入 `req_to_host_pool[R1, 0:P]`），仅为 extend 部分做 DMA 建立新的 host indices
- Decode 阶段的 `swap_in_selected_pages` 从 `req_to_host_pool` 按 top-k positions 读取 host indices，不区分 prefix/extend，整个过程不涉及 radix tree

### 4.4 Prefix-Hit 请求的完整生命周期

prefix 长度 P、总长度 N，对比冷启动 R0 和 prefix-hit R1：

```
R0（冷启动）                              R1（prefix-hit, prefix_len=P）
═══════════                              ══════════════════════════════

1. match_prefix → 无命中                  1. match_prefix → 命中 P tokens
   extend_input_len = N                     extend_input_len = N-P

2. alloc_extend:                          2. alloc_extend:
   N 个 logical + hisparse indices           prefix P 个由 radix tree 提供
                                             仅分配 (N-P) 个新 indices

3. model.forward:                         3. model.forward:
   完整 prefill N tokens                     仅 prefill (N-P) extend tokens
                                             prefix KV 在 logical pool 中已存在

4. cache_unfinished_req → 插入 radix       4. cache_unfinished_req → 更新 radix

5. admit_request_into_staging:            5. admit_request_into_staging【改造】:
   分配 N 个 host slots, DMA 全量              prefix: 从 prefix_host_cache 复用
   注册到 prefix_host_cache(ref=1)            extend: 分配 (N-P) host slots + DMA
                                             req_to_host_pool = [prefix | extend]

6. alloc_device_buffer:                   6. alloc_device_buffer【改造】:
   从全部 hisparse indices 取                  仅从 extend hisparse indices 取
   buffer tokens 初始化 [0..buf_size-1]        buffer tokens 初始化 -1（首步填充）

7. decode: swap_in top-k from host        7. decode: 同（不变）

8. request_finished:                      8. request_finished【改造】:
   device buffer + hisparse mapping 释放      同左
   host: prefix dec_ref, extend 释放          host: prefix dec_ref, extend 释放
```

### 4.5 组件交互关键细节

**Prefill Attention**：prefix-hit 时 `extend_input_len = N-P`，model forward 只计算 extend。FlashInfer kernel 通过 `req_to_token_pool` 看到完整 N tokens 的 logical indices（prefix 由 radix tree 提供，KV 数据仍在 logical pool 中）。HiSparse prefill 使用标准 attention 路径（logical indices），不依赖 `full_to_hisparse_device_index_mapping`，prefix 部分的 hisparse mapping 被前一请求清零不影响正确性。

**Staging**：prefix-hit 时 `admit_request_into_staging` 将 `req_to_host_pool` 拼接为 `[prefix_host_indices | extend_host_indices]`。DMA 量从 N 降为 N-P。

**Device Buffer**：prefix-hit 时初始化为空（`-1`）。首步 decode 的 swap-in 从 host pool 按 top-k 加载，后续 LRU 正常。

`**_eager_backup_previous_token`**：无需修改。`_skip_first_backup` flag 在 `collect_ready_reqs` 中自动设置。

### 4.6 代码改动清单（Phase 1）


| #   | 文件                        | 改动                                                               | 行数       |
| --- | ------------------------- | ---------------------------------------------------------------- | -------- |
| 1   | `server_args.py`          | 去掉 `assert self.disable_radix_cache`，改为 warning                  | ~2       |
| 2   | `hisparse_coordinator.py` | `PrefixHostCacheEntry` + `prefix_host_cache` / `req_prefix_info` | ~30      |
| 3   | `hisparse_coordinator.py` | `admit_request_into_staging` prefix-hit/冷启动双路径                   | ~60      |
| 4   | `hisparse_coordinator.py` | `alloc_device_buffer` 只传 extend indices + buffer tokens 初始化      | ~20      |
| 5   | `hisparse_coordinator.py` | `request_finished` 引用计数释放                                        | ~25      |
| 6   | `hisparse_memory_pool.py` | `alloc_device_buffer` 接口适配                                       | ~10      |
|     | **合计**                    |                                                                  | **~150** |


### 4.7 冲突解决总结


| 冲突                     | 处理                                                             |
| ---------------------- | -------------------------------------------------------------- |
| hisparse mapping 清零    | `alloc_device_buffer` 只处理 extend indices；decode 从 host swap-in |
| Host pool 全量分配         | prefix-hit 复用 `prefix_host_cache`，只分配 extend                   |
| `request_finished` 全释放 | 引用计数：prefix dec_ref，ref=0 时释放                                  |
| `alloc_extend` 双分配     | 自动正确：radix cache prefix_lens 控制                                |
| Device buffer 私有       | 不变：prefix-hit 从 host swap-in 填充                                |
| Scheduler staging 路径   | 不变，DMA 量减少                                                     |


### 4.8 Radix Cache 驱逐与 Host Pool 的交互

Radix cache eviction（`RadixCache.evict()`）只释放 GPU 侧资源：调用 `token_to_kv_pool_allocator.free()` 回收 logical index 和 hisparse device index，然后从 radix tree 中删除节点。**不触及 host pool**——host pool 由 `HiSparseCoordinator` 独立管理。

驱逐 prefix 节点后的影响：
- **正在 decode 的请求不受影响**：通过 `req_to_host_pool` 直接访问 host indices，不经过 radix tree
- **新请求退化为冷启动**：radix tree 中无该 prefix，`match_prefix()` 返回空，需完整 prefill
- **`prefix_host_cache` 中的 host KV 变为孤立**：ref_count 最终降为 0，可被清理

**实验中不是问题**：所有请求共享固定 prefix，radix tree 中该节点被持续 `inc_lock_ref`，不会被驱逐。即使驱逐也只是性能退化，不影响正确性。

**生产环境扩展方向**（非本次范围）：
1. **Host→device reload**：radix miss 时检查 `prefix_host_cache`，从 host DMA 回 device 重建 radix 节点（类似 HiCache load-back）
2. **Host pool LRU 淘汰**：ref_count=0 的 prefix entry 按 LRU 释放，防止 host pool 耗尽

### 4.9 实验原型 Workaround（最小化可跑系统）

为尽快在单机上跑通端到端对比、控制改动面，当前原型与「理想化 prefix-hit 完全跳过 GPU 计算」有两点刻意简化，实验结论应在此前提下解读。

**（1）Prefix-hit 后首步仍是 EXTEND，仅恢复 tail page**

即便 radix cache `match_prefix` 命中，调度上**第一个 forward batch 仍然是 EXTEND**（`alloc_for_extend` / `prepare_for_extend` 路径），而不是整段 prefix 当作已算完跳过。为降低显存与 DMA 压力、让系统先跑通，HiSparse coordinator 在「长 prefix + host-extend」路径上**只把 prefix 的最后一个 page（tail page）的 logical→hisparse 映射与 KV 加载进 GPU**，满足 `alloc_extend` 对尾页对齐的依赖；其后行为更接近「从该状态进入 decode」：后续步仍按 HiSparse decode 从 host 做 top-k swap-in。这与「prefix 全量在 GPU 上、首步即纯 decode」的理想形态不等价，但足以对比 **CXL 按需读 vs RDMA 预取** 在 decode 阶段的主差异。

**（2）DPA 下每 DP 独立 radix cache：`e2e_bench_fast.py` 的 warmup 策略**

开启 **DP Attention（DPA）** 时，`dp_size > 1` 表示多个数据并行组，**每组各自一份 radix tree / HiSparse 状态**，prefix cache **不能跨 DP rank 共享**。若 Round 1 只发 **1 条** HTTP 请求，负载均衡只会 warm **其中一个 rank** 的 cache，Round 2 发到其他 rank 的请求仍会走完整长上下文 prefill（日志上可见部分 DP `#cached-token: 0`、部分 DP 有高 `#cached-token`）。

基准脚本 `test/cxl_utils/e2e_bench_fast.py` 的 workaround：

- 启动时通过 `/server_info` 读取 `dp_size`；
- **Round 1**：对每个 unique prompt 发送 **`dp_size` 份**相同请求（并行度受 `--max-concurrency` 限制），使 round-robin 下**每个 DP rank 至少执行过一次**该 prompt 的冷启动，从而各 rank 的 radix / prefix_host 均被填充；
- **Round 2**：仍按 `--num-requests` 与 `repeat-mode` **重复同一批 prompt**，在已 warm 的多 rank 上测量 prefix-hit 行为。

因此：实验报告中若涉及 DPA，应写明「Round 1 请求数 = `num_unique_prompts × dp_size`」，并与单 DP（`dp_size=1`）时的语义区分说明。

---

## 5. 需要开发的代码

> 本节描述 RDMA/CXL 后端相关代码（Phase 3）。HiSparse + Prefix-Caching 兼容代码（Phase 1，~150 行）见[第 3 节](#3-分阶段开发计划)和[第 4 节](#4-hisparse--prefix-caching-设计方案)。

### 5.1 Phase 3 代码量（RDMA 后端 + Benchmark）


| 模块                 | 新代码量       | 修改/新增                           | 说明                                       |
| ------------------ | ---------- | ------------------------------- | ---------------------------------------- |
| **RDMA Host Pool** | ~300 行     | 新增 `rdma_memory_pool.py`        | 方案 B 核心：page-first 远端 → bulk RDMA → transpose/scatter + 本地缓存分层 |
| **Coordinator 适配** | ~40 行      | 修改 `hisparse_coordinator.py`    | 根据配置选择 A/B/C 后端 + decode prefetch + cleanup |
| **Config 扩展**      | ~15 行      | 修改 `factory.py` + `SparseConfig` | rdma_pool JSON 解析 + 互斥校验                   |
| **ModelRunner**     | ~2 行       | 修改 `model_runner.py`           | 传递 rdma_config                            |
| **RDMA C 扩展**      | ~60 行      | 修改 `.c` + `.h` + `_py.cpp`     | bulk read + 多实例 RDMAContext                |
| **Benchmark 脚本**   | ~350 行     | 新增 `e2e_bench.py`              | 两轮实验自动化，ShareGPT 多样化请求，TTFT/TBT/TPOT      |
| **自动化脚本**         | ~150 行     | 新增 `run_all_experiments.sh`     | 四方案 × 多 prefix 完整实验矩阵                    |
| **合计**             | **~920 行** | 3 新增 + 5 修改                    |                                          |


### 5.2 模块 1：RDMAMLATokenToKVPoolHost（~300 行）

**新文件**：`python/sglang/srt/mem_cache/rdma_memory_pool.py`

方案 B 的核心：继承 `MLATokenToKVPoolHost`，用本机 **NIC loopback + MR** 模拟「部分请求 KV 在远端」。**远端使用 page-first 布局，本地 host pool 保持 layer-first 布局**，RDMA 传输后做布局转换写入 host pool。远端请求在进入 decode 前 **只做一次** 全量预取；之后与本地请求一样，HiSparse 只从统一的 `kv_buffer`（layer-first）做 swap-in。

**数据路径**分为两种场景：**Cold（首次 prefill，无 prefix cache）** 和 **Warm（prefix cache 命中）**。

#### Cold 路径（Round 1 / 无 prefix cache）

1. Prefill / staging 阶段照常把 KV 写入 `kv_buffer`（layer-first）。
2. Staging 完成后，对被判为 remote 的请求，调用 **`simulate_remote_write`**：把 `kv_buffer` 中有效 token 的 KV **转换为 page-first 布局后写入 `remote_buffer`**（注册为 MR），模拟「远端 prefill 节点产生了 KV 并存入分布式内存池」。
3. 进入 decode 时调用 **`prefetch_for_decode`**：**bulk RDMA READ**（`remote_buffer` 连续 page-first → loopback → `landing_staging`）→ **transpose + scatter**（page-first → `kv_buffer` layer-first slot）。
4. 此后 decode 全部从本地 `kv_buffer` 读取。

#### Warm 路径（Round 2 / prefix cache 命中）

prefix cache 命中时，**prefix 部分的 KV 在 `remote_buffer` 中已经存在**（Round 1 的 `simulate_remote_write` 写入的），**不需要再次 `simulate_remote_write`**。这更贴近真实分布式场景：decode 节点发现本地 radix cache 命中了某个 prefix，但 KV 不在本地 DRAM，需要从远端内存池拉取。

1. Coordinator 检测 prefix cache hit → prefix 部分的 host indices 来自 `prefix_host_cache` entry。
2. **跳过** prefix 部分的 `simulate_remote_write`（远端已有）。
3. 对 prefix 部分直接调用 **`prefetch_for_decode`**：从 `remote_buffer`（Round 1 已写入）做 **bulk RDMA READ → transpose + scatter** 到 `kv_buffer`。
4. 对 extend 部分（新 token）走正常 staging DMA 路径。
5. 此后 decode 全部从本地 `kv_buffer` 读取。

```
Round 1 (Cold, remote 请求):
  GPU prefill → kv_buffer (layer-first)
             → simulate_remote_write → remote_buffer (page-first)  ✓ 写入远端
             → prefetch_for_decode → RDMA → kv_buffer              ✓ 拉回本地
             → decode...

Round 2 (Warm, prefix cache hit, remote 请求):
  Radix cache hit → prefix KV 已在 remote_buffer (Round 1 写入)
                  → 【跳过 simulate_remote_write】                    ✗ 不再重复写
                  → prefetch_for_decode → RDMA → kv_buffer           ✓ 直接从远端拉
                  → extend 部分正常 staging
                  → decode...
```

**与 SGLang / HiSparse 的接口契约**（对上层透明，仅 pool + coordinator 约定）：

| 入口 | 调用时机 | 作用 |
| --- | --- | --- |
| `init_kv_buffer()` | Model runner 建 pool | 分配 `kv_buffer`（layer-first，返回值）、`remote_buffer`（page-first）、`landing_staging`，并完成 RDMA MR 注册 |
| `mark_request_locality(req_pool_idx)` | 某请求 staging 完成、即将标为 decode-ready | 按 `local_ratio` **请求级** 决定 local / remote |
| `simulate_remote_write(req_pool_idx, host_indices)` | Cold 路径：staging 完成后 | 仅在 **首次 prefill（无 prefix cache hit）** 时调用；将 `kv_buffer`（layer-first）中有效 slot 的 KV **转换为 page-first** 写入 `remote_buffer` |
| `prefetch_for_decode(req_pool_idx, host_indices)` | Cold 和 Warm 路径均调用 | bulk RDMA READ（`remote_buffer` → `landing_staging`）+ transpose + scatter → `kv_buffer`；local 请求 no-op |

**设计要点**：
- **远端 page-first / 本地 layer-first**：`remote_buffer` 形态 `[num_tokens, layer_num, 1, kv_dim]`，`kv_buffer` 形态 `[layer_num, num_tokens, 1, kv_dim]`。
- **Cold 写一次、Warm 直接拉**：`simulate_remote_write` 仅在 Cold 路径调用一次，将 KV 写入远端。Warm 路径（prefix cache hit）时，prefix KV 已在 `remote_buffer` 中（Round 1 写入的 host indices 保留在 `prefix_host_cache` 中），`prefetch_for_decode` 用同一组 host indices 直接从 `remote_buffer` RDMA 拉取，**不重复 simulate_remote_write**。
- **RDMA 传输无需 gather**：远端同一请求的 token 在 `remote_buffer` 中按连续地址排列（page-first），NIC 直接做大块连续 READ。
- **布局转换是真实开销**：page-first → layer-first 的 transpose/scatter 反映了真实 decode 节点接收到 KV 后重组的代价。
- 请求级 `local_ratio`；HiSparse decode 路径不区分 local/remote；staging 大小由 `max_seq_len` 等配置封顶；`local_ratio=1.0` 时行为等同纯 DRAM 方案 A。

### 5.3 模块 2：Coordinator 适配（~60 行）

**修改文件**：`python/sglang/srt/managers/hisparse_coordinator.py`

- **`__init__`**：`hisparse_config["rdma_pool"]` 存在且启用时构造 `RDMAMLATokenToKVPoolHost`（并传入 `rdma_config`；多网卡与请求级 striping 见 §9），否则保持现有 DRAM / CXL 分支。

- **`admit_request_into_staging`（Cold 路径）**：在 staging DMA 完成后、请求加入 `ack_staging_queue` 之前，若请求被判为 remote 且 **不是** prefix cache hit（即走的是 Cold-start 分支），调用 `simulate_remote_write` 将 KV 写入 `remote_buffer`。**Prefix-hit 路径不调用 `simulate_remote_write`**——prefix 部分的 KV 已在 Round 1 的 cold-start 时写入了 `remote_buffer`。

- **`collect_ready_reqs`（Cold 和 Warm 路径均适用）**：对每个新就绪请求，若该请求是 remote，调用 `prefetch_for_decode`（bulk RDMA READ → transpose + scatter 到 `kv_buffer`）。无论是 Cold 还是 Warm 请求，**进入 decode 前都通过 RDMA 从 `remote_buffer` 拉取 KV**。Decode 循环内不再触发 RDMA。

- **`request_finished` / `clear_request_locality`**：请求结束后清理 locality 状态。注意 `remote_buffer` 中该请求写入的 KV **不需要主动清除**——其 slot 由 host pool allocator 管理，后续 alloc 覆写即可；prefix cache 引用计数归零后 slot 自然可复用。

### 5.4 模块 3：端到端 Benchmark 脚本（~350 行）

**新文件**：`test/cxl_utils/e2e_bench.py`

自动化两轮实验，向运行中的 SGLang 服务器发送请求并收集精细指标。

#### 5.4.1 设计原则

1. **两轮请求完全相同**：Round 1 和 Round 2 发送**完全一致**的 N 个请求（同一批 prompt，同一顺序），Round 2 因此能命中 Round 1 建立的 prefix cache
2. **单 round 内请求各不相同**：N 个 prompt 来自 ShareGPT 数据集的 N 条不同对话，每条内容唯一，不存在公共 system prompt
3. **cache 粒度为请求自身**：每个请求的 prefix cache key 是其自身 token 序列，Round 2 复用 Round 1 留下的完整 KV，prefill 跳过，仅计算 extend（即新生成的 output tokens 中的 decode 部分）
4. **精细指标**：通过 streaming API 逐 token 计时，测量 TTFT、TBT、TPOT、E2E latency、吞吐
5. **复用 bench_serving 生态**：使用 sglang 的 `benchmark.utils` 自动下载 ShareGPT 数据集、tokenizer

#### 5.4.2 请求构造

```
N 个 prompt（从 ShareGPT 采样，一次采样，两轮共用）:
  prompt_0 = sharegpt_conversation[0].user_turn   ← 独立对话，互不相关
  prompt_1 = sharegpt_conversation[1].user_turn
  ...
  prompt_{N-1} = sharegpt_conversation[N-1].user_turn

Round 1 (Cold):  发送 prompt_0 .. prompt_{N-1}  → 填充 prefix cache
Round 2 (Warm):  重发 prompt_0 .. prompt_{N-1}  → 全部 cache hit

关键特性：
  ✓ 两轮请求数量一致（都是 N）
  ✓ 两轮发送完全相同的 N 个请求
  ✓ Round 内部每个请求内容各不相同（不同 ShareGPT 对话）
  ✗ 没有人工构造的共享 system prompt（每条请求独立）
```

#### 5.4.3 指标对比矩阵

| 指标 | Round 1 (Cold) | Round 2 (Warm) | 预期变化 |
| --- | --- | --- | --- |
| **TTFT** | 完整 prefill 延迟 | prefix cache 命中，仅 extend | **大幅下降** |
| **TBT** | decode 逐 token 延迟 | decode，但 KV 已在 host pool | 方案间对比核心 |
| **TPOT** | (latency - TTFT) / tokens | 同 | 反映 decode 效率 |
| **Throughput** | output tok/s | output tok/s | Round 2 应更高（prefill 省去） |
| **E2E Latency** | 整体请求延迟 | 整体请求延迟 | 综合指标 |

#### 5.4.4 使用方式

```bash
# 单次实验（ShareGPT 自动下载）
python test/cxl_utils/e2e_bench.py \
    --server-url http://localhost:30000 \
    --model deepseek-ai/DeepSeek-V3.2 \
    --num-requests 16 \
    --output-tokens 128 \
    --min-prompt-tokens 64 \
    --max-prompt-tokens 8192 \
    --request-rate 4 \
    --output results.json

# 指定本地 ShareGPT 路径
python test/cxl_utils/e2e_bench.py \
    --dataset-path /data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-requests 32 \
    --output-tokens 0    # 0 = use dataset ground truth output length
    ...

# 完整实验矩阵（自动化）
bash test/cxl_utils/run_all_experiments.sh \
    --model deepseek-ai/DeepSeek-V3.2 \
    --tp 8 \
    --num-requests 16
```

#### 5.4.5 自动化脚本

**新文件**：`test/cxl_utils/run_all_experiments.sh`

对四种方案 (DRAM / RDMA-0% / RDMA-30% / CXL) 自动启停服务器、运行 benchmark、汇总结果：

```bash
# 流程：
for scheme in dram rdma_0 rdma_30 cxl; do
    launch_server --hisparse-config $config
    python e2e_bench.py --output results/${scheme}.json
    kill_server
done
# 最后打印 TTFT/TBT 对比汇总表
```

---

## 6. RDMA Loopback 实现要点

### 6.1 基础设施与 bulk 路径

复用 `test/cxl_utils/sparse_kv_bench.py` 中的 loopback 与 pybind 封装；C 侧在 `rdma_scatter_read.c` 中提供 **`rdma_bulk_read(rctx, total_bytes)`**：将 `total_bytes` 按固定上限（如 8MB）拆成若干 **RDMA READ** WR，完成后 poll CQ 并返回耗时（微秒）。WR 数量随 **总字节数 / chunk**，而非 token 数。

### 6.2 数据流（page-first 远端 → bulk RDMA → layer-first 本地）

```
remote_buffer (page-first, 连续)  ---bulk RDMA READ (loopback)--->  landing_staging (page-first)
                                                                          |
                                                                   transpose + scatter
                                                                          |
                                                                          v
                                                                   kv_buffer (layer-first, 离散 slot)
```

**Step 1 — `simulate_remote_write`**（prefill 完成后、进入 decode 前）：
- 从 `kv_buffer`（layer-first）中读取该请求有效 token 的 KV
- 转换为 **page-first** 布局后连续写入 `remote_buffer` 中该请求对应的 token 区域
- 模拟「远端节点以 page-first 存储 KV」

**Step 2 — `prefetch_for_decode`**（一次性 RDMA 预取）：
- **bulk RDMA READ**：从 `remote_buffer` 该请求的连续 page-first 区域经 NIC loopback 读到 `landing_staging`
- **transpose + scatter**：将 `landing_staging` 中 page-first 数据（`[num_tokens, layer_num, 1, kv_dim]`）转换为 layer-first 后写回 `kv_buffer`（`[layer_num, token_slots, 1, kv_dim]`）

**关键区别**：不需要 gather staging。远端 page-first 布局意味着同一请求的 KV 本身已连续，NIC 直接做大块 READ。

### 6.3 布局转换的实现

`simulate_remote_write` 中 layer-first → page-first 转换：
- 遍历有效 token，对每个 token 把所有层的 KV 拼成连续块写入 `remote_buffer[token_idx]`
- 等价于 `remote_buffer[token_idx, :, :, :] = kv_buffer[:, slot_idx, :, :]`（维度 transpose）

`prefetch_for_decode` 中 page-first → layer-first 转换：
- 从 `landing_staging` 按 token 遍历，将每个 token 的各层 KV 分别写回 `kv_buffer[layer, slot]`
- 等价于 `kv_buffer[:, slot_idx, :, :] = landing_staging[token_idx, :, :, :]`（反向 transpose）

这两个转换都是 **CPU 内存操作**（pinned DRAM 间拷贝），延迟主要由 **总字节数 / DRAM 带宽** 决定。对于 32K tokens × 61 layers × 656B ≈ 1.2 GB 数据量，预期耗时约 **几毫秒到十几毫秒**。在 loopback / 本机 DRAM 场景下，这部分开销**不一定总是显著小于** RDMA 传输本身，是否需要进一步优化应以 `rdma_us` / `transpose_us` 实测为准。

### 6.4 `local_ratio` 与触发时机

- **请求级** `local_ratio`：见 §5.2；典型实验点：0.0 / 0.3 / 1.0（等同 DRAM 上界）。
- **`prefetch_for_decode`**：仅在 **staging 完成 → 进入 decode** 时调用 **一次**；后续 decode 步仅访问 `kv_buffer`，无 RDMA。代价集中在 **请求级首段**（NIC 传输 + 布局转换），用于解释 Round 2 中相对 CXL 的 TTFT / 首步 decode 差异。

### 6.5 相比旧方案（双向 gather/scatter + layer-first 远端）的改进

| 维度 | 旧方案 | 新方案（page-first 远端） |
| --- | --- | --- |
| 远端存储布局 | layer-first（与本地相同） | **page-first**（同一 token 全层连续） |
| RDMA 前 CPU 工作 | **gather**: 逐层从 remote 离散 slot → gather_staging | **无**：远端本身连续 |
| RDMA 传输 | gather_staging → landing_staging | remote_buffer → landing_staging |
| RDMA 后 CPU 工作 | **scatter**: landing_staging → kv_buffer 逐层离散 slot | **transpose + scatter**: page→layer 写回 |
| staging 数量 | 2 块（gather + landing） | **1 块**（仅 landing） |
| 真实性 | 远端与本地同布局，需额外 gather/scatter | **远端 page-first 更贴近生产分布式池** |

---

## 7. 配置接口设计

### 7.1 hisparse-config JSON 扩展

```json
{
    "top_k": 2048,
    "device_buffer_size": 4096,
    "host_to_device_ratio": 2,

    "cxl": {
        "enabled": true,
        "dev_path": "/dev/dax0.0",
        "map_bytes": 68719476736
    },

    "rdma_pool": {
        "enabled": true,
        "ib_dev": "mlx5_0",
        "local_ratio": 0.3
    }
}
```

**互斥关系**：`cxl.enabled` 和 `rdma_pool.enabled` 不能同时为 true。同时为 false 时使用默认 DRAM 后端。

### 7.2 启动命令示例

```bash
# 方案 A: DRAM (默认)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
    --enable-hisparse \
    --hisparse-config '{"top_k":2048,"device_buffer_size":4096}'

# 方案 B: RDMA 分布式池 (0% 本地缓存)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
    --enable-hisparse \
    --hisparse-config '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.0}}'

# 方案 B: RDMA 分布式池 (30% 本地缓存)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
    --enable-hisparse \
    --hisparse-config '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.3}}'

# 方案 C: CXL
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
    --enable-hisparse \
    --hisparse-config '{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_path":"/dev/dax0.0","map_bytes":68719476736}}'
```

---

## 8. 两轮实验执行流程

### 8.1 请求设计

**核心设计原则**：两轮发送完全相同的请求，round 内部各请求内容各不相同，不使用共享 system prompt。

```
N 个 prompt（一次采样，两轮共用）:
  来自 ShareGPT 数据集的 N 条不同对话的用户提问
  每条 prompt 内容唯一，相互之间没有公共前缀

Round 1 (N 个请求，cold start):
  发送 prompt_0 .. prompt_{N-1}  → 填充 radix prefix cache

Round 2 (N 个请求，warm start):
  重发完全相同的 prompt_0 .. prompt_{N-1}
  → radix cache 全部命中，prefill 跳过，仅 extend

关键特性：
  ✓ 两轮请求数量一致（都是 N）
  ✓ 两轮发送完全相同的 N 个请求（same prompts, same order）
  ✓ Round 内部每个请求内容各不相同（N 条不同 ShareGPT 对话）
  ✗ 没有人工构造的公共 system prompt（每条请求独立）
  ✓ Cache 命中粒度：每条请求自身的完整 token 序列
```

### 8.2 测量指标

通过 OpenAI-compatible **streaming API** 逐 token 计时：

| 指标 | 定义 | 意义 |
| --- | --- | --- |
| **TTFT** | 首 token 到达时间 | 反映 prefill 延迟，prefix cache 命中时大幅下降 |
| **TBT** | 相邻 token 间延迟 (inter-token latency) | 反映 decode 阶段效率，CXL 按需读取 vs RDMA 全量预取核心对比 |
| **TPOT** | (latency - TTFT) / (output_len - 1) | 每 output token 平均延迟 |
| **E2E Latency** | 整体请求延迟 | 综合指标 |
| **Throughput** | output_tokens / wall_time | 系统吞吐 |

> 注：评估 §9 `request_striping` 时，建议同时记录 **TTFT**、`get_rdma_stats()` 中的 **prefetch / transpose** 分项，以及（若接入）**单次 striped prefetch 的 wall time**，避免单一指标掩盖预取段变化。

### 8.3 完整流程

```bash
# 对每种方案 (A / B-0 / B-30 / C)：

# 1. 启动服务器（方案 X 配置）
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
    --enable-hisparse --hisparse-config "$config" &
wait_for_server_ready

# 2. 运行两轮 benchmark（两轮发送相同的 N 条 ShareGPT 请求）
python test/cxl_utils/e2e_bench.py \
    --server-url http://localhost:30000 \
    --model deepseek-ai/DeepSeek-V3.2 \
    --num-requests 16 \
    --output-tokens 128 \
    --min-prompt-tokens 64 \
    --max-prompt-tokens 8192 \
    --request-rate 4 \
    --output results/${scheme}.json

# 3. 收集系统指标
free -h > results/${scheme}_mem.txt
nvidia-smi > results/${scheme}_gpu.txt

# 4. 关闭服务器，切换方案
```

### 8.4 自动化脚本

`test/cxl_utils/run_all_experiments.sh` 自动化完整实验矩阵：

```bash
# 一键运行所有实验
bash test/cxl_utils/run_all_experiments.sh \
    --model deepseek-ai/DeepSeek-V3.2 \
    --tp 8 \
    --num-requests 16

# 可配置参数：
#   --model           模型名
#   --tp              tensor parallelism
#   --num-requests    每轮请求数（两轮相同）
#   --output-tokens   每请求输出 tokens（0=dataset ground truth）
#   --request-rate    请求发送速率 (req/s)
#   --results-dir     结果输出目录

# 自动执行：
#   1. 对 dram / rdma_0 / rdma_30 / cxl 四种方案
#   2. 每种方案启动服务器 → 运行 e2e_bench（两轮相同请求）→ 收集系统指标 → 关闭服务器
#   3. 最后打印 TTFT/TBT 对比汇总表
```

---

## 9. 多网卡 RDMA：请求级分片与 RNIC 聚合（设计方案）

### 9.1 设计目标

- **主目标**：在 remote 全量 prefetch 路径上，使**单请求**能够并行占用多张 RNIC，缩短预取关键路径上的传输时间。
- **并存模式**：保留可选的 **`rank_interleave`**（`ib_devs[tp_rank % len(ib_devs)]`，每 rank 绑定一张 RNIC、单请求仍走单卡）；本章重点描述新增的 **`request_striping`**（按 token 区间把同一请求拆到多张 RNIC 上并行 bulk read）。
- **非目标**：不改动 HiSparse decode 主循环与 swap-in 语义；仅在进入 decode 前的 prefetch 阶段增强并行度。

### 9.2 硬件拓扑与瓶颈分析

```
典型 8×H20 + 8×ConnectX-7 拓扑：

CPU Socket 0                          CPU Socket 1
┌─────────────────────┐              ┌─────────────────────┐
│        RC 0         │              │        RC 1         │
│   ~64 GB/s 双向      │              │   ~64 GB/s 双向      │
├──────────┬──────────┤              ├──────────┬──────────┤
│ Switch 0 │ Switch 1 │              │ Switch 2 │ Switch 3 │
│ ┌──┬──┐  │ ┌──┬──┐  │              │ ┌──┬──┐  │ ┌──┬──┐  │
│ │G0│G1│  │ │G2│G3│  │              │ │G4│G5│  │ │G6│G7│  │
│ │N0│N1│  │ │N2│N3│  │              │ │N4│N5│  │ │N6│N7│  │
│ └──┴──┘  │ └──┴──┘  │              │ └──┴──┘  │ └──┴──┘  │
└──────────┴──────────┘              └──────────┴──────────┘
G0~G7 = GPU (H20), N0~N7 = NIC (mlx5_0 ~ mlx5_7)
```

| 瓶颈层级 | 带宽上限 | 说明 |
| --- | --- | --- |
| 单网卡 | ~25 GB/s | ConnectX-7 200Gbps 线速 |
| PCIe Switch (2 NIC 共享) | ~32 GB/s | 同一 Switch 上行带宽 |
| PCIe RC (4 NIC 共享) | ~64 GB/s | 同 Socket 到 DRAM 的路径 |
| 双 Socket UPI | ~128 GB/s | 跨 Socket 时的 UPI 总线 |

**推论**：即使 8 张网卡理论聚合 200 GB/s，实际写入 DRAM 的带宽仍受 PCIe RC / Switch / NUMA 路径约束，**扩展效率应为 sub-linear**；`request_striping` 的设计目标是在该物理上限内尽量吃满多 RNIC 的并行度。

### 9.3 Request-level striping：单请求多 RNIC 并行预取

#### 9.3.1 核心思路

利用远端 page-first 布局：

```text
remote_buffer[token, layer, ...]
```

同一请求的 token 区间天然适合按 **token range** 分片。对长度为 `N` 的 remote 请求，可把其 KV 划成 `S` 个 shard：

- shard 0: tokens `[0, n0)`
- shard 1: tokens `[n0, n1)`
- ...
- shard S-1: tokens `[n_{S-1}, N)`

每个 shard：

- 绑定一张 RNIC / 一个 RDMA context / 一个 QP
- 独立发起 bulk RDMA READ
- 独立落到自己的 `landing_staging_shard`
- 独立完成 page-first → layer-first 的局部 transpose/scatter

这样，多 RNIC 才能缩短**单请求关键路径**。

#### 9.3.2 建议的数据流

```text
request KV in remote_buffer (page-first)
    ├─ shard 0 ──bulk_read on NIC 0──> landing_staging_0 ──transpose/scatter──> kv_buffer[:, slots_0]
    ├─ shard 1 ──bulk_read on NIC 1──> landing_staging_1 ──transpose/scatter──> kv_buffer[:, slots_1]
    ├─ shard 2 ──bulk_read on NIC 2──> landing_staging_2 ──transpose/scatter──> kv_buffer[:, slots_2]
    └─ ...
```

相比 **rank 仅绑定单张 RNIC、单请求只走一张卡** 的路径，这里是**同一请求内部并行占用多张 RNIC**。

#### 9.3.3 page-first 与分片天然对齐

page-first 下同一 token 的所有层连续，按 token 范围切 shard 时：

- 每个 shard 在远端地址上仍是**连续区间**
- 每张 RNIC 看到的仍是 bulk READ，而不是离散 scatter
- 不需要额外的 per-token gather

也就是说，`page-first` 远端布局使 **request-level striping 可以直接在连续地址上完成**，是分片并行读的理想前提。

### 9.4 本地重排分片与 overlap

如果只把 RDMA READ 拆成多 NIC，但最终仍由一个线程串行做整段 transpose/scatter，则本地重排会成为新的瓶颈。因此建议同时做以下优化：

1. **每个 shard 独立 transpose/scatter**  
   每个 RNIC 对应自己的 `landing_staging_shard`，读完后只写自己负责的 slot 范围，减少单点串行 copy。

2. **双缓冲 / 流水 overlap**  
   当 shard A 还在 RDMA READ 时，CPU 可以开始处理 shard B 的 transpose/scatter；或当请求 i 的 shard 在转置时，请求 i+1 的 shard 已开始 RDMA。  
   目标是 overlap：
   - RNIC 传输
   - CPU transpose/scatter
   - device buffer 准备

3. **尽量保持 slot 连续性**  
   若 allocator 能尽量为一个请求分配连续 host slot，则 RDMA 与本地 scatter 都更容易接近大块顺序访问。

### 9.5 Python / C 层改造要点

#### 9.5.1 C / pybind 层

在已实现的可实例化 `RDMAContext` / `RDMAContextHandle` 基础上扩展为**每 RNIC 一实例**，使单次 prefetch 可同时挂多个 QP。建议：

- 为 `ib_devs` 中每张参与 striping 的 RNIC 维护独立 `RDMAContextHandle`
- `bulk_read` 支持指定 **remote offset / local offset / length**（同一 MR 内子区间 READ）
- 在 pybind 内释放 GIL，允许多个 shard **并行**发起与等待 completion

#### 9.5.2 `rdma_memory_pool.py`

新增/区分两种 `rdma_pool.mode`：

- **`rank_interleave`**：`ib_devs[tp_rank % len(ib_devs)]`；每个 rank 绑定一张 RNIC，单请求预取只走该 rank 所选设备。
- **`request_striping`**：单请求按 token 区间切 shard，并行占用多张 RNIC（shard 数 ≤ `max_rnics_per_request`）。

建议新增接口与状态：

- `self.rdma_handles: List[RDMAContextHandle]`（或与 `ib_devs` 对齐的句柄表）
- `self.landing_stagings: List[Tensor]`（每 shard / 每 NIC 一块 page-first staging）
- `prefetch_for_decode_striped(req_pool_idx, host_indices)`

`prefetch_for_decode_striped()` 建议步骤：

1. 根据序列长度、`ib_devs` 与 `max_rnics_per_request` 决定 shard 数 `S`
2. 将 `host_indices` 切成 `S` 段连续 token 区间（若 slot 不连续，则按区间合并或回退到子块 READ）
3. 对每个 shard 计算 `remote_buffer` 中的字节偏移与长度，并行 `bulk_read`
4. 各 shard 在对应 `landing_staging_i` 上独立完成 page-first → layer-first 的局部 transpose/scatter，写回 `kv_buffer` 对应列
5. **全部 shard 完成**后返回，再由 coordinator 进入 decode

#### 9.5.3 Coordinator

remote 请求仍在 **staging 完成 → 进入 decode 前** 调用一次预取；仅将调用点从 `prefetch_for_decode()` 切换为 **`prefetch_for_decode_striped()`**（由 `rdma_pool.mode` 决定），scheduler / HiSparse decode 主流程其余契约不变。

### 9.6 配置接口

`rdma_pool` 同时描述**可用 RNIC 集合**与**使用方式**：

```json
{
    "rdma_pool": {
        "enabled": true,
        "local_ratio": 0.0,
        "ib_devs": ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"],
        "mode": "request_striping",
        "max_rnics_per_request": 4,
        "min_tokens_per_rnic": 8192
    }
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `ib_dev` | string | 单卡模式（向后兼容） |
| `ib_devs` | list | 可用 RNIC 列表 |
| `mode` | string | `rank_interleave` 或 `request_striping`（本方案主推荐） |
| `max_rnics_per_request` | int | 单请求 striping 并行上限 |
| `min_tokens_per_rnic` | int | 低于该阈值时不拆 shard，避免消息过碎 |

### 9.7 验证与实验矩阵（面向 striping）

1. **固定输入长度，扫描 `max_rnics_per_request`**：例如 32K / 64K / 128K tokens，`S ∈ {1,2,4,8}`，记录单次 `prefetch_for_decode_striped` 总耗时及 `get_rdma_stats()` 中分项。
2. **固定 `S`，扫描 `ib_devs` 子集**：验证 shard 与物理 NIC 一一映射时的带宽与稳定性。
3. **对比 `mode=rank_interleave` 与 `mode=request_striping`**：同硬件、同序列长度下，比较预取阶段 wall time 与 TTFT（客户端或 server 日志）。
4. **可选**：在 `e2e_bench_fast.py` 中增加短输出子配置，使单次请求 wall time 更贴近 prefetch 段，便于自动化回归 striping 收益。

### 9.8 预期与物理边界

在 `request_striping` 生效且 shard 粒度合理时：

- 单请求 prefetch 的**传输段**应随并行 RNIC 数量增加而缩短，直至触及 **PCIe RC / Switch / NUMA / DRAM 写入**上限。
- 扩展曲线仍为 **sub-linear**；当本地 transpose/scatter 成为瓶颈时，需依赖 §9.4 的分片与 overlap 继续压榨。

### 9.9 代码改动量（估算）

| 模块 | 改动 | 行数 |
| --- | --- | --- |
| `rdma_scatter_read_py.cpp` | 多 context + offset-aware bulk read + 并行等待 | ~80-120 行 |
| `rdma_memory_pool.py` | request-level striping / 多 staging / shard transpose | ~120-180 行 |
| `hisparse_coordinator.py` | 可选切换到 striped prefetch | ~10-20 行 |
| `factory.py` / `SparseConfig` | 解析 `mode` / `max_rnics_per_request` 等字段 | ~10 行 |
| benchmark / scripts | 增加 TTFT-sensitive 子实验与统计打印 | ~20-40 行 |
| **合计** | | **~240-370 行** |

---
