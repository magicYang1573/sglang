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

方案 B 模拟"分布式内存池中，X% 的请求 KV 在本地、其余在远端"的场景。`local_ratio` 控制的是**请求级别**的本地命中率——同一请求的全部 KV 要么全在本地，要么全在远端：

```
本机 DRAM 布局（方案 B）:

┌─────────────────────────────┐
│  模型权重 + 其他                │
├─────────────────────────────┤
│  Host Pool (pinned DRAM)     │  ← HiSparse 统一视图，所有请求的 KV
│  (staging DMA 写入 + decode  │    local 请求直接可用
│   swap-in 读取)              │    remote 请求 RDMA 预取后可用
├─────────────────────────────┤
│  Remote Buffer (RDMA MR)     │  ← 与 host pool 等大
│  (ibv_reg_mr, 模拟远端节点)    │    remote 请求的 KV 副本存放于此
├─────────────────────────────┤
│  Staging Buffer (pinned)     │  ← RDMA READ 目标
│  (一次性预取写入)              │    与 host pool 等大
└─────────────────────────────┘

请求进入 Decode 前（按请求粒度）:
  local  请求 → 无操作，KV 已在 host pool
  remote 请求 → RDMA READ: remote MR → NIC loopback → staging
                staging → 写回 host pool
                （一次性，之后与 local 请求行为一致）

Decode 每步（local/remote 一致）:
  HiSparse swap-in: host pool → GPU (仅 top-k miss)
```

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
  ├─ rdma_memory_pool.py（新增）
  ├─ rdma_scatter_read.c bulk read 扩展
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
| `rdma_memory_pool.py`（新增）       | ~260 行     | RDMA loopback host pool（local/remote/staging 分层） |
| `hisparse_coordinator.py`       | ~40 行      | 后端选择 + decode prefetch + cleanup           |
| `factory.py` + `SparseConfig`   | ~15 行      | rdma_pool JSON 解析 + 互斥校验                   |
| `model_runner.py`               | ~2 行       | 传递 rdma_config                            |
| `rdma_scatter_read.c/.h/_py.cpp` | ~60 行      | bulk read 扩展                               |
| `e2e_bench.py`（新增）              | ~350 行     | 两轮实验自动化，ShareGPT 多样化请求，TTFT/TBT/TPOT      |
| `run_all_experiments.sh`（新增）    | ~150 行     | 四方案 × 多 prefix 实验矩阵自动化                    |
| **Phase 3 合计**                  | **~880 行** |                                          |


### 3.5 代码总量


| Phase       | 代码量        | 累计      | 关键产出                                            |
| ----------- | ---------- | ------- | ----------------------------------------------- |
| **Phase 1** | **~150 行** | ~150 行  | HiSparse + radix cache 兼容，prefix-hit 跳过 prefill |
| Phase 2     | ~0 行       | ~150 行  | CXL 端到端验证                                       |
| Phase 3     | ~880 行     | ~1030 行 | RDMA 后端 + 多样化 e2e benchmark + 实验自动化              |
| Phase 4     | ~0 行       | ~1030 行 | 完整实验数据                                          |


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

---

## 5. 需要开发的代码

> 本节描述 RDMA/CXL 后端相关代码（Phase 3）。HiSparse + Prefix-Caching 兼容代码（Phase 1，~150 行）见[第 3 节](#3-分阶段开发计划)和[第 4 节](#4-hisparse--prefix-caching-设计方案)。

### 5.1 Phase 3 代码量（RDMA 后端 + Benchmark）


| 模块                 | 新代码量       | 修改/新增                           | 说明                                       |
| ------------------ | ---------- | ------------------------------- | ---------------------------------------- |
| **RDMA Host Pool** | ~260 行     | 新增 `rdma_memory_pool.py`        | 方案 B 的核心：RDMA loopback + 本地缓存分层          |
| **Coordinator 适配** | ~40 行      | 修改 `hisparse_coordinator.py`    | 根据配置选择 A/B/C 后端 + decode prefetch + cleanup |
| **Config 扩展**      | ~15 行      | 修改 `factory.py` + `SparseConfig` | rdma_pool JSON 解析 + 互斥校验                   |
| **ModelRunner**     | ~2 行       | 修改 `model_runner.py`           | 传递 rdma_config                            |
| **RDMA C 扩展**      | ~60 行      | 修改 `.c` + `.h` + `_py.cpp`     | rdma_bulk_read 接口                         |
| **Benchmark 脚本**   | ~350 行     | 新增 `e2e_bench.py`              | 两轮实验自动化，ShareGPT 多样化请求，TTFT/TBT/TPOT      |
| **自动化脚本**         | ~150 行     | 新增 `run_all_experiments.sh`     | 四方案 × 多 prefix 完整实验矩阵                    |
| **合计**             | **~880 行** | 3 新增 + 5 修改                    |                                          |


### 5.2 模块 1：RDMAMLATokenToKVPoolHost（~200 行）

**新文件**：`python/sglang/srt/mem_cache/rdma_memory_pool.py`

这是方案 B 的核心组件。继承 `MLATokenToKVPoolHost`，将 host pool 分为"本地缓存"+"远端 MR"两部分。在请求进入 decode 阶段之前，**一次性**通过 RDMA loopback 将该请求的远端 KV cache 全量预取到本地 staging buffer。

```python
"""RDMA distributed memory pool simulation for HiSparse KV Cache.

Simulates a distributed KV Cache pool where only a fraction of
KV data is cached locally in DRAM, and the rest resides on a
"remote" node and must be fetched via RDMA.

When a request transitions from prefill/staging to decode,
`prefetch_for_decode()` is called ONCE to bulk-transfer all
remote KV data into local staging buffer. After this one-time
prefetch, all subsequent decode steps read from local memory
(local_cache + staging_buffer) without further RDMA operations.

Uses real RDMA NIC loopback — the "remote" data is stored in a
local DRAM region registered as an RDMA MR, and fetched through
the NIC hardware (ibv_post_send RDMA READ → ibv_poll_cq).
This produces real hardware latency, which is a lower bound of
cross-machine RDMA latency.
"""

class RDMALoopbackContext:
    """Manages RDMA resources for loopback simulation.

    Creates:
      - ibv_context, pd, cq, qp (RC, connected to self)
      - remote_mr: registers the "remote" portion of KV Cache
      - staging_mr: the staging buffer for RDMA READ results

    Uses the same libibverbs pybind11 wrapper from
    test/cxl_utils/sparse_kv_bench.py (rdma_ext).
    """

    def __init__(self, ib_dev: str, remote_buf_ptr: int,
                 remote_buf_size: int, staging_buf_ptr: int):
        ...

    def bulk_read(self, size: int) -> float:
        """Execute RDMA READ of `size` bytes from remote_mr to staging_mr.

        Posts a single RDMA READ WR (or a few large ones if size > max_msg),
        polls CQ for completion. Returns elapsed time in microseconds.
        """
        ...


class RDMAMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA KV Cache host pool with RDMA distributed pool simulation.

    KV data is split into two regions:
      - local_cache: X% of tokens, pinned DRAM, directly accessible
      - remote_region: (1-X)% of tokens, RDMA MR, simulates remote node

    Lifecycle:
      1. Prefill: KV written to local_cache + remote_buffer
         (remote_buffer simulates "KV stored on remote node")
      2. Staging→Decode transition: `prefetch_for_decode()` called ONCE
         per request, bulk RDMA READ remote_buffer → staging_buffer
      3. All decode steps: HiSparse reads from unified local view
         = [local_cache | staging_buffer], no further RDMA operations

    This models the real PD-disaggregation scenario: decode instance
    fetches full KV from remote prefill instance once before decoding.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        rdma_config: dict,     # {"ib_dev": "mlx5_0", "local_ratio": 0.3}
        override_kv_cache_dim=None,
    ):
        self.local_ratio = rdma_config.get("local_ratio", 0.0)
        self.ib_dev = rdma_config.get("ib_dev", "mlx5_0")
        # ... init buffers, RDMA context ...
        super().__init__(...)

    def init_kv_buffer(self):
        """Allocate host pool + shadow remote buffer + staging buffer."""
        # host_pool: pinned DRAM, HiSparse 的统一视图
        self.kv_buffer = torch.empty(
            (self.layer_num, self.size, 1, self.kv_cache_dim),
            dtype=self.dtype, pin_memory=True)

        # remote_buffer: regular DRAM, registered as RDMA MR
        # 与 host_pool 等大，用于存放"远端请求"的 KV 副本
        self.remote_buffer = torch.empty_like(self.kv_buffer)
        # Register remote_buffer with RDMA NIC ...

        # staging_buffer: pinned DRAM, RDMA READ 目标
        self.staging_buffer = torch.empty_like(self.kv_buffer,
                                                pin_memory=True)

        # 跟踪哪些请求被标记为"远端"
        self._remote_reqs = set()
        return self.kv_buffer

    def mark_request_locality(self, req_pool_idx: int):
        """Staging 完成时调用，按 local_ratio 概率决定请求是 local 还是 remote."""
        if random.random() >= self.local_ratio:
            self._remote_reqs.add(req_pool_idx)

    def simulate_remote_write(self, req_pool_idx: int, seq_len: int):
        """将 remote 请求的 KV 从 host_pool 拷贝到 remote_buffer（模拟 KV 写入远端）."""
        if req_pool_idx not in self._remote_reqs:
            return
        host_indices = self.req_to_host_pool[req_pool_idx, :seq_len]
        # kv_buffer[:, host_indices] → remote_buffer[:, host_indices]
        self.remote_buffer[:, host_indices] = self.kv_buffer[:, host_indices]

    def prefetch_for_decode(self, req_pool_idx: int, seq_len: int):
        """One-time RDMA READ: fetch remote request's full KV into local.

        Called ONCE per request when transitioning to decode.
        Local requests skip this entirely (no RDMA overhead).
        Returns: RDMA transfer time in microseconds (for logging).
        """
        if req_pool_idx not in self._remote_reqs:
            return 0.0  # local request, no RDMA needed
        host_indices = self.req_to_host_pool[req_pool_idx, :seq_len]
        num_tokens = len(host_indices)
        remote_bytes = (num_tokens * self.layer_num
                        * self.kv_cache_dim * self.dtype_size)
        rdma_us = self.rdma_ctx.bulk_read(remote_bytes)
        # RDMA READ: remote_buffer → staging_buffer, then copy back
        self.kv_buffer[:, host_indices] = self.staging_buffer[:, host_indices]
        return rdma_us
```

**设计要点**：

1. **按请求划分 local/remote**：`local_ratio` 控制的是**请求级别**的本地命中率——每个请求的全部 KV 要么全在本地，要么全在远端。`mark_request_locality()` 在 staging 完成时以 `local_ratio` 概率将请求标记为 local，其余标记为 remote。这符合真实分布式场景：同一请求的 KV 在同一个节点上
2. **HiSparse 视角统一**：`init_kv_buffer()` 返回统一的 pinned DRAM host pool，HiSparse 的 staging DMA 和 decode swap-in 正常工作，无需感知 local/remote 区分
3. **Remote 模拟**：remote 请求的 KV 在 staging 完成后拷贝到 `remote_buffer`（RDMA MR），模拟"KV 存储在远端节点"
4. **一次性 RDMA 预取**：remote 请求进入 decode 前，`prefetch_for_decode()` 调用**一次**，RDMA READ 全量 KV 回本地。Local 请求直接跳过，零 RDMA 开销
5. **本地比例可配**：`local_ratio=0.0`（全部远端，每个请求都需 RDMA）→ `local_ratio=1.0`（全部本地，等同方案 A）

### 5.3 模块 2：Coordinator 适配（~20 行）

**修改文件**：`python/sglang/srt/managers/hisparse_coordinator.py`

在 Coordinator 的初始化中根据配置选择 host pool 后端，并在请求进入 decode 之前一次性 RDMA 预取：

```python
# __init__ 中：
if hisparse_config.get("rdma_pool"):
    from sglang.srt.mem_cache.rdma_memory_pool import RDMAMLATokenToKVPoolHost
    self.mem_pool_host = RDMAMLATokenToKVPoolHost(
        device_pool=self.mem_pool_device,
        rdma_config=hisparse_config["rdma_pool"],
        ...
    )
elif hisparse_config.get("cxl", {}).get("enabled"):
    # 现有 CXL 路径
    ...
else:
    # 现有 DRAM 路径
    ...

# collect_ready_reqs 中，staging 完成、进入 decode 之前：
def collect_ready_reqs(self) -> List[Req]:
    # ... (原有 staging 完成检查) ...
    for req in newly_ready:
        seq_len = len(req.fill_ids)
        if hasattr(self.mem_pool_host, 'mark_request_locality'):
            # 1. 按 local_ratio 概率标记请求为 local/remote
            self.mem_pool_host.mark_request_locality(req.req_pool_idx)
            # 2. remote 请求：将 host_pool KV 拷贝到 remote_buffer
            self.mem_pool_host.simulate_remote_write(req.req_pool_idx, seq_len)
            # 3. remote 请求：一次性 RDMA READ 拉回本地
            self.mem_pool_host.prefetch_for_decode(req.req_pool_idx, seq_len)
        self.alloc_device_buffer(req)
        # ...
```

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

## 6. RDMA Loopback 实现细节

### 6.1 复用已有 RDMA 基础设施

`test/cxl_utils/sparse_kv_bench.py` 中已有 RDMA loopback 的 pybind11 模块（`rdma_ext`）。`RDMAMLATokenToKVPoolHost` 复用同一套 libibverbs 封装，仅扩展为支持"bulk read"（大块连续读取，而非逐条目离散读取）。

需要扩展的接口：

```c
// rdma_scatter_read.c 新增
double rdma_bulk_read(struct rdma_context *rctx, size_t total_bytes) {
    // 将 total_bytes 拆分为多个 max_msg_size 的 WR
    // 批量 ibv_post_send → poll CQ → 返回总延迟
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    size_t offset = 0;
    int wr_count = 0;
    while (offset < total_bytes) {
        size_t chunk = (total_bytes - offset > MAX_MSG_SIZE)
                       ? MAX_MSG_SIZE : (total_bytes - offset);
        struct ibv_sge sge = {
            .addr = (uint64_t)rctx->local_mr->addr + offset,
            .length = chunk,
            .lkey = rctx->local_mr->lkey,
        };
        struct ibv_send_wr wr = {
            .sg_list = &sge,
            .num_sge = 1,
            .opcode = IBV_WR_RDMA_READ,
            .send_flags = 0,
            .wr.rdma = {
                .remote_addr = rctx->remote_addr + offset,
                .rkey = rctx->remote_rkey,
            },
        };
        offset += chunk;
        wr_count++;
        // Signal last WR
        if (offset >= total_bytes)
            wr.send_flags = IBV_SEND_SIGNALED;
        struct ibv_send_wr *bad_wr;
        ibv_post_send(rctx->qp, &wr, &bad_wr);
    }

    // Poll for completion
    struct ibv_wc wc;
    while (ibv_poll_cq(rctx->cq, 1, &wc) == 0) {}

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
}
```

### 6.2 本地缓存比例的实现方式

`local_ratio` 控制的是**请求级别**的本地命中率，而非 token 级别的内存划分。同一个请求的全部 KV 要么全在本地 DRAM，要么全在远端——这符合真实分布式内存池的语义（同一请求的 KV 存储在同一个节点上）。

```
请求到达，staging 完成时：
  以 local_ratio 概率 → 标记为 local（KV 已在本地 DRAM，无需 RDMA）
  以 (1-local_ratio) 概率 → 标记为 remote

Local 请求:
  staging → host_pool                      （直接可用）
  decode: swap-in 从 host_pool 读取         （本地 DRAM 访问）

Remote 请求:
  staging → host_pool → 拷贝到 remote_buffer （模拟"KV 写入远端"）
  进入 decode 前（一次性）:
    RDMA READ: remote_buffer → staging_buffer → 写回 host_pool
  decode: swap-in 从 host_pool 读取          （已拉回本地）
```

**实验中的配置**：
- `local_ratio=0.0`（B-0）：所有请求都需 RDMA 预取，最大化暴露 RDMA 开销
- `local_ratio=0.3`（B-30）：30% 请求本地命中，70% 需 RDMA，模拟部分缓存命中
- `local_ratio=1.0`（B-100）：等同方案 A（全部本地），作为 RDMA 方案的上界基线

### 6.3 RDMA 预取的触发时机

`prefetch_for_decode()` 在请求从 staging 完成进入 decode 阶段时**调用一次**，而非每步 decode 重复调用：

```python
# hisparse_coordinator.py — collect_ready_reqs（简化）:

def collect_ready_reqs(self) -> List[Req]:
    ready_reqs = []
    for req in staging_completed_reqs:
        seq_len = len(req.fill_ids)
        if hasattr(self.mem_pool_host, 'mark_request_locality'):
            # ★ 按 local_ratio 标记 local/remote
            self.mem_pool_host.mark_request_locality(req.req_pool_idx)
            # ★ remote 请求：host_pool → remote_buffer（模拟远端写入）
            self.mem_pool_host.simulate_remote_write(req.req_pool_idx, seq_len)
            # ★ remote 请求：RDMA READ remote → local（一次性预取）
            #   local 请求：直接跳过，零开销
            self.mem_pool_host.prefetch_for_decode(req.req_pool_idx, seq_len)

        self.alloc_device_buffer(req)
        ready_reqs.append(req)
    return ready_reqs

# 之后的每步 decode（不涉及 RDMA，local/remote 请求行为一致）:
def _process_decode_step(self, batch):
    self._eager_backup_previous_token(batch)
    for layer_id in range(self.num_layers):
        self.swap_in_selected_pages(layer_id, ...)
        # swap-in 从 host_pool 读取，全部是本地 DRAM 操作
```

**关键**：
- **按请求划分**：同一请求的全部 KV 要么全在本地、要么全需 RDMA，符合真实分布式语义
- **一次性开销**：remote 请求的 RDMA READ 在进入 decode 前完成一次，之后所有 decode step 都是本地内存操作
- **开销体现**：RDMA 的代价是**请求级的一次性延迟**（影响该请求的首步 decode / TTFT），local 请求无任何额外开销

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
