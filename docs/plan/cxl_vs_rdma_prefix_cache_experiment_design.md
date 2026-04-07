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
│   │ (100%    │  │ │本地 DRAM  │ │  │ 全量 KV   │                  │
│   │  本地)    │  │ │(X% 缓存) │ │  │ 在 CXL   │                  │
│   │          │  │ ├──────────┤ │  │          │                  │
│   │          │  │ │"远端"DRAM │ │  │          │                  │
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

方案 B 模拟"分布式内存池中，本地 DRAM 缓存了 X% 的 KV Cache"的场景：

```
本机 DRAM 布局（方案 B）:

┌─────────────────────────────┐
│  模型权重 + 其他                │
├─────────────────────────────┤
│  本地 KV Cache 缓存            │  ← X% 的 KV，pinned DRAM
│  (直接可用，无需 RDMA)          │    配置: rdma_local_ratio
├─────────────────────────────┤
│  "远端" KV Cache              │  ← (1-X%) 的 KV，注册为 RDMA MR
│  (ibv_reg_mr, 模拟远端节点)    │    decode 前需 RDMA READ 到 staging
├─────────────────────────────┤
│  Staging Buffer              │  ← RDMA READ 目标，pinned DRAM
│  (每步 decode 前写入)          │    大小 = "远端"部分大小
└─────────────────────────────┘

每步 Decode 数据流:
  1. RDMA READ: "远端" MR → NIC loopback → staging buffer
  2. 合并: staging + 本地缓存 → 完整 KV host pool view
  3. HiSparse swap-in: host pool → GPU (仅 top-k miss)
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


| 文件                        | 改动量        | 说明                                               |
| ------------------------- | ---------- | ------------------------------------------------ |
| `rdma_memory_pool.py`（新增） | ~200 行     | RDMA loopback host pool（local/remote/staging 分层） |
| `hisparse_coordinator.py` | ~20 行      | 后端选择 + decode prefetch 调用                        |
| `rdma_scatter_read.c`     | ~40 行      | bulk read 扩展                                     |
| `e2e_bench.py`（新增）        | ~150 行     | 两轮实验自动化                                          |
| **Phase 3 合计**            | **~410 行** |                                                  |


### 3.5 代码总量


| Phase       | 代码量        | 累计     | 关键产出                                            |
| ----------- | ---------- | ------ | ----------------------------------------------- |
| **Phase 1** | **~150 行** | ~150 行 | HiSparse + radix cache 兼容，prefix-hit 跳过 prefill |
| Phase 2     | ~0 行       | ~150 行 | CXL 端到端验证                                       |
| Phase 3     | ~410 行     | ~560 行 | RDMA 后端 + 实验工具链                                 |
| Phase 4     | ~0 行       | ~560 行 | 完整实验数据                                          |


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


| 模块                 | 新代码量       | 修改/新增                        | 说明                                |
| ------------------ | ---------- | ---------------------------- | --------------------------------- |
| **RDMA Host Pool** | ~200 行     | 新增 `rdma_memory_pool.py`     | 方案 B 的核心：RDMA loopback + 本地缓存分层   |
| **Coordinator 适配** | ~20 行      | 修改 `hisparse_coordinator.py` | 根据配置选择 A/B/C 后端 + decode prefetch |
| **Benchmark 脚本**   | ~150 行     | 新增 `e2e_bench.py`            | 两轮实验自动化                           |
| **RDMA C 扩展**      | ~40 行      | 修改 `rdma_scatter_read.c`     | bulk read 接口                      |
| **合计**             | **~410 行** | 2 新增 + 2 修改                  |                                   |


### 5.2 模块 1：RDMAMLATokenToKVPoolHost（~200 行）

**新文件**：`python/sglang/srt/mem_cache/rdma_memory_pool.py`

这是方案 B 的核心组件。继承 `MLATokenToKVPoolHost`，将 host pool 分为"本地缓存"+"远端 MR"两部分，并在每步 decode 前执行 RDMA loopback 全量预取。

```python
"""RDMA distributed memory pool simulation for HiSparse KV Cache.

Simulates a distributed KV Cache pool where only a fraction of
KV data is cached locally in DRAM, and the rest must be fetched
via RDMA before each decode step.

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
      - remote_region: (1-X)% of tokens, RDMA MR, must be RDMA-READ
        to staging_buffer before HiSparse can read them

    Before each decode step, `prefetch_for_decode()` is called to
    RDMA READ the remote portion into the staging buffer.

    After prefetch, HiSparse sees a unified host pool view:
      - tokens in local_cache → read from local DRAM
      - tokens in remote_region → read from staging buffer
      (implemented by updating data_ptrs to point to the right buffer)
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
        """Allocate KV buffer split into local + remote portions."""
        total_dims = (self.layer_num, self.size, 1, self.kv_cache_dim)

        # Split token slots: first X% → local, rest → remote
        local_size = int(self.size * self.local_ratio)
        remote_size = self.size - local_size

        # local_buffer: pinned DRAM
        self.local_buffer = torch.empty(
            (self.layer_num, local_size, 1, self.kv_cache_dim),
            dtype=self.dtype, pin_memory=True)

        # remote_buffer: regular DRAM, registered as RDMA MR
        self.remote_buffer = torch.empty(
            (self.layer_num, remote_size, 1, self.kv_cache_dim),
            dtype=self.dtype)
        # Register remote_buffer with RDMA NIC
        # ...

        # staging_buffer: pinned DRAM, target for RDMA READ
        self.staging_buffer = torch.empty_like(self.remote_buffer,
                                                pin_memory=True)

        # Combine into unified view for HiSparse
        full_buffer = torch.cat([self.local_buffer, self.staging_buffer], dim=1)
        return full_buffer

    def prefetch_for_decode(self, req_token_indices: torch.Tensor):
        """RDMA READ remote KV data into staging buffer.

        Called before each decode step. This is the critical path
        that adds RDMA overhead — the full remote portion is fetched
        regardless of which tokens will actually be selected by top-k.

        Returns: RDMA transfer time in microseconds (for logging).
        """
        remote_bytes = self.remote_buffer.numel() * self.remote_buffer.element_size()
        rdma_us = self.rdma_ctx.bulk_read(remote_bytes)
        # Copy remote data to staging (RDMA READ already did this)
        # Now staging_buffer contains fresh "remote" KV data
        return rdma_us
```

**设计要点**：

1. **KV buffer 物理分离**：local_buffer（直接可用）+ remote_buffer（RDMA MR）+ staging_buffer（RDMA READ 目标）
2. **HiSparse 视角统一**：`init_kv_buffer()` 返回 `cat([local_buffer, staging_buffer])`，HiSparse 看到的是一个连续的 host pool，`data_ptrs` 指向这个统一视图
3. **Prefill 写入**：prefill 时正常写入 full_buffer（= local + staging），然后将 staging 部分拷贝到 remote_buffer（模拟"KV 写入远端"）
4. **Decode 预取**：每步 decode 前调用 `prefetch_for_decode()`，RDMA READ remote → staging，更新统一视图
5. **本地比例可配**：`local_ratio` 控制本地缓存多少 KV，0% = 全部远端，100% = 全部本地

### 5.3 模块 2：Coordinator 适配（~20 行）

**修改文件**：`python/sglang/srt/managers/hisparse_coordinator.py`

在 Coordinator 的初始化中根据配置选择 host pool 后端，并在 decode 步开始时调用 RDMA 预取：

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

# decode 步回调中（如已有 _eager_backup_previous_token 同级别位置）：
def _on_decode_step(self, batch):
    if hasattr(self.mem_pool_host, 'prefetch_for_decode'):
        self.mem_pool_host.prefetch_for_decode(batch.req_pool_indices)
```

### 5.4 模块 3：端到端 Benchmark 脚本（~150 行）

**新文件**：`test/cxl_utils/e2e_bench.py`

自动化两轮实验，向运行中的 SGLang 服务器发送请求并收集指标。

```python
"""Two-round end-to-end benchmark: cold start + prefix cache hit.

Round 1 (Cold):
  Send N requests with shared prefix → full prefill → KV stored
  Measure: TTFT, Decode TBT, Total Throughput

Round 2 (Warm):
  Send M new requests with same prefix → prefix cache hit
  Measure: TTFT (should drop), Decode TBT, Max Concurrency

Usage:
  python e2e_bench.py \
    --server-url http://localhost:30000 \
    --prefix-tokens 32768 \
    --query-tokens 256 \
    --output-tokens 256 \
    --round1-requests 8 \
    --round2-requests 32 \
    --concurrency 1,2,4,8,16 \
    --output results.json
"""

import argparse, json, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


def generate_prefix(num_tokens: int) -> str:
    """Generate a synthetic prefix of approximately num_tokens tokens."""
    # Use a fixed seed for reproducibility
    # Each English word ≈ 1.3 tokens, pad to target
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
             "dog", "and", "runs", "across", "the", "vast", "open", "field"]
    text_words = []
    for i in range(num_tokens * 2):  # over-generate, will be trimmed by server
        text_words.append(words[i % len(words)])
    return " ".join(text_words)


def make_query(prefix: str, query_id: int) -> str:
    """Append a unique query to the shared prefix."""
    return prefix + f"\n\nQuestion {query_id}: " \
           f"Based on the text above, provide a detailed analysis #{query_id}."


def send_and_measure(url: str, prompt: str, max_new_tokens: int) -> dict:
    """Send generation request, return timing info."""
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    }
    t0 = time.monotonic()
    resp = requests.post(f"{url}/generate", json=payload, timeout=600)
    t1 = time.monotonic()
    meta = resp.json().get("meta_info", {})
    return {
        "wall_time_s": t1 - t0,
        "prompt_tokens": meta.get("prompt_tokens", 0),
        "completion_tokens": meta.get("completion_tokens", 0),
    }


def run_round(url, prefix, query_ids, max_new_tokens, concurrency):
    """Run one round of benchmark with given concurrency."""
    prompts = [make_query(prefix, qid) for qid in query_ids]
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_and_measure, url, p, max_new_tokens)
                   for p in prompts]
        for f in as_completed(futures):
            results.append(f.result())

    times = np.array([r["wall_time_s"] for r in results])
    tokens = np.array([r["completion_tokens"] for r in results])
    total_tokens = int(tokens.sum())
    wall_time = float(times.max())

    return {
        "num_requests": len(results),
        "concurrency": concurrency,
        "total_output_tokens": total_tokens,
        "wall_time_s": round(wall_time, 3),
        "throughput_tok_per_s": round(total_tokens / wall_time, 1),
        "avg_latency_ms": round(float(times.mean()) * 1000, 1),
        "p50_latency_ms": round(float(np.percentile(times, 50)) * 1000, 1),
        "p99_latency_ms": round(float(np.percentile(times, 99)) * 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--prefix-tokens", type=int, default=32768)
    parser.add_argument("--query-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--round1-requests", type=int, default=8)
    parser.add_argument("--round2-requests", type=int, default=32)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    prefix = generate_prefix(args.prefix_tokens)
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    all_results = {"round1": [], "round2": []}

    # Round 1: Cold start
    print("=" * 60)
    print("Round 1: Cold start (no prefix cache)")
    print("=" * 60)
    for conc in concurrencies:
        n = min(args.round1_requests, conc * 4)
        query_ids = list(range(n))
        r = run_round(args.server_url, prefix, query_ids,
                      args.output_tokens, conc)
        all_results["round1"].append(r)
        print(f"  conc={conc}: {r['throughput_tok_per_s']} tok/s, "
              f"p50={r['p50_latency_ms']}ms, p99={r['p99_latency_ms']}ms")

    # Brief pause to let KV settle in cache
    time.sleep(2)

    # Round 2: Prefix cache hit
    print("\n" + "=" * 60)
    print("Round 2: Warm start (prefix cache hit)")
    print("=" * 60)
    for conc in concurrencies:
        n = min(args.round2_requests, conc * 4)
        # Use different query IDs to avoid exact match
        query_ids = list(range(1000, 1000 + n))
        r = run_round(args.server_url, prefix, query_ids,
                      args.output_tokens, conc)
        all_results["round2"].append(r)
        print(f"  conc={conc}: {r['throughput_tok_per_s']} tok/s, "
              f"p50={r['p50_latency_ms']}ms, p99={r['p99_latency_ms']}ms")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
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

KV Cache 的 token slot 按比例划分为"本地"和"远端"：

```
total_slots = max_total_num_tokens × host_to_device_ratio

local_slots  = int(total_slots × local_ratio)
remote_slots = total_slots - local_slots

Prefill 写入:
  token 0 ~ local_slots-1     → 写入 local_buffer (pinned DRAM)
  token local_slots ~ total-1 → 写入 remote_buffer (RDMA MR)
                                + 拷贝到 staging_buffer 初始化

Decode 每步:
  1. RDMA READ: remote_buffer → staging_buffer (全量, 模拟远端预取)
  2. HiSparse 从 unified_view = [local_buffer | staging_buffer] 读 top-k
```

**token 分配策略**：token 按 slot index 自然落入 local 或 remote 区域。由于 NSA 的 top-k 选择是 query-aware 的随机模式，本地 vs 远端的 token 被选中的概率与 `local_ratio` 成正比——这自然模拟了"热数据在本地、冷数据在远端"的分布式缓存语义。

### 6.3 RDMA 预取的触发时机

在 HiSparse 的 decode 路径中，`prefetch_for_decode()` 在以下位置调用：

```python
# hisparse_coordinator.py 中的 decode 步流程（简化）:

def _process_decode_step(self, batch):
    # 1. 备份上一步新生成的 token KV 到 host
    self._eager_backup_previous_token(batch)

    # 2. [新增] RDMA 预取（仅方案 B）
    if hasattr(self.mem_pool_host, 'prefetch_for_decode'):
        self.mem_pool_host.prefetch_for_decode(batch)

    # 3. 逐层执行 sparse attention
    for layer_id in range(self.num_layers):
        self.swap_in_selected_pages(layer_id, ...)
        # attention computation ...
```

**关键**：RDMA 预取在所有层的 swap-in 之前一次性完成（全量预取），这正是 RDMA 方案的标准做法——因为 RDMA 无法逐层按需取。

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

### 8.1 完整流程

```bash
# 对每种方案 (A / B-0 / B-30 / C)：

# 1. 启动服务器（方案 X 配置）
python -m sglang.launch_server ... &
wait_for_server_ready

# 2. 运行两轮 benchmark
python test/cxl_utils/e2e_bench.py \
    --server-url http://localhost:30000 \
    --prefix-tokens 32768 \
    --output-tokens 256 \
    --round1-requests 16 \
    --round2-requests 64 \
    --concurrency 1,2,4,8,16 \
    --output results_${scheme}.json

# 3. 收集系统指标
free -h > mem_${scheme}.txt
nvidia-smi > gpu_${scheme}.txt

# 4. 关闭服务器，切换方案
```

### 8.2 自动化脚本

```bash
#!/bin/bash
# run_all_experiments.sh

SCHEMES=("dram" "rdma_0" "rdma_30" "cxl")
CONFIGS=(
    '{"top_k":2048,"device_buffer_size":4096}'
    '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.0}}'
    '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.3}}'
    '{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_path":"/dev/dax0.0","map_bytes":68719476736}}'
)

for i in "${!SCHEMES[@]}"; do
    scheme="${SCHEMES[$i]}"
    config="${CONFIGS[$i]}"
    echo "=== Running scheme: $scheme ==="

    python -m sglang.launch_server \
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
        --enable-hisparse --hisparse-config "$config" \
        --port 30000 &
    SERVER_PID=$!
    sleep 120  # wait for model loading

    for prefix_len in 4096 16384 32768 65536; do
        python test/cxl_utils/e2e_bench.py \
            --server-url http://localhost:30000 \
            --prefix-tokens $prefix_len \
            --output-tokens 256 \
            --concurrency 1,2,4,8,16 \
            --output "results/${scheme}_prefix${prefix_len}.json"
    done

    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    sleep 10
done
```

---
