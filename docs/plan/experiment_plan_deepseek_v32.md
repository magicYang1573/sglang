# **DeepSeek-V3.2 原生稀疏注意力 × CXL 内存池化实验方案**

---

## 1. 动机

### 1.1 长上下文推理与 KV Cache 池化

长上下文 LLM 推理（128K–1M+ tokens）产生巨大的 KV Cache，单 GPU 显存放不下，池化到远端内存是必然趋势。现有 RDMA 池化方案（Mooncake、LMCache 等）的标准模式是：在 decode 开始前，将该请求所需的 KV Cache **全量预取**到本地 DRAM，之后 attention 计算从本地 DRAM 读取。对于 dense attention 模型，这是合理的——取来的 KV Cache 100% 都会被使用。

### 1.2 RDMA 全量预取在 Sparse Attention 下的根本矛盾

随着稀疏注意力的兴起（DeepSeek-V3.2 NSA 等），每层 attention 只选取 top-k（1–10%）KV Cache 条目参与计算。如果仍然全量预取，产生两个根本矛盾：

- **矛盾一：本地 DRAM 空间浪费**。全量预取意味着 90%+ 的 KV Cache 取来后不会被使用，白白占用本地 DRAM → 可并发请求数受限 → 吞吐无法提升。Sparse attention 在计算侧节省了复杂度，但在内存侧的收益被全量预取完全抵消。
- **矛盾二：网络带宽浪费**。RDMA 带宽被大量无效数据传输占满，在超长上下文场景下，KV Cache 迁移成为系统瓶颈。

**为什么不能只预取 top-k？** top-k 选取是 query-aware 的，与当前层的 Q 向量高度相关，具有极强的实时性（每层都不同），无法在 decode 开始前预知。RDMA 的消息语义（显式 send/recv、μs 级 per-message 开销、适合大块批量传输）无法满足这种逐层、实时、细粒度的按需读取需求。RDMA 逐层按需取 top-k 的延迟为 per-message 开销（μs 级）× 每层一次 × 模型层数（60–100+层），累积延迟不可接受。

因此，在 sparse attention 场景下，**全量预取不经济，部分预取 RDMA 又无法胜任**。

### 1.3 CXL 是唯一合理选择

CXL 的 load/store 语义、cache-line 粒度访问、低 per-access 开销，天然适配 sparse top-k 的按需实时读取：

- 无需全量预取 → 本地 DRAM 只保留实际使用的 top-k → 内存效率大幅提升 → 可并发更多请求 → 吞吐释放
- 无需提前预知 top-k → 计算到哪层取到哪层，CXL 实时响应
- 层间访问相似性是已有观察，可利用它在 VRAM 中做 buffer 来进一步优化 CXL 取数量，但这不是本文的主要创新点

### 1.4 为什么在 DeepSeek-V3.2 上实验

DeepSeek-V3.2 是当前最具代表性的**原生稀疏注意力**大模型（NSA + MLA），其 indexer 产生的逐层 query-aware top-k 是真实训练出的稀疏模式。在该模型上用真实 CXL 和 RDMA 硬件实测，是最直接的实验证据。

SGLang 已合并的 **HiSparse**（PR#20343）恰好实现了 DeepSeek-V3.2 NSA 的 KV Cache host↔device 交换框架——只需将其 host 内存后端替换为 CXL 或 RDMA 内存池，即可在不改变 sparse attention 逻辑的前提下对比不同内存方案。

---

## 2. 实验目标

1. 在 DeepSeek-V3.2 上，端到端实测 **CXL 内存池** vs. **Mooncake RDMA 内存池（全量预取）** vs. **Local DRAM** 的吞吐和延迟
2. 通过 micro-benchmark 精确测量三种内存后端在 sparse KV 访问模式下的读取延迟和带宽
3. 验证核心论点：sparse attention 下 CXL 按需读取 top-k 显著优于 RDMA 全量预取

---

## 3. 硬件要求


| 组件       | 要求                                    | 说明                                  |
| -------- | ------------------------------------- | ----------------------------------- |
| GPU      | 8 × H20 96GB（或更高规格的 8 卡机器）            | DeepSeek-V3.2 为 671B MoE，TP=8 可完整加载 |
| CXL 设备   | CXL 2.0 type-3 内存扩展 ≥64GB             | 存放 offload 的 KV Cache               |
| RDMA NIC | ConnectX-5/6/7, 100/200Gbps IB 或 RoCE | 本机 loopback 模拟分布式内存池              |
| 本地 DRAM  | ≥512GB                                | 模型权重 + HiSparse host pool + OS 开销   |

> **注意**：所有实验均在**单机**完成，不需要远端节点。RDMA 方案使用本机 NIC loopback 模拟分布式内存池的全量预取开销。


---

## 4. 设计方案

### 4.1 总体架构

```
┌──────────────────────────────────────────────────────────┐
│                     DeepSeek-V3.2                         │
│              NSA Indexer → top-k indices                  │
│              NSA Attention (sparse MLA)                   │
└─────────────────────┬────────────────────────────────────┘
                      │
              HiSparse Coordinator
              (swap_in_selected_pages)
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ 方案 A   │  │ 方案 B   │  │ 方案 C   │
    │ Local    │  │ Mooncake │  │  CXL     │
    │ DRAM     │  │ RDMA Pool│  │  Memory  │
    │          │  │ (全量预取)│  │  Pool    │
    └─────────┘  └─────────┘  └─────────┘
```

三种方案共享同一条 sparse attention 路径（NSA indexer → top-k → HiSparse swap-in → MLA kernel），**唯一差异是 KV Cache 的 host 内存来源和搬运方式**。这保证了实验的公平性——稀疏计算逻辑完全一致，只有数据搬运路径不同。

### 4.2 方案 A：Local DRAM（上限基线）

HiSparse 的默认模式。Prefill 后全量 KV 备份到 CPU pinned memory（`cudaHostRegister`），decode 时 HiSparse JIT kernel 按 top-k 从 host 拷贝 miss 条目到 device buffer。

- **数据路径**：GPU ↔ CPU DRAM（PCIe DMA）
- **定位**：性能上限基线——本地 DRAM 无网络开销，延迟最低，但容量受限于单机物理内存，可支撑的并发请求数有限

### 4.3 方案 B：RDMA 分布式内存池（本机 Loopback 模拟）

这是当前 RDMA 池化系统（Mooncake、LMCache 等）的标准做法，也是本文要批判的对象。

**模拟场景**：KV Cache 存储在"远端"分布式内存池中，本地 DRAM 仅缓存一部分。每步 decode 开始前，需要通过 RDMA 将不在本地的 KV Cache 全量预取到本地 DRAM staging area，然后 HiSparse 从本地 staging area 按 top-k 加载到 GPU。

**单机模拟方式**：在同一台机器上，将 KV Cache 分为两部分：

- **本地缓存部分**：存在 pinned DRAM 中，可直接访问（模拟分布式内存池中本地节点缓存的热数据）
- **"远端"部分**：也存在本机 DRAM 中，但 decode 前需通过 RDMA NIC loopback 读取到 staging buffer（模拟从远端节点的全量预取）

通过调节"本地缓存比例"（0%–100%），可以模拟不同的分布式内存池配置。

- **数据路径**：本机 DRAM →（RDMA loopback）→ 本地 staging DRAM →（PCIe DMA）→ GPU
- **问题**：预取了 100% 数据，attention 只用了 1–10%。带宽和本地 DRAM 均被浪费

> RDMA 是消息语义，每次传输都有 μs 级的 per-message 协议开销。DeepSeek-V3.2 有 61 层 transformer，如果每层都发一次 RDMA 请求取 top-k，仅协议开销就是 61 × 数μs = 数百μs，远超 sparse attention 本身的计算时间。因此 RDMA 只能做全量预取，逐层按需取不可行。
>
> **关于 loopback 的有效性**：本机 NIC loopback 下 per-message 开销（~1-2 μs）与跨机场景量级一致（仅少了线缆传播延迟 ~0.5-1 μs）。因此 loopback 结果是 RDMA 性能的**下界**，跨机场景延迟只会更高。

### 4.4 方案 C：CXL 共享内存池

将 HiSparse 的 host pool 分配到 CXL 扩展内存。CXL 内存通过 mmap（DAX 设备 / NUMA node）映射到进程地址空间，再 `cudaHostRegister` 使 GPU 可直接 DMA 访问。

- **数据路径**：GPU ↔ CXL Memory（PCIe/CXL DMA，load/store 语义）
- HiSparse JIT kernel 的 host→device 操作无需修改——它只通过 host pointer 读取数据，不关心底层是 DRAM 还是 CXL
- **仅传输 top-k**：每层 miss 的条目直接从 CXL DMA 到 device buffer，无全量预取，无 RDMA per-message 开销
- **本地 DRAM 几乎不占**：KV Cache 全部在 CXL 中，本地 DRAM 释放给模型权重和更多并发请求

### 4.5 三种方案的数据流对比


| 环节               | A: DRAM                 | B: RDMA 分布式池（loopback 模拟） | C: CXL                 |
| ---------------- | ----------------------- | ------------------------------ | ---------------------- |
| Prefill 后 backup | GPU → DRAM              | GPU → DRAM（本地+"远端"两部分）        | GPU → CXL              |
| Decode 每步预取      | 无（数据已在本地）               | "远端"部分 → RDMA loopback → staging（**全量**） | 无（CXL 按需访问）     |
| 每层 swap-in       | DRAM → GPU (top-k miss) | staging DRAM → GPU (top-k miss) | CXL → GPU (top-k miss) |
| 每步总传输量           | top-k × layers          | **全量 KV** + top-k × layers     | top-k × layers         |
| 本地 DRAM 占用       | 全量 KV                   | 本地缓存部分 + staging               | ≈ 0（KV 在 CXL）          |
| 有效带宽利用率          | 100%                    | **1–10%**                      | ≈ 100%                 |


---

## 5. 关键设计决策

### 5.1 复用 HiSparse，仅替换内存后端

HiSparse 已实现完整的 NSA sparse KV offloading 栈（coordinator、JIT CUDA kernel、device/host pool 映射、LRU buffer、CUDA Graph 支持）。不重写这些逻辑，方案 C 只需替换 host pool 的内存分配器，方案 B 需在 decode 路径中插入 Mooncake 全量预取步骤。

### 5.2 CXL 内存的高效管理策略

CXL 内存虽然对 CPU 呈现为普通可寻址空间，但其延迟和带宽特性与本地 DRAM 不同。如何高效管理 sparse KV Cache 在 CXL 内存上的布局，是方案 C 需要探索的核心问题。主要考量：

**Interleave 与 NUMA 策略**

CXL 2.0 type-3 设备通常作为独立的 NUMA node 暴露给 OS。当系统中存在**多块 CXL 内存卡**时，每块卡对应一个 NUMA node，可以通过 interleave 策略将 KV Cache 分散到多块卡上，**最大化聚合上行带宽**——单块 CXL 卡的带宽可能成为瓶颈，但多块卡并行读取时总带宽线性叠加。

需要探索的分配策略：

- **单卡 CXL-only**：KV Cache 全部分配到一块 CXL 卡，本地 DRAM 完全释放给模型权重和其他用途。简单但带宽受限于单卡。
- **多卡 CXL interleave**：通过 `numactl --interleave=cxl_node0,cxl_node1,...` 将 KV Cache 页面交错分配到多块 CXL 卡上。sparse top-k 的离散访问天然分散到多卡，聚合带宽随卡数增长。这是最大化 CXL 上行带宽的关键策略。
- **热冷分层**：利用 HiSparse 的 LRU 统计信息，将高频访问的 KV 条目迁移到 VRAM buffer，低频条目留在 CXL。这与 HiSparse 已有的 device buffer LRU 机制天然配合。

实验中需对比上述策略，尤其关注多卡 CXL interleave 在不同卡数（1/2/4）下的带宽扩展性。

**页面大小与 TLB 优化**

CXL 内存的 mmap 通常通过 DAX 设备完成，支持大页（2MB / 1GB huge pages）。sparse top-k 的离散访问模式会导致大量 TLB miss，使用大页可以显著减少 TLB 压力：

- 默认 4KB 页：TLB 覆盖范围小，离散 KV 访问容易 TLB miss
- 2MB 大页：TLB 覆盖范围扩大 512 倍，适合 KV Cache 的大块连续分配
- 1GB 大页：进一步减少 TLB miss，但粒度过粗可能浪费

实验中需测量不同页面大小对 CXL KV 读取延迟的影响。

**KV Cache 布局优化**

HiSparse 的 host pool 使用 `layer_first` 或 `page_first` 布局。不同布局在 CXL 上的表现可能不同：

- `layer_first`：同一层所有 token 的 KV 连续存放。sparse top-k 访问同一层的多个离散 token，空间局部性较差但 layer 内预取友好。
- `page_first`：同一 token 所有层的 KV 连续存放。层间相似性高时，访问相邻层的同一 token 可利用 cache-line 预取。

需要在 CXL 上实测两种布局的性能差异。

---

## 6. 实验设计

### 6.1 Part 1：Micro-benchmark

脱离 SGLang serving 栈，直接测量三种内存后端在 sparse KV 访问模式下的原始性能，为端到端实验提供基础数据。

#### 6.1a 离散 KV 读取延迟与带宽

模拟 sparse top-k 的访问模式：从大块 KV buffer 中随机读取 N 个离散条目。


| 变量            | 取值                                              |
| ------------- | ----------------------------------------------- |
| 读取条目数 (top-k) | 128, 512, 2048, 4096                            |
| 每条目大小         | DeepSeek-V3.2 MLA KV dim × dtype_size           |
| KV buffer 总大小 | 模拟 128K tokens 的单层 KV                           |
| 访问模式          | 随机离散读（均匀随机选取 token 位置）                          |
| 内存后端          | Local DRAM, CXL mmap, RDMA one-sided read（单次全量） |
| 指标            | 总延迟 (μs)、有效带宽 (GB/s)                            |


用 CUDA kernel（GPU 发起 DMA）和 CPU memcpy 两种方式分别测量。

### 6.2 Part 2：端到端 Serving 实验

在 SGLang + HiSparse 上运行 DeepSeek-V3.2，实测三种方案的端到端推理性能。实验采用**两轮执行**设计：第一轮冷启动（无 prefix-cache），第二轮利用第一轮产生的 KV Cache 作为 prefix-cache 热数据。两轮结果均展示，形成完整的"KV Cache 从生成到复用"全链路对比。

#### 6.2.0 实验总体流程

```
┌─────────────────────────────────────────────────────────┐
│  Round 1：冷启动（无 prefix-cache）                        │
│                                                         │
│  发送 N 个共享前缀的请求（prefix + query_i）                │
│  → 每个请求完整 prefill → KV Cache 写入内存池              │
│  → decode 从内存池读取 sparse KV                          │
│                                                         │
│  测量指标：TTFT、Prefill 吞吐、Decode TBT、总吞吐          │
│  实验结束后 KV Cache 保留在内存池中（radix cache + host     │
│  prefix cache 双重保留）                                   │
├─────────────────────────────────────────────────────────┤
│  Round 2：热启动（prefix-cache 命中）                      │
│                                                         │
│  发送 M 个共享同一前缀的新请求（prefix + query_j）           │
│  → prefix 部分命中 radix cache，跳过 prefix 的 prefill     │
│    计算，仅对 query_j 做 extend prefill（数百 tokens）      │
│  → staging 仅备份 extend 部分的 KV，prefix KV 从 host      │
│    prefix cache 直接复用                                   │
│  → decode 从内存池读取 sparse KV（与 Round 1 相同路径）      │
│                                                         │
│  测量指标：TTFT（应显著降低）、Decode TBT、总吞吐            │
│  重点关注：GPU 几乎全部用于 decode，RDMA 预取开销被充分暴露   │
└─────────────────────────────────────────────────────────┘
```

**为什么要两轮？**

- Round 1 展示"无 prefix cache"场景下三种方案的**全链路差异**（含 prefill + KV 写入 + decode）
- Round 2 展示"prefix cache 命中"场景下的差异——prefix 的 prefill 计算被完全跳过（通过 SGLang radix cache），GPU 几乎 100% 用于 decode，此时 **decode 阶段的 KV 读取效率成为唯一瓶颈**，三种方案的差异被充分放大
- 两轮结合体现了真实生产场景的完整链路：首次请求生成 KV Cache → 后续请求复用
- 如果不跳过 prefix 的 prefill 计算，GPU 被 prefill 计算占据，RDMA 带宽不会成为瓶颈，三种方案的 decode throughput 趋于一致，无法区分

#### 6.2.1 Dataset 选择

| Dataset | 说明 | 适用场景 |
|---------|------|---------|
| **LongBench-v2** | 真实长文本 QA，涵盖多种任务类型，输入 4K–128K tokens | 主实验，prefix = 文档内容 |
| **RULER** | 合成长上下文基准（needle-in-haystack 等），可精确控制 context 长度 | 控制变量实验 |
| **ShareGPT 长对话子集** | 真实多轮对话，共享 system prompt + 历史 | prefix = system prompt |
| **合成 Prefix Dataset** | 人工构造固定前缀 + 随机 query | 精确控制 prefix 长度 |

**推荐主实验使用合成 Prefix Dataset**，理由：

1. 可精确控制 prefix 长度（4K/16K/32K/64K tokens），排除 dataset 本身的 variance
2. 所有请求共享完全相同的 prefix，保证 Round 2 的 prefix-cache 命中率 = 100%
3. query 部分随机生成，保证 NSA indexer 的 top-k 选择具有多样性

LongBench-v2 作为**补充实验**展示真实工作负载下的性能。

**合成 Prefix Dataset 构造**：

```
prefix = [随机英文文本, 填充到目标 token 数]
query_i = "Based on the above document, answer question {i}: {random_question}"
请求 = prefix + query_i
output = 生成 256 tokens
```

#### 6.2.2 指标定义

| 指标 | 定义 | Round 1 含义 | Round 2 含义 |
|------|------|-------------|-------------|
| **TTFT** (ms) | Time To First Token | 完整 prefill 时间 | **极低**（prefix 命中 radix cache，跳过 prefix prefill，仅计算 extend 的数百 tokens） |
| **Decode TBT** (ms) | Time Between Tokens | decode 步间延迟 | 同 Round 1（**核心对比指标**） |
| **P99 TBT** (ms) | 第 99 百分位 TBT | 尾延迟 | 尾延迟 |
| **Decode Throughput** (tokens/s) | decode 产生的 token 数 / decode 总时间 | **主要对比指标** | **主要对比指标** |
| **Total Throughput** (tokens/s) | 所有 output token / 端到端总时间 | 含 prefill 开销 | prefix cache 命中后更高 |
| **Max Concurrency** | 不 OOM 的最大并发请求数 | 受限于内存池容量 | 同 Round 1 |
| **DRAM Usage** (GB) | 本地 DRAM 被 KV Cache 占用量 | — | — |

#### 6.2.3 方案 B（RDMA）的本地缓存比例设置

方案 B 模拟分布式内存池场景。关键变量是**本地 DRAM 缓存了多少比例的 KV Cache**，剩余部分需要通过 RDMA loopback 全量取回。

| 配置名称 | 本地 DRAM 比例 | RDMA 预取量 | 模拟场景 |
|---------|-------------|-----------|---------|
| B-0 | 0% | 100% 全量 RDMA | 极端：所有 KV 在"远端"，每步全量预取 |
| B-30 | 30% | 70% RDMA | 典型：本地缓存热数据，多数仍在远端 |
| B-70 | 70% | 30% RDMA | 乐观：大部分在本地，少量远端 |
| B-100 | 100% | 0% RDMA | 等价于方案 A（对照） |

**RDMA 预取的执行方式**：

每步 decode 开始前，将"远端部分"（(1 - 本地比例) × 全量 KV）通过 RDMA NIC loopback 从本机"远端 MR"读取到 staging buffer。这真实地执行了 RDMA 传输，产生的延迟是硬件实测值而非模拟注入。

#### 6.2.4 Round 1：冷启动实验


| 变量            | 取值                                                  |
| ------------- | --------------------------------------------------- |
| 模型            | DeepSeek-V3.2 (671B MoE, 61 layers, NSA, MLA)       |
| prefix 长度     | 4K, 16K, 32K, 64K tokens                            |
| query 长度      | 256, 1024 tokens                                    |
| output 长度     | 256 tokens                                          |
| 并发请求数         | 1, 2, 4, 8, 16                                      |
| top-k         | 2048（模型默认 `index_topk`）                             |
| device buffer | 2 × top_k = 4096                                    |
| 方案            | A (DRAM) / B-0, B-30 (RDMA loopback) / C (CXL)     |
| 指标            | TTFT, Decode TBT, P99 TBT, Decode Throughput, Total Throughput, DRAM Usage |

**Round 1 预期结果**：

- 三种方案 TTFT 基本相同（都在本地 GPU 执行 prefill）
- Decode TBT：A ≈ C < B（B 被 RDMA 全量预取延迟拖慢）
- Total Throughput：A ≈ C > B
- DRAM Usage：A > B > C

#### 6.2.5 Round 2：Prefix-Cache 命中实验

Round 1 结束后，KV Cache 保留在内存池中。发送**新一批请求**，共享与 Round 1 相同的 prefix：


| 变量            | 取值                                                  |
| ------------- | --------------------------------------------------- |
| prefix 长度     | 与 Round 1 相同                                       |
| 新 query       | 不同于 Round 1 的 query（确保 decode 路径不同）                |
| 并发请求数         | 1, 2, 4, 8, 16, 32（可测更多，因 TTFT 更短释放资源更快）          |
| 方案            | A (DRAM) / B-0, B-30 (RDMA loopback) / C (CXL)     |
| 指标            | TTFT（应大幅降低）, Decode TBT, P99 TBT, Decode Throughput, Max Concurrency |

**Round 2 预期结果**：

- **TTFT 大幅降低**：prefix 命中 radix cache → prefill 跳过 prefix 计算，仅处理 extend 部分（256-1024 tokens）。三种方案的 TTFT 均大幅降低（prefill 不涉及 host 介质差异）
- **Decode TBT：A ≈ C << B**——这是核心对比：
  - GPU 几乎完全被 decode 占据（prefill 开销可忽略），RDMA 预取延迟不再被 prefill 计算掩盖
  - 方案 A/C 每步按需读取 top-k（~几十 μs），方案 B 每步需 RDMA 全量预取（~几百 μs ~ 几 ms）
  - Round 2 的 Decode TBT 差异应**显著大于** Round 1（Round 1 中 prefill 间隔掩盖了部分差异）
- **Max Concurrency**：C 远超 A/B
  - A：本地 DRAM 存全量 KV → 并发数受限
  - B：本地 DRAM 存 staging + 部分缓存 → 与 A 类似
  - C：KV 在 CXL，本地 DRAM 释放 → 可扩展到更多并发
- **总吞吐**：C > A >> B（C 有高并发优势 + 低延迟，B 被 RDMA 预取锁死）

#### 6.2.6 Round 1 + Round 2 结果展示

两轮结果并列展示，形成如下对比图：

```
  图 1: Decode TBT (ms) — prefix_len=32K, concurrency=4

        Round 1 (冷启动)            Round 2 (prefix cache hit)
   TBT │                      TBT │
   (ms)│  ┌──┐                (ms)│  ┌──┐
       │  │A │ ┌──┐               │  │A │ ┌──┐
       │  │  │ │C │               │  │  │ │C │     (A≈C, 因 decode 路径不变)
       │  │  │ │  │ ┌────┐        │  │  │ │  │ ┌────┐
       │  │  │ │  │ │ B  │        │  │  │ │  │ │ B  │  (B 仍被 RDMA 拖慢)
       │  └──┘ └──┘ └────┘        │  └──┘ └──┘ └────┘
       └──────────────────        └──────────────────

  图 2: Max Concurrency — prefix_len=32K, seq_len=64K

   Max  │
   Conc │                     ┌──┐
       │              ┌──┐   │C │
       │  ┌──┐ ┌──┐   │B │   │  │  (C >> A ≈ B)
       │  │A │ │B │   │  │   │  │
       │  └──┘ └──┘   └──┘   └──┘
       └──────────────────────────
          Round 1     Round 2
          (所有方案内存占满)  (C 因 CXL 释放 DRAM 可扩展)
```

#### 6.2.7 内存效率与可扩展性


| 变量      | 取值                         |
| ------- | -------------------------- |
| prefix 长度 | 32K, 64K                 |
| 递增并发请求数 | 1, 2, 4, 8, 16, ... 直到 OOM |
| 方案      | A / B-0 / B-30 / C        |
| 指标      | 最大可并发请求数、总吞吐量、DRAM 占用     |


**预期结果**：

- A（DRAM）：每请求需在本地存全量 KV → 可并发数少
- B-0（RDMA 0% 本地）：本地仍需 staging area 存全量 KV → 与 A 类似
- B-30（RDMA 30% 本地）：本地缓存 30% + staging → 介于 A 和 B-0 之间
- C（CXL）：本地仅需 VRAM device buffer → 可并发请求数远超 A/B

#### 6.2.8 消融实验

| 实验           | 变量                                              | 指标                              |
| ------------ | ----------------------------------------------- | ------------------------------- |
| Buffer 大小    | device_buffer_size: 512, 1024, 2048, 4096, 8192 | Buffer 命中率、CXL 实际读取量 / top-k 总量 |
| Sparse ratio | top-k: 512, 1024, 2048, 4096                    | 各方案吞吐对比                         |
| CXL 内存策略     | 单卡 CXL-only / 多卡 CXL interleave / 不同页面大小        | TBT、吞吐                          |
| RDMA 本地缓存比例  | B-0 (0%) / B-30 / B-70 / B-100                  | Decode TBT、RDMA 传输延迟            |
| 层间相似性        | 记录每层 top-k indices                              | 相邻层 overlap 率、热力图               |
| 计算与取数重叠      | 开启/关闭 overlap                                   | TBT 差异、Nsight Systems trace     |

#### 6.2.9 RDMA 全量预取延迟速查表

以 100Gbps IB NIC loopback 为例，RDMA 方案每步 decode 前需全量预取的数据量与延迟：

| prefix 长度 | 每 token 大小 | 层数 | 全量 KV 大小 | RDMA 传输延迟 (100Gbps) | 备注 |
|------------|-------------|------|------------|---------------------|------|
| 4K tokens | 656B | 61 | 155 MB | ~12.4 ms | |
| 16K tokens | 656B | 61 | 620 MB | ~49.6 ms | |
| 32K tokens | 656B | 61 | 1.24 GB | ~99.2 ms | **每步 decode 都要等** |
| 64K tokens | 656B | 61 | 2.48 GB | ~198.4 ms | |

而 CXL 方案每步 decode 只需读取 top-k × layers 的数据：

| top-k | 层数 | 实际读取量 | CXL 延迟 (估) | vs RDMA 全量 (32K prefix) |
|-------|------|----------|-------------|--------------------------|
| 2048 | 61 | 78 MB | ~3-5 ms | **20-33× 更快** |
| 1024 | 61 | 39 MB | ~1.5-2.5 ms | **40-66× 更快** |
| 512 | 61 | 19.5 MB | ~0.8-1.3 ms | **76-124× 更快** |

#### 6.2.10 预期结果总结

```
              Decode Throughput (tokens/s) vs 并发请求数
              prefix_len=32K, top-k=2048, Round 2 (prefix cache hit)

  tokens/s │
           │       ○──○──○──○──○  A: DRAM (OOM at 8 req)
           │      ╱
           │     ╱  ●──●──●──●──●──●──●  C: CXL (scales to 32 req)
           │    ╱  ╱
           │   ╱  ╱
           │  ╱  ╱    ×──×──×──×  B: RDMA (bandwidth-limited)
           │ ╱  ╱    ╱
           │╱  ╱    ╱
           └─┬──┬──┬──┬──┬──┬──┬──→ 并发请求数
             1  2  4  8  16 24 32
```

| 对比维度 | A: DRAM | B: RDMA 分布式池 | C: CXL |
|---------|---------|---------------|--------|
| 每步 decode 数据传输量 | top-k only | **全量 KV** (100%) | top-k only |
| 本地 DRAM 占用 | 全量 KV | staging + 本地缓存 | ≈ 0 |
| 最大并发请求数 | 受限于 DRAM | 受限于 staging DRAM | 受限于 CXL（远大于 DRAM） |
| Decode TBT | **最低** | 被 RDMA 延迟拖慢 | 接近 A |
| 有效带宽利用率 | 100% | **1-10%** | ≈ 100% |
| 部署复杂度 | 最简单 | 需 RDMA NIC | 需 CXL 设备 |


