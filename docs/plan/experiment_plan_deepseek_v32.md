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
| RDMA NIC | ConnectX-5/6/7, 100/200Gbps IB 或 RoCE | Mooncake 传输后端                       |
| 远端节点     | ≥1 台具有 RDMA NIC + ≥256GB DRAM 的机器     | Mooncake 远端 KV Cache 存储             |
| 本地 DRAM  | ≥512GB                                | 模型权重 + HiSparse host pool + OS 开销   |


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

### 4.3 方案 B：Mooncake RDMA 远端内存池（全量预取）

这是当前 RDMA 池化系统（Mooncake、LMCache 等）的标准做法，也是本文要批判的对象。

KV Cache 的持久存储位于远端节点。每步 decode 开始前，通过 Mooncake RDMA 将该请求的**全量 KV Cache** 从远端预取到本地 DRAM staging area，然后 HiSparse 从本地 staging area 按 top-k 加载到 GPU。

- **数据路径**：远端 DRAM →（RDMA）→ 本地 DRAM →（PCIe DMA）→ GPU
- **问题**：预取了 100% 数据，attention 只用了 1–10%。带宽和本地 DRAM 均被浪费

> RDMA 是消息语义，每次传输都有 μs 级的 per-message 协议开销。DeepSeek-V3.2 有 61 层 transformer，如果每层都发一次 RDMA 请求取 top-k，仅协议开销就是 61 × 数μs = 数百μs，远超 sparse attention 本身的计算时间。因此 RDMA 只能做全量预取，逐层按需取不可行。

### 4.4 方案 C：CXL 共享内存池

将 HiSparse 的 host pool 分配到 CXL 扩展内存。CXL 内存通过 mmap（DAX 设备 / NUMA node）映射到进程地址空间，再 `cudaHostRegister` 使 GPU 可直接 DMA 访问。

- **数据路径**：GPU ↔ CXL Memory（PCIe/CXL DMA，load/store 语义）
- HiSparse JIT kernel 的 host→device 操作无需修改——它只通过 host pointer 读取数据，不关心底层是 DRAM 还是 CXL
- **仅传输 top-k**：每层 miss 的条目直接从 CXL DMA 到 device buffer，无全量预取，无 RDMA per-message 开销
- **本地 DRAM 几乎不占**：KV Cache 全部在 CXL 中，本地 DRAM 释放给模型权重和更多并发请求

### 4.5 三种方案的数据流对比


| 环节               | A: DRAM                 | B: RDMA 全量预取               | C: CXL                 |
| ---------------- | ----------------------- | -------------------------- | ---------------------- |
| Prefill 后 backup | GPU → DRAM              | GPU → DRAM → RDMA → 远端     | GPU → CXL              |
| Decode 每步预取      | 无（数据已在本地）               | 远端 → RDMA → DRAM（**全量**）   | 无（CXL 按需访问）            |
| 每层 swap-in       | DRAM → GPU (top-k miss) | DRAM → GPU (top-k miss)    | CXL → GPU (top-k miss) |
| 每步总传输量           | top-k × layers          | **全量 KV** + top-k × layers | top-k × layers         |
| 本地 DRAM 占用       | 全量 KV                   | 全量 KV (staging)            | ≈ 0（KV 在 CXL）          |
| 有效带宽利用率          | 100%                    | **1–10%**                  | ≈ 100%                 |


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

在 SGLang + HiSparse 上运行 DeepSeek-V3.2，实测三种方案的端到端推理性能。

#### 6.2a 吞吐与延迟对比


| 变量            | 取值                                                  |
| ------------- | --------------------------------------------------- |
| 模型            | DeepSeek-V3.2 (671B MoE, 61 layers, NSA, MLA)       |
| 序列长度          | 32K, 64K, 128K                                      |
| 并发请求数         | 1, 2, 4, 8                                          |
| top-k         | 2048（模型默认 `index_topk`）                             |
| device buffer | 2 × top_k = 4096                                    |
| 方案            | A (DRAM) / B (RDMA 全量预取) / C (CXL)                  |
| 指标            | Throughput (tokens/s), TTFT (ms), TBT (ms), P99 TBT |


**预期结果**：

- A（DRAM）：延迟最低，但受限于本地 DRAM 容量，可并发请求数有限
- B（RDMA 全量预取）：每步 decode 的 RDMA 全量传输时间远超 sparse 计算时间，吞吐被 RDMA 带宽锁死
- C（CXL）：延迟接近 A，但本地 DRAM 占用接近 0，可扩展更多并发

#### 6.2b 内存效率与可扩展性


| 变量      | 取值                         |
| ------- | -------------------------- |
| 序列长度    | 128K                       |
| 递增并发请求数 | 1, 2, 4, 8, 16, ... 直到 OOM |
| 指标      | 最大可并发请求数、总吞吐量              |


**预期结果**：

- A（DRAM）：每请求需在本地存全量 KV → 可并发数少
- B（RDMA）：本地仍需 staging area 存全量 KV → 与 A 类似
- C（CXL）：本地仅需 VRAM buffer → 可并发请求数远超 A/B

#### 6.2c 消融实验


| 实验           | 变量                                              | 指标                              |
| ------------ | ----------------------------------------------- | ------------------------------- |
| Buffer 大小    | device_buffer_size: 512, 1024, 2048, 4096, 8192 | Buffer 命中率、CXL 实际读取量 / top-k 总量 |
| Sparse ratio | top-k: 512, 1024, 2048, 4096                    | 各方案吞吐对比                         |
| CXL 内存策略     | 单卡 CXL-only / 多卡 CXL interleave / 不同页面大小        | TBT、吞吐                          |
| 层间相似性        | 记录每层 top-k indices                              | 相邻层 overlap 率、热力图               |
| 计算与取数重叠      | 开启/关闭 overlap                                   | TBT 差异、Nsight Systems trace     |


