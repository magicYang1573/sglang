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

1. **Micro-benchmark（§6.1）**：在脱离 serving 栈的条件下，测量 **DRAM、CXL、RDMA** 三种后端在 sparse attention KV 访存模式下的**读取延迟**（及有效带宽）；脚本 `test/cxl_utils/sparse_kv_bench.py`。
2. **端到端 DeepSeek-V3.2（§6.2）**：对比 **Local DRAM、CXL、RDMA（`local_ratio` = 0% 与 50%）**；**第一轮为 warmup**，**主要分析与展示第二轮（prefix / radix 已热）** 的 TTFT、TBT、吞吐等；客户端 **512 请求、最大并发 64**；**第二轮**每条请求的 `max_tokens` 在 **0～1024（均匀）** 上采样（0 在客户端会钳制为 1 以满足 API）；脚本 `test/cxl_utils/e2e_bench_fast.py`，矩阵批跑 `test/cxl_utils/run_all_experiments.sh`。
3. **消融与系统扩展（§6.3–§6.5）**：多卡 **CXL interleave**（设计待落地）、**多 RDMA 网卡** 扩展、**本地 DRAM 容量受限** 下并发对吞吐的影响。
4. **成本分析（§6.6）**：在 **TTFT、TBT** 等约束下，对比 DRAM / CXL / RDMA 等方案的 **单位有效吞吐成本**。
5. 验证核心论点：sparse attention 下 **CXL 按需读 top-k** 相对 **RDMA 全量预取** 在延迟与带宽利用上更优。

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

**模拟场景**：KV Cache 存储在"远端"分布式内存池中。Remote 请求在进入 decode 阶段前，需一次性通过 RDMA 将全量 KV Cache 预取到本地 DRAM staging area，之后 HiSparse 从本地 DRAM 按 top-k 加载到 GPU。

**单机模拟方式**：在同一台机器上，按**请求粒度**划分 local/remote：

- **Local 请求**（概率 = `local_ratio`）：KV 在 host pool（pinned DRAM）中直接可用，无需 RDMA
- **Remote 请求**（概率 = `1 - local_ratio`）：KV 存在本机 DRAM 的 RDMA MR 中模拟远端，decode 前需一次性 RDMA loopback 全量拉回

同一请求的全部 KV 要么全在本地，要么全在远端，符合真实分布式语义。通过调节 `local_ratio`（0%–100%）模拟不同的命中率配置。

- **数据路径**：remote 请求进入 decode 前一次性预取："远端" DRAM →（RDMA loopback）→ 本地 DRAM；之后每步 decode：本地 DRAM →（PCIe DMA）→ GPU
- **问题**：一次性预取了 100% 数据，但每步 attention 只用了 1–10%。带宽和本地 DRAM 均被浪费

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
| Decode 前预取（一次性） | 无（数据已在本地）               | remote 请求：全量 KV → RDMA loopback → 本地 DRAM（**一次性**） | 无（CXL 按需访问）     |
| 每层 swap-in       | DRAM → GPU (top-k miss) | 本地 DRAM → GPU (top-k miss)（预取后与 A 相同） | CXL → GPU (top-k miss) |
| 一次性传输量           | 0                       | remote 请求：**全量 KV**（一次性）       | 0                      |
| 本地 DRAM 占用       | 全量 KV                   | 全量 KV（预取后占用与 A 相同）             | ≈ 0（KV 在 CXL）          |
| 有效带宽利用率          | 100%                    | 预取：**1–10%**（取 100% 用 1-10%） | ≈ 100%                 |


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

多卡 interleave 的对比实验单独列为 **§6.3（ablation-1）**；与 §5.2 中的分配策略设计对应，**实现与跑数尚未完成**。

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

## 6. 实验设计（总览）

主实验分为 **六类（§6.1–§6.6）**；**§6.7** 为与 §4 对照的简要汇总表。

| 小节 | 主题 | 脚本 / 入口 |
|------|------|-------------|
| **§6.1** | KVCache Sparse 读取 latency（micro） | `test/cxl_utils/sparse_kv_bench.py` |
| **§6.2** | End-to-End DeepSeek-V3.2（两轮；**主看 Round 2**） | `test/cxl_utils/e2e_bench_fast.py`，矩阵 `test/cxl_utils/run_all_experiments.sh` |
| **§6.3** | Ablation-1：CXL interleave（多卡带宽） | **设计未实现** |
| **§6.4** | RDMA expand（多网卡） | 待跑；见假设 |
| **§6.5** | Under memory limitation | 待跑；限制本地 DRAM |
| **§6.6** | Cost analysis | 建模 / 报表；与 §6.2 指标衔接 |

---

### 6.1 KVCache Sparse 读取实验（latency）

**目的**：脱离 SGLang serving 栈，在 **与 DeepSeek-V3.2 MLA 一致的条目尺寸与离散读模式** 下，对比 **DRAM、CXL、RDMA** 三种后端读写 sparse attention 所用 KV 的访存 **延迟与有效带宽**，为 §6.2 提供下界/上界参照。

**脚本**：`test/cxl_utils/sparse_kv_bench.py`（GPU scatter 读、CXL mmap + `cudaHostRegister`、RDMA one-sided 等路径以脚本实现为准）。

**典型扫参**（可与脚本 CLI 对齐微调）：

| 变量 | 取值（示例） |
|------|----------------|
| 读取条目数 (top-k) | 128, 512, 2048, 4096 |
| 每条目大小 | DeepSeek-V3.2 FP8 packed MLA（如 656 B / 条目，以代码为准） |
| KV buffer 规模 | 模拟长上下文单层或多层驻留 |
| 访问模式 | 随机离散索引（均匀），对应 sparse top-k |
| 后端 | Local pinned DRAM；CXL；RDMA（离散读或对照性 bulk，以实现为准） |
| 指标 | 总延迟 (μs)、有效带宽 (GB/s)、可分位延迟 |

可同时记录 **GPU 发起 DMA** 与 **CPU 侧** 测量方式（若脚本支持），便于区分 PCIe/CXL 与 kernel 开销。

---

### 6.2 End-to-End DeepSeek-V3.2

**目的**：在 **SGLang + HiSparse + DeepSeek-V3.2** 上，对比 **Local DRAM、CXL、RDMA（`local_ratio` = 0% 与 50%）** 的真实 serving 行为。**第一轮为 warmup**（填满各 DP rank 的 radix / prefix 相关状态；见 `e2e_bench_fast.py` 文档字符串），**论文与主表以第二轮（Round 2）为主**：此时 prefix 已热，GPU 时间更多落在 decode，**host 侧 KV 路径差异**更易暴露。

**脚本**：

- 单次跑：`test/cxl_utils/e2e_bench_fast.py`
- 全矩阵（多 context × 多 scheme）：`test/cxl_utils/run_all_experiments.sh`

**与当前批跑脚本一致的默认负载**（均可通过环境变量覆盖，以脚本头部注释为准）：

| 参数 | 默认值（`run_all_experiments.sh`） | 说明 |
|------|-------------------------------------|------|
| Round 2 总请求数 | **512**（`NUM_REQUESTS`） | `--num-requests` |
| 客户端最大并发 | **64**（`MAX_CONCURRENCY`） | `--max-concurrency`；仅限制同时 in-flight HTTP |
| Round 2 输出长度 | **0～1024 均匀**（`ROUND2_OUTPUT_MIN=0`, `ROUND2_OUTPUT_MAX=1024`） | 每条请求独立采样 `max_tokens`；0 在客户端钳为 1 以满足 API |
| Round 1 输出长度 | **512**（`OUTPUT_TOKENS`） | 与 warmup 用的 synthetic 请求一致 |
| 上下文长度扫描 | **16K / 32K / 64K / 128K**（`CONTEXT_LENGTHS`） | `--target-input-tokens` |
| Pool / 方案 | **dram, cxl, rdma_local_0, rdma_local_50** | RDMA 两档对应 **0% / 50%** 请求级本地命中 |
| 其它 | `NUM_UNIQUE_PROMPTS`（默认 1）、`REPEAT_MODE=interleaved` 等 | 详见脚本 |

**模型与 HiSparse**：批跑脚本侧为 DeepSeek-V3.2 AWQ、TP=8、dp-size=8 等（见 `run_all_experiments.sh` 内注释）；HiSparse JSON 含 `top_k: 2048`、`device_buffer_size: 4096`，CXL/RDMA 子段按 scheme 注入。

**两轮流程（摘要）**：

```
Round 1 — Warmup：按 dp_size 与各 rank 需求发送唯一 prompt 副本，使 cache 在各 DP 上就绪；指标可作参考，不与 Round 2 直接比吞吐。
Round 2 — Measurement：将同一批 unique prompt 按 repeat 模式展开为 512 条请求，在已预热 cache 上测量；输出长度按 [0,1024] 均匀（实现上 min≥1）。
```

**核心指标**（与 `e2e_bench.py` 汇总一致）：**TTFT**、**Decode TBT**（及 P99）、**Decode / Total throughput**、可选 **DRAM 占用**。**主结论优先使用 Round 2**；Round 1 用于说明冷启动与 warmup 成本。

**RDMA 全量预取数据量参考**（remote 请求进入 decode 前一次性传输；100 Gbps 量级估算，与 context 长度线性相关）：

| 输入长度 (tokens) | 全量 KV 约 (61×656B×tokens) | ~100Gbps 仅传输时间量级 |
|-------------------|-----------------------------|-------------------------|
| 16K | ~620 MB | ~50 ms |
| 32K | ~1.24 GB | ~100 ms |
| 64K | ~2.48 GB | ~200 ms |
| 128K | ~4.96 GB | ~400 ms |

CXL/DRAM 路径以 **按步 top-k** 为主，有效搬运量随 sparse 比例显著小于上表「全量」列（与 §6.1 micro 对照）。

**预期（Round 2，定性）**：Decode TBT **DRAM ≲ CXL ≪ RDMA（尤其 local_ratio=0）**；总吞吐与尾延迟趋势与同序。若提高并发且受 **本地 DRAM 容量** 约束，CXL 可能在 **可承载并发** 上相对纯 DRAM 更有空间（需与 §6.5 联调实测）。

**可选扩展**（与主矩阵正交，另行成表）：固定 context 下 **递增并发直至 OOM**；`device_buffer_size` / `top_k` 消融；LongBench-v2、RULER 等真实或合成数据集。主矩阵 **不包含** RDMA B-25 档；若需三档 RDMA，可在 `e2e_bench_fast` 与 server 配置上单独加列。

---

### 6.3 Ablation-1：CXL interleave（多卡带宽）

**目的**：使用 **多张 CXL 内存扩展卡**，通过 **NUMA interleave**（或其它条带策略）聚合上行带宽，与 **单张 CXL 卡** 对比 sparse KV 与端到端指标是否随卡数近似扩展。

**状态**：**设计与自动化实验尚未在仓库中落地**（§5.2 中的分配策略与此对应；实现后可复用 `run_all_experiments.sh` 的 scheme 循环，增加 `numactl --interleave=...` 或等价内存策略）。

**计划扫参**：卡数 1 / 2 / 4；可选与 **大页（2MB/1GB）**、`layer_first` vs `page_first` 布局（§5.2）组合。

**指标**：§6.1 延迟/带宽；§6.2 Round 2 的 TBT、吞吐。

---

### 6.4 RDMA expand（多网卡）

**目的**：增加参与 loopback / 预取的 **RDMA NIC 数量**（多队列、多设备并行），观察 **RDMA 全量预取** 是否随网卡数线性提升。

**假设**：**可能无明显收益或收益饱和**——预取路径上数据往往需进入 **本地 DRAM**，写侧受 **Root Complex / 内存子系统** 带宽与争用限制，NIC 数增加不一定提高「远端 → 本地 DRAM」的有效速率；需用 **Nsight / perfetto、NIC 计数器** 验证是否触顶 PCIe 或 DRAM 写带宽。

**指标**：单次全量预取延迟、Round 2 中带 remote 标记请求的 TTFT 尾部分、总吞吐；与单 NIC 基线对比。

---

### 6.5 Under memory limitation（受限本地 DRAM）

**目的**：在 **限制进程可见的本地 DRAM**（如 cgroup、`membind` 仅绑小容量 node、或压低 `mem-fraction-static` / 可配置 host pool 上限）的前提下，扫描 **客户端并发**，比较 **DRAM vs CXL vs RDMA** 的 **总吞吐、P99 TBT、OOM 前最大并发**。

**动机**：DRAM 方案在 KV 全量驻留本地时 **容量瓶颈** 先出现；CXL 将 KV 主要放在扩展内存，**相同本地 DRAM 预算** 下可能支撑 **更高并发**。RDMA 方案仍受 **全量预取 + staging** 对本地 DRAM 的冲击。

**指标**：与 §6.2 相同，外加 **失败请求数 / OOM 事件**。

---

### 6.6 Cost analysis（成本分析）

**目的**：在可比 **SLA** 下对比 **DRAM、CXL、RDMA** 的 **单位成本 / 单位有效吞吐**（例如 **$/GB/s decode 吞吐** 或 **$/1M output tokens**），约束可包括：

- **TTFT P99**、**TBT P99**（或 mean）上限；
- **单机 GPU 数、NIC 数、CXL 卡数与容量** 的 CAPEX + 粗略 OPEX（电力、机位）。

**输入数据**：§6.2 Round 2 的吞吐与延迟；§6.5 在受限内存下的 **最大可持续并发**；硬件清单单价（市场或内部采购价）。

**输出**：在固定 SLA 下 **哪种方案单位吞吐成本最低**；灵敏度分析（电价、CXL 单价、RDMA 端口数）。

---

### 6.7 方案对比小结（与 §4 呼应）

| 对比维度 | A: DRAM | B: RDMA 池（全量预取） | C: CXL |
|---------|---------|------------------------|--------|
| 每步 decode 有效搬运 | top-k 为主 | 预取全量；decode 仍 top-k | top-k 为主 |
| 本地 DRAM 占用 | 高（KV 常驻） | remote：staging + 预取后高 | 低（KV 主要在 CXL） |
| Decode TBT（Round 2） | **最低（基线）** | **remote 显著变差** | **介于其间，通常远优于 B** |
| 有效网络/内存带宽利用 | 高 | 预取阶段 **大量无效字节** | 高（按需） |
| 部署 | 简单 | NIC + 分布式栈 | CXL 硬件 + NUMA/DAX 配置 |


