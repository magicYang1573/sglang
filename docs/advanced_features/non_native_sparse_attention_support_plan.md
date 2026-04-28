# 非原生 Sparse Attention：维度、分类与支持范围

本文档介绍 sparse attention 的基本概念、设计维度、常见算法分类，以及当前非原生 sparse attention 路线中计划涉及、后续可支持、暂缓支持和原则上不支持的算法范围。

## 1. Sparse Attention 基本知识

### 1.1 问题背景

标准自回归 Transformer 在 decode 阶段，每生成一个新 token，当前 query 需要与历史上所有 KV 做 attention：

```text
output_t = softmax(q_t K_{1..t}^T / sqrt(d)) V_{1..t}
```

随着上下文长度增长，decode 单步 attention 的计算量和访存量都会随历史长度增长；整个生成过程累计下来，attention 成本接近二次增长。KV Cache 本身也随上下文长度线性增长，长上下文场景下常见瓶颈包括：

- GPU KV Cache 显存占用过高，限制 batch size 和并发请求数。
- Decode attention 每步需要扫描更长历史 KV，吞吐下降。
- 长上下文请求与短请求混合时，调度和缓存管理更复杂。

Sparse attention 的核心思想是：**不是每个 decode step 都必须读取完整历史 KV；如果能选出对当前输出最重要的一部分 KV，就可以减少 attention 的计算和访存。**

### 1.2 原生与非原生 Sparse Attention

Sparse attention 可以分成两类：

1. **原生稀疏注意力**：模型结构自带 sparse indexer、router 或 sparse attention 语义。稀疏选择结果来自模型内部，通常需要与模型结构匹配的 backend 或 kernel。
2. **非原生稀疏注意力**：模型本身仍是标准 MHA/GQA dense attention 架构，但运行时通过额外算法选择部分 KV 参与 attention，例如 Quest、StreamingLLM、ChunkKV、H2O、SnapKV 等。

本文关注第二类：**不修改模型结构，在推理阶段通过外部算法决定哪些历史 KV 参与当前 attention。**

### 1.3 非原生 Sparse Attention 的基本流程

非原生 sparse attention 通常包含以下步骤：

1. **Prefill 阶段**：仍执行 dense attention，保证 prompt 编码质量，同时收集后续选择所需的表示或统计量。
2. **表示构建**：基于 K/V、位置、page/chunk 统计量或历史状态，为每个请求和每层维护轻量索引。
3. **Decode 选择**：每个 decode step 或每层 decode 前，根据当前 query、固定规则或历史统计选择一部分 KV。
4. **Attention 执行**：attention 只读取选中的 KV 子集。
5. **状态更新**：新 token 生成后，算法更新表示和缓存状态。

不同算法的主要差异在于：保留哪些 KV、以什么粒度选择、多久选择一次、根据什么信号选择。

## 2. Sparse Attention 设计维度

非原生 sparse attention 算法虽然名称很多，但从系统接入角度可以拆成几个正交维度。框架设计不应该为每个算法写特例，而应判断算法在这些维度上的选择是否能被统一表达。

### 2.1 D1：保留策略

| 类型 | 含义 | 说明 |
|---|---|---|
| **仅选择可见子集** | 历史 KV 仍存在，只是本步 attention 看不到未选 token。 | 实现相对简单，不一定降低 KV 存储量。 |
| **分层 KV / 热冷分离** | 完整 KV 可以放在较慢层级，GPU 只保留当前 attention 需要的热 KV。 | 同时减少 GPU 常驻 KV 和 attention 扫描量。 |
| **物理驱逐 KV** | 被淘汰 token 的 KV 槽位提前释放。 | 需要与请求长度、缓存复用、allocator 释放语义严格一致。 |
| **压缩或摘要 KV** | 历史 KV 不完整保留，而是量化、合并或摘要。 | 可能影响质量，需要单独评估误差来源。 |

当前更适合优先考虑“不改变模型语义、可回退、可验证”的选择型策略；物理驱逐和压缩型策略需要更强的调度与正确性约束。

### 2.2 D2：选择粒度

| 粒度 | 代表算法 / 策略 | 特点 | 支持判断 |
|---|---|---|---|
| **Token** | StreamingLLM、H2O、SnapKV | 精度高，可表达任意 token 子集。 | 最通用，但选择和排序开销可能更高。 |
| **Page / Block / Chunk** | Quest、ChunkKV | 评分和表示更高效，适合长上下文。 | 工程上友好，但需要处理块内冗余和边界对齐。 |
| **Cluster / Semantic Group** | PQCache 类方法、聚类检索 | 适合作为近似检索结构。 | 更适合作为算法内部索引，最终仍需映射到可执行的 token/page 子集。 |

粒度越粗，选择开销通常越低，但有效稀疏度可能下降；粒度越细，选择更精确，但检索、排序和 metadata 组织成本更高。

### 2.3 D3：选择频率

| 频率 | 含义 | 代表 |
|---|---|---|
| **每请求一次** | Prefill 后固定选择结果，decode 全程复用或只做轻量更新。 | Prompt compression、static pruning、部分观察窗口策略。 |
| **每 decode step 一次** | 每个 decode step 重新计算可见集合，通常跨层共享。 | StreamingLLM、固定窗口、部分 query-unaware 策略。 |
| **每层一次** | 每个 decode step 的每一层都重新选择。 | Quest、ChunkKV 等 query-aware 策略。 |

选择频率越高，算法越能贴合当前生成状态，但开销也越大。每层选择尤其需要关注检索和 top-k 的成本。

### 2.4 D4：选择依据

| 类型 | 是否依赖当前 Query | 是否依赖 attention weights | 代表算法 / 策略 | 当前态度 |
|---|---|---|---|---|
| **固定策略** | 否 | 否 | StreamingLLM、sink + window、sliding window | 可支持 |
| **Query-Unaware** | 否 | 否 | 基于 K/V 范数、位置、静态统计量的策略 | 可支持 |
| **Query-Aware** | 是 | 否 | Quest、ChunkKV | 首版重点 |
| **Attention-Score-Dependent** | 通常是 | 是 | H2O、SnapKV | 暂缓支持 |
| **原生学习型稀疏** | 由模型结构决定 | 视模型而定 | 模型内置 indexer/router | 不纳入非原生范围 |

D4 是最关键的分类维度。是否依赖完整 attention weights，直接决定算法能否以低成本接入高性能 fused attention 路径。

## 3. 算法分类

### 3.1 固定策略

固定策略不依赖当前 query，也不依赖 attention weights，仅根据位置规则选择 KV。

常见模式包括：

- **Sliding Window**：只保留最近若干 token。
- **Sink + Window**：保留开头少量 sink token，再保留最近窗口。
- **Pattern Mask**：按固定稀疏模式保留局部、跨块或周期性位置。

这类方法实现简单、开销低、行为稳定，适合作为较早支持的策略。但它无法根据当前语义动态选择远处 token。

### 3.2 Query-Unaware 统计策略

Query-Unaware 策略不看当前 query，而是根据历史 K/V 或位置上的统计量判断重要性。

可能使用的信号包括：

- Key 或 value 的范数、方差、离群程度。
- 位置衰减、段落边界、特殊 token。
- 与 attention weights 无关的历史状态。

这类方法比固定策略更内容相关，但选择结果不会随当前 query 细粒度变化。只要不依赖完整 attention weights，通常可以作为后续支持方向。

### 3.3 Query-Aware 策略

Query-Aware 策略使用当前 step 或当前层的 query 作为选择输入，再与历史 KV 的紧凑表示交互打分。

常见做法包括：

- 为 page/chunk 维护 min/max、centroid、摘要向量等表示。
- 用当前 query 对每个 page/chunk 估计相关性或 attention score 上界。
- 选择 top-k page/chunk/token，再只对这些位置执行 attention。

Quest、ChunkKV 都属于这类路线。它们能根据当前生成内容动态选择远处上下文，是当前非原生 sparse attention 最值得优先落地的一类。

### 3.4 Attention-Score-Dependent 策略

Attention-Score-Dependent 策略依赖逐 token attention score，或依赖可等价反映每个历史 token 贡献的细粒度统计。

典型思路包括：

- 累积历史 decode attention score，把长期高分 token 视为重要。
- 在 prefill 的观察窗口内统计 attention 投票，选择后续 decode 保留集合。
- 根据不同层或不同头的 attention 分布动态更新 token 重要性。

H2O、SnapKV 等属于这一类。它们语义上直观，但工程接入更困难：高性能 fused attention 通常不会物化完整 attention weight 矩阵。如果为了拿到逐 token weights 额外写出大矩阵，可能抵消 sparse attention 的收益。

### 3.5 原生学习型稀疏

原生学习型稀疏不是在已有 dense 模型外部外挂选择器，而是把稀疏模式、indexer、router 或专用 attention 结构写进模型本身。

这类方法通常具有更强的模型协同能力，但也意味着：

- 需要特定模型结构或训练流程。
- 需要与模型结构匹配的 kernel 或 backend。
- 很难作为通用推理时插件直接套到普通 dense attention 模型上。

因此，本文把它单独列为分类，不纳入非原生 sparse attention 的通用支持范围。

## 4. 支持范围规划

### 4.1 首版涉及

| 算法 / 策略 | 类型 | 内部粒度 | 选择频率 | 说明 |
|---|---|---|---|---|
| **Quest** | Query-Aware | Page / Chunk | 每层 | 基于当前 query 与 page/chunk 表示打分，不依赖 attention weights。 |

Quest 适合作为首版落地算法，因为它满足三个条件：

1. 不需要完整 attention weights。
2. 可以在 prefill 阶段基于 K 构建每层表示。
3. Decode 每层只依赖当前 query 与预计算表示，选择结果可转成可执行的 KV 子集。

首版范围建议保持收敛：先验证 query-aware、page/chunk 级选择在 decode 阶段的正确性、质量和性能收益，再扩展更多算法。

### 4.2 后续优先支持

| 算法 / 策略 | 类型 | 预计难度 | 支持条件 |
|---|---|---|---|
| **ChunkKV** | Query-Aware | 中 | chunk/page 表示可在算法内部维护，最终输出可执行的 token/page 子集。 |
| **StreamingLLM** | 固定策略 | 低 | sink + sliding window 可直接生成可见集合，不需要复杂表示构建。 |
| **Query-Unaware KV 统计策略** | Query-Unaware | 低到中 | 只依赖 K/V 范数、位置或静态统计量，不依赖 attention weights。 |
| **PQCache 类方法** | 近似检索 / cluster 内部实现 | 中到高 | cluster 只能作为算法内部索引，最终需要映射到 token/page 子集。 |

这些方向的共同点是：选择逻辑不需要 fused attention 输出完整 attention weights，因此可以避免破坏主 attention 路径。

### 4.3 暂缓支持

| 算法 / 能力 | 暂缓原因 | 可能方向 |
|---|---|---|
| **H2O** | 依赖历史 attention score 累积；逐 token attention weights 获取成本高。 | 研究 block-level 近似统计，或算法内部自行计算轻量 score。 |
| **SnapKV** | 依赖 prefill 观察窗口内的 attention weights 投票。 | 研究 prefill 采样、近似 score 或离线 profiling 路径。 |
| **物理 KV 驱逐类算法** | 与请求长度、cache 复用、allocator 释放语义耦合强。 | 等系统层具备正式 eviction 协议后再考虑。 |

暂缓支持不代表算法方向不成立，而是当前阶段接入成本、正确性风险或性能风险较高。

### 4.4 原则上不支持

以下类型不建议纳入非原生 sparse attention 的通用支持范围：

1. **要求 attention backend 输出完整 attention weight 矩阵的算法**
   - 完整 attention weights 的写出会产生额外显存带宽和存储开销。
   - 这会破坏 fused attention 避免中间矩阵物化的核心优势。

2. **要求改变模型结构或训练额外 indexer 的算法**
   - 这类方法更接近原生稀疏注意力。
   - 它们应该按模型专用路径评估，而不是作为普通 dense 模型的推理时插件。

3. **要求 backend 按非 token/page 物理粒度访问 KV 的算法**
   - 例如把 cluster、semantic group、hash bucket 作为 backend 级访问单位。
   - cluster 可以作为算法内部检索结构，但最终应映射回 token/page 子集。

4. **要求 decode 主路径中执行大规模 CPU/GPU 同步的算法**
   - Sparse attention 的收益依赖 decode 低延迟。
   - 如果每步都需要 CPU 排序、跨设备同步或读取大块慢速内存，可能抵消收益。

5. **首阶段就要求 prefill attention 也稀疏化的算法**
   - Prefill sparse 会影响 prompt 编码质量、batching 策略和 attention backend 设计。
   - 首阶段应先解决 decode sparse，再单独评估 prefill sparse。

## 5. 接入新算法的判断标准

新增一个非原生 sparse attention 算法前，应先回答以下问题：

| 问题 | 期望答案 |
|---|---|
| 是否能在不修改模型结构的情况下运行？ | 是 |
| 是否能在不读取完整 attention weights 的情况下选择 KV？ | 是 |
| 是否能最终输出 token/page 级可执行子集？ | 是 |
| 选择开销是否小于节省的 attention 计算与访存？ | 是 |
| 算法状态是否能按 request/layer 清理，不污染其他请求？ | 是 |
| 对短上下文、低收益场景是否可以禁用或回退 dense？ | 是 |

如果某个算法内部使用 page、chunk 或 cluster，不一定被排除；关键是它能否在算法边界内把结果映射为 attention 后端可执行的 token/page 子集。

## 6. 小结

Sparse attention 可以先按四个维度理解：

1. **D1 保留策略**：未选 KV 只是不可见，还是被移走、压缩或驱逐？
2. **D2 选择粒度**：选择的是 token、page/chunk，还是更高层的 cluster？
3. **D3 选择频率**：每请求、每 decode step，还是每层都重新选择？
4. **D4 选择依据**：固定策略、Query-Unaware、Query-Aware、Attention-Score-Dependent，还是原生学习型稀疏？

当前最适合优先支持的是 **Query-Aware 且不依赖 attention weights** 的方法，例如 Quest；后续可以扩展固定策略、Query-Unaware 策略和 ChunkKV/PQCache 类方法。依赖完整 attention weights、要求改变模型结构、要求 backend 支持新物理粒度或引入大规模同步的算法，应暂缓或不纳入通用非原生 sparse attention 支持范围。
