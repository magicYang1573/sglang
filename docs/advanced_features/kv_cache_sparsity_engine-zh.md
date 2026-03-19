# KVCache 稀疏引擎 (KSE) 设计文档

## 1. 架构概览

### 1.1 问题陈述

随着上下文长度的增长，attention 计算复杂度以 O(n²) 增长，KV Cache 存储需求以 O(n) 线性增长。KVCache 稀疏化技术通过仅选择部分 KV 条目参与 attention 计算来解决这一问题，但 sglang 中现有的实现（Double Sparsity、NSA、Hierarchical Sparse）与特定后端和算法紧密耦合。本文档提出一个**统一的 KVCache 稀疏引擎 (KSE)**，目标是：

- 支持稀疏化维度（驱逐策略、粒度、频率、选择策略）的任意组合
- 对现有 sglang 前向路径的改动最小化
- 为未来算法提供清晰的扩展点

### 1.2 设计维度

KSE 的设计围绕 KVCache 稀疏化的四个正交维度展开：

| 维度 | 选项 | KSE 中的对应职责 |
|---|---|---|
| **D1: 驱逐策略** | 部分保留（驱逐）/ 全量保留（仅选择） | `EvictionPolicy` — 是否物理释放 KV 槽位 |
| **D2: 粒度** | Token / Block(Page) / Cluster | `SelectionResult.granularity` — 选择的单位 |
| **D3: 频率** | 每请求 / 每 token（decode step）/ 每层 | `SparsityPolicy.frequency` — 何时执行选择 |
| **D4: 策略** | 固定策略（滑动窗口、sink）/ 不含 Query / 含 Query（Query-Aware） | `SparsityPolicy.select()` — 如何选择 KV |

### 1.3 在 sglang 拓扑中的位置

KSE 是一个附加在现有前向路径上的 **sidecar 模块**，在两个明确定义的位置进行拦截：

1. **Scheduler 层**（用于 prefill 后的每请求驱逐）
2. **Attention 层**（用于 decode 阶段的每 token / 每层选择）

无需修改 `AttentionBackend` 基类接口、`KVCache` 或 `ModelRunner.forward()`。KSE 通过在现有后端处理之前**重写 attention 元数据**（page table、seq_lens、mask）来实现功能。

```mermaid
graph TB
    subgraph Scheduler
        S[Scheduler] -->|"分配 KV，构建 batch"| SB[ScheduleBatch]
        SB -->|"get_model_worker_batch()"| MWB[ModelWorkerBatch]
    end

    subgraph ModelRunner
        MWB -->|"ForwardBatch.init_new()"| FB[ForwardBatch]
        FB -->|"init_forward_metadata()"| AB[AttentionBackend]
    end

    subgraph 模型前向传播
        AB --> L1[Layer 0: Attention]
        L1 --> L2[Layer 1: Attention]
        L2 --> LN[Layer N: Attention]
    end

    subgraph "KVCache 稀疏引擎 (KSE)"
        KSE_CTRL[KSEController]
        SP[SparsityPolicy]
        MA[MetadataAdapter]
        EP[EvictionPolicy]
    end

    S -. "on_prefill_complete()" .-> KSE_CTRL
    KSE_CTRL -. "evict()" .-> EP
    EP -. "释放 token 槽位" .-> S

    FB -. "on_forward_begin()" .-> KSE_CTRL
    L1 -. "on_attention_begin(q, layer)" .-> KSE_CTRL
    KSE_CTRL -. "select()" .-> SP
    SP -. "SelectionResult" .-> MA
    MA -. "重写元数据" .-> AB

    style KSE_CTRL fill:#e1f5fe
    style SP fill:#e8f5e9
    style MA fill:#fff3e0
    style EP fill:#fce4ec
```

### 1.4 核心设计原则：元数据重写

核心洞察在于：所有 attention 后端（FlashInfer、FlashAttention、Triton）最终都消费**索引数组**（`kv_indices`、`page_table`、`kv_indptr`）和**长度数组**（`seq_lens`、`cache_seqlens`）来决定访问哪些 KV 条目。KSE 不触碰 KV 数据本身 — 仅重写这些元数据数组，使其指向选中的子集。

这意味着：
- **零拷贝** — 对于仅选择（非驱逐）的策略，不产生任何 KV 数据搬运
- **后端无关** — 适用于任何使用分页/索引 KV 访问的后端
- **CUDA Graph 兼容** — 元数据缓冲区可预分配并原地重写

---

## 2. 核心抽象与接口

### 2.1 SelectionResult

任何稀疏策略的通用输出，描述哪些 KV 条目应保留用于 attention 计算。

```python
@dataclass
class SelectionResult:
    """稀疏策略 select() 调用的输出。

    所有索引张量使用*逻辑*位置（每个请求序列内从 0 开始的偏移量）。
    MetadataAdapter 通过 req_to_token 将其转换为物理 KV Cache 位置。
    """

    class Granularity(Enum):
        TOKEN = "token"      # 索引为单个 token 位置
        PAGE = "page"        # 索引为 page 索引（乘以 page_size 得到 token 位置）

    granularity: Granularity

    # [batch_size, max_selected] — 逻辑索引，用 -1 填充
    selected_indices: torch.Tensor

    # [batch_size] — 每个请求的实际有效条目数
    valid_lengths: torch.Tensor

    # [batch_size] — 哪些请求实际使用稀疏（其余使用全量 KV）
    sparse_mask: torch.Tensor

    # 可选的逐层覆盖：若设置，仅对这些 layer id 生效
    layer_ids: Optional[List[int]] = None
```

### 2.2 SparsityPolicy（抽象基类）

所有稀疏策略的算法无关接口。

```python
class SparsityPolicy(ABC):
    """所有 KV Cache 稀疏策略的基类。

    一个策略回答一个问题：给定当前状态，哪些 KV 条目应参与 attention 计算？

    生命周期：
        1. __init__()              — 解析配置，分配缓冲区
        2. on_request_begin(req)   — 注册每请求状态
        3. on_prefill_complete()   — 构建初始表征（可选）
        4. select()                — 按频率调用，产生 SelectionResult
        5. on_attention_complete() — 更新表征（可选）
        6. on_request_end(req)     — 清理资源
    """

    class Frequency(Enum):
        PER_REQUEST = "per_request"   # prefill 后执行一次，结果复用于所有 decode step
        PER_STEP = "per_step"         # 每个 decode step 执行一次，跨层共享
        PER_LAYER = "per_layer"       # 每个 decode step 的每层执行一次

    @abstractmethod
    def frequency(self) -> "SparsityPolicy.Frequency":
        """该策略的 select() 需要以什么频率被调用？"""
        ...

    @abstractmethod
    def select(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> SelectionResult:
        """产生一组 KV 条目的选择结果。

        Args:
            query: [batch, num_heads, head_dim]，对于不含 Query 的策略可为 None。
            layer_id: 当前层索引。
            req_pool_indices: [batch] req_to_token_pool 中的索引。
            seq_lens: [batch] 当前序列长度。
            forward_batch: 完整的 batch 上下文。

        Returns:
            描述哪些 KV 参与 attention 的 SelectionResult。
        """
        ...

    def on_request_begin(self, req) -> None:
        """当新请求进入系统时调用。"""
        pass

    def on_request_end(self, req) -> None:
        """当请求完成或中止时调用。"""
        pass

    def on_prefill_complete(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """在每层的 prefill attention 之后调用。构建初始表征。"""
        pass

    def on_attention_complete(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """在每次 decode attention 之后调用。增量更新表征。"""
        pass
```

### 2.3 EvictionPolicy（可选 Mixin）

用于永久删除 KV 条目的算法（D1: 部分保留）。

```python
class EvictionPolicy(ABC):
    """用于永久驱逐 KV Cache 条目的策略 Mixin。

    驱逐会物理释放 TokenToKVPoolAllocator 中的 token 槽位，
    降低内存压力，但被驱逐的 token 不可恢复。
    """

    @abstractmethod
    def compute_eviction(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """返回每个请求需要驱逐的 token 位置。

        Returns:
            evict_indices: [batch, max_evict] 逻辑 token 位置，用 -1 填充
        """
        ...
```

### 2.4 MetadataAdapter

将 `SelectionResult` 转换为后端特定的元数据重写。

```python
class MetadataAdapter(ABC):
    """将 SelectionResult 适配为后端特定的 attention 元数据。

    每个 attention 后端有自己的元数据格式（FlashInfer 使用
    kv_indptr/kv_indices，FlashAttention 使用 page_table/cache_seqlens，
    Triton 使用 kv_indptr/kv_indices）。适配器知道如何重写这些结构
    以反映稀疏选择。
    """

    @abstractmethod
    def save_dense_metadata(self, forward_metadata: Any) -> None:
        """在任何稀疏重写之前快照原始（稠密）元数据。
        在每次前向传播开始时（第一个稀疏层）调用一次。
        """
        ...

    @abstractmethod
    def apply(
        self,
        result: SelectionResult,
        forward_metadata: Any,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Any:
        """原地重写 forward_metadata 以反映稀疏选择。

        对于非稀疏请求（result.sparse_mask == False），元数据保持不变（稠密 attention）。
        """
        ...

    @abstractmethod
    def restore_dense_metadata(self, forward_metadata: Any) -> None:
        """在稀疏层完成后恢复原始稠密元数据。
        当仅部分层使用稀疏 attention 时调用。
        """
        ...
```

### 2.5 KSEController

连接策略、适配器和前向路径的中央协调器。

```python
class KSEController:
    """KVCache 稀疏引擎的中央协调器。

    管理稀疏策略和元数据适配器的生命周期。
    提供 hook 方法，通过最小的代码改动从现有 sglang 前向路径中调用。

    集成点（现有代码仅需 4 行改动）：
        1. Scheduler.on_prefill_complete()  → controller.after_prefill()
        2. ModelRunner.forward_decode()     → controller.before_forward()
        3. AttentionBackend.forward()       → controller.before_attention()
        4. AttentionBackend.forward()       → controller.after_attention()
    """

    def __init__(
        self,
        policy: SparsityPolicy,
        adapter: MetadataAdapter,
        eviction: Optional[EvictionPolicy],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        config: KSEConfig,
    ):
        self.policy = policy
        self.adapter = adapter
        self.eviction = eviction
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.config = config

        # 缓存上一次的 SelectionResult，用于 PER_REQUEST 和 PER_STEP 频率
        self._cached_result: Optional[SelectionResult] = None
        self._cached_step: int = -1
        self._metadata_saved: bool = False

    # ---- Hook: 请求生命周期 ----

    def on_request_begin(self, req) -> None:
        self.policy.on_request_begin(req)

    def on_request_end(self, req) -> None:
        self.policy.on_request_end(req)

    # ---- Hook: Prefill 完成后 ----

    def after_prefill(
        self,
        forward_batch: ForwardBatch,
    ) -> None:
        """在 prefill 完成后调用。触发表征构建和可选的驱逐。"""
        for layer_id in range(self.config.start_layer, self.config.end_layer):
            k_buf = self.token_to_kv_pool.get_key_buffer(layer_id)
            v_buf = self.token_to_kv_pool.get_value_buffer(layer_id)
            self.policy.on_prefill_complete(
                layer_id,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                k_buf, v_buf,
                forward_batch,
            )

        if self.eviction is not None:
            self._do_eviction(forward_batch)

        # 对于 PER_REQUEST 策略，仅计算一次选择
        if self.policy.frequency() == SparsityPolicy.Frequency.PER_REQUEST:
            self._cached_result = self.policy.select(
                query=None,
                layer_id=-1,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_batch=forward_batch,
            )

    # ---- Hook: Decode 前向传播之前 ----

    def before_forward(self, forward_batch: ForwardBatch) -> None:
        """在每次 decode 前向传播之前调用。"""
        # 对于 PER_STEP 策略，每步计算一次选择
        if self.policy.frequency() == SparsityPolicy.Frequency.PER_STEP:
            self._cached_result = None  # 将在第一次 attention_begin 时计算
        self._metadata_saved = False

    # ---- Hook: 每层 Attention 之前 ----

    def before_attention(
        self,
        query: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
        forward_metadata: Any,
    ) -> Any:
        """在 attention 计算之前调用。返回重写后的元数据。"""
        if not self._should_apply(layer_id, forward_batch):
            return forward_metadata

        # 在第一个稀疏层保存稠密元数据
        if not self._metadata_saved:
            self.adapter.save_dense_metadata(forward_metadata)
            self._metadata_saved = True

        # 获取或计算选择结果
        result = self._get_selection(query, layer_id, forward_batch)

        # 重写元数据
        return self.adapter.apply(result, forward_metadata, forward_batch, layer_id)

    # ---- Hook: 每层 Attention 之后 ----

    def after_attention(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> None:
        """在 attention 之后调用。更新表征。"""
        if not forward_batch.forward_mode.is_decode():
            return
        k_buf = self.token_to_kv_pool.get_key_buffer(layer_id)
        v_buf = self.token_to_kv_pool.get_value_buffer(layer_id)
        self.policy.on_attention_complete(
            layer_id,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            k_buf, v_buf,
            forward_batch,
        )

    # ---- 内部方法 ----

    def _get_selection(
        self, query, layer_id, forward_batch
    ) -> SelectionResult:
        freq = self.policy.frequency()
        if freq == SparsityPolicy.Frequency.PER_REQUEST:
            return self._cached_result
        if freq == SparsityPolicy.Frequency.PER_STEP:
            if self._cached_result is None:
                self._cached_result = self.policy.select(
                    query=query,
                    layer_id=layer_id,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    forward_batch=forward_batch,
                )
            return self._cached_result
        # PER_LAYER: 始终重新计算
        return self.policy.select(
            query=query,
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_batch=forward_batch,
        )

    def _should_apply(self, layer_id, forward_batch) -> bool:
        if not forward_batch.forward_mode.is_decode():
            return False
        if layer_id < self.config.start_layer or layer_id >= self.config.end_layer:
            return False
        return True

    def _do_eviction(self, forward_batch):
        evict_indices = self.eviction.compute_eviction(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch,
        )
        # 通过 allocator 释放被驱逐的 token 槽位
        # （实现委托给 TokenToKVPoolAllocator.free）
        ...
```

### 2.6 KSEConfig 与工厂

```python
@dataclass
class KSEConfig:
    """KVCache 稀疏引擎的配置。"""
    policy_name: str                    # 例如 "quest", "h2o", "streaming_llm"
    backend_name: str                   # 例如 "flashattention", "flashinfer", "triton"
    start_layer: int = 0                # 应用稀疏的起始层
    end_layer: int = -1                 # 结束层（不含），-1 表示所有层
    min_seq_len: int = 2048             # 激活稀疏的最小 seq_len
    page_size: int = 64                 # page 粒度策略的 page 大小
    policy_kwargs: dict = field(default_factory=dict)  # 算法特定参数


# ---- 注册表与工厂 ----

_POLICY_REGISTRY: Dict[str, Type[SparsityPolicy]] = {}
_ADAPTER_REGISTRY: Dict[str, Type[MetadataAdapter]] = {}

def register_policy(name: str):
    """装饰器：注册一个 SparsityPolicy 实现。"""
    def wrapper(cls):
        _POLICY_REGISTRY[name] = cls
        return cls
    return wrapper

def register_adapter(name: str):
    """装饰器：注册一个 MetadataAdapter 实现。"""
    def wrapper(cls):
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return wrapper

def create_kse_controller(
    config: KSEConfig,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool: KVCache,
    device: torch.device,
) -> KSEController:
    policy_cls = _POLICY_REGISTRY[config.policy_name]
    adapter_cls = _ADAPTER_REGISTRY[config.backend_name]

    policy = policy_cls(config, device)
    adapter = adapter_cls(device)
    eviction = policy if isinstance(policy, EvictionPolicy) else None

    return KSEController(
        policy=policy,
        adapter=adapter,
        eviction=eviction,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        config=config,
    )
```

### 2.7 维度覆盖矩阵

| 算法 | D1 驱逐 | D2 粒度 | D3 频率 | D4 策略 |
|---|---|---|---|---|
| **H2O** | 驱逐 | Token | 每步 | Query-Aware（attention score 累积） |
| **StreamingLLM** | 驱逐 | Token | 每请求 | 固定策略（sink + 滑动窗口） |
| **Quest** | 全量保留 | Page | 每层 | Query-Aware（bounding-box） |
| **SnapKV** | 驱逐 | Token | 每请求 | Query-Aware（观察窗口投票） |
| **ChunkKV** | 全量保留 | Page/Chunk | 每层 | Query-Aware（chunk 重要性） |
| **DeepSeek NSA** | 全量保留 | Page | 每层 | 原生（模型内置 indexer） |

**不支持的组合及原因：**

- **Cluster 粒度 + 每层频率**：Cluster 分配（如 k-means）在每层重新计算的代价过高。框架通过 `Granularity.TOKEN` 配合 cluster-aware 策略来支持——预先将 token 分配到 cluster 并按 cluster 选择，但 cluster 结构本身应在 `on_prefill_complete` 时构建。
- **驱逐 + 每层频率**：驱逐是破坏性操作，不应在前向传播中途发生。框架强制 `EvictionPolicy.compute_eviction()` 仅在 `after_prefill()` 或 `before_forward()` 中调用，不会在 `before_attention()` 内部调用。

---

## 3. 工作流实现

### 3.1 整体工作流

```mermaid
sequenceDiagram
    participant Scheduler
    participant ModelRunner
    participant KSEController
    participant SparsityPolicy
    participant MetadataAdapter
    participant AttentionBackend

    Note over Scheduler: 请求到达
    Scheduler->>KSEController: on_request_begin(req)
    KSEController->>SparsityPolicy: on_request_begin(req)

    Note over Scheduler: Prefill 阶段
    Scheduler->>ModelRunner: forward_extend(batch)
    ModelRunner->>AttentionBackend: forward_extend(q, k, v, layer, batch)
    Note over AttentionBackend: 正常的稠密 prefill attention

    Note over Scheduler: Prefill 完成后
    Scheduler->>KSEController: after_prefill(batch)
    KSEController->>SparsityPolicy: on_prefill_complete(layer, ...)
    Note over KSEController: [若为驱逐策略] 驱逐 token

    Note over Scheduler: Decode 阶段开始
    loop 每个 decode step
        Scheduler->>ModelRunner: forward_decode(batch)
        ModelRunner->>KSEController: before_forward(batch)

        loop 每层
            ModelRunner->>AttentionBackend: forward_decode(q, k, v, layer, batch)
            AttentionBackend->>KSEController: before_attention(q, layer_id, metadata)
            KSEController->>SparsityPolicy: select(q, layer_id, ...)
            SparsityPolicy-->>KSEController: SelectionResult
            KSEController->>MetadataAdapter: apply(result, metadata)
            MetadataAdapter-->>AttentionBackend: 重写后的元数据
            Note over AttentionBackend: 在选中的 KV 上执行稀疏 attention
            AttentionBackend->>KSEController: after_attention(layer_id)
            KSEController->>SparsityPolicy: on_attention_complete(layer_id, ...)
        end
    end

    Note over Scheduler: 请求完成
    Scheduler->>KSEController: on_request_end(req)
```

### 3.2 具体示例：Quest 算法

Quest 是一个 Query-Aware、Page 粒度、每层频率的稀疏 attention 算法。它为每个 page 维护 key 的 bounding box（min/max），并通过上界估计 attention score 来对 page 评分。

**步骤 1：注册**

```python
@register_policy("quest")
class QuestPolicy(SparsityPolicy):
    def __init__(self, config: KSEConfig, device: torch.device):
        self.page_size = config.page_size
        self.sparsity_ratio = config.policy_kwargs.get("sparsity_ratio", 0.7)
        self.num_recent_pages = config.policy_kwargs.get("num_recent_pages", 4)
        self.page_k_min = {}   # layer_id -> [num_pages, num_heads, head_dim]
        self.page_k_max = {}
        self.page_valid = {}
        ...

    def frequency(self) -> SparsityPolicy.Frequency:
        return SparsityPolicy.Frequency.PER_LAYER
```

**步骤 2：在 prefill 时构建表征**

```python
    def on_prefill_complete(self, layer_id, req_pool_indices, seq_lens,
                            k_buffer, v_buffer, forward_batch):
        # 对每个请求，计算每个 page 的 key min/max
        num_pages = seq_lens // self.page_size
        for i, req_idx in enumerate(req_pool_indices):
            token_indices = req_to_token[req_idx, :seq_lens[i]]
            keys = k_buffer[token_indices]  # [seq_len, num_heads, head_dim]
            # 重塑为 [num_pages, page_size, num_heads, head_dim]
            paged_keys = keys[:num_pages[i] * self.page_size].view(
                num_pages[i], self.page_size, -1, keys.shape[-1]
            )
            phys_pages = token_indices[::self.page_size][:num_pages[i]] // self.page_size
            self.page_k_min[layer_id][phys_pages] = paged_keys.amin(dim=1)
            self.page_k_max[layer_id][phys_pages] = paged_keys.amax(dim=1)
            self.page_valid[layer_id][phys_pages] = True
```

**步骤 3：decode 阶段的每层选择**

```python
    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kwargs) -> SelectionResult:
        bs = query.shape[0]
        all_indices = []
        all_lengths = []

        for i in range(bs):
            num_pages = seq_lens[i] // self.page_size
            # 对每个 page 评分：通过 bounding box 计算 q·k 的上界
            q_i = query[i]  # [num_heads, head_dim]
            k_min = self.page_k_min[layer_id][:num_pages]
            k_max = self.page_k_max[layer_id][:num_pages]
            scores = torch.where(q_i >= 0, q_i * k_max, q_i * k_min).sum(dim=(-2, -1))

            # 选择 top-k pages + 最近的 pages
            k = int(num_pages * self.sparsity_ratio)
            topk = scores.topk(k).indices
            recent = torch.arange(num_pages - self.num_recent_pages, num_pages)
            selected = torch.cat([topk, recent]).unique().sort().values
            all_indices.append(selected)
            all_lengths.append(len(selected))

        # 填充并堆叠
        max_len = max(all_lengths)
        indices = torch.full((bs, max_len), -1, dtype=torch.int32)
        for i, sel in enumerate(all_indices):
            indices[i, :len(sel)] = sel

        return SelectionResult(
            granularity=SelectionResult.Granularity.PAGE,
            selected_indices=indices,
            valid_lengths=torch.tensor(all_lengths, dtype=torch.int32),
            sparse_mask=seq_lens >= self.config.min_seq_len,
        )
```

**步骤 4：MetadataAdapter 重写 page_table**

以 FlashAttention 后端为例：

```python
@register_adapter("flashattention")
class FlashAttentionAdapter(MetadataAdapter):
    def apply(self, result, forward_metadata, forward_batch, layer_id):
        if result.granularity == SelectionResult.Granularity.PAGE:
            # 重写 page_table：仅保留选中的 pages
            for i in range(result.selected_indices.shape[0]):
                if not result.sparse_mask[i]:
                    continue
                n_sel = result.valid_lengths[i]
                logical_pages = result.selected_indices[i, :n_sel]
                # 通过 req_to_token 将逻辑 page 索引映射为物理 pages
                phys_pages = self._logical_to_physical(
                    logical_pages, forward_batch, i
                )
                forward_metadata.page_table[i, :n_sel] = phys_pages
            # 更新 cache_seqlens 以反映缩减后的 KV 长度
            forward_metadata.cache_seqlens_int32 = torch.where(
                result.sparse_mask,
                result.valid_lengths * forward_batch.token_to_kv_pool.page_size,
                forward_metadata.cache_seqlens_int32,
            )
        return forward_metadata
```

### 3.3 具体示例：StreamingLLM（驱逐策略）

StreamingLLM 保留固定数量的 "sink" token（初始 token）加上一个滑动窗口的最近 token，驱逐中间的所有 token。

```python
@register_policy("streaming_llm")
class StreamingLLMPolicy(SparsityPolicy, EvictionPolicy):
    def __init__(self, config: KSEConfig, device: torch.device):
        self.num_sink_tokens = config.policy_kwargs.get("num_sink_tokens", 4)
        self.window_size = config.policy_kwargs.get("window_size", 1024)

    def frequency(self) -> SparsityPolicy.Frequency:
        return SparsityPolicy.Frequency.PER_REQUEST

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kwargs) -> SelectionResult:
        # 驱逐后，所有剩余 token 都参与计算 — 返回全量范围
        # （驱逐已经移除了中间的 token）
        return SelectionResult(
            granularity=SelectionResult.Granularity.TOKEN,
            selected_indices=...,  # 所有剩余 token 位置
            valid_lengths=seq_lens,
            sparse_mask=torch.ones(len(seq_lens), dtype=torch.bool),
        )

    def compute_eviction(self, req_pool_indices, seq_lens, forward_batch):
        # 驱逐 sink 和窗口之间的 token
        evict_list = []
        for i in range(len(req_pool_indices)):
            n = seq_lens[i]
            if n <= self.num_sink_tokens + self.window_size:
                evict_list.append(torch.empty(0, dtype=torch.int32))
                continue
            evict_start = self.num_sink_tokens
            evict_end = n - self.window_size
            evict_list.append(torch.arange(evict_start, evict_end))
        # 填充并返回
        ...
```

### 3.4 现有代码中的集成点

KSE 在现有 sglang 代码库中仅需 **4 个最小插入点**：

| 位置 | 改动 | 代码 |
|---|---|---|
| `Scheduler.run_batch()` | Prefill 完成后 | `if kse: kse.after_prefill(forward_batch)` |
| `ModelRunner.forward_decode()` | 模型前向之前 | `if kse: kse.before_forward(forward_batch)` |
| `AttentionBackend.forward_decode()` | Attention 内核之前 | `if kse: metadata = kse.before_attention(q, layer_id, batch, metadata)` |
| `AttentionBackend.forward_decode()` | Attention 内核之后 | `if kse: kse.after_attention(layer_id, batch)` |

无需修改 `AttentionBackend` 基类、`KVCache`、`ReqToTokenPool`、`ForwardBatch` 或任何模型代码。

---

## 4. 可扩展性与性能

### 4.1 原生稀疏注意力（DeepSeek NSA）支持

DeepSeek NSA 是一种**模型原生**的稀疏注意力机制，其稀疏模式由模型架构内部的学习型 indexer 模块决定（而非事后优化）。这带来了独特的挑战：indexer 作为模型前向传播的一部分运行，产生的稀疏索引直接输入到专用的 sparse MLA 内核中。

**KSE 如何适配 NSA：**

NSA 可以作为 `PER_LAYER` 频率的 `SparsityPolicy` 集成，其中 `select()` 委托给模型的原生 indexer：

```python
@register_policy("deepseek_nsa")
class NSAPolicy(SparsityPolicy):
    def frequency(self):
        return SparsityPolicy.Frequency.PER_LAYER

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kwargs):
        indexer = kwargs["indexer"]
        # indexer 是一个模型组件，产生 top-k page 索引
        sparse_indices = indexer(
            x=kwargs["x"], q_lora=kwargs["q_lora"],
            positions=kwargs["positions"],
            forward_batch=forward_batch, layer_id=layer_id,
        )
        return SelectionResult(
            granularity=SelectionResult.Granularity.PAGE,
            selected_indices=sparse_indices,
            valid_lengths=...,
            sparse_mask=torch.ones(len(req_pool_indices), dtype=torch.bool),
        )
```

然而，NSA 还使用**专用内核**（通过 TileLang/Triton 实现的 sparse MLA），这些内核与其元数据格式紧密耦合。对于 NSA，`MetadataAdapter` 必须生成 NSA 特有的元数据（`NSAMetadata`，包含 `page_table_1`、`real_page_table` 等），而非通用的 page table 重写。

这通过注册 NSA 专用适配器来处理：

```python
@register_adapter("nsa")
class NSAMetadataAdapter(MetadataAdapter):
    def apply(self, result, forward_metadata, forward_batch, layer_id):
        # 将 SelectionResult 转换为 NSA 特有的 page tables
        # (page_table_1, real_page_table, nsa_cache_seqlens 等)
        ...
```

**限制**：NSA 的 indexer 深度嵌入在模型的前向传播中（它使用中间隐藏状态，而不仅仅是 Q）。KSE 可以包装它，但无法将其与模型架构完全解耦。这是模型原生稀疏性的固有属性。

### 4.2 选择开销分析

KSE 引入的主要开销是 `SparsityPolicy.select()` 中的**选择计算**。按频率分析如下：

| 频率 | 每 decode step 的开销 | 摊销方式 |
|---|---|---|
| PER_REQUEST | 0（在 prefill 时计算一次） | 完全摊销到所有 decode step |
| PER_STEP | 1 × select() | 跨所有层共享 |
| PER_LAYER | N_layers × select() | 无摊销 — 每步承担全部开销 |

**定量估算**（Quest，PER_LAYER，batch_size=32，seq_len=8K，page_size=64）：
- 每请求的 page 数：128
- Score 计算：32 × 128 次 [num_heads, head_dim] 矩阵乘 ≈ 0.1ms（A100）
- Top-k 选择：32 × topk(128) ≈ 0.01ms
- 元数据重写：page_table scatter ≈ 0.02ms
- **总计：每层约 0.13ms**，而 8K 长度下全量 attention 约 0.5ms
- **净加速**：attention 从 0.5ms 降至约 0.2ms（稀疏），每层净收益约 0.17ms

### 4.3 优化策略

**1. 融合选择内核**

对于 PER_LAYER 策略，选择开销在关键路径上。将 score 计算和 top-k 融合为单个 Triton 内核，可消除内核启动开销和中间内存分配：

```python
@triton.jit
def fused_page_score_topk_kernel(
    query_ptr, k_min_ptr, k_max_ptr, output_indices_ptr,
    num_pages, k, BLOCK_SIZE: tl.constexpr,
):
    # 在一次 pass 中计算 bounding-box score 并选择 top-k
    ...
```

**2. PER_STEP 缓存选择**

对于 PER_STEP 策略，选择仅计算一次并跨所有层复用。`KSEController._cached_result` 机制确保零冗余计算。

**3. CUDA Graph 兼容性**

元数据重写操作的是预分配的张量（`page_table`、`kv_indptr`、`cache_seqlens`），这些张量已经是 CUDA Graph 捕获的一部分。适配器执行原地更新，使其与 CUDA Graph 回放兼容，无需重新捕获。

**4. 异步表征更新**

对于在 attention 后更新表征的算法（例如 Quest 为新 token 更新 page bounding box），更新可以与同一层的 MLP 计算使用 CUDA stream 重叠：

```
Layer L: [Attention] → [KSE 更新 (异步)] → [MLP]
                         ↓ (重叠)
Layer L: [Attention] → [MLP + KSE 更新在 alt stream 上]
```

**5. 批量化选择**

当前 `select()` 中的逐请求循环是瓶颈。使用填充张量和掩码操作的批量化实现可消除 Python 循环：

```python
# 批量 page 评分：[batch, num_pages, num_heads, head_dim]
scores = torch.where(q.unsqueeze(1) >= 0,
                     q.unsqueeze(1) * k_max_batched,
                     q.unsqueeze(1) * k_min_batched).sum(dim=(-2, -1))
topk_indices = scores.topk(k, dim=1).indices  # [batch, k]
```

### 4.4 内存开销

KSE 本身引入的内存开销极小：

| 组件 | 内存 | 备注 |
|---|---|---|
| `SelectionResult` 张量 | O(batch × max_selected) | 跨步复用 |
| 稠密元数据快照 | O(batch × max_pages) | page_table 的一份拷贝 |
| 策略表征 | 取决于算法 | Quest: O(num_pages × num_heads × head_dim × num_layers) |

以 Quest 为例，100K pages、8 heads、128 dim、80 layers：表征约 8GB。可通过跨 GQA 组共享表征或使用低精度（FP16 → FP8）来减少。
