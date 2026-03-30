# 实验方案：CXL 内存池化 × 稀疏注意力 KV Cache 推理实验

> 目标：在当前 SGLang 代码库上，以**最小改动**实现论文所需的全部实验。使用 **真实 RDMA 和 CXL 2.0 硬件**，在小模型上以 Quest 稀疏注意力证明 CXL 内存池化在 sparse attention 场景下相比 RDMA 的优势。

---

## 0. 核心设计思路

### 0.1 关键发现：SGLang 已有可复用的组件

| 组件 | 位置 | 状态 | 说明 |
|------|------|------|------|
| **SparseCoordinator 框架** | `mem_cache/sparsity/` | ✅ 已有 | 通用稀疏注意力协调框架（未接入 attention 路径） |
| **Quest 算法** | `mem_cache/sparsity/algorithms/quest_algorithm.py` | ✅ 已实现 | Page-wise query-aware top-k 选择 |
| **FlashAttention 适配器** | `mem_cache/sparsity/backend/backend_adaptor.py` | ✅ 已实现 | 将 sparse top-k 映射为 FA page table 修改 |
| **HiCache 分层缓存** | `mem_cache/memory_pool_host.py` | ✅ 已有 | GPU(L1) → CPU(L2) 传输（pinned memory + sgl_kernel） |
| **HostTensorAllocator** | `mem_cache/memory_pool_host.py` | ✅ 可插件化 | 自定义 host 内存分配器（如 Mooncake 的 RDMA 注册内存） |
| **HiCache 存储后端** | `mem_cache/storage/backend_factory.py` | ✅ 可扩展 | 支持注册新后端（`"dynamic"` 模式可加载自定义类） |
| **Mooncake RDMA** | `mem_cache/storage/mooncake_store/` | ✅ 已有 | 完整的 RDMA KV 传输实现（可参考接口模式） |

### 0.2 实验核心方法论

**使用真实硬件进行实测**：

- **Local DRAM**：SGLang 已有的 `cudaHostRegister` pinned memory，CPU 直接访问
- **RDMA 远端内存池**：利用 SGLang 已有的 Mooncake RDMA 基础设施，或直接用 RDMA verbs 注册远端内存
- **CXL 共享内存池**：通过 CXL 2.0 设备暴露的共享内存区域，实现自定义 `HostTensorAllocator` 将 KV Cache 分配到 CXL 内存

### 0.3 整体架构

```
                        ┌──────────────────────────────────────┐
                        │              GPU (VRAM)               │
                        │   模型权重 + Decode hot KV buffer      │
                        └───────┬────────────────┬─────────────┘
                      sgl_kernel│transfer        │sgl_kernel transfer
                   (load_to_device)            (backup_from_device)
                        ┌───────▼────────────────▼─────────────┐
                        │         KV Cache 存储层               │
                        │                                       │
                 ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
                 │ 方案A: DRAM  │  │ 方案B: RDMA  │  │ 方案C: CXL  │
                 │ (pinned mem) │  │ (远端内存池)  │  │ (共享内存池) │
                 │ cudaHostReg  │  │ RDMA WRITE/  │  │ mmap CXL    │
                 │              │  │ READ verbs   │  │ load/store  │
                 └──────────────┘  └──────────────┘  └──────────────┘
```

**关键差异点**：三种方案的区别仅在于 **KV Cache host 内存的来源（分配器）**，以及 **host↔remote 的传输方式**。GPU↔host 的传输路径（sgl_kernel / JIT hicache）保持不变。

---

## 1. 三种内存后端的具体实现

### 1.1 方案 A：Local DRAM（基线）

**无需任何代码改动**。使用 SGLang 已有的 HiCache：

```
GPU ←──sgl_kernel──→ CPU pinned memory (cudaHostRegister)
```

- `HostTensorAllocator` 默认分配 `torch.empty(..., device="cpu")`
- `alloc_with_host_register` 对其执行 `cudaHostRegister`
- 延迟特征：~100ns 本地 DRAM 访问

**对应论文场景**：§5.2 方案 A（Sparse Attention + DRAM Offloading，上限基线）

### 1.2 方案 B：RDMA 远端内存池

**两种实现路径（按优先级）**：

#### 路径 B1：复用 Mooncake（推荐，改动最小）

SGLang 已有完整的 Mooncake RDMA HiCache 集成。数据流：

```
GPU ←──sgl_kernel──→ CPU pinned memory ←──Mooncake RDMA──→ 远端内存池
                     (MooncakeHostTensor-                    (另一台机器)
                      Allocator 分配)
```

- Mooncake `MooncakeDistributedStore` 提供 `batch_get_into` / `batch_put_from` 零拷贝 RDMA
- `MooncakeHostTensorAllocator` 通过 `mooncake.store.MooncakeHostMemAllocator` 分配 RDMA 注册内存
- 传输路径：HiCache controller 的 prefetch（storage→host）和 backup（host→storage）

**关键操作流程**：
```
Prefill:
  1. KV 计算后写入 GPU KV buffer
  2. GPU → host pinned memory (sgl_kernel backup)
  3. host → RDMA 远端 (Mooncake batch_put_from) [异步]

Decode (全量预取模式):
  1. RDMA 远端 → host pinned memory (Mooncake batch_get_into)
  2. host → GPU (sgl_kernel load)
  3. Quest 选 top-k → FlashAttention

Decode (按需取模式):
  1. Quest 选 top-k pages
  2. 仅 top-k pages: RDMA → host (batch_get_into, 逐层)
  3. host → GPU (sgl_kernel load per layer)
  4. FlashAttention on selected pages
```

**配置**：
```bash
# 启动 Mooncake storage (远端节点)
# ... Mooncake 集群配置 ...

# SGLang 服务端
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=mlx5_0  # RDMA NIC

python -m sglang.launch_server \
  --model $MODEL \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake \
  --hicache-ratio 2 \
  --page-size 16 \
  # ... Quest 相关参数 ...
```

#### 路径 B2：自定义 RDMA allocator（如果 Mooncake 不可用）

实现一个 `RDMAHostTensorAllocator`，通过 RDMA verbs 注册远端内存：

```python
class RDMAHostTensorAllocator(HostTensorAllocator):
    """将 host KV 分配到 RDMA 注册的远端内存"""
    def __init__(self, rdma_config):
        super().__init__()
        self.rdma_conn = RDMAConnection(rdma_config)

    def allocate(self, dims, dtype, device):
        size = int(np.prod(dims)) * torch.tensor([], dtype=dtype).element_size()
        ptr = self.rdma_conn.alloc_remote(size)  # RDMA malloc on remote
        buf = ctypes_buffer_from_ptr(ptr, size)
        tensor = torch.frombuffer(buf, dtype=dtype).view(dims)
        return tensor
```

### 1.3 方案 C：CXL 共享内存池

**核心思路**：CXL 2.0 设备将一段内存暴露为 CPU 可直接 load/store 的地址空间。我们通过 `mmap` 将 CXL 设备的共享内存区域映射到进程地址空间，然后实现一个 `CXLHostTensorAllocator`。

```
GPU ←──sgl_kernel──→ CXL 内存 (mmap 到进程地址空间)
                     load/store 语义，cache-line 粒度
```

**CXL 共享内存池操作模式**（待用户提供具体 API 示例后细化）：

```python
class CXLHostTensorAllocator(HostTensorAllocator):
    """将 host KV 分配到 CXL 共享内存池"""
    def __init__(self, cxl_config):
        super().__init__()
        # 打开 CXL 设备，mmap 共享内存区域
        # 方式取决于具体 CXL 设备的 API：
        #   方式 1: /dev/dax 设备 → mmap
        #   方式 2: CXL shared memory library API
        #   方式 3: numactl + 指定 CXL NUMA node
        self.cxl_pool = CXLMemoryPool(cxl_config)

    def allocate(self, dims, dtype, device):
        size = int(np.prod(dims)) * torch.tensor([], dtype=dtype).element_size()
        # 从 CXL 池分配内存
        ptr = self.cxl_pool.alloc(size)
        # 包装为 torch.Tensor（零拷贝，直接映射 CXL 地址）
        buf = (ctypes.c_char * size).from_address(ptr)
        tensor = torch.frombuffer(buf, dtype=dtype).view(dims)
        return tensor
```

**CXL 的关键优势**：
- GPU 通过 `sgl_kernel` 将 KV 写入 CXL 内存时，走的是 PCIe/CXL 链路，**无需额外的 RDMA 传输步骤**
- Decode 时 Quest 选出 top-k 后，GPU 直接从 CXL 读取所需 KV page，**无全量预取**
- 总路径：`GPU ←→ CXL memory`（单跳），而 RDMA 是 `GPU → host DRAM → RDMA NIC → 远端 → RDMA NIC → host DRAM → GPU`（多跳）

**关于 `cudaHostRegister` 与 CXL 内存**：
- CXL 2.0 内存在 CPU 端呈现为普通物理地址空间（通过 DAX 或 NUMA node 暴露）
- `cudaHostRegister` 可以注册 CXL mmap 的内存，使 GPU 可以直接 DMA 访问
- 这是 CXL 天然适配 GPU KV offloading 的关键特性

---

## 2. Quest 稀疏注意力接入方案

### 2.1 为什么选 Quest + 小模型

- Quest 是 query-aware、page 粒度、逐层动态的稀疏注意力算法
- 其访存模式与 DeepSeek-V3.2 NSA 本质一致（论文 §7 论证）
- SGLang 已有 `QuestAlgorithm` 实现和 `FlashAttentionAdaptor`
- 可运行在 Llama-3-8B / Qwen2.5-7B 等单卡小模型上

### 2.2 技术路线

```
                ┌─────────────────────────────────────┐
                │      FlashAttention Backend          │
                │      forward_decode()                │
                │                                      │
                │  ① save_original_metadata (layer 0)  │
                │  ② Quest retrieve_topk               │──→ QuestAlgorithm
                │  ③ adapt_for_attn_metadata           │──→ FlashAttentionAdaptor
                │  ④ [新增] offload: load top-k pages  │──→ HostKVCache.load_to_device
                │  ⑤ flash_attn_with_kvcache           │    (从 DRAM/RDMA/CXL)
                │  ⑥ construct/update representations  │
                └─────────────────────────────────────┘
```

### 2.3 Attention Backend 选择：必须使用 FlashAttention (FA3)

`FlashAttentionAdaptor` 直接修改 `FlashAttentionMetadata` 中的 `page_table` 和 `cache_seqlens_int32`，因此 **必须使用 `--attention-backend fa3`**（或 `flashattention`）。FlashInfer 使用不同的 metadata 结构（wrapper-based plan），不兼容当前 adaptor。

### 2.4 代码改动详解

#### 改动 1：在 FlashAttention Backend 中接入 SparseCoordinator

在 `flashattention_backend.py` 的 `forward_decode()` 中，于 `flash_attn_with_kvcache` 调用前后插入 hook：

```python
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
    # ... 原有 KV save 逻辑 ...

    sparse_coordinator = forward_batch.sparse_coordinator
    if sparse_coordinator is not None:
        attn_metadata = sparse_coordinator.attention_begin(
            q, k, v, layer, forward_batch, self.forward_metadata
        )

    # ... 原有 flash_attn_with_kvcache 逻辑 ...

    if sparse_coordinator is not None:
        sparse_coordinator.attention_end(output, layer, forward_batch)

    return output
```

#### 改动 2：在 SparseCoordinator 中集成 KV offloading

扩展 `sparse_coordinator.py`，在 `_handle_sparse_retrieve()` 中加入 host→device 数据搬运：

```python
def _handle_sparse_retrieve(self, query, layer, forward_batch, attn_metadata, **kwargs):
    # 已有：计算 top-k
    selected_indices, valid_lengths = self.algorithm.retrieve_topk(...)

    # [新增] 如果 KV 在 host 上，将 selected pages 加载到 device
    if self.kv_offloader is not None:
        self.kv_offloader.load_selected_pages(
            selected_indices, valid_lengths, layer.layer_id,
            forward_batch.req_pool_indices, forward_batch.seq_lens
        )

    # 已有：适配 attention metadata
    return self.backend_adaptor.adapt_for_attn_metadata(...)
```

#### 改动 3：实现 KV Offloader

新增 `mem_cache/sparsity/kv_offloader.py`，封装 host↔device KV 搬运：

```python
class SparseKVOffloader:
    """管理 sparse attention 场景下的 KV Cache host↔device 搬运"""

    def __init__(self, host_pool, device_pool, req_to_token_pool, page_size):
        self.host_pool = host_pool      # HostKVCache (DRAM/RDMA/CXL)
        self.device_pool = device_pool  # GPU KV pool
        self.req_to_token_pool = req_to_token_pool
        self.page_size = page_size

    def backup_after_prefill(self, device_indices, stream):
        """Prefill 结束后，将全量 KV 从 GPU 备份到 host"""
        host_indices = self.host_pool.alloc(len(device_indices))
        self.host_pool.backup_from_device_all_layer(
            self.device_pool, host_indices, device_indices, io_backend="kernel"
        )
        return host_indices

    def load_selected_pages(self, selected_page_indices, valid_lengths,
                            layer_id, req_pool_indices, seq_lens):
        """Decode 时，仅加载 top-k pages 对应的 KV 到 device"""
        # 将 logical page indices → physical token indices → host slot indices
        # 调用 host_pool.load_to_device_per_layer() 传输
        host_indices, device_indices = self._resolve_indices(
            selected_page_indices, valid_lengths, req_pool_indices
        )
        self.host_pool.load_to_device_per_layer(
            self.device_pool, host_indices, device_indices,
            layer_id, io_backend="kernel"
        )
```

#### 改动 4：初始化和生命周期管理

在 `model_runner.py` 中初始化：

```python
if self.server_args.enable_sparse_offload:
    from sglang.srt.mem_cache.sparsity import create_sparse_coordinator
    self.sparse_coordinator = create_sparse_coordinator(
        device=self.device,
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool=self.token_to_kv_pool,
        start_layer=self.start_layer,
        end_layer=self.end_layer,
        server_args=self.server_args,
    )
```

在 `ForwardBatch` 中添加 `sparse_coordinator` 字段，在 `model_runner._forward_raw()` 中赋值。

#### 改动 5：自定义内存分配器

**CXL 分配器** — 新增 `mem_cache/storage/cxl/cxl_allocator.py`：

```python
class CXLHostTensorAllocator(HostTensorAllocator):
    def __init__(self, cxl_device_path, pool_size_gb):
        super().__init__()
        self.pool = CXLMemoryPool(cxl_device_path, pool_size_gb)

    def allocate(self, dims, dtype, device):
        nbytes = int(np.prod(dims)) * torch.tensor([], dtype=dtype).element_size()
        ptr = self.pool.alloc(nbytes)
        buf = (ctypes.c_char * nbytes).from_address(ptr)
        tensor = torch.frombuffer(buf, dtype=dtype).view(dims)
        return tensor
```

在 `memory_pool_host.py` 的 `get_allocator_from_storage()` 中注册：

```python
def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        ...
    elif allocator_type == "cxl":
        from sglang.srt.mem_cache.storage.cxl.cxl_allocator import CXLHostTensorAllocator
        return CXLHostTensorAllocator(...)
    else:
        return HostTensorAllocator()
```

---

## 3. 三种方案的关键差异对比

| 维度 | A: DRAM | B: RDMA | C: CXL |
|------|---------|---------|--------|
| **内存分配** | `torch.empty` + `cudaHostRegister` | Mooncake allocator (RDMA MR) | CXL mmap + `cudaHostRegister` |
| **GPU→host 传输** | sgl_kernel (PCIe DMA) | sgl_kernel → host → RDMA WRITE | sgl_kernel (PCIe/CXL DMA) |
| **host→GPU 传输** | sgl_kernel (PCIe DMA) | RDMA READ → host → sgl_kernel | sgl_kernel (PCIe/CXL DMA) |
| **全量预取？** | 否（KV 已在本地） | **是**（RDMA 需预取到本地） | 否（GPU 直接 DMA 到 CXL） |
| **sparse decode 开销** | 仅 GPU 计算 | RDMA 传输 + GPU 计算 | CXL DMA + GPU 计算 |
| **sparse 收益** | 计算量减少 | 计算量减少，**但传输量不减** | 计算量减少 **+ 传输量减少** |
| **code path** | 默认 HiCache | Mooncake HiCache | 自定义 CXL allocator + HiCache |

### 3.1 RDMA 方案的两种模式

**B1: RDMA 全量预取（对应论文批评的模式）**

```
每步 decode 开始:
  1. 将请求所需的 全部 KV Cache 从 RDMA 远端取到本地 host
  2. Quest 选 top-k pages
  3. 仅 top-k 对应的 KV 从 host → GPU
  ⚠️ 步骤 1 传输了 100% 数据，但步骤 3 只用了 1-10%
```

**B2: RDMA 按需取 top-k（理论上可行但高延迟）**

```
每步 decode 每层:
  1. Quest 选 top-k pages
  2. 逐个/批量 RDMA READ top-k pages 到本地 host
  3. host → GPU
  ⚠️ per-message RDMA 开销 × layers 数 → 延迟累积
```

### 3.2 CXL 方案的优势

```
每步 decode 每层:
  1. Quest 选 top-k pages
  2. sgl_kernel 直接从 CXL 内存 DMA 到 GPU（或 GPU 直接访问）
  ⚠️ 仅传输 top-k 数据，无全量预取，无 per-message 开销
```

---

## 4. 代码修改总表

### 4.1 必须修改的现有文件

| 文件 | 修改内容 | 改动量 |
|------|---------|--------|
| `layers/attention/flashattention_backend.py` | 在 `forward_decode()` 中加入 SparseCoordinator hook | ~25 行 |
| `mem_cache/sparsity/core/sparse_coordinator.py` | 扩展 `_handle_sparse_retrieve` 加入 offload load 调用 | ~30 行 |
| `model_executor/model_runner.py` | 初始化 SparseCoordinator + KVOffloader | ~40 行 |
| `model_executor/forward_batch_info.py` | 新增 `sparse_coordinator` 字段 | ~5 行 |
| `managers/scheduler.py` | request begin/end lifecycle 调用 | ~20 行 |
| `managers/scheduler_output_processor_mixin.py` | prefill 结束后触发 backup | ~10 行 |
| `mem_cache/memory_pool_host.py` | 在 `get_allocator_from_storage` 中注册 `"cxl"` | ~10 行 |
| `server_args.py` | 新增 `--enable-sparse-offload` 及配置参数 | ~25 行 |

### 4.2 需要新增的文件

| 文件 | 内容 | 代码量 |
|------|------|--------|
| `mem_cache/sparsity/kv_offloader.py` | Sparse KV Offloader（host↔device 搬运封装） | ~200 行 |
| `mem_cache/storage/cxl/cxl_allocator.py` | CXL 内存分配器 | ~100 行 |
| `mem_cache/storage/cxl/__init__.py` | 模块初始化 | ~5 行 |
| `scripts/run_cxl_experiment.py` | 实验自动化脚本 | ~300 行 |
| `scripts/analyze_results.py` | 结果分析 + 图表生成 | ~200 行 |

### 4.3 总代码改动估算

- 修改现有文件：~165 行
- 新增文件：~805 行
- **总计约 970 行**（其中核心逻辑约 500 行，脚本约 500 行）

---

## 5. 实验矩阵

### 5.1 论文实验 → 代码实验映射

| 论文实验 | 对应方案 | 内存后端 | 稀疏算法 | 关键指标 |
|---------|---------|---------|---------|---------|
| §5.2 端到端吞吐/延迟 | A/B/C 全跑 | DRAM/RDMA/CXL 真实硬件 | Quest | Throughput, TTFT, TBT |
| §5.3 DRAM 效率与可扩展性 | 固定 GPU 内存，增并发 | DRAM vs CXL | Quest | max_concurrent_reqs, throughput |
| §5.4 RDMA 带宽瓶颈 | RDMA 全量预取 vs 按需 | RDMA 不同配比 | Quest | 有效带宽利用率, TTFT 分解 |
| §5.5a Buffer 消融 | CXL + 不同 device buffer | CXL | Quest | hit rate vs buffer size |
| §5.5b Sparse ratio | 不同 top-k | 全部 | Quest | 各方案吞吐对比 |
| §5.5c 层间相似性 | 记录 top-k overlap | CXL | Quest | 层间 top-k 重叠率 |

### 5.2 实验参数

```yaml
models:
  - meta-llama/Meta-Llama-3-8B      # MHA, 32 layers, 8B params, 单卡可跑
  - Qwen/Qwen2.5-7B                 # GQA, 28 layers, 7B params, 单卡可跑

seq_lengths: [32768, 65536, 131072]

quest_configs:
  - {top_k_ratio: 0.02, page_size: 16, min_sparse_prompt_len: 4096}
  - {top_k_ratio: 0.05, page_size: 16, min_sparse_prompt_len: 4096}
  - {top_k_ratio: 0.10, page_size: 16, min_sparse_prompt_len: 4096}

memory_backends:
  - name: dram
    allocator: default
  - name: rdma
    allocator: mooncake         # 或自定义 RDMA allocator
    rdma_device: mlx5_0
    modes: [full_prefetch, topk_per_layer]
  - name: cxl
    allocator: cxl
    cxl_device: /dev/dax0.0     # 或具体 CXL 设备路径

concurrency: [1, 2, 4, 8, 16]

device_buffer_sizes: [512, 1024, 2048, 4096, 8192]  # 消融实验
```

### 5.3 实验启动命令

```bash
MODEL=meta-llama/Meta-Llama-3-8B

# ===== 方案 A: Sparse + DRAM =====
python -m sglang.launch_server \
  --model $MODEL \
  --attention-backend fa3 \
  --enable-sparse-offload \
  --hisparse-config '{"algorithm":"quest","backend":"fa3","top_k":2048,"page_size":16,"min_sparse_prompt_len":4096}' \
  --enable-hierarchical-cache \
  --hicache-ratio 4 \
  --disable-cuda-graph \
  --port 30000

# ===== 方案 B: Sparse + RDMA (Mooncake) =====
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=mlx5_0
python -m sglang.launch_server \
  --model $MODEL \
  --attention-backend fa3 \
  --enable-sparse-offload \
  --hisparse-config '{"algorithm":"quest","backend":"fa3","top_k":2048,"page_size":16,"min_sparse_prompt_len":4096}' \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake \
  --hicache-ratio 4 \
  --disable-cuda-graph \
  --port 30000

# ===== 方案 C: Sparse + CXL =====
python -m sglang.launch_server \
  --model $MODEL \
  --attention-backend fa3 \
  --enable-sparse-offload \
  --hisparse-config '{"algorithm":"quest","backend":"fa3","top_k":2048,"page_size":16,"min_sparse_prompt_len":4096}' \
  --enable-hierarchical-cache \
  --hicache-storage-backend cxl \
  --cxl-device-path /dev/dax0.0 \
  --hicache-ratio 4 \
  --disable-cuda-graph \
  --port 30000

# ===== Dense baseline (对照) =====
python -m sglang.launch_server \
  --model $MODEL \
  --attention-backend fa3 \
  --enable-hierarchical-cache \
  --hicache-ratio 4 \
  --disable-cuda-graph \
  --port 30000
```

### 5.4 Benchmark 命令

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-input 131072 \
  --random-output 1024 \
  --num-prompts 32 \
  --request-rate 2 \
  --host 127.0.0.1 \
  --port 30000
```

---

## 6. 数据采集与统计

### 6.1 需采集的指标

```python
@dataclass
class ExperimentMetrics:
    # === 端到端指标（bench_serving） ===
    throughput_tokens_per_sec: float
    avg_ttft_ms: float
    avg_tbt_ms: float
    p99_tbt_ms: float
    p99_ttft_ms: float

    # === 内存效率指标 ===
    peak_gpu_memory_gb: float
    host_memory_used_gb: float
    max_concurrent_requests: int

    # === Sparse attention 指标（SparseCoordinator 内部统计） ===
    avg_pages_selected_per_layer: float     # 平均每层选取的 page 数
    sparsity_ratio: float                    # 实际稀疏率
    quest_overhead_ms_per_layer: float       # Quest 选取开销

    # === 内存传输指标（KVOffloader 内部统计） ===
    total_bytes_transferred: int             # 总传输字节
    effective_bytes_used: int                # 实际被 attention 使用的字节
    bandwidth_utilization: float             # effective / total
    avg_transfer_latency_us_per_layer: float # 每层平均传输延迟

    # === 层间相似性指标 ===
    inter_layer_topk_overlap: float          # 相邻层 top-k 重叠率
    delta_pages_per_layer: float             # 实际需新加载的 pages/层
```

### 6.2 统计收集方式

在 `SparseCoordinator` 和 `SparseKVOffloader` 中添加 instrumentation：

```python
class SparseKVOffloader:
    def __init__(self, ...):
        ...
        self.stats = OffloaderStats()

    def load_selected_pages(self, ...):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # ... 实际传输 ...

        end.record()
        torch.cuda.synchronize()
        self.stats.record_transfer(
            bytes_transferred=...,
            latency_ms=start.elapsed_time(end),
            layer_id=layer_id,
        )
```

---

## 7. 实现路线图

### Phase 0: 硬件环境确认（0.5 天）

- [ ] 确认 GPU 型号和数量
- [ ] 确认 RDMA 网卡型号和连通性（`ibstat`, `ib_write_bw`）
- [ ] 确认 CXL 2.0 设备信息（`lspci`, `/dev/dax*`, NUMA topology）
- [ ] 获取 CXL 共享内存池操作的具体 API 示例
- [ ] 安装 sglang，确保基线可运行
- [ ] 下载 Llama-3-8B / Qwen2.5-7B 权重

### Phase 1: Quest + FA3 接入（2 天）

- [ ] 在 `flashattention_backend.py` 中插入 SparseCoordinator hook
- [ ] 在 `model_runner.py` 中初始化 SparseCoordinator
- [ ] 在 `forward_batch_info.py` 中添加 `sparse_coordinator` 字段
- [ ] 在 `scheduler.py` / `scheduler_output_processor_mixin.py` 中添加 lifecycle
- [ ] 在 `server_args.py` 中添加 `--enable-sparse-offload` 参数
- [ ] 验证 Quest sparse attention 在 DRAM 基线下的正确性

### Phase 2: KV Offloader 实现（1.5 天）

- [ ] 实现 `mem_cache/sparsity/kv_offloader.py`
- [ ] 集成 host↔device 传输（复用 `HostKVCache` 的 backup/load API）
- [ ] 在 SparseCoordinator 中插入 offload 调用
- [ ] 添加 instrumentation（统计 transfer 延迟、bytes、hit rate）
- [ ] 验证 DRAM offload 功能正确性

### Phase 3: CXL 内存分配器（1 天）

- [ ] 根据用户提供的 CXL API 实现 `CXLHostTensorAllocator`
- [ ] 在 `get_allocator_from_storage` 中注册
- [ ] 验证 `cudaHostRegister` 对 CXL mmap 内存的兼容性
- [ ] 跑通 CXL 方案的完整流程

### Phase 4: RDMA 方案（1 天）

- [ ] 配置 Mooncake RDMA 环境（或自定义 RDMA allocator）
- [ ] 实现全量预取模式的数据流
- [ ] 实现逐层按需取模式（在 offloader 中添加 RDMA per-layer fetch）
- [ ] 验证 RDMA 方案正确性

### Phase 5: 全量实验运行（2 天）

- [ ] 编写实验自动化脚本 `scripts/run_cxl_experiment.py`
- [ ] 运行 §5.2 端到端吞吐/延迟实验
- [ ] 运行 §5.3 DRAM 效率实验
- [ ] 运行 §5.4 RDMA 带宽瓶颈实验
- [ ] 运行 §5.5 消融实验
- [ ] 收集层间相似性数据

### Phase 6: 分析和图表（1 天）

- [ ] 实现 `scripts/analyze_results.py`
- [ ] 生成论文所需图表
- [ ] 验证实验数据支持论文论点

**总工期：约 8-9 天**

---

## 8. 代码架构图

```
sglang/
├── srt/
│   ├── server_args.py                         # [修改] +--enable-sparse-offload, --cxl-device-path
│   ├── managers/
│   │   ├── scheduler.py                       # [修改] request lifecycle + sparse_coordinator
│   │   └── scheduler_output_processor_mixin.py # [修改] prefill→backup 触发
│   ├── model_executor/
│   │   ├── model_runner.py                    # [修改] 初始化 SparseCoordinator + KVOffloader
│   │   └── forward_batch_info.py              # [修改] +sparse_coordinator 字段
│   ├── layers/
│   │   └── attention/
│   │       └── flashattention_backend.py      # [修改] SparseCoordinator hook
│   └── mem_cache/
│       ├── memory_pool_host.py                # [修改] 注册 cxl allocator
│       ├── sparsity/
│       │   ├── core/
│       │   │   └── sparse_coordinator.py      # [修改] 加入 offload 调用
│       │   ├── algorithms/
│       │   │   └── quest_algorithm.py         # [不变] Quest 已实现
│       │   ├── backend/
│       │   │   └── backend_adaptor.py         # [不变] FA adaptor 已实现
│       │   └── kv_offloader.py                # [新增] Sparse KV Offloader
│       └── storage/
│           └── cxl/
│               ├── __init__.py                # [新增]
│               └── cxl_allocator.py           # [新增] CXL 内存分配器
├── scripts/
│   ├── run_cxl_experiment.py                  # [新增] 实验自动化
│   └── analyze_results.py                     # [新增] 结果分析
└── docs/
    └── plan/
        └── experiment_plan.md                 # 本文件
```

---

## 9. CXL 内存分配器详细设计（待用户提供 API 后细化）

### 9.1 预期的 CXL 设备接入方式

CXL 2.0 共享内存池通常通过以下方式暴露给用户态：

**方式 A: DAX 设备**
```python
import mmap, os
fd = os.open("/dev/dax0.0", os.O_RDWR)
mm = mmap.mmap(fd, size, mmap.MAP_SHARED)
ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
```

**方式 B: NUMA node 绑定**
```python
import numa  # 或 libnuma ctypes binding
# CXL 内存通常作为一个新的 NUMA node 出现
ptr = numa.alloc_on_node(size, cxl_numa_node_id)
```

**方式 C: 专用库 API**
```python
from cxl_shm_lib import CXLPool  # 用户提供的具体库
pool = CXLPool("/dev/dax0.0", size_gb=32)
ptr = pool.alloc(nbytes)
```

### 9.2 与 SGLang 集成的关键点

无论 CXL 暴露方式如何，集成到 SGLang 的关键步骤：

1. **分配**：`CXLHostTensorAllocator.allocate()` 返回 `torch.Tensor`（backed by CXL 内存）
2. **注册**：`alloc_with_host_register()` 对 CXL tensor 执行 `cudaHostRegister` → GPU 可 DMA
3. **传输**：`sgl_kernel` 的 `transfer_kv_*` 函数不感知底层是 DRAM 还是 CXL — 它只操作 host pointer + device pointer，因此**无需修改传输逻辑**
4. **释放**：`CXLHostTensorAllocator` 需实现 `free()` 或依赖 `__del__` 归还 CXL 池

### 9.3 `cudaHostRegister` 对 CXL 的兼容性

- CXL 2.0 type-3 设备的内存通过 HDM（Host-managed Device Memory）或 HPA 映射为 CPU 物理地址
- 通过 DAX 或 NUMA mmap 后，呈现为普通虚拟地址空间
- `cudaHostRegister` 可以注册任何合法的 CPU 虚拟地址范围（包括 CXL mmap 的）
- 注册后 GPU 可通过 BAR 或 peer mapping 直接 DMA — 这是 CXL 天然优势所在

⚠️ 需要在真实硬件上验证 `cudaHostRegister` 对具体 CXL 设备的兼容性。

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| `cudaHostRegister` 不支持 CXL mmap 内存 | 测试备选方案：使用 `cudaHostAlloc` + 手动拷贝，或 GPUDirect RDMA |
| Mooncake 部署复杂 | 备选：自定义简单 RDMA verbs wrapper |
| Quest 精度损失过大 | 调整 sparsity_ratio, page_size, num_recent_pages；先跑 PPL 验证 |
| FA3 backend 不支持当前模型 | 使用 `--prefill-attention-backend` 和 `--decode-attention-backend` 分别配置 |
| CUDA Graph 与动态 page_table 冲突 | 使用 `--disable-cuda-graph`（实验场景可接受） |
| 长序列 OOM | 降低 `hicache-ratio`、减小并发数、使用 FP8 KV cache |
| CXL 带宽不稳定 | 多次重复实验取中位数；先跑 micro-benchmark 确认 CXL 实际带宽 |

---

## 11. Quest vs NSA 访存模式等价性论证

| 特征 | NSA (DeepSeek DSA) | Quest |
|------|-------------------|-------|
| 粒度 | token-level (page_size=1) | page-level (page_size=16-64) |
| query-aware | ✅ 依赖当前 Q | ✅ 依赖当前 Q 的 bounding-box 评分 |
| 逐层动态 | ✅ 每层不同 top-k | ✅ 每层不同 page top-k |
| top-k 选取量 | ~2048 tokens/层 | ~128-256 pages/层 ≈ ~2048-4096 tokens/层 |
| 计算开销 | FP8 MQA logits (DeepGEMM) | Min/max bounding-box 评分 |
| 实时性 | 完全实时 | 完全实时 |
| KV Cache 访问模式 | 离散 token 级随机读 | page 级随机读（更粗粒度） |

**结论**：Quest 的 page 粒度 query-aware 访存模式与 NSA 的 token 粒度本质一致。page 粒度实际上更有利于 CXL/RDMA 的传输效率（减少 per-access 开销），因此如果 CXL 在 Quest 场景下已显著优于 RDMA，那在更细粒度的 NSA 场景下优势只会更大——这增强了论文结论的泛化性。

---

## 12. 硬件需求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| GPU | 1 × A100 40GB | 1 × A100 80GB |
| CPU DRAM | 64GB | 128GB+ |
| RDMA NIC | 1 × ConnectX-5/6 (25Gbps+) | 100/200Gbps IB |
| CXL 设备 | CXL 2.0 type-3 内存扩展设备 | ≥32GB CXL 内存 |
| 远端节点（RDMA） | 1 台有 RDMA NIC 的机器 | 内存 ≥64GB |
| 模型 | Llama-3-8B (FP16, ~16GB) | Qwen2.5-7B (FP16) |
| 序列长度 | 32K-64K | 128K |

对于 128K 序列长度 + Llama-3-8B：
- 模型权重：~16GB (FP16)
- 单请求 KV Cache (128K)：~16GB (32 layers × 32 heads × 128 dim × 128K × 2(K+V) × 2B)
- 使用 HiCache offload 后：GPU 仅保留 top-k 热数据 + 模型权重

---

## 13. 下一步行动

1. **请提供 CXL 共享内存池的具体操作示例**（设备路径、API、分配/释放方式）
2. **确认 RDMA 环境**：是否已有 Mooncake 或其他 RDMA 框架部署
3. **确认实验机器配置**：GPU 型号、DRAM 大小、CXL 设备型号
4. 确认后，我将开始实现代码
