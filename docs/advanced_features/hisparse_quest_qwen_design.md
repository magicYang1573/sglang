# HiSparse × Quest：为 Qwen 等非原生稀疏模型引入 HiSparse 的设计文档

本文档描述如何在 **不改动 HiSparse 现有 DSA（DeepSeek Sparse Attention）链路的前提下**，将 HiSparse 的 "prefill 后全量 KV 驻留 DRAM + GPU 仅持小 hot buffer + on-demand swap-in" 架构，扩展到 **非原生稀疏** 模型（以 Qwen2/Qwen3 的 MHA/GQA 为代表），并用 **Quest** 作为算法来源产生 decode 阶段的 topk。

> 本文的前置阅读：[`hisparse_guide.md`](./hisparse_guide.md)。本文不复述 HiSparse 的基本概念。

---

## 1. 背景与动机

### 1.1 现有 HiSparse 的限制

现有 HiSparse 实现（`python/sglang/srt/managers/hisparse_coordinator.py`）与 DSA 家族模型深度绑定：

- `server_args.py` 中有强校验：`--enable-hisparse` 仅接受 DeepSeek-NSA 模型、`flashmla_sparse` 后端、`bfloat16` KV、`--disable-radix-cache`。
- 设备/主机 KV 池使用 MLA 结构（`HiSparseNSATokenToKVPool` + `MLATokenToKVPoolHost`），一次只有一个 latent buffer。
- topk 不由 HiSparse 自身产生，而是由 DSA 的 lightning indexer 原生输出，HiSparse 仅负责根据 topk 从 host → device 做 swap-in。
- Page-table 改写仅存在于 NSA 后端（`nsa_backend.py` 中调用 `translate_loc_to_hisparse_device`）。

### 1.2 非原生稀疏模型（Qwen 系列）的情况

Qwen2 / Qwen3 属于 MHA 或 GQA，模型本身 **没有** lightning indexer，要做 decode 阶段稀疏必须显式补一个"算法"来选 topk page/token。仓库中已经存在一套通用的稀疏算法框架：

- `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` —— 通用 `SparseCoordinator`（生命周期钩子）
- `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` —— Quest 算法（页级 min/max bbox 评分）
- `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` —— `BaseSparseAlgorithmImpl` 提供通用 construct / update / retrieve 流程
- `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py` —— `FlashAttentionAdaptor` 可改写 FA3 的 page_table / cache_seqlens

该框架目前是 **骨架**：`SparseCoordinator.forward_begin/forward_end` 仍为 `pass`，`NSABackendAdaptor` 未实现，`factory.create_sparse_coordinator` 未被任何运行路径调用；它与 `HiSparseCoordinator` 尚未连通。

### 1.3 目标

基于现有 HiSparse IO 子系统 + 现有 Quest 算法，补齐"算法 → IO"的连接，使得：

1. Qwen 模型可以用 `--enable-hisparse --hisparse-config='{"algorithm":"quest", ...}'` 启动。
2. Prefill 结束后 KV 全量位于 DRAM（pinned host），GPU 上只有一个小 hot buffer。
3. Decode 每步调用 Quest 产出 topk page；topk 通过 `hisparse.cuh` kernel 从 host 装载到 GPU。
4. HiSparse 现有 DSA 路径不受影响。

---

## 2. 总体设计

### 2.1 分层与职责切分

```
┌────────────────────────────────────────────────────────────────┐
│                      SparseCoordinator (指挥层)                 │
│  生命周期钩子: on_request_begin / attention_begin / attention_  │
│                end / forward_begin / forward_end / on_request_  │
│                end                                              │
│  - 调用 algorithm.retrieve_topk 拿 topk                         │
│  - 调用 io_subsystem.swap_in(topk) 装载 KV                      │
│  - 调用 backend_adaptor 改写 attention metadata                 │
└───────────────┬──────────────────────────┬──────────────────────┘
                │                          │
                ▼                          ▼
   ┌──────────────────────┐     ┌──────────────────────────────┐
   │   SparseAlgorithm     │     │    HiSparseIOSubsystem       │
   │ (QuestAlgorithm ...)  │     │ (原 HiSparseCoordinator)     │
   │ - construct_repr      │     │ - host pool / device buffer  │
   │ - update_repr         │     │ - staging / direct admit     │
   │ - retrieve_topk → idx │     │ - swap_in_selected_pages     │
   │ (不感知 host/device)   │     │ - eager_backup / release     │
   └──────────────────────┘     └──────────────────────────────┘
                                             ▲
                                             │
                                  ┌──────────┴───────────┐
                                  │  hisparse.cuh kernel  │
                                  │ (MLA 与 MHA 两种包装) │
                                  └──────────────────────┘
```

**关键原则：**
- `SparseAlgorithm` 只负责"给定 query 和历史 K 的表示，输出 logical 页号 topk"。它对 host/device 一无所知，仅读 GPU 上的 K buffer（此 K buffer 即 hot buffer，下文 §3.2 会讨论近似性）。
- `HiSparseIOSubsystem` 只负责"给定 topk，把对应的 K（或 K/V）从 host 换入 device"，对算法无感知。
- `SparseCoordinator` 做粘合，在 `attention_begin` 里串起 `retrieve_topk → swap_in → adapt_metadata`。
- 两种链路共存：
  - **DSA 链路**（现状）：indexer 产 topk → `HiSparseCoordinator.swap_in_selected_pages` → 改写 NSA `page_table_1`。
  - **新增 Quest/Qwen 链路**：`QuestAlgorithm.retrieve_topk` 产 topk → MHA 版 `swap_in_selected_pages` → 改写 FA3 `page_table`。

### 2.2 代码落点

| 新增/修改 | 路径 | 说明 |
|-----------|------|------|
| 修改 | `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` | 填充 `forward_begin/forward_end`，让 `attention_begin` 既能走"算法 topk + HiSparse swap_in"，也能走纯算法（无 HiSparse）路径 |
| 修改 | `python/sglang/srt/mem_cache/sparsity/factory.py` | 支持按模型族装配 `io_subsystem`（DSA / MHA-HiSparse / None） |
| 新增 | `python/sglang/srt/mem_cache/hisparse_memory_pool.py` | 新增 `HiSparseMHATokenToKVPool` + `HiSparseMHATokenToKVPoolAllocator` |
| 修改 | `python/sglang/srt/managers/hisparse_coordinator.py` | 抽象为 `HiSparseIOSubsystem` 基类 + `HiSparseMLAIO` / `HiSparseMHAIO` 两个子类；或保持单类，通过 `is_mla` 分支走两套 host pool |
| 新增 | `python/sglang/jit_kernel/hisparse.py` | 增加 `load_cache_to_device_buffer_mha` wrapper（kernel 本身已支持 `IsMLA=false`） |
| 修改 | `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py` | `FlashAttentionAdaptor` 已实现，但需要接入 HiSparse 的 "logical → hisparse device" 映射 |
| 修改 | `python/sglang/srt/server_args.py` | 解除"仅 DSA + flashmla_sparse"硬断言，按算法/模型族动态校验 |
| 修改 | `python/sglang/srt/model_executor/model_runner.py` | `enable_hisparse` 时，根据模型族决定实例化哪种 IO 子系统，并构造 `SparseCoordinator` |

---

## 3. 详细设计

### 3.1 生命周期串接

`SparseCoordinator` 将 HiSparse 的事件一一映射到生命周期钩子上。以 Qwen + Quest + HiSparse 为例：

| 阶段 | 事件 | 动作 |
|------|------|------|
| 请求创建 | `on_request_begin(req)` | `states.register(...)`；若 PD disagg decode 侧，调用 `io.admit_request_direct(req)`（预约 host 槽位、分配小 device buffer）；否则先进 `io.admit_request_into_staging(req)`。 |
| Prefill/Extend 每层 | `attention_end(...)` | `algorithm.construct_representations(...)`：按页累积 K 的 per-dim min/max，填入 Quest 的 `page_k_min/max`。K buffer 此时仍是完整 KV（因为 prefill 路径）。 |
| Prefill 结束后 | —（由 scheduler 驱动 `collect_ready_reqs`） | staging 路径下 GPU→host 背景拷贝完成后，`alloc_device_buffer`：把该 req 的 device buffer 压缩到 `device_buffer_size` 大小。**此时算法的 `page_k_min/max` 仍然保留在 GPU 的算法表示池中**（见 §3.2）。 |
| Decode forward 开始 | `forward_begin(batch)` | `io.wait_for_pending_backup()`：等上一步的 eager backup 完成。 |
| Decode 每层前 | `attention_begin(q, k, v, layer, batch, meta)` | 1. `topk_logical = algorithm.retrieve_topk(q, layer, req_indices, mask, ...)`<br>2. `device_locs = io.swap_in_selected_pages(req_indices, seq_lens, topk_logical, layer_id)`（调 hisparse.cuh kernel）<br>3. `backend_adaptor.adapt_for_attn_metadata(selected=device_locs, ...)` 写回 FA3 的 `page_table` / `cache_seqlens_int32` / `cu_seqlens_k`。 |
| Decode 每层后 | `attention_end(...)` | `algorithm.update_representations(...)`：为上一步新生成的 token 所在的 page 增量更新 min/max。 |
| Decode forward 结束 | `forward_end(batch)` | 触发对"上一步新 token"的 eager backup（当前 `HiSparseCoordinator._eager_backup_previous_token` 逻辑），并调用 `map_last_loc_to_buffer` 管理新 token 在 hot buffer 里的槽位。 |
| 请求结束 | `on_request_end(req)` | `io.request_finished(req)` + `states.clear(idx)`。 |

### 3.2 Quest 表示池与 hot buffer 的关系（关键正确性问题）

`QuestAlgorithm._compute_page_representations` 从 `k_buffer[phys_tok]` 读 K 算 min/max；`phys_tok` 来自 `req_to_token_pool.req_to_token`。在非 HiSparse 场景下 K buffer 始终是全量的，这没有问题；**但在 HiSparse 下 GPU 上的 K buffer 只有 hot 区域，历史 K 在 host**。为此必须约定：

1. **Quest 的 min/max 表示 (`page_k_min/max`) 在 prefill 阶段一次性计算并长期驻留 GPU**。
   - Prefill 时 `attention_end` 是在完整 K buffer 上调用的（staging 路径的 prefill 仍然在 GPU 上；direct-to-host 路径 §3.6 会单独讨论），此时 Quest 能拿到完整 K → min/max 一次算齐。
   - Prefill 结束后即使 KV 被换到 host，`page_k_min/max` 保留在 GPU，不再依赖 K buffer 本体。
2. **Decode 阶段 `update_representations` 只处理 "新 token 所在的页"**。新 token 此时还在 hot buffer 中（hot buffer 的 `newest_slot` 保存最新 token），因此 `k_buffer[phys_tok]` 对新页合法。对已 swap 到 host 的历史页，不做 update（Quest 的 bbox 本就随 token 增加单调变大，不必回填历史页；即使需要更新也应由 eager_backup 完成后再异步打点，见 §4.3 TODO）。
3. **Quest 的 `total_num_pages` 对应的物理页空间**：`BaseSparseAlgorithmImpl.initialize_representation_pool` 用 `token_to_kv_pool.get_key_buffer(start_layer).shape[0]` 作为总 token 数，即 GPU 上的 logical 全量池大小（`_size_full = size * host_to_device_ratio`），所以 min/max 池会按 logical 槽位分配，与 `req_to_token` 里记录的 logical 索引对齐，与 hot buffer 的 hisparse 子槽位无关。✅ 这点和现状兼容，不需要改 Quest。

> 副产物：Quest 的 min/max 表示占用 GPU 显存 `O(num_pages × kv_heads × head_dim × 4B × num_layers)`。对 Qwen3-32B GQA（`kv_heads=8, head_dim=128`）在 `page_size=16, seq_len=128k` 下，每请求 ≈ 8192 页 × 8 × 128 × 4B × 2 = 64 MB 每层。这需要在文档中明示并让 `top_k`/`page_size`/`host_to_device_ratio` 可调。

### 3.3 Topk 的语义对齐

与 DSA 完全对齐：**用户配置的是 `top_k`（每步参与 attention 的 token 数量）**，kernel 期望的也是 token 级索引。区别仅在内部实现层——Quest 按页选，页内的 token 全部带出：

| 维度 | DSA 现状 | Quest + Qwen 新路径 |
|------|----------|---------------------|
| 用户 config | `top_k = 2048`（token 数） | `top_k = 2048`（token 数，同语义） |
| 算法输出 | indexer 直接给 token 级索引 | Quest 给 page 级索引（`num_topk_pages = ⌈top_k / page_size⌉`） |
| 喂给 kernel | `top_k_tokens`（int32，shape `[bs, top_k]`，-1 填充） | 同左：对 page 级索引做 `page × page_size + [0..page_size)` 展开后得到 token 级 |

Quest 路径在 `SparseCoordinator._handle_sparse_retrieve` 里对 `selected_indices`（page 级）展开为 token 级，由辅助函数 `_expand_selected_pages_to_tokens` 完成：给定 `selected_pages: [bs, num_topk_pages]` 和 `seq_lens`，输出 `[bs, top_k]`，-1 padding 并保证 token 级索引严格小于 `seq_lens`。

> 固定选 `num_topk_pages = ⌈top_k / page_size⌉` 页、固定输出宽度 `top_k`，**CUDA graph 友好**。不做按页动态 k 的变形（Quest/DSA 原论文实现也不依赖）。

### 3.4 Host Pool / Device Pool 的 MHA 版本

新增 `HiSparseMHATokenToKVPool`（与现有 `HiSparseNSATokenToKVPool` 并列）：

- Device 侧继承 `MHATokenToKVPool`（有独立 K/V buffer）；提供 `translate_loc_to_hisparse_device` 与 `register_mapping`（与 NSA 版一致的逻辑）。
- Host 侧使用现有 `MHATokenToKVPoolHost`（`layout="layer_first"`，`page_size=1`）。它内部已经有 `k_data_ptrs` / `v_data_ptrs`（已经是 CUDA 上的指针数组），正好可以喂给 kernel。

新增 `HiSparseMHATokenToKVPoolAllocator`：几乎镜像 `HiSparseTokenToKVPoolAllocator`，唯一差异是底层 `PagedTokenToKVPoolAllocator` 对应的 KV dtype/形状不同。可以通过将 `HiSparseTokenToKVPoolAllocator` 参数化（或抽共同基类）来避免大面积复制代码。

### 3.5 Kernel Wrapper（MHA 版）

`hisparse.cuh` 的 `load_cache_to_device_buffer_kernel` 模板参数 `IsMLA` 在 `false` 时会额外处理 V：

```cpp
if constexpr (!IsMLA) {
  const auto src_v = ... + src_loc * item_size_bytes;
  auto       dst_v = ... + dst_loc * item_size_bytes;
  transfer_item_warp(lane_id, src_v, dst_v, item_size_bytes);
}
```

因此只需要在 `python/sglang/jit_kernel/hisparse.py` 新增：

```python
def load_cache_to_device_buffer_mha(
    top_k_tokens, device_buffer_tokens,
    host_cache_locs, device_buffer_locs,
    host_cache_k, host_cache_v,
    device_buffer_k, device_buffer_v,
    top_k_device_locs, req_pool_indices, seq_lens, lru_slots,
    item_size_bytes, num_top_k, hot_buffer_size,
    page_size=1, block_size=256, num_real_reqs=None,
) -> None:
    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=False,
    )
    ...
    module.load_cache_to_device_buffer(
        top_k_tokens, device_buffer_tokens,
        host_cache_locs, device_buffer_locs,
        host_cache_k, host_cache_v,      # 两个 host buffer
        device_buffer_k, device_buffer_v,# 两个 device buffer
        top_k_device_locs, req_pool_indices, seq_lens, lru_slots,
        num_real_reqs, page_size, item_size_bytes,
    )
```

`item_size_bytes` 在 MHA 下取单个 token 的 K buffer 字节数（`kv_heads * head_dim * itemsize`），K 和 V 用相同 `item_size`（Qwen 保证 K/V 形状一致）。

`HiSparseIOSubsystem`（MHA 版）的 `swap_in_selected_pages` 传入 `mem_pool_host.k_buffer[layer]` 和 `v_buffer[layer]` 即可。

### 3.6 Prefill 路径与 Backend Adaptor

Qwen 默认走 FA3。FA3 的 `forward_extend` / `forward_decode` 路径里**没有** 类似 NSA 的 `translate_loc_to_hisparse_device(page_table_1)` 改写点。改写职责下放给 `SparseCoordinator.attention_begin → backend_adaptor.adapt_for_attn_metadata`：

- 复用 `FlashAttentionAdaptor`（已实现）；它接收 `selected_indices`（logical page 号）、`valid_lengths`，并在 `current_metadata` 上原地改写 `page_table / cache_seqlens_int32 / cu_seqlens_k / max_seq_len_k`。
- **但是**现有 `FlashAttentionAdaptor` 把 logical → physical 的映射用的是 `req_to_token // page_size`，这是全量池语义。HiSparse 下，这个结果还需要再经一次 `translate_loc_to_hisparse_device` 才能变成 hisparse device buffer 里的真实槽位。两种做法：
  1. **推荐**：扩展 `FlashAttentionAdaptor` 接受 `hisparse_translate_fn`，最终在生成 `physical_pages` 后按需二次转换；
  2. 或者让 `SparseCoordinator` 在调用 `adapt_for_attn_metadata` 之前就把 topk（token 级）喂给 kernel 得到 `device_locs`，再用 `device_locs` 直接覆盖 `page_table`（类似 NSA 路径那种一步到位，但要求 FA3 `page_table` 的 dtype / shape 容得下）。首版选做法 (1)。

### 3.7 PD disagg 与单机模式

- **PD disagg**：prefill 侧仍然走原生 FA3 KV，完成后 RDMA 把每请求 KV 写入 decode 侧的 host pool；decode 侧 `admit_request_direct(req)` 预留小 device buffer。**Quest 的 `page_k_min/max` 此时来不及在 prefill 阶段算**（prefill 侧不感知 HiSparse），必须在 decode 侧的 **第一次 decode 前**做一遍"冷启动 bbox 构建"。实现上有两条路：
  1. Decode 首步（forward）前，在 host → device 预装载（`_preload_to_device_buffer` 或显式循环分 page 加载），在设备 K buffer 上按页算一遍 `construct_representations`；完成后再做稀疏 decode。对长序列这是一次 O(N) 的开销，但摊到 decode 过程中可以 pipeline（每次 swap 一批 page 计一次 bbox，分层异步推进）。
  2. 更极端的做法：在 host 上直接算 min/max，再 DMA 到 GPU 的 bbox 池。这个对 CPU 计算敏感，不推荐首版。
- **单机（无 PD）**：走 staging 路径，prefill 仍在 GPU 上发生，完整 K 在 GPU 内；Quest 的 `construct_representations` 照常在 `attention_end` 里一次过完成，然后才做 KV → host 的 staging 背景拷贝 + device buffer 压缩。

### 3.8 Server Args 改造

当前 `server_args.py` 对 `--enable-hisparse` 的三处硬断言（模型必须是 DeepSeek-NSA、后端必须是 flashmla_sparse、kv_cache_dtype 必须是 bfloat16）需要改为按算法族 + 模型族决定。

新的语义：

```
--enable-hisparse
--hisparse-config='{
    "algorithm": "quest" | "deepseek_nsa",
    "backend":   "flashmla_sparse" | "fa3",
    "top_k": 2048,
    "device_buffer_size": 6144,
    "host_to_device_ratio": 10
}'
```

**与 DSA HiSparse 完全一致的配置表面**：用户只需给定 `top_k`、`device_buffer_size`、`host_to_device_ratio`，Quest 内部自动按 `num_topk_pages = ⌈top_k / page_size⌉` 选页，不需要也不接受 `sparsity_ratio / num_recent_pages / min_sparse_prompt_len`。

- `algorithm="deepseek_nsa"` 时维持现有 DSA + flashmla_sparse 约束不变（向后兼容）。
- `algorithm="quest"` 时允许 `backend in {"fa3"}`，允许非 MLA 模型；并且 `kv_cache_dtype` 从 `bfloat16/float16` 中选（kernel 的 `item_size_bytes` 动态计算，不强绑 bf16）。

### 3.9 CUDA Graph 兼容性

`HiSparseCoordinator` 当前已为 CUDA graph 做了三件事：
1. 预分配 `top_k_device_locs_buffer`；
2. 预分配 `req_device_buffer_tokens / _token_locs / lru_slots`；
3. 通过 `num_real_reqs` 标量让 kernel 里的 padded block early-return。

MHA 路径 **完全复用** 这套结构。`QuestAlgorithm.retrieve_topk` 被覆盖为一次 `torch.topk` 的批量版本（`[bs, max_num_pages]` 上一次完成，对非法/越界页用 `-inf` 掩码），输出宽度固定为 `num_topk_pages = ⌈top_k / page_size⌉`；后续的 page→token 展开以及 kernel 调用也都是固定形状，整条链路 CUDA graph 安全。

---

## 4. 最小改动集（MVP 实现清单）

为了清晰，下面按"新增/修改"列出 MVP 所需的文件变更。

### 4.1 新增文件

- `python/sglang/srt/mem_cache/hisparse_memory_pool_mha.py`
  - `HiSparseMHATokenToKVPool(MHATokenToKVPool)`
  - `HiSparseMHATokenToKVPoolAllocator`（抽公共逻辑到 `_HiSparseAllocatorMixin`）
- `python/sglang/srt/managers/hisparse_io/`（重构，可选）
  - `base.py` —— `HiSparseIOSubsystem` 抽象
  - `mla_io.py` —— 现有 `HiSparseCoordinator` 重命名并切成子类
  - `mha_io.py` —— 新 MHA 子类
- `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`（扩展）
  - 批量版 `retrieve_topk`（覆盖 `BaseSparseAlgorithmImpl.retrieve_topk` 的 Python loop）

### 4.2 关键修改点

1. `python/sglang/jit_kernel/hisparse.py`：新增 `load_cache_to_device_buffer_mha`。
2. `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py`：
   - 填充 `forward_begin` → `io.wait_for_pending_backup()`；
   - 填充 `forward_end` → `io.map_last_loc_to_buffer(...)` + eager backup；
   - `_handle_sparse_retrieve` 在拿到 `selected_indices` 后，展开为 token 级、调 `io.swap_in_selected_pages`，再交 backend_adaptor。
3. `python/sglang/srt/mem_cache/sparsity/factory.py`：根据 `config.algorithm` 与模型类型决定 `io_subsystem` 的具体类型；DSA 走 `HiSparseMLAIO`，Quest 走 `HiSparseMHAIO`（或 Quest+DSA 模型也允许走 MLA）。
4. `python/sglang/srt/model_executor/model_runner.py`：`enable_hisparse` 时同时构造 `HiSparseIOSubsystem` 与 `SparseCoordinator`；把 `forward_batch.hisparse_coordinator` 的赋值路径推广到 MHA 分支（或通过 `sparse_coordinator` 统一访问）。
5. `python/sglang/srt/server_args.py`：解除三处硬断言；改为按 `algorithm/backend/model` 交叉校验。
6. `python/sglang/srt/layers/attention/flashinfer_backend.py` / `fa3_backend.py`（择一，视 Qwen 默认 decode 后端）：**不需要改后端本体**，只要 `SparseCoordinator.attention_begin` 在 `backend_adaptor.adapt_for_attn_metadata` 里写回 metadata 即可；不新增像 NSA 那样的 `translate_loc_to_hisparse_device` 调用点。
7. `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py`：`FlashAttentionAdaptor.adapt_for_attn_metadata` 在计算 `physical_pages` 后，额外调用 `hisparse_translate_fn(physical_pages)` 把 logical 槽位翻译为 hisparse device 槽位。

### 4.3 显式 TODO / 已知不完善点

| TODO | 说明 |
|------|------|
| 直写 host 路径下 Quest bbox 冷启动 | §3.7 描述，MVP 可先只支持 staging 模式 |
| Quest `retrieve_topk` 的批量/CUDA-graph 化 | MVP 先禁用 CUDA graph 跑通；之后用 batched `torch.topk` 改写 |
| `update_representations` 是否需要回填 host 侧页 | 严格来说 decode 步 k 的页已经 backup 到 host，再回读算 min/max 代价较高；Quest 的 bbox 本身单调扩张，**MVP 跳过** |
| Quest bbox 池显存占用 | §3.2 末尾讨论，给出配置指引；后续可研究 per-layer 共享或 fp16 压缩 |
| 与 radix cache 的兼容 | 现 HiSparse 要求 `--disable-radix-cache`，MVP 继承该限制 |
| 多 `page_size` 支持 | kernel 当前已按 token 级 swap（`page_size=1`），但算法端是页级；对齐逻辑见 §3.3 |

---

## 5. 性能与显存模型

### 5.1 显存估算（以单请求、单层为例）

| 部件 | 大小 | 说明 |
|------|------|------|
| hot buffer（K+V） | `device_buffer_size × kv_heads × head_dim × itemsize × 2` | 每请求、每层 |
| Quest bbox min/max | `num_pages × kv_heads × head_dim × 4B × 2` | 全池共享（不是每请求）|
| `req_to_device_buffer` | `max_reqs × (device_buffer_size + page_size) × 8B` | 全局 |
| `lru_slots` | `num_layers × max_reqs × device_buffer_size × 2B` | 全局 |

Qwen3-32B / GQA8 / head_dim=128 / bf16 / `device_buffer_size=6144` / `page_size=16` / `seq_len=128k`：
- hot buffer ≈ 6144 × 8 × 128 × 2 × 2 = 25 MB/layer/req → 64 层 × 200 并发 ≈ 320 GB（分布在 TP 所有卡上）。作为对比，全量 KV 每请求每层 ≈ 128k × 8 × 128 × 2 × 2 = 524 MB → 全量 64×200×524MB ≈ 6.5 TB（不可行）。
- bbox：8192 pages × 8 × 128 × 4 × 2 = 64 MB/layer × 64 = **4 GB 全局**（不随并发线性涨），可接受。

### 5.2 IO 带宽模型

每 decode 步每层的 host→device 换入量：

```
miss_bytes_per_step_per_layer
  ≈ (top_k - hit_count) × item_size_bytes
  ≤ top_k × kv_heads × head_dim × itemsize × 2  (MHA：K+V)
```

Qwen3 + `top_k=2048`：每步每层 ≈ 2048 × 8 × 128 × 2 × 2 = 8 MB；64 层 ≈ 512 MB/step。在 PCIe4 (≈ 25 GB/s) 下单请求需要 ≈ 20 ms；依赖 `hisparse.cuh` 的 LRU 命中率与 batching 来摊薄。此处和 DSA 路径的结论一致，不是新引入的瓶颈。

---

## 6. 测试计划

1. **单元测试**（新增 `test/registered/unit/managers/test_hisparse_quest_unit.py`）：
   - mock `req_to_token_pool` + 小 `MHATokenToKVPool`；
   - 构造 2 个 req × 3 层的 prefill，随机 K，验证 `QuestAlgorithm.page_k_min/max` 正确；
   - 构造 decode query，验证 `retrieve_topk` 的 page 与 PyTorch 原生实现一致；
   - 把 topk 展开为 token 级，调 MHA 版 kernel，验证 `device_buffer` 被正确写入；
   - 构造 FA3 metadata，验证 `FlashAttentionAdaptor.adapt_for_attn_metadata` 改写后 page_table 指向正确的 hisparse 槽位。
2. **数值一致性测试**（新增 `test/registered/8-gpu-models/test_qwen_hisparse.py`）：
   - 小 Qwen 模型（Qwen2.5-1.5B），长上下文，用 `--enable-hisparse --hisparse-config '{"algorithm":"quest", ...}'` 跑 GSM8K / RULER 子集；
   - 对比：不开 HiSparse 的 baseline vs 开 HiSparse + Quest（sparsity=0.3/0.5）；
   - 验收：Quest sparsity=0.3 时精度下降 ≤ 2 pp（与 Quest 原论文一致）。
3. **压力测试**：`bench_serving` 跑 `random` 数据集 `40k input / 20k output / concurrency=200`，验证不 OOM、吞吐与理论估算一致。

---

## 7. 对用户的接口（与 §1 的 hisparse_guide.md 相衔接）

Qwen 启动样例：

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/Qwen3-32B \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --context-length 131072 \
    --tp-size 8 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype bfloat16 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --enable-hisparse \
    --hisparse-config='{
        "algorithm": "quest",
        "backend": "fa3",
        "top_k": 2048,
        "device_buffer_size": 6144,
        "host_to_device_ratio": 10
    }'
```

---

## 8. 回顾：方案与"直接在 sparse_coordinator.py 里集成算法"的对应

本方案没有把 Quest 的评分逻辑"内嵌"到 `SparseCoordinator`，而是保留 `algorithm: QuestAlgorithm` 作为可替换组件；`SparseCoordinator` 仅作为"指挥层"调用算法与 IO 子系统。这样做有三个好处：

1. **算法可扩展**：未来接入 SnapKV / PQCache / Look-ahead 只需新增一个 `BaseSparseAlgorithmImpl` 子类。
2. **IO 可扩展**：MLA / MHA / 未来 Gemma MLA 2 等可以各自实现 `HiSparseIOSubsystem` 子类。
3. **与现有 DSA HiSparse 路径零冲突**：DSA 走 `DeepSeekNSAAlgorithm`（不计算 bbox，topk 由 indexer 给）+ `HiSparseMLAIO`，行为与现状完全一致。

---

## 9. 致谢

本设计复用了 HiSparse 团队的 `hisparse.cuh` kernel（`IsMLA` 模板参数已原生支持 MHA 路径），以及稀疏算法框架已有的 `BaseSparseAlgorithmImpl` / `QuestAlgorithm` / `FlashAttentionAdaptor`。
