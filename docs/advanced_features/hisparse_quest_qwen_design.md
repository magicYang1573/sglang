# HiSparse × Quest：为 Qwen 等非原生稀疏模型引入 HiSparse 的设计文档

本文档描述如何在 **不改动 HiSparse 现有 DSA（DeepSeek Sparse Attention）链路的前提下**，将 HiSparse 的 "prefill 后全量 KV 驻留 DRAM + GPU 仅持小 hot buffer + on-demand swap-in" 架构，扩展到 **非原生稀疏** 模型（以 Qwen2/Qwen3 的 MHA/GQA 为代表），并用 **Quest** 作为算法来源产生 decode 阶段的 topk。

> 本文的前置阅读：[`hisparse_guide.md`](./hisparse_guide.md)。本文不复述 HiSparse 的基本概念。

---

## 1. 背景与动机

### 1.1 现有 HiSparse 的两条算法分支

**DSA（`algorithm=deepseek_nsa`，默认）** 与 `python/sglang/srt/managers/hisparse_coordinator.py` 深度绑定：

- `server_args.py` 校验：模型须为 DeepSeek-NSA 族、NSA prefill/decode 后端为 `flashmla_sparse`、`bfloat16` KV、`--disable-radix-cache`。
- 设备/主机 KV 池为 MLA（`HiSparseNSATokenToKVPool` + `MLATokenToKVPoolHost`）；topk 由 DSA lightning indexer 给出；NSA 路径在 `nsa_backend.py` 等处做 `translate_loc_to_hisparse_device` 类改写。

**Quest（`algorithm=quest`）** 针对 **MHA/GQA（如 Qwen）+ FA3**：

- 使用 `HiSparseMHATokenToKVPool` / `HiSparseMHACoordinator`（`hisparse_mha_memory_pool.py`、`hisparse_mha_coordinator.py`）与 `load_cache_to_device_buffer_mha`；topk 由 `QuestAlgorithm.retrieve_topk` 产生。
- `server_args` 要求 **`--page-size 1`**（见 §1.4）、`--disable-radix-cache`，KV 为 `bfloat16` 或 `float16`；**当前 MVP 要求 `--disable-cuda-graph`**（FA3 在稀疏改写后需重算 `scheduler_metadata`，与 graph capture 不兼容，见 §3.9）。

### 1.2 非原生稀疏模型（Qwen 系列）的情况

Qwen2 / Qwen3 属于 MHA 或 GQA，没有 lightning indexer，decode 稀疏由 **Quest** 在通用稀疏框架里完成。当前 **Quest + HiSparse + FA3** 已在运行时连通：

- `SparseCoordinator`（`sparse_coordinator.py`）在 `attention_begin` 中串联 `retrieve_topk`、（decode 且非空选择时）`swap_in_selected_pages`、`FlashAttentionAdaptor.adapt_for_attn_metadata`；`forward_begin` / `forward_end` 与 HiSparse 的 backup / `map_last_loc_to_buffer` 对齐。
- `model_runner.py` 在 `algorithm=quest` 时构造 `HiSparseMHACoordinator` 并 **`register_sparse_coordinator(create_sparse_coordinator(..., io_subsystem=mha_coord))`**。
- `QuestAlgorithm` + `BaseSparseAlgorithmImpl`：`construct_representations` / `update_representations` 维护 bbox；**bbox 读 K 时**若池子提供 `translate_loc_to_hisparse_device`，会把 `req_to_token` 给出的逻辑槽翻译成 **hisparse device 槽** 再索引 `k_buffer`（与 hot buffer 上的物理布局一致）。
- `FlashAttentionAdaptor`：在 HiSparse 下 **统一用 `req_to_device_buffer[req, token_pos]` 写入 FA3 的 `page_table`**（稀疏选中与稠密兜底皆然），避免仅依赖 `translate_loc_to_hisparse_device(logical)` 在 `alloc_device_buffer` 后 prefill 槽映射被清空而读到空槽的问题。

`NSABackendAdaptor` 仍为占位，与 Quest/FA3 路径无关。

### 1.3 目标

基于现有 HiSparse IO 子系统 + 现有 Quest 算法，补齐"算法 → IO"的连接，使得：

1. Qwen 模型可以用 `--enable-hisparse --hisparse-config='{"algorithm":"quest", ...}'` 启动。
2. Prefill 结束后 KV 全量位于 DRAM（pinned host），GPU 上只有一个小 hot buffer。
3. Decode 每步调用 Quest 产出 topk page；topk 通过 `hisparse.cuh` kernel 从 host 装载到 GPU。
4. HiSparse 现有 DSA 路径不受影响。

### 1.4 MVP 范围：FA3 路径固定 `--page-size 1`（推荐且当前设计以此为准）

**决策**：Quest + Qwen + **FA3** 这一条实验链路，**MVP 只支持 `server_args` 层面的 `--page-size 1`**（即 FlashAttention 后端构造的 `metadata.page_table` 为 **token 级物理槽位**，与「无稀疏时」一致）。

**原因**：

- 当 `--page-size > 1` 时，SGLang 的 FA3 集成会把 `page_table` 压成 **页号**（`flashattention_backend.py` 末尾对 `req_to_token` 做 strided + `// page_size`），注意力读 KV 时必须按「整页 × 页内 offset」寻址。
- HiSparse 的 swap-in kernel 在 **长序列**下按 **token** 粒度把历史 K/V 散落到 hot buffer 的各槽位，**不保证**同一 Quest「稀疏页」内的 16 个 token 在 hisparse device 上仍落在 **同一个 FA3 物理页** 里。二者叠在一起时，`FlashAttentionAdaptor` 要么写页号（与 scatter 冲突），要么写 token 槽位（与 FA3 页语义冲突）。
- 固定 **`--page-size 1`** 后，FA3 始终用 **token 槽位** 索引 `k_cache`；`FlashAttentionAdaptor` 在 HiSparse 下以 **`req_to_device_buffer`** 为权威把「请求内 token 位置」映射到 **hot buffer 槽位** 写回 `page_table`，**与 Quest 内部 `N_quest` 无关**，框架前面不被 FA3 的 server 级分页绑死。

**与 `hisparse-config` 里 `page_size` 的关系（避免混淆）**：

| 配置项 | 含义 |
|--------|------|
| **`--page-size 1`**（server） | KV 分配与 **FA3 `page_table` 语义**：MVP 下必须为 **1**。 |
| **`"page_size": N` in `--hisparse-config`**（可选） | **Quest 稀疏页大小**（bbox / `retrieve_topk` 的页粒度），可与 server 的 1 不同，例如 `N=16` 表示「每 16 个 token 一个 Quest 页」，再展开成 token 喂 swap-in 与 adaptor。 |

**实现状态**：`check_server_args` 在 `algorithm=quest` 且 `enable_hisparse` 时 **强制 `--page-size 1`**（否则 `ValueError`），并 **将 `self.page_size` 固定为 1**，避免后续按后端默认把分页改大。`HiSparseMHATokenToKVPoolAllocator` / `map_last_loc_to_buffer` 已支持该路径（含 decode 新 token 时释放被覆盖的 transient hisparse 槽，避免池子泄漏）。

**非目标（MVP 之后）**：在 **不** 改 HiSparse kernel（按页 swap / 页对齐 hot buffer）的前提下，支持 Quest + FA3 + **`--page-size > 1`** 的端到端正确性；可作为后续工作单独设计。

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
│  - 调用 io.swap_in_selected_pages(...) 装载 KV                 │
│  - 调用 backend_adaptor 改写 attention metadata                 │
└───────────────┬──────────────────────────┬──────────────────────┘
                │                          │
                ▼                          ▼
   ┌──────────────────────┐     ┌──────────────────────────────┐
   │   SparseAlgorithm     │     │ HiSparseCoordinator /         │
   │ (QuestAlgorithm ...)  │     │ HiSparseMHACoordinator        │
   │ - construct_repr      │     │ - host pool / device buffer   │
   │ - update_repr         │     │ - staging / direct admit      │
   │ - retrieve_topk → idx │     │ - swap_in_selected_pages      │
   │ - bbox 读 k_buffer    │     │ - eager_backup / release      │
   └──────────────────────┘     └──────────────────────────────┘
                                             ▲
                                             │
                                  ┌──────────┴───────────┐
                                  │  hisparse.cuh kernel  │
                                  │ (MLA 与 MHA 两种包装) │
                                  └──────────────────────┘
```

**关键原则：**
- `SparseAlgorithm` 只负责"给定 query 和历史 K 的表示，输出 **Quest 逻辑页号** topk"（页大小由 `hisparse-config` 的 `page_size` 决定，与 §1.4 中 server `--page-size` 解耦）。它不直接做 host→device IO；读 K 算 bbox 时走 **当前层 `k_buffer`**，在 HiSparse 下通过 **`translate_loc_to_hisparse_device`**（若池子提供）把逻辑槽换到 **device 上实际存 K 的槽**（通常为 hot buffer 区），见 §3.2。
- `HiSparseCoordinator`（MLA）/ `HiSparseMHACoordinator`（MHA）只负责"给定 token 级 topk 索引，把对应 K/V 从 host 换入 device hot buffer"，对 Quest 评分无感知。
- `SparseCoordinator` 做粘合，在 `attention_begin` 里串起 `retrieve_topk → swap_in → adapt_metadata`。
- 两种链路共存：
  - **DSA 链路**（现状）：indexer 产 topk → `HiSparseCoordinator.swap_in_selected_pages` → 改写 NSA `page_table_1`。
  - **新增 Quest/Qwen 链路**：`QuestAlgorithm.retrieve_topk` 产 topk → MHA 版 `swap_in_selected_pages` → 改写 FA3 `page_table`。

### 2.2 代码落点（与当前仓库一致）

| 类型 | 路径 | 说明 |
|------|------|------|
| 核心编排 | `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` | `attention_begin` / `attention_end`、`forward_begin` / `forward_end`；`_handle_sparse_retrieve`：空选择时 **跳过** `swap_in_selected_pages`，仍调用 adaptor 做 **稠密 + `req_to_device_buffer` 寻址** |
| 工厂 | `python/sglang/srt/mem_cache/sparsity/factory.py` | `create_sparse_coordinator`：解析 `hisparse-config`，挂接 `QuestAlgorithm` + `FlashAttentionAdaptor` + 可选 `io_subsystem` |
| MHA 池 | `python/sglang/srt/mem_cache/hisparse_mha_memory_pool.py` | `HiSparseMHATokenToKVPool`、`HiSparseMHATokenToKVPoolAllocator`（含 **`--page-size 1`** 下 logical + hisparse 双分配与映射） |
| MHA 协调 | `python/sglang/srt/managers/hisparse_mha_coordinator.py` | `HiSparseMHACoordinator`：`swap_in_selected_pages`、`req_to_device_buffer`、`map_last_loc_to_buffer` 等，API 面对齐原 `HiSparseCoordinator` 以便 scheduler 复用 |
| JIT | `python/sglang/jit_kernel/hisparse.py` | `load_cache_to_device_buffer_mha`（`IsMLA=false`） |
| FA3 适配 | `python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py` | `FlashAttentionAdaptor`：`backend_page_size!=1` 直接报错；HiSparse 下 **`page_table` ← hot-buffer 槽**（稀疏 `_write_token_level` / 兜底 `_write_dense_translated`） |
| 参数校验 | `python/sglang/srt/server_args.py` | `deepseek_nsa` 与 `quest` 分支分别校验；Quest 分支 **`--page-size 1`** 且 `self.page_size = 1` |
| 装配 | `python/sglang/srt/model_executor/model_runner.py` | `enable_hisparse`：`quest` → `HiSparseMHACoordinator` + `create_sparse_coordinator(..., io_subsystem=mha_coord)`；`deepseek_nsa` → 原 `HiSparseCoordinator` |
| 算法基类 | `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` | `num_pages` / `end_page` 等使用 **ceil_div**，短 prompt 与尾页也能建 bbox |
| DSA 现状 | `python/sglang/srt/managers/hisparse_coordinator.py` | MLA HiSparse **未**拆成独立 `hisparse_io/` 包；与 MHA 路径并列存在 |

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
| Decode 每层前 | `attention_begin(q, k, v, layer, batch, meta)` | 1. `retrieve_topk` → Quest 页级 `selected_indices`。<br>2. 若 **非空选择** 且为 decode：`swap_in_selected_pages`（token 级 `top_k_tokens`）。若 **空选择**（warmup / 尚无 bbox）：**不调** swap-in，由 adaptor 稠密兜底。<br>3. `adapt_for_attn_metadata`：用 `req_to_device_buffer` 等写 FA3 `page_table` / `cache_seqlens_int32` / `cu_seqlens_k`（稀疏时缩短 K 长度；稠密时全长 + hot-buffer 寻址）。 |
| Decode 每层后 | `attention_end(...)` | `algorithm.update_representations(...)`：为上一步新生成的 token 所在的 page 增量更新 min/max。 |
| Decode forward 结束 | `forward_end(batch)` | 触发对「上一步新 token」的 eager backup（`HiSparseCoordinator` / `HiSparseMHACoordinator` 各自的 `_eager_backup_previous_token`），并调用 `map_last_loc_to_buffer` 管理新 token 在 hot buffer 里的槽位。 |
| 请求结束 | `on_request_end(req)` | `io.request_finished(req)` + `states.clear(idx)`。 |

### 3.2 Quest 表示池与 hot buffer 的关系（关键正确性问题）

`QuestAlgorithm._compute_page_representations` 从 `k_buffer[phys_tok]` 读 K 算 min/max：先用 `req_to_token` 得到逻辑槽，若 `token_to_kv_pool` 提供 **`_translate_loc_to_hisparse_device`** 则再映射到 **device 上实际存 K 的槽** 再索引 `k_buffer`。在非 HiSparse 场景下 K buffer 通常可视为全量逻辑视图；**在 HiSparse 下历史 K 主要在 host，GPU 上为 hot 区**。为此必须约定：

1. **Quest 的 min/max 表示 (`page_k_min/max`) 在 prefill 阶段一次性计算并长期驻留 GPU**。
   - Prefill 时 `attention_end` 是在完整 K buffer 上调用的（staging 路径的 prefill 仍在 GPU 上；PD / direct-to-host 见 §3.7），此时 Quest 能拿到完整 K → min/max 一次算齐。
   - Prefill 结束后即使 KV 被换到 host，`page_k_min/max` 保留在 GPU，不再依赖 K buffer 本体。
2. **Decode 阶段 `update_representations` 只处理 "新 token 所在的页"**。新 token 此时还在 hot buffer 中（hot buffer 的 `newest_slot` 保存最新 token），因此 `k_buffer[phys_tok]` 对新页合法。对已 swap 到 host 的历史页，不做 update（Quest 的 bbox 本就随 token 增加单调变大，不必回填历史页；即使需要更新也应由 eager_backup 完成后再异步打点，见 §4.2）。
3. **Quest 的 `total_num_pages` 对应的页空间**：`BaseSparseAlgorithmImpl.initialize_representation_pool` 用 `token_to_kv_pool.get_key_buffer(start_layer).shape[0]` 等推导池规模；bbox 张量按 **Quest 页数**（与 `N_quest`、ceil_div 后的序列页数）分配，与 §1.4 中 server `--page-size` 独立。

> 副产物：Quest 的 min/max 表示占用 GPU 显存 `O(num_pages × kv_heads × head_dim × 4B × num_layers)`。对 Qwen3-32B GQA（`kv_heads=8, head_dim=128`）在 **`N_quest=16`**、`seq_len=128k` 下，每请求 ≈ 8192 页 × 8 × 128 × 4B × 2 = 64 MB 每层。这需要在文档中明示并让 `top_k` / `hisparse-config` 的 `page_size` / `host_to_device_ratio` 可调（与 server `--page-size` 无关，见 §1.4）。

### 3.3 Topk 的语义对齐

与 DSA 完全对齐：**用户配置的是 `top_k`（每步参与 attention 的 token 数量）**，HiSparse swap-in kernel 期望的也是 **token 级** 逻辑位置索引。Quest **内部**按 **Quest 稀疏页**（`hisparse-config` 里的 `page_size`，记为 `N_quest`）选页，再把选中页展开为 token：

| 维度 | DSA 现状 | Quest + Qwen 新路径 |
|------|----------|---------------------|
| 用户 config | `top_k = 2048`（token 数） | `top_k = 2048`（token 数，同语义） |
| 算法输出 | indexer 直接给 token 级索引 | Quest 给 **Quest 页号**（`num_topk_pages = ⌈top_k / N_quest⌉`） |
| 喂给 HiSparse kernel | `top_k_tokens`（int32，`[bs, top_k]`，-1 填充） | 同左：`page × N_quest + [0..N_quest)` 展开得到 **逻辑 token 位置** |
| 喂给 FA3（MVP，§1.4） | （NSA 路径） | `FlashAttentionAdaptor` 在 **`--page-size 1`** 下写 **token 级 hot-buffer（hisparse device）槽位** 到 `page_table`（来自 `req_to_device_buffer`，而非仅 `req_to_token` + `translate_loc_to_hisparse_device`） |

Quest 路径在 `SparseCoordinator._handle_sparse_retrieve` 里对 `selected_indices`（Quest 页级）展开为 token 级，由 `_expand_selected_pages_to_tokens` 完成（其中的 `page_size` 参数为 **`N_quest`**，来自 `SparseCoordinator.config.page_size`，与 server `--page-size` 解耦）。

> 固定 `num_topk_pages = ⌈top_k / N_quest⌉`、固定输出宽度，便于 eager 下稳定形状；**与 FA3 CUDA graph 同时启用仍不兼容**（§3.9）。

### 3.4 Host Pool / Device Pool 的 MHA 版本

已实现 `HiSparseMHATokenToKVPool`（与 `HiSparseNSATokenToKVPool` 并列，见 `hisparse_mha_memory_pool.py`）：

- Device 侧继承 `MHATokenToKVPool`（独立 K/V buffer）；提供 `translate_loc_to_hisparse_device` 与 `register_mapping`（与 NSA 版同一套「logical ↔ hisparse device」思想）。
- Host 侧使用 `MHATokenToKVPoolHost`（`layout="layer_first"` 等），`k_data_ptrs` / `v_data_ptrs` 供 kernel 使用。

`HiSparseMHATokenToKVPoolAllocator` 负责 logical 与 hisparse device 双槽分配及 `full_to_hisparse_device_index_mapping`；**`server_args.page_size == 1`** 时 prefill/decode 均走该 allocator 路径，并与 `map_last_loc_to_buffer` 协同回收 decode 时被覆盖的 transient hisparse 槽。

### 3.5 Kernel Wrapper（MHA 版）

`hisparse.cuh` 的 `load_cache_to_device_buffer_kernel` 模板参数 `IsMLA` 在 `false` 时会额外处理 V：

```cpp
if constexpr (!IsMLA) {
  const auto src_v = ... + src_loc * item_size_bytes;
  auto       dst_v = ... + dst_loc * item_size_bytes;
  transfer_item_warp(lane_id, src_v, dst_v, item_size_bytes);
}
```

`python/sglang/jit_kernel/hisparse.py` 已提供 `load_cache_to_device_buffer_mha`，接口形如：

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

`HiSparseMHACoordinator.swap_in_selected_pages` 向 kernel 传入 host 侧 `k_buffer[layer]` / `v_buffer[layer]`（与 JIT 包装一致）。

### 3.6 Prefill / Decode 与 `FlashAttentionAdaptor`（当前实现）

Qwen 默认走 FA3；FA3 本体不在 NSA 那样提前改写 `page_table`。在 **稀疏协调器命中** 时，由 `FlashAttentionAdaptor.adapt_for_attn_metadata` 在 **`--page-size 1`** 下原地改写 `page_table`、`cache_seqlens_int32`、`cu_seqlens_k`、`max_seq_len_k`。

**HiSparse 下寻址约定（与早期设计稿的差异）**：

- 写入 `page_table` 的必须是 **K/V 实际驻留的 device 槽**（hot buffer）。**权威映射** 为 `HiSparseMHACoordinator.req_to_device_buffer[req_pool_idx, pos_in_req]`：prefill 压缩进 hot buffer 时由 `alloc_device_buffer` 填好；decode 每步由 `map_last_loc_to_buffer` / `_grow_device_buffers` 维护。
- **稀疏选中**（Quest 返回有效页）：`_write_token_level` 将页展开为 token 位置，再用 **`req_to_device_buffer`** 取槽写入 `page_table`，并缩短 `cache_seqlens`。
- **稠密兜底**：若 `sparse_mask` 全 False、或 Quest **空选择**（全 `-1` / 无表示）、或（无 HiSparse 时）仅翻译需求：`page_table` 仍可对全长请求写成 **hot-buffer 槽**（`_write_dense_translated`：用 `req_to_device_buffer` 覆盖 `page_table` 前 `copy_w` 列，`cache_seqlens` 保持全长）。这修复了「仅用 `translate_loc_to_hisparse_device(logical)`，在 mapping 被 `alloc_device_buffer` 清掉后 prefill 槽变 0 → FA3 读空 KV」的问题。
- **`backend_page_size != 1`**：直接 `ValueError`（与 `server_args` 对 Quest 的锁定一致）。
- **`--page-size > 1` 不作为 MVP**：见 §1.4 非目标。

### 3.7 PD disagg 与单机模式

- **PD disagg**：prefill 侧仍然走原生 FA3 KV，完成后 RDMA 把每请求 KV 写入 decode 侧的 host pool；decode 侧 `admit_request_direct(req)` 预留小 device buffer。**Quest 的 `page_k_min/max` 此时来不及在 prefill 阶段算**（prefill 侧不感知 HiSparse），必须在 decode 侧的 **第一次 decode 前**做一遍"冷启动 bbox 构建"。实现上有两条路：
  1. Decode 首步（forward）前，在 host → device 预装载（`_preload_to_device_buffer` 或显式循环分 page 加载），在设备 K buffer 上按页算一遍 `construct_representations`；完成后再做稀疏 decode。对长序列这是一次 O(N) 的开销，但摊到 decode 过程中可以 pipeline（每次 swap 一批 page 计一次 bbox，分层异步推进）。
  2. 更极端的做法：在 host 上直接算 min/max，再 DMA 到 GPU 的 bbox 池。这个对 CPU 计算敏感，不推荐首版。
- **单机（无 PD）**：走 staging 路径，prefill 仍在 GPU 上发生，完整 K 在 GPU 内；Quest 的 `construct_representations` 照常在 `attention_end` 里一次过完成，然后才做 KV → host 的 staging 背景拷贝 + device buffer 压缩。

### 3.8 Server Args 改造

`check_server_args` 在 `--enable-hisparse` 时按 **`hisparse-config.algorithm`** 分支：

- **`deepseek_nsa`**：仍要求 DeepSeek-NSA 模型、`flashmla_sparse` NSA 后端、`bfloat16` KV；`page_size` 走 DSA 默认（可与 Quest 路径不同）。
- **`quest`**：禁止 DSA 模型；`kv_cache_dtype ∈ {bfloat16, float16}`；**强制 `--page-size 1`** 并将 `self.page_size = 1`（理由见 §1.4）。

`--hisparse-config` JSON 常用字段示例：

```
{
    "algorithm": "quest",
    "backend": "fa3",
    "top_k": 2048,
    "device_buffer_size": 6144,
    "host_to_device_ratio": 10,
    "page_size": 16
}
```

- `top_k` / `device_buffer_size` / `host_to_device_ratio` 与 DSA 路径同语义；Quest 用 **`page_size`（`N_quest`）** 做 bbox 与选页粒度，`num_topk_pages = ⌈top_k / N_quest⌉`。
- **`min_sparse_prompt_len`（可选）**：若省略或为 `0`，`sparse_mask` 对 batch 内所有请求为 True（始终尝试稀疏检索）。若设为 **正整数**，仅当 `prompt_len >= threshold` 的请求参与稀疏；其余请求走 adaptor 的 **稠密 + hot-buffer 寻址** 分支（`sparse_mask` 全 False 时也会翻译 `page_table`，保证 FA3 仍读到正确 KV）。

### 3.9 CUDA Graph 兼容性

MHA HiSparse 侧仍预分配 swap-in 相关 buffer（与 MLA 思路类似）。**但 Quest + FA3 + `FlashAttentionAdaptor` 当前与 CUDA Graph capture 不兼容**：capture 时 FA3 会把依赖 `max_seq_len_k` / `cache_seqlens` 的 `scheduler_metadata` 烘焙进 graph，而 adaptor 每步会改写这些字段；replay 时会触发形状不匹配等错误。`SparseCoordinator._handle_sparse_retrieve` 在 **`get_is_capture_mode()`** 为真时 **直接返回**，让 capture 走稠密路径。

**实操**：Quest + HiSparse + FA3 请 **`--disable-cuda-graph`**。在 FA3 侧支持 capture 后重算 `scheduler_metadata` 之前，不要把三者同时打开。

---

## 4. 实现状态与后续工作

### 4.1 已落地（摘要）

Quest + HiSparse + FA3 的端到端链路已在 **单机、staging 式 prefill、关闭 CUDA graph** 场景下打通：Quest 选页 → decode 步 `swap_in_selected_pages` → `FlashAttentionAdaptor` 用 **`req_to_device_buffer`** 写 FA3 `page_table`；空选择或 `sparse_mask` 全 False 时走 **稠密兜底** 仍保持正确寻址。`retrieve_topk` 使用批量 `torch.topk` 与固定输出宽度，便于后续 graph 相关工作的基础已具备。

主要文件见 **§2.2**。

### 4.2 已知限制与开放项

| 主题 | 说明 |
|------|------|
| PD disagg 下 Quest bbox | §3.7：decode 侧冷启动 bbox 仍属设计级开放项；当前 MVP 以单机 staging 为主。 |
| CUDA Graph | §3.9：Quest + FA3 + adaptor 与 capture 不兼容；需 **`--disable-cuda-graph`** 或 FA3 侧支持在 adaptor 后重算 `scheduler_metadata`。 |
| Radix cache | 仍要求 **`--disable-radix-cache`**（与 DSA HiSparse 相同）。 |
| FA3 `page_size > 1` + Quest + HiSparse | §1.4 非目标；需 kernel / hot buffer 与 FA3 页语义对齐。 |
| Quest 近似与精度 | bbox 上界选页会偶发漏关键 token；可通过 **`top_k`**、`N_quest`、`min_sparse_prompt_len` 与任务评测调参；小样本准确率会有统计波动。 |
| `update_representations` 与 host 上历史 K | 当前不回填 host 侧页到 bbox；依赖 bbox 单调性与 hot 区对新 token 的 update。 |

---

## 5. 性能与显存模型

### 5.1 显存估算（以单请求、单层为例）

| 部件 | 大小 | 说明 |
|------|------|------|
| hot buffer（K+V） | `device_buffer_size × kv_heads × head_dim × itemsize × 2` | 每请求、每层 |
| Quest bbox min/max | `num_pages × kv_heads × head_dim × 4B × 2` | 全池共享（不是每请求）|
| `req_to_device_buffer` | `max_reqs × (device_buffer_size + page_size) × 8B` | 全局 |
| `lru_slots` | `num_layers × max_reqs × device_buffer_size × 2B` | 全局 |

Qwen3-32B / GQA8 / head_dim=128 / bf16 / `device_buffer_size=6144` / Quest `N_quest=16`（`hisparse-config`）/ `seq_len=128k`（server `--page-size 1` 时 bbox 页数仍按 Quest 页计）：
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

## 6. 对用户的接口（与 `hisparse_guide.md` 相衔接）

Qwen 启动样例：

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/Qwen3-32B \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --context-length 131072 \
    --tp-size 8 \
    --page-size 1 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype bfloat16 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --enable-hisparse \
    --hisparse-config='{
        "algorithm": "quest",
        "backend": "fa3",
        "top_k": 2048,
        "device_buffer_size": 6144,
        "host_to_device_ratio": 10,
        "page_size": 16
    }'
```

说明：`--page-size 1` 由 `server_args` 对 Quest 路径 **强制**（§1.4）。`hisparse-config` 里的 `"page_size": 16` 表示 **Quest 稀疏页** `N_quest=16`，与 server 分页无关。可选字段 `"min_sparse_prompt_len": <int>` 见 §3.8。

---

## 7. 扩展其他非原生稀疏注意力算法

以下假设新算法仍服务于 **MHA/GQA + HiSparse + FA3** 这一类与 Quest 相同的部署形态；若目标后端或 KV 布局不同，需要额外评估 adaptor 与 IO。

### 7.1 算法实现（必做）

1. **继承 `BaseSparseAlgorithmImpl`**（`base_algorithm.py`），实现与现有生命周期一致的接口：
   - `initialize_representation_pool` / `construct_representations` / `update_representations`：在 prefill / decode 中维护算法所需的 KV 侧表示（统计量、索引结构等）。
   - **`retrieve_topk`**：签名与返回值需与协调器、swap-in、adaptor 衔接（见下）。
2. 在 `python/sglang/srt/mem_cache/sparsity/factory.py` 的 **`_ALGORITHM_REGISTRY`** 中注册名字，例如 `"snapkv": lambda config, device, **kw: SnapKVAlgorithm(...)`，并在 `hisparse-config` 里用 `"algorithm": "<name>"` 选用。

### 7.2 `retrieve_topk` 与下游的契约（与 Quest 对齐或显式分叉）

当前 `SparseCoordinator._handle_sparse_retrieve` 将 `retrieve_topk` 的返回值视为 **Quest 风格的「稀疏页号」矩阵** `selected_indices`，再用 **`SparseCoordinator.config.page_size`（即 `hisparse-config` 的 `page_size` / `N_quest`）** 通过 `_expand_selected_pages_to_tokens` 展成 **请求内 token 位置**，供 `swap_in_selected_pages` 与 `FlashAttentionAdaptor._write_token_level` 使用。

扩展时二选一：

- **推荐**：沿用同一契约——在配置里令 **`page_size` 等于算法的自然块长**（若算法本质是 **token 级** 选片，可令 **`page_size = 1`**，此时「页号」即 token 下标），这样 **无需改** `_expand_selected_pages_to_tokens` 与 MHA swap-in 的形状假设。
- **若输出语义不同**（例如直接产出 block id、或非连续 token 列表）：需要改 **`sparse_coordinator.py`** 的展开逻辑，或在新算法内先把结果 **编码成** 当前 pipeline 能消费的 `(selected_indices, valid_lengths)` 格式；必要时为该类算法增加 **专用 adaptor** 分支（见 §7.3）。

**空选择**：若某步无有效 topk，应返回与现实现一致的 **全 `-1` / 空张量** 形态，使协调器 **跳过 swap-in**，由 `FlashAttentionAdaptor` 走 **稠密 + `req_to_device_buffer`** 兜底（§3.6）。

### 7.3 注意力后端适配器

- `_create_backend_adaptor`（`factory.py`）在 **非 `DeepSeekNSAAlgorithm`** 且 `backend in {"fa3", "flashattention"}` 时使用 **`FlashAttentionAdaptor`**。
- 新算法若仍通过 **改写 FA3 `page_table` / `cache_seqlens` / `cu_seqlens_k`** 表达稀疏 K，通常 **复用** `FlashAttentionAdaptor`；若元数据形状或语义与 Quest 不同（例如需要第二组索引张量），可 **新增 `BackendAdaptor` 子类**，在 `_create_backend_adaptor` 里按 `isinstance(sparse_algorithm, YourAlgo)` 或 `(algorithm_name, backend)` 分发。
- **MVP 约束**：在 HiSparse + FA3 + 非原生算法路径上，仍应遵守 **`--page-size 1`** 与 **hot-buffer 寻址**（`req_to_device_buffer`）约定（§1.4、§3.6），否则易再现「逻辑槽未写 KV」类错误。

### 7.4 IO 与 `model_runner` / `server_args`

- **复用 `HiSparseMHACoordinator` + `load_cache_to_device_buffer_mha`**：只要稀疏侧最终仍给出 **「请求内 token 位置 + top_k 上限」** 的 swap-in 输入，现有 host→device kernel 与 LRU 逻辑可继续用。
- **`model_runner.py`**：`algorithm=quest` 分支已示范 **MHA 协调器 + `create_sparse_coordinator(..., io_subsystem=...)`**。新算法若同属 MHA HiSparse，可在同一分支内用 **`parse_hisparse_config` / `create_sparse_coordinator`** 的 `algorithm` 字段区分；若需 **不同 IO**（例如另一套 staging），再增加 `elif _algo == "..."` 并挂接对应 coordinator。
- **`check_server_args`（`server_args.py`）**：为 `algorithm=<新名>` 增加校验（模型族、KV dtype、`--page-size`、是否禁止 CUDA graph 等），并与文档 **`hisparse_guide.md`** 中的说明同步。

### 7.5 配置与调参

- 算法专有超参建议放在 **`hisparse-config` JSON 的额外键** 中，经 `_parse_sparse_config` 落入 **`SparseConfig.sparse_extra_config`**，在算法 `__init__` 中读取，避免污染全局 `server_args`。
- **`min_sparse_prompt_len`**、**`top_k`**、**`device_buffer_size`** 等为框架级字段，可直接复用以控制 **何时稀疏**、**每步参与注意力的 token 预算**、**热缓冲大小**。

按上述步骤，新算法以 **「注册名 + 统一 retrieve 契约 +（可选）adaptor/展开逻辑」** 接入即可，无需改动 DSA / `HiSparseCoordinator`（MLA）主路径。
