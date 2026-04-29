# Quest + HiSparse Sparse Decode CUDA Graph 改造设计

> 目标：在不改变现有 Quest + HiSparse sparse decode 设计意图的前提下，把当前首版“禁用 CUDA Graph”的路径逐步改造成可被 SGLang decode CUDA graph capture/replay 的实现。
>
> 本文是对当前实现的改造规划，不直接修改运行路径。后续实现应按阶段推进，每阶段先在 **不启用 CUDA Graph** 的 eager 路径验证正确性，再打开下一阶段。

---

## 1. 不变的设计约束

### 1.1 Sparse decode 的主语义不变

- `HiSparseCoordinator` 仍是外部唯一入口；`SparseAlgorithmController` 仍作为 coordinator 的内部模块存在。
- Prefill 仍走 dense FA3，prefill 结束后把完整 KV 迁移到 host pool，decode 阶段只在 GPU 保留 hot buffer。
- DSA / NSA 原生 HiSparse 路径不改，不应因为 sparse decode CUDA Graph 改造引入行为差异。
- FA3 sparse decode backend 仍只负责“每层调用 controller 改 metadata，然后委派给 FA3”，不把 Quest 逻辑写进 attention backend。

### 1.2 Quest 内部 page 粒度与后端 page 粒度必须分离

Quest 的算法内部可以、也应该继续按 page 做 score。例如 Quest 内部可使用 `quest_page_size=16` 维护 `page_k_min/page_k_max` 并按 page 选 top-k page。

但对外接口必须固定为 **token 粒度**：

- `QuestAlgorithm.retrieve_topk()` 返回 `[bs, top_k] int32` token positions，padding 为 `-1`。
- `HiSparseCoordinator.swap_in_selected_pages()` 接收 token positions。
- `Fa3SparseDecodeAdapter` 接收的是已经 swap-in 后的 token 级 device physical loc。
- attention backend 的 `page_size=1` 不应被 Quest 内部 page score 粒度污染。

因此后续需要把配置拆成两层：

| 配置 | 含义 | 建议约束 |
|---|---|---|
| `page_size` | attention backend / KV pool page size | sparse decode 首版仍固定 `1` |
| `quest_page_size` | Quest 内部 score 与 representation 粒度 | 默认 `16`，仅影响算法内部 |

---

## 2. 当前与 CUDA Graph 冲突的点

### 2.1 Quest retrieval 仍是 Python 动态控制流

当前 `QuestAlgorithm.retrieve_topk()` 中存在：

- `for i in range(bs)` 按请求循环。
- `bool(sparse_mask[i])`、`seq_lens[i].item()`、`int(...)` 等 GPU 到 CPU 同步。
- 按 runtime `num_pages` 创建 `torch.arange(num_pages)`。
- runtime `k_pages` 传入 `torch.topk()`。
- `torch.cat()`、`torch.unique()`、动态长度 token expansion。
- `token_positions.numel()` 控制写入长度。

这些都不适合放进 captured graph：

- capture 时的 Python 分支只执行一次，replay 时不会重新根据真实 batch 执行。
- `.item()` / `bool(tensor)` 会强制同步，破坏 graph hot path。
- 动态 shape 和动态 tensor 分配会让 captured graph 的指针、kernel launch shape 与 replay 输入不一致。

修复方向：

- 把 Quest retrieval 改为固定 shape 的 tensor / Triton 流程。
- 固定输出为 `[max_bs, top_k]` 或 `[bs, top_k]`，不再按请求产生 variable-length 中间结果。
- 固定内部 page score 范围为 `max_quest_pages = ceil(max_context_len / quest_page_size)`。
- 固定 `topk_pages_budget = ceil(top_k / quest_page_size)`，`torch.topk` 或 Triton top-k 的 `k` 必须是配置期常量。
- 用 mask 屏蔽超过 `seq_len`、最近窗口、无效 page，而不是 Python 分支。
- 在最后用固定 shape kernel 将 page top-k 展开成 token top-k；即使 Quest 内部 `quest_page_size=16`，返回给 HiSparse 的仍是 token positions。

### 2.2 Quest representation update 在 decode forward 内执行

当前 `Fa3SparseDecodeBackend.forward_decode()` 在每层 FA3 后调用 `controller.end_layer_decode()`，后者调用 `algorithm.update_representations()`。

`BaseSparseAlgorithmImpl.update_representations()` / `QuestAlgorithm._compute_page_representations()` 中存在：

- `forward_batch.forward_mode` Python 分支。
- `valid_mask.any()`。
- `max_pages = int((end_page - start_page).max().item())`。
- runtime shape 的 `torch.arange(max_pages)`。
- `nonzero()` 后按动态 index 写入 page representation。

如果这一段留在 captured forward 内，会和 Quest retrieval 一样破坏 graph。

修复方向有两种：

1. 首选：decode representation update 移出 captured forward。
   - graph 内只做：新 token KV 写入 hot buffer、Quest retrieve、swap-in、FA3 attention。
   - graph replay 完成后，由 `CudaGraphRunner.replay()` 或 `ModelRunner._forward_raw()` 的 graph 分支调用 `controller.after_decode_step()`，在 eager 流程中更新 representation。
   - 该 update 只影响下一 decode step，放在 replay 后仍满足算法语义。
2. 后续优化：把 update 也改成固定 shape Triton kernel 后放回 graph。

首版 CUDA Graph 兼容建议采用方案 1，改动面更小，也更利于逐步验证。

### 2.3 FA3 adapter 会动态 clone / 分配 / 同步

当前 `Fa3SparseDecodeAdapter` 中的 CUDA Graph blocker：

- `save_original_metadata()` 使用 `metadata.page_table.clone()`、`cache_seqlens.clone()`、`cu_seqlens.clone()`。
- `adapt_for_attn_metadata()` 中有 `bool(sparse_mask.any())`、`bool(invalid_sparse_rows.any())`。
- warning 日志里有 `invalid_sparse_rows.sum().item()`。
- `torch.nn.functional.pad(torch.cumsum(...))` 返回新 tensor，并重新赋给 `meta.cu_seqlens_k`。
- `meta.cache_seqlens_int32 = new_seqlens` 改变 tensor 指针。
- `meta.max_seq_len_k = int(meta.cache_seqlens_int32.max().item())` 依赖 CPU scalar。

在 CUDA Graph 下，metadata 的 tensor 指针必须稳定，Python 标量也不能依赖 replay 时的数据。

修复方向：

- `Fa3SparseDecodeAdapter.init_cuda_graph_state(max_bs, top_k)` 预分配：
  - `sparse_page_table_buf: [max_bs, top_k] int32`
  - `sparse_cache_seqlens_buf: [max_bs] int32`
  - `sparse_cu_seqlens_k_buf: [max_bs + 1] int32`
  - 必要时预分配 `sparse_scheduler_metadata_buf`
- eager adapter 可以继续支持 dense/sparse mixed fallback，但 graph adapter 不 clone、不新建 tensor、不换指针，只对预分配 buffer 做 `copy_()` 或 kernel 写入。
- graph capture/replay 中 `meta.page_table` 指向固定 sparse buffer，`meta.cache_seqlens_int32` 指向固定 sparse seqlens buffer，`meta.cu_seqlens_k` 指向固定 cu buffer。
- `meta.max_seq_len_k` 在 sparse graph 首版固定为 `top_k`，不从 runtime tensor `.item()` 得到。

### 2.4 FA3 scheduler metadata 与 sparse metadata 不一致

FA3 decode 会优先使用 `metadata.scheduler_metadata`。当前 adapter 改写了 `page_table/cache_seqlens/cu_seqlens_k`，但没有同步重算或清空 `scheduler_metadata`。

这在 eager 路径也有风险：FA3 可能使用 dense metadata 预计算出来的 scheduler metadata，但实际 page table 已变成 sparse top-k。

修复方向：

- 第一阶段先在 `fa3_sparse_decode` 路径禁用 precomputed scheduler metadata：adapter 改写 sparse metadata 后强制 `metadata.scheduler_metadata = None`。
- eager 正确性稳定后，再增加 sparse scheduler metadata buffer：
  - capture 时按 `max_bs/top_k` 分配固定 buffer。
  - replay 或 graph 内按 sparse `cache_seqlens` 更新。
  - 确认 FA3 `get_scheduler_metadata` 本身可安全用于 capture；否则保持禁用。

### 2.5 mixed dense/sparse batch 的 graph 语义不稳定

当前设计支持短请求走 dense、长请求走 sparse，`sparse_mask` 可按 row 混合。

这对 eager 是合理的，但对首版 CUDA Graph 有两个问题：

- graph 内不能用 Python `if sparse_mask.any()` 决定是否改写。
- dense row 可能需要完整 `max_seq_len_k` 和 full page table，sparse row 只需要 `top_k`。同一个 captured FA3 launch 对 `max_seq_len_k`、scheduler metadata、page table 宽度的要求更复杂。

修复方向：

- 首版 CUDA Graph 只覆盖 **all-sparse batch**。
- 如果当前 batch 任意 row 不满足 `min_sparse_prompt_len`，`CudaGraphRunner.can_run()` 返回 `False`，该 step 走 eager sparse decode，保留现有 dense fallback 语义。
- 后续再做 mixed graph 支持，建议通过 scheduler 分桶或双 graph 路径实现，而不是在同一个 graph 内同时承载 full dense 与 sparse row。

这个限制不改变算法设计意图，只是 CUDA Graph 的启用条件更保守。

### 2.6 `SparseAlgorithmController` 中的 CPU all-dense flag 不适合 graph replay

当前 `HiSparseCoordinator.map_last_loc_to_buffer()` 在 graph 外通过 CPU mirror 计算 `_all_dense_this_step`，`begin_layer_decode()` 再用 Python bool 跳过 sparse pipeline。

这在 eager 中可用，但 captured graph 中 Python 分支只在 capture 时固化，不能表达 replay 时的 batch 差异。

修复方向：

- 保留 eager 的 `_all_dense_this_step` 快路径。
- graph 首版不在 graph 内处理 all-dense / mixed-dense：
  - `CudaGraphRunner.can_run()` 先检查 batch 是否 all-sparse。
  - graph 内 controller 不再检查 `_all_dense_this_step`，或者该值在 capture/replay 必须固定为 False。
- `compute_all_dense_flag()` 只作为 graph 外的 `can_run` 判定辅助。

### 2.7 CUDA Graph capture 使用 dummy batch，当前 sparse path 没有 dummy state

`CudaGraphRunner.capture_one_batch_size()` 用固定 buffer 构造 dummy `ForwardBatch` 并捕获 decode graph。当前 sparse decode 在 capture 时会进入真实 controller：

- Quest 会读取 `RequestTrackers`、`req_to_token_pool`、page representations。
- swap-in kernel 会读取 `req_to_host_pool`、LRU、host cache loc。
- capture 时这些状态未必对应真实请求。

修复方向：

- `SparseAlgorithmController.init_cuda_graph_state(max_bs, max_context_len)` 初始化 graph capture 所需 dummy-safe state。
- capture 阶段使用固定 dummy `req_pool_indices`、`seq_lens=top_k` 或 `seq_lens_fill_value=top_k`，并预填：
  - dummy prompt lens 满足 sparse threshold。
  - dummy page representations 为合法零值。
  - dummy host loc / device buffer loc 指向合法 pad row 或预留 row。
- replay 前由现有 `map_last_loc_to_buffer()` 和 `init_forward_metadata_replay_cuda_graph()` 填真实 req/buffer/seqlen 内容。

### 2.8 `Fa3SparseDecodeBackend` 继承 dense FA3 的 graph metadata 初始化

当前 `Fa3SparseDecodeBackend` 只覆盖 `forward_decode()`，CUDA Graph 相关方法全部继承 dense FA3：

- `init_cuda_graph_state()`
- `init_forward_metadata_capture_cuda_graph()`
- `init_forward_metadata_replay_cuda_graph()`
- `get_cuda_graph_seq_len_fill_value()`

dense FA3 的 graph metadata 是按 full context page table 初始化的，不知道 sparse top-k page table、valid lengths、controller scratch buffers。

修复方向：

- `Fa3SparseDecodeBackend` 需要覆盖 CUDA Graph metadata 相关方法，但仍复用 dense FA3 的普通 eager metadata。
- `get_cuda_graph_seq_len_fill_value()` 对 sparse decode 返回 `top_k`，使 capture 的 kernel shape 对齐 sparse attention 长度。
- capture/replay metadata 中，decode path 的 `page_table/cache_seqlens/cu_seqlens_k` 指向 sparse adapter 的固定 buffers。
- 如果首版 all-sparse graph 固定 `max_seq_len_k=top_k`，capture 与 replay 的 FA3 launch 参数保持一致。

### 2.9 server args 当前主动禁用了 CUDA Graph

当前 `server_args.py` 中 `--enable-sparse-decode` 会自动：

- `self.disable_cuda_graph = True`
- `self.disable_piecewise_cuda_graph = True`

这是正确的安全保护。只有在上述阶段完成后才能放开普通 decode CUDA Graph。

Piecewise CUDA Graph 仍建议继续禁用，直到 sparse metadata 指针、per-layer update、mixed batch 策略都稳定后再单独设计。

---

## 3. 目标架构

### 3.1 Graph 外执行的部分

以下逻辑继续在 CUDA Graph 外执行：

- request admit / finish / retract。
- prefill representation construction。
- prefill KV 迁移到 host。
- `map_last_loc_to_buffer()`：
  - backup 上一步 token。
  - grow hot buffer。
  - 写 full-to-hot mapping。
  - 更新 `num_real_reqs`。
- `wait_for_pending_backup()`。
- 首版 graph 方案中的 `controller.after_decode_step()` representation update。
- `CudaGraphRunner.can_run()` 中的 all-sparse 判定。

这些逻辑有 CPU 分支、allocator、event、host pool 操作，不应塞进 graph。

### 3.2 Graph 内执行的部分

graph 内每层只保留固定 shape、固定指针的工作：

```text
q
 -> Quest retrieve_topk_fixed_shape(q, page_k_min/max, seq_lens, req_pool_indices)
 -> HiSparse swap_in_selected_pages kernel
 -> FA3 sparse metadata buffer update
 -> flash_attn_with_kvcache
```

关键要求：

- Quest output 固定 `[bs, top_k] int32` token positions。
- swap-in output 固定写入 coordinator 预分配的 `top_k_device_locs_buffer`。
- adapter metadata 固定写入预分配 sparse metadata buffers。
- FA3 看到的 metadata tensor 指针在 capture/replay 之间不变。

---

## 4. 分阶段实施方案

### 阶段 0：基线收敛与保护

目标：明确当前 eager sparse decode 正确，CUDA Graph 仍禁用。

改动：

- 保留 `server_args.py` 的 auto-disable CUDA Graph。
- 修复 eager 下明显 metadata 风险：adapter 改写 sparse metadata 后先禁用或重算 `scheduler_metadata`。
- 增加测试覆盖当前 token-level contract：
  - Quest 内部按 page 选，输出仍为 token positions。
  - `top_k_tokens.dtype == int32`。
  - shape 恒为 `[bs, config.top_k]`。

验证：

- 不启用 CUDA Graph 跑 sparse decode。
- 对比 dense FA3 与 sparse decode 的 basic correctness。
- 对短请求 dense fallback、长请求 sparse path 分别测试。

### 阶段 1：拆分 `quest_page_size`

目标：解决“Quest 内部 page 粒度”和“FA3 page_size=1”混淆。

改动：

- `SparseConfig.sparse_extra_config` 中引入 `quest_page_size`，默认 `16`。
- `BaseSparseAlgorithmImpl.page_size` 不再直接使用 backend `config.page_size` 作为算法 page 粒度；Quest 使用 `self.quest_page_size`。
- Quest page representation 的 phys page id 不能再用 `token_loc // backend_page_size` 表示算法页，应明确维护：
  - token position -> quest page id。
  - quest page id -> token positions。
  - selected quest pages -> token positions。
- `retrieve_topk()` 输出仍固定 token positions。

验证：

- `config.page_size=1, quest_page_size=16` 时，Quest 能返回 token 级 top-k。
- 构造 seq_len 非 16 对齐的请求，最后一页 token 展开和裁剪正确。
- 不启用 CUDA Graph。

### 阶段 2：Quest retrieval 固定 shape 化

目标：去掉 capture 内不能 replay 的 Python 动态控制流。

改动：

- 增加 `QuestAlgorithm.init_cuda_graph_state(max_bs, max_context_len)` 或统一 `initialize_representation_pool()` 中预分配 scratch：
  - `page_ids_buf: [max_bs, max_quest_pages]`
  - `page_scores_buf: [max_bs, max_quest_pages]`
  - `topk_pages_buf: [max_bs, topk_pages_budget]`
  - `topk_tokens_buf: [max_bs, top_k]`
  - `valid_lengths_buf: [max_bs]`
- 新增 `retrieve_topk_static()`：
  - 输入 shape 固定。
  - 不使用 `.item()`、`bool(tensor)`、`nonzero()`、`unique()`。
  - top-k 的 `k` 是配置期常量。
  - 近窗 token 用固定 mask 保证进入最终 token list。
- eager `retrieve_topk()` 可以先调用 static 版本，减少双路径分叉。

验证：

- 不启用 CUDA Graph，替换 Quest retrieval 后跑 correctness。
- 针对 batch 内不同 `seq_len`、不同 sparse mask 的输出做单测。
- 对 `quest_page_size=16` 的 page->token 展开做边界测试。

### 阶段 3：FA3 sparse metadata 固定 buffer 化

目标：adapter 不再动态 clone、分配、替换 metadata tensor 指针。

改动：

- `Fa3SparseDecodeAdapter` 增加 graph state：
  - eager snapshot 可保留，但 graph snapshot 必须是预分配 buffer + `copy_()`。
  - graph path 的 `adapt_for_attn_metadata()` 不使用 Python bool 分支。
- `Fa3SparseDecodeBackend` 覆盖：
  - `init_cuda_graph_state()`
  - `init_forward_metadata_capture_cuda_graph()`
  - `init_forward_metadata_replay_cuda_graph()`
  - `get_cuda_graph_seq_len_fill_value()`
- 首版 graph metadata 只支持 all-sparse batch：
  - `cache_seqlens = valid_lengths`。
  - `page_table[:, :top_k] = top_k_device_locs`。
  - `max_seq_len_k = top_k`。
  - `scheduler_metadata = None`。

验证：

- 不启用 CUDA Graph，强制使用固定 sparse metadata buffer 跑 eager。
- adapter 单测确认 metadata tensor 指针不变。
- 确认 FA3 在 `scheduler_metadata=None` 下输出正确。

### 阶段 4：representation update 移出 captured forward

目标：graph 内不再执行 `update_representations()` 的 Python 动态逻辑。

改动：

- `Fa3SparseDecodeBackend.forward_decode()` 在 `forward_batch.forward_mode.is_cuda_graph()` 时不调用 `controller.end_layer_decode()`。
- `CudaGraphRunner.replay()` 完成后调用 `controller.after_decode_step(forward_batch)`。
- eager decode 仍保留当前每层 `end_layer_decode()`，或也统一切到 step 后 update，二者选一：
  - 若统一切换，验证更简单。
  - 若保留双路径，需确保 eager/graph 的 representation state 一致。
- `after_decode_step()` 中按 layer 更新新 token 所在 Quest page。

验证：

- 不启用 CUDA Graph，先将 update 从 per-layer end hook 移到 step 后，验证多 decode step 后 top-k 选择不漂移。
- 对比迁移前后同一请求前 N step 的 selected token set。

### 阶段 5：打开普通 decode CUDA Graph

目标：只在 all-sparse batch 上启用 CUDA Graph。

改动：

- `CudaGraphRunner.can_run()` 增加 sparse decode 判定：
  - coordinator 存在且 `mode == "sparse_decode"` 时，只有 all-sparse batch 才允许 graph。
  - mixed 或 all-dense batch 走 eager。
- capture 前初始化 controller / algorithm / adapter 的 graph state。
- replay 前填充真实 sparse metadata 输入 buffer，设置 `num_real_reqs=raw_bs`。
- 移除 `server_args.py` 对普通 CUDA Graph 的 auto-disable；保留 piecewise CUDA Graph disable。

验证：

- all-sparse 单请求、batch、多 batch size replay。
- batch padding：`raw_bs < captured_bs` 时 padded row 不读非法 host/device loc。
- 与 eager sparse decode 比较 logits 或 generated token。
- 长上下文、多步 decode，确认 host backup、swap-in、LRU、Quest page rep 都正常。

### 阶段 6：后续 mixed batch 与 piecewise graph

目标：扩大 graph 覆盖率。

可选方案：

- scheduler 层把 sparse decode batch 分成 all-sparse graph batch 与 dense/eager batch。
- 或在 sparse graph metadata 中保留 full-context page table 宽度，允许 dense row 和 sparse row 混合，但这会显著增加 metadata 与 FA3 scheduler 复杂度。
- Piecewise CUDA Graph 单独设计，不能简单继承普通 graph 的 assumptions。

---

## 5. 需要优先落地的代码位置

| 模块 | 当前问题 | 首个改造动作 |
|---|---|---|
| `quest_algorithm.py` | retrieval Python loop、动态 shape、`.item()` | 加 `quest_page_size`，再写 fixed-shape retrieval |
| `base_algorithm.py` | 算法 page size 与 backend page size 混用 | 明确算法 page size contract |
| `sparse_coordinator.py` | graph 内 all-dense Python 分支、end-layer update | graph path 跳过 dense fallback，update 移到 step 后 |
| `fa3_sparse_decode_adapter.py` | clone、pad、new tensor、`.item()` | 预分配 sparse metadata buffer，in-place 更新 |
| `fa3_sparse_decode_backend.py` | 继承 dense FA3 graph metadata | 覆盖 graph metadata 初始化与 fill value |
| `flashattention_backend.py` | scheduler metadata 与 sparse 改写不一致 | sparse decode 先禁用 scheduler metadata |
| `hisparse_coordinator.py` | graph 外/内边界需要明确 | 保持 backup/grow 外置，swap-in kernel 保持 graph 内 |
| `cuda_graph_runner.py` | can_run 不知道 sparse all-sparse 条件 | 增加 sparse decode graph gating 与 replay 后 update hook |
| `server_args.py` | 当前禁用 CUDA Graph | 最后阶段再移除普通 graph auto-disable |

---

## 6. 验证策略

### 6.1 每阶段 eager 验证

每个阶段都先不启用 CUDA Graph：

1. 单元测试：Quest page->token 输出、adapter metadata、controller call order。
2. 小模型端到端：dense FA3 vs sparse decode。
3. 长 prompt：超过 `min_sparse_prompt_len` 的 sparse path。
4. 短 prompt：dense fallback path。
5. 多 step decode：确认 representation update 影响下一步 retrieval。

### 6.2 CUDA Graph 验证

只在阶段 5 开始：

1. all-sparse 单请求 graph replay。
2. all-sparse batch，覆盖 `raw_bs < captured_bs` padding。
3. mixed batch 确认 `can_run=False`，自动走 eager。
4. 对同一 seed 比较 eager sparse decode 与 graph sparse decode 的 logits / token。
5. 打开 `CUDA_LAUNCH_BLOCKING=1` 和 debug graph，定位非法地址或 metadata 指针问题。

---

## 7. 待确认问题

当前建议的首版 CUDA Graph 策略是 **只 graph all-sparse batch，mixed batch 继续 eager**。这不会改变 sparse decode 的功能语义，但会影响 graph 覆盖率。如果希望 mixed dense/sparse batch 也必须首版 graph 化，需要额外设计同一 graph 内 full-context dense row 与 top-k sparse row 共存的 metadata 方案。

另一个建议是 **把 Quest decode representation update 移出 captured forward**。这保持“本 step 结束后更新、下一 step 生效”的算法语义，但会改变当前代码里“每层 FA3 后立即 update”的实现位置。如果后续算法需要同一 forward 内后续层依赖前面层的 update，需要重新评估；Quest 当前不需要这个依赖。
