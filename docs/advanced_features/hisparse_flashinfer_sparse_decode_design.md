# Quest + HiSparse FlashInfer Sparse Decode 设计

> 目标：把当前 `Quest + HiSparse sparse_decode + FA3` 路径改为 `Quest + HiSparse sparse_decode + FlashInfer` decode backend。新 backend 专门服务 HiSparse：从 HiSparse device hot buffer 读取 KV，接收 Quest + swap-in 产出的 token 级 hot-buffer loc，并填入 FlashInfer paged decode 的 `kv_indices`。
>
> 核心决定：
>
> 1. **FlashInfer decode page_size 强制为 1**，与 HiSparse sparse decode 的 attention backend page_size=1 对齐。
> 2. Quest 内部仍可使用 `quest_page_size=16` 做 page-level score，但对 backend 输出必须是 token 级。
> 3. 首版不使用 `flashinfer.sparse.BlockSparseAttentionWrapper` 作为主线，而是新增 HiSparse 专用 FlashInfer paged decode backend。

---

## 1. 结论

这个方案可行，而且比继续改 FA3 adapter 更符合 HiSparse sparse decode 的数据流。

原因：

- HiSparse swap-in kernel 已经把 Quest 选中的 token 拷到 GPU hot buffer。
- `page_size=1` 时，一个 token loc 就等价于 FlashInfer paged decode 的一个 page index。
- FlashInfer `BatchDecodeWithPagedKVCacheWrapper` 本来就通过 `kv_indptr / kv_indices / kv_last_page_len` 指定每个 request 需要看的 KV pages。
- 因此我们不需要像 FA3 那样每层改 `page_table / cache_seqlens / cu_seqlens_k / scheduler_metadata`，只需要把每层 top-k hot-buffer loc 写进 FlashInfer 的 `kv_indices`。

最终 decode 数据流：

```text
Quest.retrieve_topk(q)
  -> token positions [bs, top_k]
  -> HiSparseCoordinator.swap_in_selected_pages(...)
  -> hot-buffer physical token locs [bs, top_k]
  -> FlashInfer kv_indices
  -> BatchDecodeWithPagedKVCacheWrapper.forward(q, HiSparseMHATokenToKVPool.kv_buffer)
```

---

## 2. Page Size 语义

### 2.1 FlashInfer page_size 必须强制为 1

本设计要求：

- ServerArgs `--page-size=1`
- `HiSparseMHATokenToKVPool.page_size == 1`
- FlashInfer paged decode wrapper 的 page size 也按 1 组织
- `kv_last_page_len[:] = 1`
- `kv_indices` 中的每个元素都是 **token 级物理 row loc**

这样 FlashInfer 的 page index 与 HiSparse hot buffer token loc 一一对应，不需要做 `loc // page_size` 或 token-to-page 转换。

如果 FlashInfer page_size > 1，会出现语义错位：

- HiSparse `swap_in_selected_pages()` 返回的是 token 级 hot-buffer loc。
- FlashInfer paged decode 会把 `kv_indices` 解释为 page index。
- 一个 page index 对应多个 token，无法直接表达 Quest 选出的任意 token 集合。
- 还会破坏“Quest top-k 对 backend 是 token 粒度”的接口约束。

因此 sparse decode + FlashInfer 路径必须像 FA3 路径一样强制 `page_size=1`。

### 2.2 Quest 内部 page_size 与 FlashInfer page_size 解耦

Quest 的算法内部可以使用 `quest_page_size=16`：

- `page_k_min/page_k_max` 按 Quest page 维护。
- score 按 Quest page 计算。
- top-k 先选 Quest pages。
- 返回前把 Quest pages 展开成 token positions。

但 FlashInfer backend 只看到 token 级结果：

```text
Quest 内部:
  quest_page_size = 16
  selected_quest_pages = [...]

对外:
  top_k_tokens = [token_pos_0, token_pos_1, ...]

FlashInfer:
  page_size = 1
  kv_indices = top_k_device_locs
```

这与当前 FA3 sparse decode 的接口原则一致。

---

## 3. 为什么不用 `flashinfer.sparse` 作为首版主线

FlashInfer sparse API (`BlockSparseAttentionWrapper`) 是 block-sparse attention mask API。它表达的是：

```text
query block i 可以 attend 到哪些 KV block j
```

而 HiSparse sparse decode 要表达的是：

```text
request i 在 layer l 只读取这些已经 swap-in 的 token hot-buffer loc
```

二者可以转化，但不是同一个接口。直接用 `flashinfer.sparse` 会带来几个问题：

- 每层每步 Quest top-k 都变，BSR `indices` 也变，可能需要频繁 `plan()`。
- 文档没有显示 sparse wrapper 像 paged decode wrapper 那样提供 `use_cuda_graph=True` 和固定 `indices` buffer。
- BSR 列空间若使用整个 hot buffer 物理空间，`N` 会很大。
- 它绕开 SGLang 现有 FlashInfer paged decode 的 CUDA Graph buffer 体系。

因此首版采用 FlashInfer paged decode wrapper。后续如果验证 `flashinfer.sparse` 对动态 sparse pattern + CUDA Graph 有更好支持，再单独评估。

---

## 4. 新 Backend 架构

### 4.1 新增 backend

新增文件：

```text
python/sglang/srt/layers/attention/flashinfer_hisparse_decode_backend.py
```

类名：

```python
class FlashInferHiSparseDecodeBackend(FlashInferAttnBackend):
    ...
```

职责：

- 只作为 sparse decode backend。
- Prefill 不走该 backend，prefill 仍走 dense FA3 或 dense FlashInfer。
- Decode 时：
  - 写当前 token KV 到 `HiSparseMHATokenToKVPool` hot buffer。
  - 调 controller 运行 Quest + swap-in。
  - 把 `top_k_device_locs` 填入 FlashInfer paged decode wrapper 的 `kv_indices`。
  - 调 `BatchDecodeWithPagedKVCacheWrapper.forward()`。

### 4.2 注册名

新增 attention backend 名称：

```text
flashinfer_hisparse_decode
```

在 `attention_registry.py` 中注册：

```python
@register_attention_backend("flashinfer_hisparse_decode")
def create_flashinfer_hisparse_decode_backend(runner):
    from sglang.srt.layers.attention.flashinfer_hisparse_decode_backend import (
        FlashInferHiSparseDecodeBackend,
    )

    return FlashInferHiSparseDecodeBackend(runner)
```

### 4.3 ServerArgs backend 选择

backend 选择走 SGLang 外层 attention backend 参数，而不是写进
`--hisparse-config` / `--sparse-decode-config`。也就是说，Quest/HiSparse
配置只描述算法与 hot buffer 大小；decode backend 由下面的启动参数决定：

```bash
--attention-backend flashinfer
```

当外层 attention backend 为 `flashinfer`，且启用非原生 HiSparse
（例如 `--enable-hisparse --hisparse-config '{"algorithm":"quest", ...}'`）
时：

```text
prefill_attention_backend = "fa3" 或 "flashinfer"
decode_attention_backend = "flashinfer_hisparse_decode"
```

首版建议固定：

```text
prefill = "fa3"
decode  = "flashinfer_hisparse_decode"
```

如果外层 attention backend 未设置或为 `fa3`，decode 仍走现有
`fa3_sparse_decode`。首版不要求用户在 sparse config 里配置 backend。

原因是 backend 是运行时 kernel 选择，应该与 `--attention-backend` 的语义一致；sparse config 只保留算法参数，如 `top_k`、`device_buffer_size`、`quest_page_size`。

---

## 5. Adapter 设计

### 5.1 新增 `FlashInferHiSparseAdapter`

新增文件：

```text
python/sglang/srt/mem_cache/sparsity/backend/flashinfer_hisparse_adapter.py
```

职责：

- 接收 `top_k_device_locs: [bs, top_k] int32`。
- 接收 `valid_lengths: [bs] int32`。
- 写入 FlashInfer decode wrapper 的：
  - `kv_indptr`
  - `kv_indices`
  - `kv_last_page_len`
- 不返回 FA3 metadata。

### 5.2 Eager 模式下的 indices 写法

eager 可以保留真实 `valid_lengths`：

```text
kv_indptr[0] = 0
kv_indptr[i + 1] = kv_indptr[i] + valid_lengths[i]
kv_indices[kv_indptr[i] : kv_indptr[i + 1]] =
    top_k_device_locs[i, :valid_lengths[i]]
kv_last_page_len[i] = 1
```

注意：

- `top_k_device_locs` 必须是 hot-buffer physical token loc。
- `top_k_device_locs` 中的 `-1` 不能写入有效区间。
- FlashInfer `page_size=1`，所以 `kv_last_page_len` 恒为 1。

### 5.3 CUDA Graph 模式下的 indices 写法

CUDA Graph 首版建议固定 shape：

```text
kv_indptr[i] = i * top_k
kv_indices[i * top_k : (i + 1) * top_k] = safe(top_k_device_locs[i])
kv_last_page_len[i] = 1
```

约束：

- 只 graph all-sparse batch。
- 每个 row 固定看 `top_k` 个 token。
- invalid loc 用 pad row 兜底，但 graph 首版应尽量保证 all-sparse row 的 valid length 等于 `top_k`。
- `raw_bs < captured_bs` 的 padded rows 由 coordinator 的 `num_real_reqs` 和 FlashInfer fixed buffers 共同保护。

---

## 6. Controller 接口改造

当前 controller 假设 adapter 返回 attention metadata。FlashInfer 路径需要把 adapter 改成 backend-neutral。

建议把接口改成：

```python
class SparseDecodeAdapter:
    def prepare_attention(
        self,
        top_k_device_locs: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        backend_context: object,
    ) -> object:
        ...
```

FA3 adapter：

- `backend_context` 是当前 `FlashAttentionMetadata`。
- 返回改写后的 metadata。

FlashInfer adapter：

- `backend_context` 是 FlashInfer wrapper / indices buffers。
- 写 `kv_indptr / kv_indices / kv_last_page_len`。
- 返回原 context 或 `None`。

`SparseAlgorithmController.begin_layer_decode()` 的公共流程不变：

```text
sparse_mask
  -> algorithm.retrieve_topk(...)
  -> io.swap_in_selected_pages(...)
  -> adapter.prepare_attention(...)
```

---

## 7. Decode 时序

### 7.1 Eager decode

```text
ScheduleBatch.prepare_for_decode
  -> coord.map_last_loc_to_buffer(...)

ModelRunner._forward_raw
  -> coord.wait_for_pending_backup()
  -> forward_batch.hisparse_coordinator = coord
  -> FlashInferHiSparseDecodeBackend.forward_decode per layer

Per layer:
  1. set_kv_buffer 写当前 token K/V 到 hot buffer reserved slot
  2. Quest.retrieve_topk 返回 token positions
  3. coord.swap_in_selected_pages 把选中 token swap 到 hot buffer
  4. adapter 写 FlashInfer kv_indices
  5. decode_wrapper.forward(q, HiSparseMHATokenToKVPool.get_kv_buffer(layer_id))
  6. controller.end_layer_decode 更新 Quest representation
```

### 7.2 CUDA Graph decode

首版 graph 内只保留固定 shape 操作：

```text
set_kv_buffer
Quest.retrieve_topk_static
swap_in_selected_pages kernel
fill_flashinfer_kv_indices kernel/copy
FlashInfer decode wrapper.forward
```

graph 外保留：

- backup previous token。
- grow hot buffer。
- all-sparse gating。
- representation update（建议 replay 后做）。

---

## 8. CUDA Graph 影响

### 8.1 FlashInfer 能减少的复杂度

相比 FA3 方案，FlashInfer paged decode 后端可以删除这些 CUDA Graph blocker：

- 每层 clone / restore FA3 dense metadata。
- 改写 `FlashAttentionMetadata.page_table`。
- 改写 `cache_seqlens_int32` 和 `cu_seqlens_k`。
- `max_seq_len_k = int(tensor.max().item())`。
- FA3 `scheduler_metadata` 与 sparse page table 不一致。

FlashInfer 的 graph fixed-buffer 语义更贴合 HiSparse：

```text
固定 kv_indices buffer
每层写入当前 top_k_device_locs
wrapper.forward 读取这个 buffer
```

### 8.2 仍未自动解决的问题

换 FlashInfer 后仍必须处理：

- Quest retrieval 的 Python loop / `.item()` / 动态 `topk`。
- Quest representation update 的动态 shape。
- mixed dense/sparse batch 的 graph 策略。
- FlashInfer wrapper 是否在 replay 时读取 `paged_kv_indices_buffer` 的最新内容。

所以 FlashInfer 是降低 backend metadata 风险，不是自动完成 CUDA Graph 支持。

### 8.3 必须验证的 FlashInfer 行为

在 CUDA Graph 前必须写 micro-test 验证：

1. 创建 `BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True, paged_kv_indices_buffer=buf, ...)`。
2. capture 后，在 replay 前修改 `buf` 的内容。
3. 确认 `wrapper.forward()` 输出随 `buf` 改变。

如果输出不变，说明 wrapper plan 缓存了 indices 内容，则需要：

- 调整 FlashInfer wrapper 使用方式；
- 或把 indices update / plan 纳入 graph；
- 或暂时只支持 eager FlashInfer sparse decode。

---

## 9. 实施计划

### 阶段 0：Micro Benchmark 验证

目标：验证 FlashInfer paged decode wrapper 可以用 `page_size=1` 的 hot-buffer loc 做 sparse attention。

测试：

- 构造 `q`。
- 构造完整 `k/v` hot buffer。
- 构造 `kv_indices = top_k_device_locs`。
- FlashInfer wrapper 输出与手写 gather 后 dense attention 输出对齐。
- 覆盖 MHA/GQA、bf16、`page_size=1`。

### 阶段 1：新增 Eager Backend

改动：

- 新增 `FlashInferHiSparseDecodeBackend`。
- 新增 `FlashInferHiSparseAdapter`。
- 注册 `flashinfer_hisparse_decode`。
- factory 在 sparse config 未写 backend 时从外层 `--attention-backend` 推断 adapter。
- server args 在非原生 HiSparse / sparse decode + `--attention-backend flashinfer` 时选择新 decode backend。

验证：

- 不启用 CUDA Graph。
- Qwen + Quest + HiSparse + FlashInfer decode 跑通。
- 与 FA3 sparse decode 对比：
  - Quest selected token positions。
  - swap-in device loc。
  - logits / generated tokens。

### 阶段 2：统一 Adapter 接口

改动：

- 把 `adapt_for_attn_metadata()` 抽象为 `prepare_attention()`。
- FA3 adapter 继续返回 metadata。
- FlashInfer adapter 写 indices buffer。
- controller 不再依赖 adapter 一定返回 FA3 metadata。

验证：

- FA3 sparse decode 原路径不回归。
- FlashInfer sparse decode eager 跑通。

### 阶段 3：Quest 固定 Shape 化

改动：

- 引入 `quest_page_size`。
- `retrieve_topk()` 固定输出 `[bs, top_k]`。
- 去掉 graph path 中的 Python loop / `.item()` / dynamic allocation。

验证：

- 不启用 CUDA Graph。
- `quest_page_size=16`，FlashInfer `page_size=1`。
- selected Quest pages 展开后的 token positions 正确。

### 阶段 4：FlashInfer CUDA Graph Fixed Buffer

改动：

- `FlashInferHiSparseDecodeBackend.init_cuda_graph_state()` 预分配：
  - `sparse_kv_indptr: [max_bs + 1]`
  - `sparse_kv_indices: [max_bs * top_k]`
  - `sparse_kv_last_page_len: [max_bs]`
- capture 时创建 `use_cuda_graph=True` 的 decode wrapper。
- `CudaGraphRunner.can_run()` 只允许 all-sparse batch。
- replay 前设置 `num_real_reqs=raw_bs`。

验证：

- all-sparse 单请求。
- all-sparse batch。
- `raw_bs < captured_bs` padding。
- 修改 `kv_indices` 后 replay 输出变化。

### 阶段 5：Representation Update 移出 Graph

改动：

- graph path 的 `forward_decode` 不调用 `end_layer_decode()`。
- `CudaGraphRunner.replay()` 之后调用 `controller.after_decode_step()`。

验证：

- 多 step decode。
- Quest representation 与 eager path 对齐。

---

## 10. 测试计划

### 10.1 单元测试

- `FlashInferHiSparseAdapter`：
  - `top_k_device_locs -> kv_indices`
  - `valid_lengths` padding
  - page_size=1 invariant
- `FlashInferHiSparseDecodeBackend`：
  - mock controller 调用顺序
  - `set_kv_buffer` 在 attention 前执行
- `QuestAlgorithm`：
  - `quest_page_size=16`
  - 输出 token-level top-k

### 10.2 端到端测试

- Qwen2/Qwen3 text-only。
- 长 prompt 触发 sparse。
- 短 prompt fallback。
- batch decode。
- 与 FA3 sparse decode 对比输出。

### 10.3 CUDA Graph 测试

- all-sparse graph replay。
- padded batch replay。
- replay 前修改 `kv_indices`，确认输出变化。
- mixed batch 确认 `can_run=False`，走 eager。

---

## 11. 最终约束清单

首版 `flashinfer_hisparse_decode` 必须满足：

- 仅 sparse decode 使用。
- `--page-size=1` 强制开启。
- FlashInfer paged decode page size 语义固定为 token 级。
- Quest 内部 page score 粒度通过 `quest_page_size` 单独控制。
- `QuestAlgorithm.retrieve_topk()` 对外返回 token positions。
- `HiSparseCoordinator.swap_in_selected_pages()` 返回 hot-buffer physical token loc。
- FlashInfer `kv_indices` 填入 hot-buffer physical token loc。
- 不使用 `flashinfer.sparse.BlockSparseAttentionWrapper` 作为首版主线。
- CUDA Graph 首版只支持 all-sparse batch。
