# HiSparse × Quest（Qwen 等）：简明设计说明

本文是 [`hisparse_quest_qwen_design.md`](./hisparse_quest_qwen_design.md) 的**中文精简版**，只保留部署与原理上必知的点；细节、公式与文件级说明以完整版为准。HiSparse 基本概念见 [`hisparse_guide.md`](./hisparse_guide.md)。

---

## 1. 要解决什么问题

- **DSA 模型**：原生稀疏 + MLA，HiSparse 沿用原 `HiSparseCoordinator`，topk 来自模型 indexer。
- **Qwen 等非原生稀疏模型**：没有 indexer，在 **`--enable-hisparse` 且 `hisparse-config` 里 `algorithm=quest"`** 时，用 **Quest** 在 decode 阶段选「该看哪些历史 KV」，再由 HiSparse 从 **host 全量 KV** 换入 **GPU hot buffer**，FA3 只在这批槽位上做注意力。

DSA 与 Quest **两条线并存**，互不影响。

---

## 2. 运行时一条线（decode）

`SparseCoordinator` 在每层 attention 前大致做三件事：

1. **Quest** `retrieve_topk` → 得到稀疏「页」索引（页大小见下节 `N_quest`）。
2. 若有有效选中且为 decode：**`swap_in_selected_pages`** 把对应 token 的 K/V 从 host 装进 hot buffer。
3. **`FlashAttentionAdaptor`** 改写 FA3 的 `page_table` / `cache_seqlens` 等，使 FA3 读到 **hot buffer 里真实有数据** 的槽位。

若某步 **没有选中**（warmup、尚无 bbox 等）：**不做 swap-in**，adaptor 走 **稠密长度 + 仍用 hot-buffer 正确寻址**，避免读空槽。

---

## 3. 必知约束（踩坑最多）

| 项 | 说明 |
|----|------|
| **`--page-size 1`** | Quest + HiSparse + FA3 **强制**；与 `hisparse-config` 里的 Quest 页大小 **不是** 一回事（见下表）。 |
| **`--disable-radix-cache`** | 与 DSA HiSparse 相同。 |
| **`--disable-cuda-graph`** | 稀疏改写与 FA3 CUDA Graph 当前不兼容。 |
| **KV dtype** | `bfloat16` 或 `float16`。 |
| **模型** | Quest 分支 **不要** 用在 DSA 模型上；DSA 仍用 `deepseek_nsa`。 |

---

## 4. 两个「page_size」别混

| 配置 | 含义 |
|------|------|
| **`--page-size 1`**（启动参数） | KV 分配与 FA3 `page_table`：**token 级槽位**；本路径下写死为 1。 |
| **`"page_size": N`（JSON）** | **Quest 内部** bbox / 选页的块长 `N_quest`（例如 16）；选完再展成 token 做 swap 与 FA3。 |

---

## 5. 常用 JSON 字段

- **`top_k`**：每步参与注意力的 **token 数** 上限（与 DSA HiSparse 语义一致）。
- **`device_buffer_size`**：每请求 GPU hot buffer 容量（token 数）；应 **≥ top_k**，且长序列场景下与 `host_to_device_ratio` 一起调显存与命中。
- **`host_to_device_ratio`**：逻辑 KV 池相对单卡 hot 容量的倍率，影响 host 侧能放多少全量 KV。
- **`page_size`**：上表中的 **`N_quest`**。
- **`min_sparse_prompt_len`（可选）**：仅当 prompt 长度 ≥ 该阈值时才做稀疏检索；更短则走稠密注意力（仍保证寻址正确）。

算法专有参数可放在 JSON **多余字段**里，经 `sparse_extra_config` 传给算法类（完整版 §7.5）。

---

## 6. 启动示例

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/Qwen3-32B \
    --trust-remote-code \
    --attention-backend fa3 \
    --page-size 1 \
    --kv-cache-dtype bfloat16 \
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

---

## 7. 精度与成本（一句话）

Quest 用 bbox **上界** 估分，是 **近似稀疏**；`top_k`、`N_quest` 过小可能掉点，过大则接近稠密、换入带宽增加。Quest 的 min/max 池占一块 **与序列页数相关** 的显存，长上下文 + 小 `N_quest` 时更费显存——完整版 §5 有估算思路。

---

## 8. 想接另一种非原生稀疏算法？

继承 `BaseSparseAlgorithmImpl`，实现 `retrieve_topk` 等与 Quest **相同的输出契约**（或先编码成该契约），在 `factory._ALGORITHM_REGISTRY` 注册；多数情况 **复用** `HiSparseMHACoordinator` 与 `FlashAttentionAdaptor`，并在 `server_args` / `model_runner` 为新区名增加分支与校验。细节见完整版 **§7**。
