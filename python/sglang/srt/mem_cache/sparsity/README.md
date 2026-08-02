# HBM-resident KV-cache sparsity

This package contains the common policy/controller/adaptor contracts for
post-hoc sparse attention. The initial runtime keeps every KV entry allocated
in HBM and changes only the subset visible to attention. It does not free,
compact, migrate, or swap KV-cache storage.

## Initial supported path

- NVIDIA CUDA, decoder-only MHA/GQA text models
- FA3 prefill and decode backends
- eager prefill; StreamingLLM and Quest eager or Breakable CUDA Graph decode
- normal prefill followed by non-speculative decode
- page-granular StreamingLLM-style sink + recent visibility
- optimized per-layer Quest key bounding-box selection (requires `page_size >= 2`)

Other backends, speculative decoding, local/hybrid attention, PD
disaggregation, DP/CP attention, physical eviction, and hierarchical placement
are rejected explicitly.

The implementation keeps logical policy output independent from physical KV
locations:

1. `SparsityPolicy` returns request-relative logical page indices.
2. `KVSparsityController` applies request/step/layer lifecycle rules.
3. `HBMResidentPlacement` translates logical pages through `req_to_token`.
4. `FlashAttentionVisibilityAdaptor` rewrites and later restores FA3 metadata.

During prefill, the controller publishes lifecycle contexts without changing
attention visibility. This lets a later KV-derived policy build auxiliary
representations without adding another model-forward hook.

## Launch

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --attention-backend fa3 \
  --enable-kv-cache-sparsity \
  --kv-cache-sparsity-config '{
    "policy": "streaming_llm",
    "backend": "fa3",
    "min_sparse_tokens": 4096,
    "policy_config": {
      "sink_pages": 4,
      "recent_pages": 2048
    }
  }'
```

The runtime `--page-size` and optional JSON `page_size` must match. With the
default page size of one, `sink_pages` and `recent_pages` are token counts.
StreamingLLM decode defaults to Breakable CUDA Graph. Because its output width
is always `sink_pages + recent_pages`, it needs only the normal batch-size graph
variants and does not multiply capture shapes by context-length buckets.
On NVIDIA CUDA, sink/recent construction, sparse-mask evaluation, exact partial
page lengths, and fixed-width padding are fused into one Triton launch. The
portable policy path remains the fallback for other devices and unusually wide
windows.

Quest uses a total `page_budget` that includes `recent_pages`. It stores two
key vectors (page minimum and maximum) per physical KV page and layer in the
KV-cache dtype. For example:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --attention-backend fa3 \
  --page-size 16 \
  --enable-kv-cache-sparsity \
  --kv-cache-sparsity-config '{
    "policy": "quest",
    "backend": "fa3",
    "page_size": 16,
    "min_sparse_tokens": 2048,
    "policy_config": {
      "page_budget": 128,
      "recent_pages": 4,
      "cuda_graph_context_buckets": [10240, 33792]
    }
  }'
```

On NVIDIA CUDA, Quest uses Triton kernels for complete-page representation
updates, bounding-box scoring, fixed-width selection finalization, and direct
FA3 metadata writes. Decode defaults to Breakable CUDA Graph. The graph key
includes both a batch-size bucket and a context-page-capacity bucket; the
configured context buckets and the model maximum context are captured at
startup. Portable PyTorch implementations remain as correctness fallbacks.
The GQA score kernel evaluates KV heads and dimensions in parallel while taking
the exact maximum over every query head in a group; MHA/MQA retain the
lower-register serial kernel. This changes scheduling only, not Quest scores or
selected pages.

The FA3 adaptor snapshots and restores only the selected page-table prefix.
The framework-facing `SelectionResult`, controller lifecycle, placement, and
adaptor interfaces are unchanged by these fast paths.

Quest preallocates two KV-dtype bounding vectors per physical page and sparse
layer. More context buckets reduce replay padding but increase graph capture
time and memory, so production values should follow the request-length
distribution rather than copying the example unchanged.

## Validation

Run the CPU policy/controller/adaptor conformance tests with:

```bash
python -m pytest -q \
  test/registered/unit/mem_cache/test_kv_sparsity_framework.py
```

For an end-to-end dense-path parity smoke test, first set
`min_sparse_tokens` above the tested context length. Then lower it below the
context length and benchmark a genuinely sparse budget. For Quest, this gives
the most direct comparison because the controller, kernels, and CUDA Graph
runtime remain enabled while only FA3 visibility changes. A separate pure
dense eager run is also useful for quantifying the total sidecar cost.

```bash
python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name random-ids \
  --random-input-len 8192 \
  --random-output-len 256 \
  --num-prompts 64 \
  --max-concurrency 16 \
  --warmup-requests 4 \
  --tokenize-prompt
```

For either policy, set `min_sparse_tokens` above the tested context to measure
the same CUDA Graph runtime with dense visibility. Explicitly lock decode to
`disabled` only when an eager comparison is desired.

Report input/output throughput, inter-token latency, the exact sparse budget,
GPU type, dtype, batch/concurrency, and whether prefix caching was warm. This
mode reduces attention work and memory traffic but does not increase KV-cache
capacity because the complete cache remains resident.
