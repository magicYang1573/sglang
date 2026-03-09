# Engram — N-gram Embedding Module for SGLang

Engram is an experimental module that augments Transformer models with
n-gram hash-based embedding lookups. It runs **alongside** the existing
KV cache system (HiCache) rather than replacing any backend — the KV cache
continues to manage paged attention, while Engram independently manages its
own embedding tables through a dedicated storage layer.

> **Prototype status**: The current implementation is a simulation prototype.
> We support Qwen2 and Qwen3 model integration as a proof-of-concept
> (the DeepSeek model variant is not publicly released at this time).
> The module is used for simulating the Engram architecture and is not yet
> optimised for production workloads.

## Architecture Overview

Engram is split into two packages with a clear separation of concerns:

```
sglang/srt/
├── models/engram/              # Computation side
│   ├── engram.py               # Core module (hashing, gating, projections)
│   └── triton_ops/             # Optional Triton kernel implementations
│       ├── __init__.py
│       └── engram_triton.py
│
└── mem_cache/engram/           # Storage side (this directory)
    ├── __init__.py             # Package exports
    ├── engram_store.py         # EngramStore abstract base & EngramStoreConfig
    ├── engram_store_manager.py # EngramStoreManager + global accessors
    └── local_engram_store.py   # CPU DRAM-backed store

test/srt/engram/                # Integration tests
    ├── run_all_tests.py
    ├── test_import_paths.py
    ├── test_local_store_manager.py
    └── test_engram_e2e.py
```

### Computation side (`models/engram/`)

| Component | Description |
|---|---|
| `NgramHashMapping` | Builds n-gram hash keys from token IDs using prime-modulo hashing. Supports both NumPy (offline) and Torch (online) paths. |
| `MultiHeadEmbedding` | Looks up embeddings from an `EngramStore` given hashed indices. Each n-gram order gets multiple heads with prime-sized tables. |
| `Engram` | Top-level module that combines hashing, embedding lookup, gating, value projection, and a short depthwise convolution. Supports async prefetch on a separate CUDA stream. |
| `ShortConv` | Grouped depthwise 1-D convolution with RMSNorm and SiLU activation. |
| Triton ops | Optional fused kernels for gate-value computation, grouped RMSNorm, and short-conv preprocessing. Activated via `ENGRAM_TRITON=1`. |

### Storage side (`mem_cache/engram/`)

| Component | Description |
|---|---|
| `EngramStore` (ABC) | Abstract interface: `put_sharded`, `get_one`, `get_many`, `close`. |
| `LocalEngramStore` | Default backend — stores embedding tables as CPU tensors (`torch.nn.Embedding`) in process DRAM. No external dependencies. |
| `EngramStoreManager` | Registry and factory for per-layer `EngramStore` instances. Provides global accessor functions for singleton lifecycle management. |

## Supported Models

| Model | Entry Class | How to Enable |
|---|---|---|
| Qwen2 | `Qwen2ForCausalLM` | Requires swapping `Qwen2Model` to `Qwen2MoelEngram` in the model class |
| Qwen3 | `Qwen3ForCausalLM` | Set environment variable `ENABLE_ENGRAM=1` |

In Qwen3, enabling Engram is controlled by the `ENABLE_ENGRAM` environment
variable. When set, `Qwen3ForCausalLM` instantiates `Qwen3ModelEngram`
(which inherits from `Qwen2MoelEngram`) instead of the standard
`Qwen3Model`.

## Storage Backend

Engram uses a single storage backend: **Local DRAM**.

### Local (default)

Stores all embedding tables as CPU tensors (`torch.nn.Embedding`) in the
current process's DRAM. No external dependencies; suitable for single-node
development and testing.

**How it works:**

- `put_sharded(vocab_table)` — validates shape, casts to the configured
  dtype, and copies the table into an `nn.Embedding` weight buffer.
- `get_one(index, layer_id, device)` — returns a single embedding vector;
  returns a zero vector for out-of-range indices.
- `get_many(indices, layer_id, device)` — batch lookup via `nn.Embedding`,
  returns a tensor of shape `(*indices.shape, embedding_dim)`.
- `close()` — no-op (memory is released when the Python object is garbage
  collected).

**Configuration:**

The backend is selected via `EngramConfig.store_backend` (default `"local"`).
No environment variables are required.

## Quick Start

### Running integration tests

```bash
python test/srt/engram/run_all_tests.py
```

This runs three test suites in sequence:

1. **Import path verification** — checks all modules are importable
2. **LocalEngramStore + Manager lifecycle** — put/get correctness, global accessor lifecycle
3. **End-to-end Engram module** — forward pass with `EngramStoreManager`, backward compatibility, multi-layer scenarios

### Launching SGLang with Engram (Qwen3)

> **Note:** CUDA graph capture is not yet supported with Engram. You must
> pass `--disable-cuda-graph` when launching the server.

```bash
ENABLE_ENGRAM=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-XXX \
    --disable-cuda-graph \
    --port 30000
```

## Configuration

Global defaults are defined in `models/engram/engram.py` via the
`EngramConfig` and `BackBoneConfig` dataclasses. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `store_backend` | `"local"` | Storage backend (currently only `"local"` is supported) |
| `engram_vocab_size` | `[1024, 1024]` | Per-ngram-order hash table sizes |
| `max_ngram_size` | `3` | Maximum n-gram order (bigram + trigram) |
| `n_embed_per_ngram` | `512` | Total embedding dimension per n-gram |
| `n_head_per_ngram` | `8` | Number of hash heads per n-gram order |
| `layer_ids` | `[1, 15]` | Transformer layer indices where Engram is injected |
| `enable_prefetch` | `True` | Async embedding prefetch on a separate CUDA stream |

## TODO

- [ ] Support CUDA graph capture with Engram enabled (currently requires `--disable-cuda-graph`)
- [ ] Production-ready DeepSeek model integration
- [ ] Custom CUDA kernels for hash mapping and embedding gather
- [ ] Distributed TP (tensor parallel) support for Engram embeddings across ranks
