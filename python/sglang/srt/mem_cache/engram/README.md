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
>
> For technical details on the CXL and Mooncake storage backends, see our
> paper:
> [*Pooling Engram Conditional Memory in Large Language Models using CXL*](https://doi.org/10.5281/zenodo.18883519)

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
    ├── local_engram_store.py   # CPU DRAM-backed store (default)
    ├── mooncake_engram_store.py# Mooncake distributed store backend
    ├── cxl_engram_store.py     # CXL shared-memory store backend
    └── cxl_utils/              # CXL C++/CUDA extension sources
        ├── cxl_mem.cpp / .hpp / .cu
        ├── cxl_mem_pybind.cpp
        └── demo_cxl.py

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
| `LocalEngramStore` | Default backend — stores embedding tables as CPU tensors in process memory. |
| `MooncakeEngramStore` | Distributed backend using Mooncake's key-value store for cross-node sharing. |
| `CxlEngramStore` | Shared-memory backend using CXL DAX devices via a C++/CUDA extension. |
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

## Storage Backends

Engram supports three storage backends for embedding tables. The backend is
selected via the `store_backend` field in `EngramConfig` (or environment
variables in each backend's config).

### Local (default)

Stores all embedding tables as CPU tensors in the current process's DRAM.
No external dependencies; suitable for single-node development and testing.

### Mooncake — distributed RDMA store

Uses [Mooncake](https://github.com/kvcache-ai/Mooncake)'s
`MooncakeDistributedStore` for cross-node distributed embedding storage.
Embedding tables are sharded into fixed-size blocks and distributed across
Mooncake servers. This backend is designed for multi-node scenarios where
Engram tables are too large for a single machine or need to be shared
across inference workers.

**Prerequisites:**
- The `mooncake` Python package must be installed (`pip install mooncake`).
- A running Mooncake metadata server (default: `http://127.0.0.1:8080/metadata`).
- RDMA-capable network hardware for optimal performance.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `MOONCAKE_LOCAL_HOSTNAME` | `localhost` | Local hostname / IP for Mooncake |
| `MOONCAKE_METADATA_SERVER` | `http://127.0.0.1:8080/metadata` | Metadata server endpoint |
| `MOONCAKE_MASTER_SERVER` | `127.0.0.1:50051` | Master server address |
| `MOONCAKE_PROTOCOL` | `tcp` | Transport protocol (`tcp` or `rdma`) |
| `MOONCAKE_DEVICE_NAME` | `""` | RDMA device name |
| `MOONCAKE_SERVER_ID` | `0` | This server's shard ID |
| `MOONCAKE_NUM_SERVERS` | `1` | Total number of Mooncake servers |
| `MOONCAKE_VOCAB_BLOCK_SIZE` | `64` | Number of embedding rows per block |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `17179869184` (16 GB) | Global segment size in bytes |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | `4294967296` (4 GB) | Local buffer size in bytes |

**Launch example:**

```bash
ENABLE_ENGRAM=1 \
MOONCAKE_LOCAL_HOSTNAME=192.168.1.10 \
MOONCAKE_METADATA_SERVER=http://192.168.1.1:8080/metadata \
MOONCAKE_MASTER_SERVER=192.168.1.1:50051 \
MOONCAKE_PROTOCOL=rdma \
MOONCAKE_DEVICE_NAME=mlx5_0 \
MOONCAKE_SERVER_ID=0 \
MOONCAKE_NUM_SERVERS=2 \
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-XXX \
    --disable-cuda-graph \
    --port 30000
```

### CXL — shared-memory via CXL DAX device

Uses [Compute Express Link (CXL)](https://www.computeexpresslink.org/)
memory exposed as a Linux DAX device for ultra-low-latency shared-memory
embedding storage. A custom C++/CUDA extension (`cxl_mem_ext`) provides
memory-mapped I/O with optional CUDA-registered access for GPU-direct reads.

CXL's fine-grained, byte-addressable access is well suited to Engram's
sparse, discrete retrieval patterns — unlike RDMA which is optimised for
bulk transfers. This makes CXL the ideal backend when the hardware is
available.

**Prerequisites:**
- A CXL memory device exposed as a DAX device (e.g. `/dev/dax0.0`).
- Read/write permissions on the DAX device.
- The C++ extension is auto-compiled on first use via `torch.utils.cpp_extension.load`
  (requires a C++17 compiler with SSE4.1 and clflushopt support).

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `CXL_DEV_PATH` | `/dev/dax0.0` | Path to the CXL DAX device |
| `CXL_MAP_BYTES` | `8589934592` (8 GB) | Bytes to mmap from the device |
| `CXL_OFFSET` | `0` | Starting byte offset in the device |
| `CXL_REGISTER_CUDA` | `true` (if CUDA available) | Register CXL memory with CUDA for GPU-direct access |
| `CXL_GPU_ID` | `0` | GPU device ID for CUDA registration |
| `CXL_INIT_ENGRAM` | `true` | Whether this process initialises (writes) the embedding tables |
| `CXL_BUILD_DIR` | `<auto>` | Directory for the compiled C++ extension |
| `CXL_VERBOSE_BUILD` | `false` | Verbose output during extension compilation |
| `CXL_READY_TIMEOUT_S` | `0` | Timeout (seconds) for reader to wait for writer's ready flag |
| `CXL_READY_POLL_S` | `0.1` | Poll interval (seconds) when waiting for ready flag |

**Launch example:**

```bash
ENABLE_ENGRAM=1 \
CXL_DEV_PATH=/dev/dax0.0 \
CXL_MAP_BYTES=8589934592 \
CXL_REGISTER_CUDA=true \
CXL_GPU_ID=0 \
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-XXX \
    --disable-cuda-graph \
    --port 30000
```

For detailed benchmarks and analysis of CXL vs RDMA for Engram storage,
see our paper:
[*Pooling Engram Conditional Memory in Large Language Models using CXL*](https://doi.org/10.5281/zenodo.18883519).

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

To select a different storage backend, set `store_backend` in `EngramConfig`
or configure the corresponding environment variables before launch (see
[Storage Backends](#storage-backends) above).

## Configuration

Global defaults are defined in `models/engram/engram.py` via the
`EngramConfig` and `BackBoneConfig` dataclasses. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `store_backend` | `"local"` | `"local"`, `"mooncake"`, or `"cxl_shm"` |
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

## References

- Ma, R. et al., *Pooling Engram Conditional Memory in Large Language
  Models using CXL*, 2026.
  DOI: [10.5281/zenodo.18883519](https://doi.org/10.5281/zenodo.18883519)
