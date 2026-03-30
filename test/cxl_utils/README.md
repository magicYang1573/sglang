# CXL / DRAM / RDMA Sparse KV Cache Read Latency Benchmark

Micro-benchmark comparing GPU scatter-read latency of **per-layer discrete top-k KV Cache** entries from three memory backends, using DeepSeek-V3.2 FP8 packed MLA parameters (`item_size=656B`).

See the full experiment design at `docs/plan/microbench/sparse_kv_read_latency_bench.md`.

## Directory Layout

```
test/cxl_utils/
├── sparse_kv_bench.py           # Main benchmark script
├── sparse_kv_kernel.cu          # GPU scatter read kernel (JIT compiled)
├── rdma_scatter_read.h          # RDMA C library header
├── rdma_scatter_read.c          # RDMA libibverbs loopback implementation
├── rdma_scatter_read_py.cpp     # pybind11 wrapper for RDMA
├── Makefile                     # Build RDMA extension
└── README.md                    # This file
```

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA H20 / H100 / A100 (CUDA ≥ 12.0) |
| CXL | CXL 2.0 type-3 device with DAX (≥ 16GB) |
| RDMA NIC | Mellanox ConnectX-5/6/7 (100/200 Gbps IB or RoCE) |
| OS | Linux 5.15+, `libibverbs-dev` installed |
| Python | PyTorch with CUDA, pybind11 |

## Build

```bash
cd test/cxl_utils

# Build RDMA extension (requires libibverbs-dev)
make rdma
```

The GPU scatter kernel and CXL extension are JIT-compiled by the benchmark script on first run.

## Usage

```bash
# Full benchmark (DRAM + CXL + RDMA loopback)
python sparse_kv_bench.py \
    --backends dram,cxl,rdma \
    --cxl-dev /dev/dax0.0 \
    --ib-dev mlx5_0 \
    --rdma-mode loopback \
    --seq-lens 32768,65536,131072 \
    --top-ks 64,128,256,512,1024,2048 \
    --item-size 656 \
    --warmup 50 \
    --iters 200 \
    --output results/

# DRAM + CXL only (no RDMA NIC)
python sparse_kv_bench.py \
    --backends dram,cxl \
    --cxl-dev /dev/dax0.0 \
    --top-ks 64,128,256,512,1024,2048 \
    --output results/

# DRAM only (quick sanity check)
python sparse_kv_bench.py \
    --backends dram \
    --seq-lens 32768 \
    --top-ks 64,128 \
    --iters 50
```

## RDMA Loopback Note

The RDMA backend uses **single-machine NIC loopback** (RC QP connected to itself). Data traverses the NIC hardware but not a physical cable. This is sufficient to demonstrate RDMA's per-message overhead in discrete access patterns. Cross-machine RDMA latency would be even higher.
