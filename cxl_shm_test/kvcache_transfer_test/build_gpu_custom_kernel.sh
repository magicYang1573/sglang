#!/usr/bin/env bash
set -euo pipefail

# Build GPU KV cache latency benchmark using the custom kernel copy path.
# The binary will be named kvcache_latency_cxl2gpu_custom_kernel.exe.

nvcc -O2 -std=c++17 -Xcompiler "-fopenmp" -x cu \
  kvcache_latency_cxl2gpu_custom_kernel.cpp -lcudart -o kvcache_latency_cxl2gpu_custom_kernel.exe

# Example run (adjust to your environment):
./kvcache_latency_cxl2gpu_custom_kernel.exe --dax /dev/dax0.0 --token-total 2048 --tokens 16 \
  --layers 64 --layers-total 64 --heads 8 --head-dim 128 --token-offset 0 --layer-offset 0 \
  --warmup 3 --iterations 20 -v

