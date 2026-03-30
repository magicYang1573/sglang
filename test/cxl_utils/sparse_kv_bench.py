#!/usr/bin/env python3
"""
Single-Layer Sparse KV Cache Read Latency Micro-Benchmark
==========================================================

Compares GPU scatter-read latency from three memory backends:
  - DRAM  (CPU pinned memory, cudaHostRegister)
  - CXL   (CXL type-3 device mmap, cudaHostRegister, GPU direct read)
  - RDMA  (discrete per-token RDMA READs via NIC loopback, then GPU copy)

Uses DeepSeek-V3.2 FP8 packed MLA parameters:
  item_size = 656 bytes, store_dtype = uint8

Reference: docs/plan/microbench/sparse_kv_read_latency_bench.md
"""

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
#  JIT compile the GPU scatter read kernel
# ---------------------------------------------------------------------------

_scatter_module = None


def get_scatter_module():
    global _scatter_module
    if _scatter_module is not None:
        return _scatter_module

    build_dir = ROOT / "build" / "scatter_kernel"
    build_dir.mkdir(parents=True, exist_ok=True)

    _scatter_module = load(
        name="sparse_kv_scatter_ext",
        sources=[str(ROOT / "sparse_kv_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=False,
        with_cuda=True,
    )
    return _scatter_module


def launch_scatter_kernel(
    src_ptr: int,
    dst_ptr: int,
    indices_ptr: int,
    item_size: int,
    top_k: int,
    stream: Optional[int] = None,
):
    """Launch the scatter_read_kernel via raw CUDA driver API."""
    # We use the torch custom op path: the .cu is compiled as a plain CUDA
    # file with extern "C", so we call it through ctypes / cuLaunchKernel.
    # However, for simplicity we wrap it with a small inline extension.
    raise NotImplementedError("Use _launch_scatter_kernel_torch instead")


# Torch-friendly wrapper that calls the CUDA kernel via a thin C++ shim
# compiled at the same time as the .cu file.  We compile a combined module.

_scatter_torch_module = None


def _build_scatter_torch_module():
    """Build a combined .cu + wrapper that exposes a torch-callable function."""
    global _scatter_torch_module
    if _scatter_torch_module is not None:
        return _scatter_torch_module

    build_dir = ROOT / "build" / "scatter_torch"
    build_dir.mkdir(parents=True, exist_ok=True)

    wrapper_src = build_dir / "scatter_wrapper.cu"
    wrapper_src.write_text(r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void scatter_read_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const int64_t* __restrict__ indices,
    int64_t item_size_bytes);

void launch_scatter(int64_t src_ptr, torch::Tensor dst, torch::Tensor indices,
                    int64_t item_size) {
    int top_k = indices.size(0);
    dim3 grid(top_k);
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_read_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_ptr),
        dst.data_ptr<uint8_t>(),
        indices.data_ptr<int64_t>(),
        item_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_scatter", &launch_scatter,
          "Launch scatter read kernel: src_ptr(int64), dst(Tensor), indices(Tensor), item_size(int64)");
}
""")

    _scatter_torch_module = load(
        name="scatter_torch_ext",
        sources=[
            str(ROOT / "sparse_kv_kernel.cu"),
            str(wrapper_src),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=False,
        with_cuda=True,
    )
    return _scatter_torch_module


# ---------------------------------------------------------------------------
#  CXL extension (reuse sgl-kernel/cxl_utils)
# ---------------------------------------------------------------------------

_cxl_ext = None


def build_cxl_extension():
    global _cxl_ext
    if _cxl_ext is not None:
        return _cxl_ext

    cxl_root = ROOT.parent.parent / "sgl-kernel" / "cxl_utils"
    if not (cxl_root / "cxl_mem.cpp").exists():
        return None

    build_dir = ROOT / "build" / "cxl_ext"
    build_dir.mkdir(parents=True, exist_ok=True)

    _cxl_ext = load(
        name="cxl_mem_ext",
        sources=[
            str(cxl_root / "cxl_mem.cpp"),
            str(cxl_root / "cxl_mem_cuda.cu"),
            str(cxl_root / "cxl_mem_pybind.cpp"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-mclflushopt", "-mclwb", "-msse4.1", "-fopenmp"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lgomp"],
        build_directory=str(build_dir),
        verbose=False,
        with_cuda=True,
    )
    return _cxl_ext


# ---------------------------------------------------------------------------
#  RDMA extension
# ---------------------------------------------------------------------------

_rdma_ext = None


def load_rdma_extension():
    global _rdma_ext
    if _rdma_ext is not None:
        return _rdma_ext

    so_pattern = list(ROOT.glob("rdma_scatter_ext*.so"))
    if not so_pattern:
        print("[WARN] RDMA extension not found. Build with 'make rdma' first.")
        return None

    sys.path.insert(0, str(ROOT))
    try:
        import rdma_scatter_ext
        _rdma_ext = rdma_scatter_ext
    except ImportError as e:
        print(f"[WARN] Failed to import RDMA extension: {e}")
        return None
    return _rdma_ext


# ---------------------------------------------------------------------------
#  Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(latencies: List[float], top_k: int, item_size: int) -> Dict:
    arr = np.array(latencies)
    data_bytes = top_k * item_size
    mean_us = float(np.mean(arr))
    bw_gbs = (data_bytes / 1e9) / (mean_us / 1e6) if mean_us > 0 else 0.0
    return {
        "mean_us": mean_us,
        "std_us": float(np.std(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "p99_us": float(np.percentile(arr, 99)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "eff_bw_gbs": bw_gbs,
    }


def fmt_stats_row(name: str, s: Dict) -> str:
    return (
        f"{name:<16s}| {s['mean_us']:>9.2f} | {s['std_us']:>8.2f} | "
        f"{s['p50_us']:>9.2f} | {s['p99_us']:>9.2f} | {s['eff_bw_gbs']:>13.2f}"
    )


# ---------------------------------------------------------------------------
#  DRAM backend
# ---------------------------------------------------------------------------

def init_dram_pool(seq_len: int, item_size: int) -> torch.Tensor:
    buf = torch.randint(0, 256, (seq_len, item_size), dtype=torch.uint8)
    buf = buf.contiguous()
    torch.cuda.cudart().cudaHostRegister(
        buf.data_ptr(), buf.numel() * buf.element_size(), 0
    )
    return buf


def cleanup_dram_pool(buf: torch.Tensor):
    torch.cuda.cudart().cudaHostUnregister(buf.data_ptr())


def bench_dram(
    scatter_mod,
    host_buf: torch.Tensor,
    item_size: int,
    top_k: int,
    warmup: int,
    iters: int,
) -> List[float]:
    seq_len = host_buf.shape[0]
    src_ptr = host_buf.data_ptr()
    device_buf = torch.empty(top_k, item_size, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        indices = torch.randint(0, seq_len, (top_k,), dtype=torch.int64, device="cuda")
        scatter_mod.launch_scatter(src_ptr, device_buf, indices, item_size)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, seq_len, (top_k,), dtype=torch.int64, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scatter_mod.launch_scatter(src_ptr, device_buf, indices, item_size)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000.0)  # ms → μs

    return latencies


# ---------------------------------------------------------------------------
#  CXL backend
# ---------------------------------------------------------------------------

def bench_cxl(
    scatter_mod,
    cxl_ext,
    cxl_dev: str,
    seq_len: int,
    item_size: int,
    top_k: int,
    gpu_id: int,
    warmup: int,
    iters: int,
) -> List[float]:
    total_bytes = seq_len * item_size
    # DAX devices require mmap size aligned to huge page boundary (2MB)
    HUGE_PAGE_SIZE = 2 * 1024 * 1024
    map_bytes = ((total_bytes + HUGE_PAGE_SIZE - 1) // HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE
    ok = cxl_ext.cxl_init(
        dev_path=cxl_dev,
        map_bytes=map_bytes,
        register_cuda=True,
        gpu_id=gpu_id,
    )
    if not ok:
        raise RuntimeError(f"cxl_init failed for {cxl_dev}")

    src_data = torch.randint(0, 256, (seq_len, item_size), dtype=torch.uint8)
    cxl_ext.tensor_to_cxl(src_data.view(-1), offset=0)

    # After cxl_init with register_cuda=True, the device_ptr is the
    # CUDA-accessible host pointer.  We need to get it.  The current
    # cxl_mem_pybind does not expose cxl_get_base_ptr / device_ptr directly,
    # so we use cxl2vram_noflush_parallel_contiguous_dst's kernel path which
    # reads from device_ptr.  For our scatter kernel we need the raw pointer.
    #
    # Workaround: allocate a tiny GPU tensor via cxl_to_tensor to confirm the
    # path works, then use the device_ptr obtained from CUDA registration.
    # Since cxl_mem.cpp stores g_cxl globally, we can retrieve device_ptr
    # by calling the existing copy functions.  But for the scatter kernel
    # we actually need the HOST pointer (which is what cudaHostRegister
    # exposes).  The base host pointer IS the mmap pointer = g_cxl.base.
    #
    # We'll pass the host pointer (= data_ptr of a CPU tensor at the CXL
    # mmap region).  Since we registered it with cudaHostRegister, the GPU
    # kernel can read it via PCIe/CXL using ld.global.nc.
    #
    # To get this pointer, we rely on the fact that tensor_to_cxl copies
    # data INTO the CXL mmap and we need the mmap base address.  The
    # cxl_to_tensor_noflush_parallel_contiguous_dst path uses device_ptr
    # (from cudaHostGetDevicePointer) in its kernel.  For ld.global.nc
    # from a kernel, the device_ptr works identically.
    #
    # Approach: we'll do a small probe to find the device_ptr.
    # Actually, the simplest approach is to add a helper.  For now, we
    # use the CXL extension's scatter-gather kernel path directly:
    # cxl_to_tensor_noflush_parallel_contiguous_dst already launches a
    # kernel with (device_ptr, dst, offsets, segment_size) which is
    # essentially our scatter read.  We can benchmark that directly.

    device_buf = torch.empty(top_k * item_size, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        indices = torch.randint(0, seq_len, (top_k,), dtype=torch.int64)
        offsets = (indices * item_size).to(torch.int64)
        cxl_ext.cxl_to_tensor_noflush_parallel_contiguous_dst(
            device_buf, item_size, offsets
        )
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, seq_len, (top_k,), dtype=torch.int64)
        offsets = (indices * item_size).to(torch.int64)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cxl_ext.cxl_to_tensor_noflush_parallel_contiguous_dst(
            device_buf, item_size, offsets
        )
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000.0)

    cxl_ext.cxl_close()
    return latencies


# ---------------------------------------------------------------------------
#  RDMA backend
# ---------------------------------------------------------------------------

def bench_rdma(
    scatter_mod,
    rdma_ext,
    ib_dev: str,
    seq_len: int,
    item_size: int,
    top_k: int,
    warmup: int,
    iters: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Returns (rdma_latencies, gpu_latencies, total_latencies) in μs."""

    remote_size = seq_len * item_size
    local_size = top_k * item_size

    remote_buf = torch.randint(0, 256, (seq_len, item_size), dtype=torch.uint8)
    remote_buf = remote_buf.contiguous()

    staging_buf = torch.zeros(top_k, item_size, dtype=torch.uint8)
    staging_buf = staging_buf.contiguous()

    # Pin both buffers
    torch.cuda.cudart().cudaHostRegister(
        remote_buf.data_ptr(), remote_buf.numel(), 0
    )
    torch.cuda.cudart().cudaHostRegister(
        staging_buf.data_ptr(), staging_buf.numel(), 0
    )

    # Initialise RDMA loopback
    rdma_ext.rdma_init(
        ib_dev,
        staging_buf.data_ptr(),
        local_size,
        remote_buf.data_ptr(),
        remote_size,
        top_k,
    )

    device_buf = torch.empty(top_k, item_size, dtype=torch.uint8, device="cuda")
    contiguous_indices = torch.arange(top_k, dtype=torch.int64, device="cuda")

    rdma_latencies = []
    gpu_latencies = []
    total_latencies = []

    for it in range(-warmup, iters):
        indices_cpu = torch.randint(0, seq_len, (top_k,), dtype=torch.int64)

        # Phase 1: RDMA discrete reads (remote → local staging)
        rdma_us = rdma_ext.rdma_scatter_read(
            indices_cpu.data_ptr(), top_k, item_size
        )

        # Phase 2: GPU scatter read (staging → GPU)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scatter_mod.launch_scatter(
            staging_buf.data_ptr(), device_buf, contiguous_indices, item_size
        )
        end.record()
        torch.cuda.synchronize()
        gpu_us = start.elapsed_time(end) * 1000.0

        if it >= 0:
            rdma_latencies.append(rdma_us)
            gpu_latencies.append(gpu_us)
            total_latencies.append(rdma_us + gpu_us)

    rdma_ext.rdma_destroy()
    torch.cuda.cudart().cudaHostUnregister(remote_buf.data_ptr())
    torch.cuda.cudart().cudaHostUnregister(staging_buf.data_ptr())

    return rdma_latencies, gpu_latencies, total_latencies


# ---------------------------------------------------------------------------
#  Printing
# ---------------------------------------------------------------------------

def print_header(seq_len: int, top_k: int, item_size: int):
    data_kb = top_k * item_size / 1024
    unit = "KB" if data_kb < 1024 else "MB"
    data_val = data_kb if data_kb < 1024 else data_kb / 1024
    print(f"\n[Config: seq_len={seq_len}, top_k={top_k}, "
          f"data={data_val:.2f}{unit}]")
    print()
    hdr = (
        f"{'Backend':<16s}| {'Mean (μs)':>9s} | {'Std (μs)':>8s} | "
        f"{'P50 (μs)':>9s} | {'P99 (μs)':>9s} | {'Eff.BW (GB/s)':>13s}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_comparison(results: Dict[str, Dict]):
    names = list(results.keys())
    if "dram" in results and "cxl" in results:
        overhead = (results["cxl"]["mean_us"] / results["dram"]["mean_us"] - 1) * 100
        print(f"  CXL vs DRAM overhead:  +{overhead:.1f}%")
    if "cxl" in results and "rdma_total" in results:
        ratio = results["rdma_total"]["mean_us"] / results["cxl"]["mean_us"]
        overhead = (ratio - 1) * 100
        print(f"  RDMA vs CXL overhead:  +{overhead:.1f}%  ({ratio:.1f}x slower)")
    if "dram" in results and "rdma_total" in results:
        ratio = results["rdma_total"]["mean_us"] / results["dram"]["mean_us"]
        overhead = (ratio - 1) * 100
        print(f"  RDMA vs DRAM overhead: +{overhead:.1f}%  ({ratio:.1f}x slower)")
    print()


def print_sweep_table(
    sweep_results: Dict[int, Dict[str, Dict]],
    backends: List[str],
):
    print("\n[Sweep: varying top-k]\n")
    has_rdma = "rdma" in backends
    if has_rdma:
        hdr = "top-k |  DRAM (μs) |   CXL (μs) |  RDMA (μs) | CXL/DRAM | RDMA/CXL"
        sep = "------|------------|------------|------------|----------|----------"
    else:
        hdr = "top-k |  DRAM (μs) |   CXL (μs) | CXL/DRAM"
        sep = "------|------------|------------|----------"
    print(hdr)
    print(sep)

    for tk in sorted(sweep_results.keys()):
        r = sweep_results[tk]
        dram_us = r.get("dram", {}).get("mean_us", float("nan"))
        cxl_us = r.get("cxl", {}).get("mean_us", float("nan"))
        cxl_ratio = cxl_us / dram_us if dram_us > 0 else float("nan")

        if has_rdma:
            rdma_us = r.get("rdma_total", {}).get("mean_us", float("nan"))
            rdma_ratio = rdma_us / cxl_us if cxl_us > 0 else float("nan")
            print(
                f"{tk:>5d} | {dram_us:>10.2f} | {cxl_us:>10.2f} | "
                f"{rdma_us:>10.2f} | {cxl_ratio:>7.2f}x | {rdma_ratio:>8.1f}x"
            )
        else:
            print(
                f"{tk:>5d} | {dram_us:>10.2f} | {cxl_us:>10.2f} | {cxl_ratio:>7.2f}x"
            )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run_single_config(
    backends: List[str],
    scatter_mod,
    cxl_ext,
    rdma_ext,
    seq_len: int,
    top_k: int,
    item_size: int,
    cxl_dev: str,
    ib_dev: str,
    gpu_id: int,
    warmup: int,
    iters: int,
) -> Dict[str, Dict]:
    """Run benchmarks for one (seq_len, top_k) configuration."""
    results = {}

    # --- DRAM ---
    if "dram" in backends:
        host_buf = init_dram_pool(seq_len, item_size)
        lats = bench_dram(scatter_mod, host_buf, item_size, top_k, warmup, iters)
        stats = compute_stats(lats, top_k, item_size)
        results["dram"] = stats
        print(fmt_stats_row("DRAM", stats))
        cleanup_dram_pool(host_buf)
        del host_buf

    # --- CXL ---
    if "cxl" in backends:
        lats = bench_cxl(
            scatter_mod, cxl_ext, cxl_dev, seq_len, item_size, top_k,
            gpu_id, warmup, iters,
        )
        stats = compute_stats(lats, top_k, item_size)
        results["cxl"] = stats
        print(fmt_stats_row("CXL", stats))

    # --- RDMA ---
    if "rdma" in backends:
        rdma_lats, gpu_lats, total_lats = bench_rdma(
            scatter_mod, rdma_ext, ib_dev, seq_len, item_size, top_k,
            warmup, iters,
        )
        stats_total = compute_stats(total_lats, top_k, item_size)
        stats_rdma = compute_stats(rdma_lats, top_k, item_size)
        stats_gpu = compute_stats(gpu_lats, top_k, item_size)
        results["rdma_total"] = stats_total
        results["rdma_read"] = stats_rdma
        results["rdma_gpu"] = stats_gpu
        print(fmt_stats_row("RDMA (total)", stats_total))
        print(fmt_stats_row("  └─ rdma read", stats_rdma))
        print(fmt_stats_row("  └─ gpu copy", stats_gpu))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Single-layer sparse KV cache read latency benchmark"
    )
    parser.add_argument(
        "--backends", type=str, default="dram,cxl,rdma",
        help="Comma-separated list of backends to test (dram,cxl,rdma)",
    )
    parser.add_argument("--cxl-dev", type=str, default="/dev/dax0.0")
    parser.add_argument("--ib-dev", type=str, default="mlx5_0")
    parser.add_argument("--rdma-mode", type=str, default="loopback",
                        choices=["loopback"])
    parser.add_argument(
        "--seq-lens", type=str, default="32768,65536,131072",
        help="Comma-separated sequence lengths",
    )
    parser.add_argument(
        "--top-ks", type=str, default="64,128,256,512,1024,2048",
        help="Comma-separated top-k values",
    )
    parser.add_argument("--item-size", type=int, default=656,
                        help="Per-token KV entry size in bytes")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save JSON results")

    args = parser.parse_args()

    backends = [b.strip().lower() for b in args.backends.split(",")]
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    top_ks = [int(k) for k in args.top_ks.split(",")]

    torch.cuda.set_device(args.gpu_id)

    print("=" * 60)
    print("Single-Layer Sparse KV Read Latency Benchmark")
    print("=" * 60)
    print(f"Model: DeepSeek-V3.2 (MLA FP8, item_size={args.item_size}B, "
          f"store_dtype=uint8)")
    print(f"Backends: {backends}")
    print(f"Seq lens: {seq_lens}")
    print(f"Top-k:    {top_ks}")
    print(f"Warmup:   {args.warmup}, Iters: {args.iters}")
    if "rdma" in backends:
        print(f"RDMA:     {args.rdma_mode} mode, device={args.ib_dev}")

    # Build extensions
    print("\nBuilding GPU scatter kernel...")
    scatter_mod = _build_scatter_torch_module()

    cxl_ext = None
    if "cxl" in backends:
        print("Building CXL extension...")
        cxl_ext = build_cxl_extension()
        if cxl_ext is None:
            print("[WARN] CXL extension build failed; removing 'cxl' from backends.")
            backends.remove("cxl")

    rdma_ext = None
    if "rdma" in backends:
        print("Loading RDMA extension...")
        rdma_ext = load_rdma_extension()
        if rdma_ext is None:
            print("[WARN] RDMA extension not available; removing 'rdma' from backends.")
            backends.remove("rdma")

    all_results = {}

    for seq_len in seq_lens:
        print(f"\n{'='*60}")
        print(f"  seq_len = {seq_len} ({seq_len // 1024}K)")
        print(f"{'='*60}")

        sweep_results = {}

        for top_k in top_ks:
            if top_k > seq_len:
                print(f"\n  [SKIP] top_k={top_k} > seq_len={seq_len}")
                continue

            print_header(seq_len, top_k, args.item_size)
            results = run_single_config(
                backends=backends,
                scatter_mod=scatter_mod,
                cxl_ext=cxl_ext,
                rdma_ext=rdma_ext,
                seq_len=seq_len,
                top_k=top_k,
                item_size=args.item_size,
                cxl_dev=args.cxl_dev,
                ib_dev=args.ib_dev,
                gpu_id=args.gpu_id,
                warmup=args.warmup,
                iters=args.iters,
            )
            print_comparison(results)
            sweep_results[top_k] = results

        if len(sweep_results) > 1:
            print_sweep_table(sweep_results, backends)

        all_results[seq_len] = sweep_results

    # Save JSON
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"sparse_kv_bench_{ts}.json"
        serializable = {}
        for sl, sr in all_results.items():
            serializable[str(sl)] = {}
            for tk, res in sr.items():
                serializable[str(sl)][str(tk)] = res
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "backends": backends,
                        "item_size": args.item_size,
                        "seq_lens": seq_lens,
                        "top_ks": top_ks,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "rdma_mode": args.rdma_mode if "rdma" in backends else None,
                    },
                    "results": serializable,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
