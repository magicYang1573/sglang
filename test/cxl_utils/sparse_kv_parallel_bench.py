#!/usr/bin/env python3
"""
Multi-GPU Parallel Sparse KV Cache Read Latency Micro-Benchmark
================================================================

Measures **concurrent** GPU scatter-read latency to evaluate CXL interleave
benefit under the DPA (Data-Parallel Attention) model, where multiple ranks
(GPUs) read KV cache from host memory simultaneously.

Iterates over GPU counts (e.g. 2, 4, 8) to show how contention scales:

Backends:
  - DRAM:   N GPUs concurrently read from pinned DRAM (baseline)
  - CXL-x1: N GPUs concurrently read from the SAME CXL device
  - CXL-x2: N GPUs interleaved across 2 CXL devices (gpu_rank % 2)

Each GPU reads num_tokens discrete entries of item_size bytes.
The reported latency is the wall-clock max across all GPUs per iteration.

Uses DeepSeek-V3.2 FP8 packed MLA parameters:
  item_size = 656 bytes, store_dtype = uint8
"""

import argparse
import ctypes
import ctypes.util
import json
import mmap as _mmap_mod
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
#  JIT compile GPU scatter read kernel
# ---------------------------------------------------------------------------

_scatter_torch_module = None


def _build_scatter_torch_module():
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
    int num_tokens = indices.size(0);
    dim3 grid(num_tokens);
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_read_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_ptr),
        dst.data_ptr<uint8_t>(),
        indices.data_ptr<int64_t>(),
        item_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_scatter", &launch_scatter, "Single-source scatter read");
}
""")

    _scatter_torch_module = load(
        name="scatter_torch_ext",
        sources=[str(ROOT / "sparse_kv_kernel.cu"), str(wrapper_src)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=False,
        with_cuda=True,
    )
    return _scatter_torch_module


# ---------------------------------------------------------------------------
#  CUDA runtime via ctypes (for CXL mmap registration)
# ---------------------------------------------------------------------------

_cudart_lib = None


def _get_cudart():
    global _cudart_lib
    if _cudart_lib is not None:
        return _cudart_lib

    path = ctypes.util.find_library("cudart")
    if path is None:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        for candidate in [
            os.path.join(cuda_home, "lib64", "libcudart.so"),
            os.path.join(cuda_home, "lib", "libcudart.so"),
        ]:
            if os.path.isfile(candidate):
                path = candidate
                break
    if path is None:
        raise RuntimeError("Cannot find libcudart.so")
    lib = ctypes.CDLL(path)

    lib.cudaHostRegister.restype = ctypes.c_int
    lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.cudaHostUnregister.restype = ctypes.c_int
    lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    lib.cudaHostGetDevicePointer.restype = ctypes.c_int
    lib.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
    ]

    _cudart_lib = lib
    return lib


# ---------------------------------------------------------------------------
#  CXL device region (direct mmap + cudaHostRegister)
# ---------------------------------------------------------------------------

HUGE_PAGE_SIZE = 2 * 1024 * 1024


def align_to_2mb(size: int) -> int:
    return ((size + HUGE_PAGE_SIZE - 1) // HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE


class CXLDeviceRegion:
    """Manages a single CXL DAX device: mmap + cudaHostRegister."""

    def __init__(self, dev_path: str, map_bytes: int, gpu_id: int):
        self.dev_path = dev_path
        self.map_bytes = map_bytes
        self.fd = os.open(dev_path, os.O_RDWR)
        self.mm = _mmap_mod.mmap(
            self.fd, map_bytes, _mmap_mod.MAP_SHARED,
            _mmap_mod.PROT_READ | _mmap_mod.PROT_WRITE,
        )
        self.base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

        cudart = _get_cudart()
        prev_dev = torch.cuda.current_device()
        torch.cuda.set_device(gpu_id)

        err = cudart.cudaHostRegister(self.base_ptr, map_bytes, 0x02)
        if err != 0:
            self.close()
            raise RuntimeError(f"cudaHostRegister failed for {dev_path} (err={err})")

        dev_ptr = ctypes.c_void_p()
        err = cudart.cudaHostGetDevicePointer(
            ctypes.byref(dev_ptr), self.base_ptr, 0
        )
        self.device_ptr = dev_ptr.value or 0
        torch.cuda.set_device(prev_dev)

        if err != 0 or self.device_ptr == 0:
            self.close()
            raise RuntimeError(
                f"cudaHostGetDevicePointer failed for {dev_path} (err={err})"
            )

    def write_random_data(self, nbytes: int):
        chunk = 64 * 1024 * 1024
        offset = 0
        while offset < nbytes:
            n = min(chunk, nbytes - offset)
            data = np.random.randint(0, 256, n, dtype=np.uint8).tobytes()
            self.mm[offset:offset + n] = data
            offset += n

    def close(self):
        try:
            _get_cudart().cudaHostUnregister(self.base_ptr)
        except Exception:
            pass
        try:
            self.mm.close()
        except Exception:
            pass
        try:
            os.close(self.fd)
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------

def compute_stats(latencies: List[float], num_tokens: int, item_size: int,
                  num_gpus: int = 1) -> Dict:
    arr = np.array(latencies)
    total_data = num_tokens * item_size * num_gpus
    mean_us = float(np.mean(arr))
    bw_gbs = (total_data / 1e9) / (mean_us / 1e6) if mean_us > 0 else 0.0
    return {
        "mean_us": mean_us,
        "std_us": float(np.std(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "p99_us": float(np.percentile(arr, 99)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "agg_bw_gbs": bw_gbs,
    }


FMT_W = 24


def fmt_row(name: str, s: Dict) -> str:
    return (
        f"{name:<{FMT_W}s}| {s['mean_us']:>9.2f} | {s['std_us']:>8.2f} | "
        f"{s['p50_us']:>9.2f} | {s['p99_us']:>9.2f} | {s['agg_bw_gbs']:>13.2f}"
    )


def table_hdr() -> str:
    return (
        f"{'Backend':<{FMT_W}s}| {'Mean (μs)':>9s} | {'Std (μs)':>8s} | "
        f"{'P50 (μs)':>9s} | {'P99 (μs)':>9s} | {'Agg.BW (GB/s)':>13s}"
    )


# ---------------------------------------------------------------------------
#  Per-GPU worker
# ---------------------------------------------------------------------------

def _gpu_worker(
    scatter_mod,
    gpu_id: int,
    src_ptr: int,
    seq_len: int,
    item_size: int,
    num_tokens: int,
    warmup: int,
    iters: int,
    barrier: threading.Barrier,
    result_holder: list,
    worker_idx: int,
):
    torch.cuda.set_device(gpu_id)
    dev = f"cuda:{gpu_id}"
    device_buf = torch.empty(num_tokens, item_size, dtype=torch.uint8, device=dev)

    for _ in range(warmup):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device=dev)
        scatter_mod.launch_scatter(src_ptr, device_buf, indices, item_size)
        torch.cuda.synchronize(gpu_id)

    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device=dev)
        barrier.wait()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scatter_mod.launch_scatter(src_ptr, device_buf, indices, item_size)
        end.record()
        torch.cuda.synchronize(gpu_id)
        latencies.append(start.elapsed_time(end) * 1000.0)

    result_holder[worker_idx] = latencies


def bench_parallel(
    scatter_mod,
    gpu_ids: List[int],
    src_ptrs: List[int],
    seq_len: int,
    item_size: int,
    num_tokens: int,
    warmup: int,
    iters: int,
) -> List[float]:
    """Returns wall-clock latencies: max across GPUs per iteration."""
    num_gpus = len(gpu_ids)
    assert len(src_ptrs) == num_gpus

    barrier = threading.Barrier(num_gpus)
    results = [None] * num_gpus
    threads = []

    for i in range(num_gpus):
        t = threading.Thread(
            target=_gpu_worker,
            args=(
                scatter_mod, gpu_ids[i], src_ptrs[i],
                seq_len, item_size, num_tokens,
                warmup, iters, barrier, results, i,
            ),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    wall_latencies = []
    for it in range(iters):
        max_lat = max(results[g][it] for g in range(num_gpus))
        wall_latencies.append(max_lat)

    return wall_latencies


# ---------------------------------------------------------------------------
#  DRAM pool helpers
# ---------------------------------------------------------------------------

def init_dram_pool(seq_len: int, item_size: int) -> torch.Tensor:
    buf = torch.randint(0, 256, (seq_len, item_size), dtype=torch.uint8).contiguous()
    torch.cuda.cudart().cudaHostRegister(
        buf.data_ptr(), buf.numel() * buf.element_size(), 0
    )
    return buf


def cleanup_dram_pool(buf: torch.Tensor):
    torch.cuda.cudart().cudaHostUnregister(buf.data_ptr())


# ---------------------------------------------------------------------------
#  Run one (num_gpus, num_tokens) configuration
# ---------------------------------------------------------------------------

def run_config(
    scatter_mod,
    backends: List[str],
    gpu_ids: List[int],
    seq_len: int,
    num_tokens: int,
    item_size: int,
    cxl_dev: str,
    cxl_devs: List[str],
    warmup: int,
    iters: int,
) -> Dict[str, Dict]:
    results = {}
    num_gpus = len(gpu_ids)

    # --- DRAM: all GPUs read from the same pinned DRAM ---
    if "dram" in backends:
        host_buf = init_dram_pool(seq_len, item_size)
        src_ptr = host_buf.data_ptr()
        lats = bench_parallel(
            scatter_mod, gpu_ids, [src_ptr] * num_gpus,
            seq_len, item_size, num_tokens, warmup, iters,
        )
        stats = compute_stats(lats, num_tokens, item_size, num_gpus)
        results["dram"] = stats
        print(fmt_row(f"DRAM ({num_gpus}GPU)", stats))
        cleanup_dram_pool(host_buf)
        del host_buf

    # --- CXL-x1: all GPUs read from the SAME CXL device ---
    if "cxl_x1" in backends:
        total_bytes = seq_len * item_size
        map_bytes = align_to_2mb(total_bytes)
        region = CXLDeviceRegion(cxl_dev, map_bytes, gpu_ids[0])
        region.write_random_data(total_bytes)

        lats = bench_parallel(
            scatter_mod, gpu_ids, [region.device_ptr] * num_gpus,
            seq_len, item_size, num_tokens, warmup, iters,
        )
        stats = compute_stats(lats, num_tokens, item_size, num_gpus)
        results["cxl_x1"] = stats
        print(fmt_row(f"CXL-x1 ({num_gpus}GPU)", stats))
        region.close()

    # --- CXL-x2: GPUs interleaved across 2 CXL devices (rank % 2) ---
    if "cxl_x2" in backends:
        total_bytes = seq_len * item_size
        map_bytes = align_to_2mb(total_bytes)
        num_cxl = len(cxl_devs)
        regions = []
        for dev_path in cxl_devs:
            r = CXLDeviceRegion(dev_path, map_bytes, gpu_ids[0])
            r.write_random_data(total_bytes)
            regions.append(r)

        # gpu_rank % num_cxl → which CXL device
        ptrs = [regions[i % num_cxl].device_ptr for i in range(num_gpus)]

        lats = bench_parallel(
            scatter_mod, gpu_ids, ptrs,
            seq_len, item_size, num_tokens, warmup, iters,
        )
        stats = compute_stats(lats, num_tokens, item_size, num_gpus)
        results["cxl_x2"] = stats
        # Show distribution
        dist = {}
        for i in range(num_gpus):
            d = i % num_cxl
            dist.setdefault(d, []).append(gpu_ids[i])
        dist_str = ", ".join(f"dev{k}→GPU{v}" for k, v in sorted(dist.items()))
        print(fmt_row(f"CXL-x2 ({num_gpus}GPU)", stats))
        print(f"    interleave: {dist_str}")
        for r in regions:
            r.close()

    return results


def print_comparison(results: Dict[str, Dict]):
    if "dram" in results and "cxl_x1" in results:
        ratio = results["cxl_x1"]["mean_us"] / results["dram"]["mean_us"]
        print(f"  CXL-x1 vs DRAM: {ratio:.2f}x")
    if "cxl_x1" in results and "cxl_x2" in results:
        speedup = results["cxl_x1"]["mean_us"] / results["cxl_x2"]["mean_us"]
        print(f"  CXL-x2 vs CXL-x1 speedup: {speedup:.2f}x")
    if "dram" in results and "cxl_x2" in results:
        ratio = results["cxl_x2"]["mean_us"] / results["dram"]["mean_us"]
        print(f"  CXL-x2 vs DRAM: {ratio:.2f}x")
    print()


# ---------------------------------------------------------------------------
#  Sweep table: one per (seq_len, num_gpus), varying num_tokens
# ---------------------------------------------------------------------------

def print_sweep_table(
    sweep: Dict[int, Dict[str, Dict]],
    backends: List[str],
    num_gpus: int,
):
    print(f"\n[Sweep: {num_gpus} GPUs, varying num_tokens]\n")
    has_x1 = "cxl_x1" in backends
    has_x2 = "cxl_x2" in backends

    cols = ["num_tokens", "DRAM (μs)"]
    if has_x1:
        cols.append("CXL-x1 (μs)")
    if has_x2:
        cols.append("CXL-x2 (μs)")
    if has_x1:
        cols.append("x1/DRAM")
    if has_x1 and has_x2:
        cols.append("x1/x2")
    if has_x2:
        cols.append("x2/DRAM")

    hdr = " | ".join(f"{c:>12s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    for nt in sorted(sweep.keys()):
        r = sweep[nt]
        dram_us = r.get("dram", {}).get("mean_us", float("nan"))
        vals = [f"{nt:>12d}", f"{dram_us:>12.2f}"]

        x1_us = float("nan")
        x2_us = float("nan")
        if has_x1:
            x1_us = r.get("cxl_x1", {}).get("mean_us", float("nan"))
            vals.append(f"{x1_us:>12.2f}")
        if has_x2:
            x2_us = r.get("cxl_x2", {}).get("mean_us", float("nan"))
            vals.append(f"{x2_us:>12.2f}")
        if has_x1:
            ratio = x1_us / dram_us if dram_us > 0 else float("nan")
            vals.append(f"{ratio:>11.2f}x")
        if has_x1 and has_x2:
            speedup = x1_us / x2_us if x2_us > 0 else float("nan")
            vals.append(f"{speedup:>11.2f}x")
        if has_x2:
            ratio = x2_us / dram_us if dram_us > 0 else float("nan")
            vals.append(f"{ratio:>11.2f}x")

        print(" | ".join(vals))


# ---------------------------------------------------------------------------
#  GPU-count scaling summary table
# ---------------------------------------------------------------------------

def print_gpu_scaling_summary(
    all_gpu_results: Dict[int, Dict[int, Dict[str, Dict]]],
    backends: List[str],
    num_tokens_list: List[int],
):
    """Print a compact summary table: for each num_tokens, show latency across GPU counts."""
    has_x1 = "cxl_x1" in backends
    has_x2 = "cxl_x2" in backends
    gpu_counts = sorted(all_gpu_results.keys())

    print(f"\n{'='*78}")
    print("  GPU Scaling Summary (mean latency μs)")
    print(f"{'='*78}")

    for nt in num_tokens_list:
        print(f"\n  num_tokens = {nt}")
        cols = ["#GPUs", "DRAM"]
        if has_x1:
            cols.append("CXL-x1")
        if has_x2:
            cols.append("CXL-x2")
        if has_x1:
            cols.append("x1/DRAM")
        if has_x1 and has_x2:
            cols.append("x1/x2")

        hdr = " | ".join(f"{c:>10s}" for c in cols)
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")

        for ng in gpu_counts:
            sweep = all_gpu_results[ng]
            if nt not in sweep:
                continue
            r = sweep[nt]
            dram_us = r.get("dram", {}).get("mean_us", float("nan"))
            vals = [f"{ng:>10d}", f"{dram_us:>10.2f}"]
            x1_us = float("nan")
            x2_us = float("nan")
            if has_x1:
                x1_us = r.get("cxl_x1", {}).get("mean_us", float("nan"))
                vals.append(f"{x1_us:>10.2f}")
            if has_x2:
                x2_us = r.get("cxl_x2", {}).get("mean_us", float("nan"))
                vals.append(f"{x2_us:>10.2f}")
            if has_x1:
                ratio = x1_us / dram_us if dram_us > 0 else float("nan")
                vals.append(f"{ratio:>9.2f}x")
            if has_x1 and has_x2:
                speedup = x1_us / x2_us if x2_us > 0 else float("nan")
                vals.append(f"{speedup:>9.2f}x")
            print(f"  {' | '.join(vals)}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel sparse KV cache read latency benchmark"
    )
    parser.add_argument(
        "--backends", type=str, default="dram,cxl_x1,cxl_x2",
        help="Comma-separated backends: dram, cxl_x1, cxl_x2",
    )
    parser.add_argument(
        "--gpu-ids", type=str, default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU IDs (full pool of available GPUs)",
    )
    parser.add_argument(
        "--gpu-counts", type=str, default="2,4,8",
        help="Comma-separated GPU counts to test (subset of --gpu-ids)",
    )
    parser.add_argument(
        "--cxl-dev", type=str, default="/dev/dax0.0",
        help="CXL DAX device for single-device (cxl_x1) test",
    )
    parser.add_argument(
        "--cxl-devs", type=str, default="/dev/dax0.0,/dev/dax1.0",
        help="CXL DAX devices for interleave (cxl_x2); GPUs assigned via rank %% num_devs",
    )
    parser.add_argument(
        "--seq-lens", type=str, default="32768,65536,131072",
    )
    parser.add_argument(
        "--num-tokens", type=str,
        default="64,128,256,512,1024,2048,4096,8192,16384",
    )
    parser.add_argument("--item-size", type=int, default=656)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    backends = [b.strip().lower() for b in args.backends.split(",")]
    all_gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    gpu_counts = [int(c) for c in args.gpu_counts.split(",")]
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    num_tokens_list = [int(k) for k in args.num_tokens.split(",")]
    cxl_devs = [d.strip() for d in args.cxl_devs.split(",")]

    # Validate GPU counts
    for gc in gpu_counts:
        if gc > len(all_gpu_ids):
            print(f"[ERROR] --gpu-counts {gc} > available GPUs {len(all_gpu_ids)}")
            return

    print("=" * 78)
    print("Multi-GPU Parallel Sparse KV Read Latency Benchmark")
    print("=" * 78)
    print(f"Model:       DeepSeek-V3.2 (MLA FP8, item_size={args.item_size}B)")
    print(f"GPU pool:    {all_gpu_ids}")
    print(f"GPU counts:  {gpu_counts}")
    print(f"Backends:    {backends}")
    print(f"Seq lens:    {seq_lens}")
    print(f"Num tokens:  {num_tokens_list}")
    if "cxl_x1" in backends:
        print(f"CXL (x1):    {args.cxl_dev}")
    if "cxl_x2" in backends:
        print(f"CXL (x2):    {cxl_devs} (GPUs assigned via rank %% {len(cxl_devs)})")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")

    # Validate CXL devices
    if "cxl_x1" in backends and not os.path.exists(args.cxl_dev):
        print(f"[WARN] {args.cxl_dev} not found; removing cxl_x1.")
        backends.remove("cxl_x1")
    if "cxl_x2" in backends:
        for dp in cxl_devs:
            if not os.path.exists(dp):
                print(f"[WARN] {dp} not found; removing cxl_x2.")
                backends.remove("cxl_x2")
                break

    print("\nBuilding GPU scatter kernel...")
    scatter_mod = _build_scatter_torch_module()

    all_results = {}  # seq_len → gpu_count → num_tokens → backend → stats

    for seq_len in seq_lens:
        all_results[seq_len] = {}

        for num_gpus in gpu_counts:
            gpu_ids = all_gpu_ids[:num_gpus]

            print(f"\n{'='*78}")
            print(f"  seq_len = {seq_len} ({seq_len // 1024}K), "
                  f"GPUs = {gpu_ids} ({num_gpus} concurrent)")
            print(f"{'='*78}")

            sweep = {}

            for num_tokens in num_tokens_list:
                if num_tokens > seq_len:
                    print(f"\n  [SKIP] num_tokens={num_tokens} > seq_len={seq_len}")
                    continue

                data_kb = num_tokens * args.item_size / 1024
                unit = "KB" if data_kb < 1024 else "MB"
                data_val = data_kb if data_kb < 1024 else data_kb / 1024
                print(f"\n[num_tokens={num_tokens}, data/GPU={data_val:.2f}{unit}, "
                      f"{num_gpus} GPUs]")
                print()
                print(table_hdr())
                print("-" * len(table_hdr()))

                results = run_config(
                    scatter_mod=scatter_mod,
                    backends=backends,
                    gpu_ids=gpu_ids,
                    seq_len=seq_len,
                    num_tokens=num_tokens,
                    item_size=args.item_size,
                    cxl_dev=args.cxl_dev,
                    cxl_devs=cxl_devs,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                print_comparison(results)
                sweep[num_tokens] = results

            if len(sweep) > 1:
                print_sweep_table(sweep, backends, num_gpus)

            all_results[seq_len][num_gpus] = sweep

        # GPU scaling summary for this seq_len
        print_gpu_scaling_summary(all_results[seq_len], backends, num_tokens_list)

    # Save JSON
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"sparse_kv_parallel_bench_{ts}.json"
        serializable = {}
        for sl, gpu_data in all_results.items():
            serializable[str(sl)] = {}
            for ng, sweep in gpu_data.items():
                serializable[str(sl)][str(ng)] = {}
                for nt, res in sweep.items():
                    serializable[str(sl)][str(ng)][str(nt)] = res
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "backends": backends,
                        "gpu_ids": all_gpu_ids,
                        "gpu_counts": gpu_counts,
                        "item_size": args.item_size,
                        "seq_lens": seq_lens,
                        "num_tokens": num_tokens_list,
                        "cxl_devs": cxl_devs,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    },
                    "results": serializable,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
