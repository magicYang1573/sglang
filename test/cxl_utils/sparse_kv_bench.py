#!/usr/bin/env python3
"""
Single-Layer Sparse KV Cache Read Latency Micro-Benchmark
==========================================================

Compares GPU scatter-read latency from multiple memory backends:
  - DRAM           (CPU pinned memory, cudaHostRegister)
  - CXL            (single CXL device mmap, cudaHostRegister, GPU direct read)
  - CXL-interleave (multiple CXL devices, tokens round-robin across devices)
  - RDMA           (discrete per-token RDMA READs via NIC loopback, then GPU copy)

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
#  JIT compile the GPU scatter read kernel (single-source + interleave)
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

extern "C" __global__ void scatter_read_interleave_kernel(
    const uint64_t* __restrict__ src_ptrs,
    uint8_t* __restrict__ dst,
    const int64_t* __restrict__ indices,
    const int32_t* __restrict__ device_map,
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

void launch_scatter_interleave(torch::Tensor src_ptrs, torch::Tensor dst,
                               torch::Tensor indices, torch::Tensor device_map,
                               int64_t item_size) {
    int num_tokens = indices.size(0);
    dim3 grid(num_tokens);
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_read_interleave_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(src_ptrs.data_ptr<int64_t>()),
        dst.data_ptr<uint8_t>(),
        indices.data_ptr<int64_t>(),
        device_map.data_ptr<int32_t>(),
        item_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_scatter", &launch_scatter,
          "Single-source scatter read");
    m.def("launch_scatter_interleave", &launch_scatter_interleave,
          "Multi-source interleave scatter read");
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
#  Direct mmap + cudaHostRegister for multi-device CXL interleave.
#  The cxl_mem_ext pybind module uses a global singleton (g_cxl), so we cannot
#  hold multiple devices open simultaneously. Instead we mmap each DAX device
#  directly from Python and register with CUDA ourselves.
# ---------------------------------------------------------------------------

import mmap as _mmap_mod


class CXLDeviceRegion:
    """Manages a single CXL DAX device: mmap + cudaHostRegister."""

    def __init__(self, dev_path: str, map_bytes: int, gpu_id: int):
        self.dev_path = dev_path
        self.map_bytes = map_bytes
        self.fd = os.open(dev_path, os.O_RDWR)
        self.mm = _mmap_mod.mmap(
            self.fd, map_bytes, _mmap_mod.MAP_SHARED, _mmap_mod.PROT_READ | _mmap_mod.PROT_WRITE,
        )
        self.base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

        prev_dev = torch.cuda.current_device()
        torch.cuda.set_device(gpu_id)
        # cudaHostRegisterMapped = 0x02
        ret = torch.cuda.cudart().cudaHostRegister(
            ctypes.c_void_p(self.base_ptr), map_bytes, 2
        )
        # cudaHostGetDevicePointer
        dev_ptr = ctypes.c_void_p()
        torch.cuda.cudart().cudaHostGetDevicePointer(
            ctypes.byref(dev_ptr), ctypes.c_void_p(self.base_ptr), 0
        )
        self.device_ptr = dev_ptr.value or 0
        torch.cuda.set_device(prev_dev)

        if self.device_ptr == 0:
            self.close()
            raise RuntimeError(
                f"cudaHostGetDevicePointer returned null for {dev_path}"
            )

    def write_random_data(self, nbytes: int):
        """Write random bytes into the mmap region for benchmarking."""
        chunk = 64 * 1024 * 1024
        offset = 0
        while offset < nbytes:
            n = min(chunk, nbytes - offset)
            data = np.random.randint(0, 256, n, dtype=np.uint8).tobytes()
            self.mm[offset:offset + n] = data
            offset += n

    def close(self):
        try:
            torch.cuda.cudart().cudaHostUnregister(ctypes.c_void_p(self.base_ptr))
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

HUGE_PAGE_SIZE = 2 * 1024 * 1024


def align_to_2mb(size: int) -> int:
    return ((size + HUGE_PAGE_SIZE - 1) // HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE


def compute_stats(latencies: List[float], num_tokens: int, item_size: int) -> Dict:
    arr = np.array(latencies)
    data_bytes = num_tokens * item_size
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
        f"{name:<20s}| {s['mean_us']:>9.2f} | {s['std_us']:>8.2f} | "
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
    num_tokens: int,
    warmup: int,
    iters: int,
) -> List[float]:
    seq_len = host_buf.shape[0]
    src_ptr = host_buf.data_ptr()
    device_buf = torch.empty(num_tokens, item_size, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device="cuda")
        scatter_mod.launch_scatter(src_ptr, device_buf, indices, item_size)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scatter_mod.launch_scatter(src_ptr, device_buf, indices, item_size)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000.0)

    return latencies


# ---------------------------------------------------------------------------
#  CXL backend (single device)
# ---------------------------------------------------------------------------

def bench_cxl(
    scatter_mod,
    cxl_ext,
    cxl_dev: str,
    seq_len: int,
    item_size: int,
    num_tokens: int,
    gpu_id: int,
    warmup: int,
    iters: int,
) -> List[float]:
    total_bytes = seq_len * item_size
    map_bytes = align_to_2mb(total_bytes)
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

    cxl_device_ptr = cxl_ext.get_cxl_device_ptr()
    if cxl_device_ptr == 0:
        cxl_ext.cxl_close()
        raise RuntimeError("CXL device_ptr is null; ensure register_cuda=True")

    device_buf = torch.empty(num_tokens, item_size, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device="cuda")
        scatter_mod.launch_scatter(cxl_device_ptr, device_buf, indices, item_size)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scatter_mod.launch_scatter(cxl_device_ptr, device_buf, indices, item_size)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000.0)

    cxl_ext.cxl_close()
    return latencies


# ---------------------------------------------------------------------------
#  CXL interleave backend (N devices)
# ---------------------------------------------------------------------------

def bench_cxl_interleave(
    scatter_mod,
    cxl_devs: List[str],
    seq_len: int,
    item_size: int,
    num_tokens: int,
    gpu_id: int,
    warmup: int,
    iters: int,
) -> List[float]:
    """Benchmark scatter read with tokens distributed across multiple CXL devices.

    Each CXL device holds a full seq_len pool. Tokens are round-robin assigned
    to devices, simulating the interleave pattern where different DP ranks'
    KV caches reside on different CXL devices and are accessed concurrently.
    """
    num_devs = len(cxl_devs)
    regions: List[CXLDeviceRegion] = []
    device_ptrs: List[int] = []

    total_bytes = seq_len * item_size
    map_bytes = align_to_2mb(total_bytes)

    for dev_path in cxl_devs:
        region = CXLDeviceRegion(dev_path, map_bytes, gpu_id)
        region.write_random_data(total_bytes)
        regions.append(region)
        device_ptrs.append(region.device_ptr)

    src_ptrs_gpu = torch.tensor(device_ptrs, dtype=torch.int64, device="cuda")
    device_buf = torch.empty(num_tokens, item_size, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device="cuda")
        device_map = torch.randint(0, num_devs, (num_tokens,), dtype=torch.int32, device="cuda")
        scatter_mod.launch_scatter_interleave(
            src_ptrs_gpu, device_buf, indices, device_map, item_size,
        )
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64, device="cuda")
        device_map = torch.randint(0, num_devs, (num_tokens,), dtype=torch.int32, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scatter_mod.launch_scatter_interleave(
            src_ptrs_gpu, device_buf, indices, device_map, item_size,
        )
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000.0)

    for region in regions:
        region.close()

    return latencies


# ---------------------------------------------------------------------------
#  RDMA backend
# ---------------------------------------------------------------------------

RDMA_MAX_BATCH = 2048


def bench_rdma(
    scatter_mod,
    rdma_ext,
    ib_dev: str,
    seq_len: int,
    item_size: int,
    num_tokens: int,
    warmup: int,
    iters: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Returns (rdma_latencies, gpu_latencies, total_latencies) in μs.

    When num_tokens > RDMA_MAX_BATCH, the RDMA scatter reads are split into
    batches of RDMA_MAX_BATCH to stay within QP/CQ hardware limits.  The
    staging buffer is sized for one batch; after each RDMA batch completes,
    its data is GPU-copied into the correct slice of device_buf, then the
    staging buffer is reused for the next batch.
    """

    batch_size = min(num_tokens, RDMA_MAX_BATCH)
    num_batches = (num_tokens + batch_size - 1) // batch_size
    remote_size = seq_len * item_size
    staging_size = batch_size * item_size

    remote_buf = torch.randint(0, 256, (seq_len, item_size), dtype=torch.uint8)
    remote_buf = remote_buf.contiguous()

    staging_buf = torch.zeros(batch_size, item_size, dtype=torch.uint8)
    staging_buf = staging_buf.contiguous()

    torch.cuda.cudart().cudaHostRegister(
        remote_buf.data_ptr(), remote_buf.numel(), 0
    )
    torch.cuda.cudart().cudaHostRegister(
        staging_buf.data_ptr(), staging_buf.numel(), 0
    )

    rdma_ext.rdma_init(
        ib_dev,
        staging_buf.data_ptr(),
        staging_size,
        remote_buf.data_ptr(),
        remote_size,
        batch_size,
    )

    device_buf = torch.empty(num_tokens, item_size, dtype=torch.uint8, device="cuda")
    # Contiguous indices [0..batch_size) for GPU copy from staging → device slice
    batch_indices_gpu = torch.arange(batch_size, dtype=torch.int64, device="cuda")

    if num_batches > 1:
        print(f"  (RDMA: {num_batches} batches x {batch_size} tokens)")

    rdma_latencies = []
    gpu_latencies = []
    total_latencies = []

    for it in range(-warmup, iters):
        indices_cpu = torch.randint(0, seq_len, (num_tokens,), dtype=torch.int64)

        rdma_us_total = 0.0
        gpu_us_total = 0.0
        offset = 0

        for b in range(num_batches):
            chunk = min(batch_size, num_tokens - offset)
            batch_idx = indices_cpu[offset:offset + chunk].contiguous()

            # RDMA phase: scatter read into staging_buf[0:chunk]
            rdma_us = rdma_ext.rdma_scatter_read(
                batch_idx.data_ptr(), chunk, item_size
            )
            rdma_us_total += rdma_us

            # GPU phase: copy staging_buf → device_buf[offset:offset+chunk]
            dst_slice = device_buf[offset:offset + chunk]
            idx_slice = batch_indices_gpu[:chunk]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            scatter_mod.launch_scatter(
                staging_buf.data_ptr(), dst_slice, idx_slice, item_size
            )
            end.record()
            torch.cuda.synchronize()
            gpu_us_total += start.elapsed_time(end) * 1000.0

            offset += chunk

        if it >= 0:
            rdma_latencies.append(rdma_us_total)
            gpu_latencies.append(gpu_us_total)
            total_latencies.append(rdma_us_total + gpu_us_total)

    rdma_ext.rdma_destroy()
    torch.cuda.cudart().cudaHostUnregister(remote_buf.data_ptr())
    torch.cuda.cudart().cudaHostUnregister(staging_buf.data_ptr())

    return rdma_latencies, gpu_latencies, total_latencies


# ---------------------------------------------------------------------------
#  Printing
# ---------------------------------------------------------------------------

def print_header(seq_len: int, num_tokens: int, item_size: int):
    data_kb = num_tokens * item_size / 1024
    unit = "KB" if data_kb < 1024 else "MB"
    data_val = data_kb if data_kb < 1024 else data_kb / 1024
    print(f"\n[Config: seq_len={seq_len}, num_tokens={num_tokens}, "
          f"data={data_val:.2f}{unit}]")
    print()
    hdr = (
        f"{'Backend':<20s}| {'Mean (μs)':>9s} | {'Std (μs)':>8s} | "
        f"{'P50 (μs)':>9s} | {'P99 (μs)':>9s} | {'Eff.BW (GB/s)':>13s}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_comparison(results: Dict[str, Dict]):
    if "dram" in results and "cxl" in results:
        overhead = (results["cxl"]["mean_us"] / results["dram"]["mean_us"] - 1) * 100
        print(f"  CXL vs DRAM overhead:  +{overhead:.1f}%")
    if "cxl" in results and "cxl_x2" in results:
        speedup = results["cxl"]["mean_us"] / results["cxl_x2"]["mean_us"]
        print(f"  CXL-x2 vs CXL-x1 speedup:  {speedup:.2f}x")
    if "cxl" in results and "rdma_total" in results:
        ratio = results["rdma_total"]["mean_us"] / results["cxl"]["mean_us"]
        print(f"  RDMA vs CXL overhead:  +{(ratio - 1) * 100:.1f}%  ({ratio:.1f}x slower)")
    if "dram" in results and "rdma_total" in results:
        ratio = results["rdma_total"]["mean_us"] / results["dram"]["mean_us"]
        print(f"  RDMA vs DRAM overhead: +{(ratio - 1) * 100:.1f}%  ({ratio:.1f}x slower)")
    print()


def print_sweep_table(
    sweep_results: Dict[int, Dict[str, Dict]],
    backends: List[str],
):
    has_cxl = "cxl" in backends
    has_cxl_x2 = "cxl_x2" in backends
    has_rdma = "rdma" in backends

    print("\n[Sweep: varying num_tokens]\n")

    cols = ["num_tokens", "DRAM (μs)"]
    if has_cxl:
        cols.append("CXL-x1 (μs)")
    if has_cxl_x2:
        cols.append("CXL-x2 (μs)")
    if has_rdma:
        cols.append("RDMA (μs)")
    if has_cxl:
        cols.append("CXL/DRAM")
    if has_cxl_x2 and has_cxl:
        cols.append("x2/x1")

    hdr = " | ".join(f"{c:>12s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    for nt in sorted(sweep_results.keys()):
        r = sweep_results[nt]
        dram_us = r.get("dram", {}).get("mean_us", float("nan"))
        vals = [f"{nt:>12d}", f"{dram_us:>12.2f}"]
        cxl_us = float("nan")
        cxl_x2_us = float("nan")
        if has_cxl:
            cxl_us = r.get("cxl", {}).get("mean_us", float("nan"))
            vals.append(f"{cxl_us:>12.2f}")
        if has_cxl_x2:
            cxl_x2_us = r.get("cxl_x2", {}).get("mean_us", float("nan"))
            vals.append(f"{cxl_x2_us:>12.2f}")
        if has_rdma:
            rdma_us = r.get("rdma_total", {}).get("mean_us", float("nan"))
            vals.append(f"{rdma_us:>12.2f}")
        if has_cxl:
            ratio = cxl_us / dram_us if dram_us > 0 else float("nan")
            vals.append(f"{ratio:>11.2f}x")
        if has_cxl_x2 and has_cxl:
            speedup = cxl_us / cxl_x2_us if cxl_x2_us > 0 else float("nan")
            vals.append(f"{speedup:>11.2f}x")

        print(" | ".join(vals))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run_single_config(
    backends: List[str],
    scatter_mod,
    cxl_ext,
    rdma_ext,
    seq_len: int,
    num_tokens: int,
    item_size: int,
    cxl_dev: str,
    cxl_devs_x2: List[str],
    ib_dev: str,
    gpu_id: int,
    warmup: int,
    iters: int,
) -> Dict[str, Dict]:
    results = {}

    # --- DRAM ---
    if "dram" in backends:
        host_buf = init_dram_pool(seq_len, item_size)
        lats = bench_dram(scatter_mod, host_buf, item_size, num_tokens, warmup, iters)
        stats = compute_stats(lats, num_tokens, item_size)
        results["dram"] = stats
        print(fmt_stats_row("DRAM", stats))
        cleanup_dram_pool(host_buf)
        del host_buf

    # --- CXL (single device) ---
    if "cxl" in backends:
        lats = bench_cxl(
            scatter_mod, cxl_ext, cxl_dev, seq_len, item_size, num_tokens,
            gpu_id, warmup, iters,
        )
        stats = compute_stats(lats, num_tokens, item_size)
        results["cxl"] = stats
        print(fmt_stats_row("CXL (x1)", stats))

    # --- CXL interleave (2 devices) ---
    if "cxl_x2" in backends:
        lats = bench_cxl_interleave(
            scatter_mod, cxl_devs_x2, seq_len, item_size, num_tokens,
            gpu_id, warmup, iters,
        )
        stats = compute_stats(lats, num_tokens, item_size)
        results["cxl_x2"] = stats
        print(fmt_stats_row("CXL-interleave (x2)", stats))

    # --- RDMA ---
    if "rdma" in backends:
        rdma_lats, gpu_lats, total_lats = bench_rdma(
            scatter_mod, rdma_ext, ib_dev, seq_len, item_size, num_tokens,
            warmup, iters,
        )
        stats_total = compute_stats(total_lats, num_tokens, item_size)
        stats_rdma = compute_stats(rdma_lats, num_tokens, item_size)
        stats_gpu = compute_stats(gpu_lats, num_tokens, item_size)
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
        "--backends", type=str, default="dram,cxl,cxl_x2,rdma",
        help="Comma-separated backends: dram, cxl, cxl_x2, rdma",
    )
    parser.add_argument("--cxl-dev", type=str, default="/dev/dax0.0",
                        help="CXL DAX device for single-device tests")
    parser.add_argument(
        "--cxl-devs", type=str, default="/dev/dax0.0,/dev/dax1.0",
        help="Comma-separated CXL DAX devices for interleave (cxl_x2)",
    )
    parser.add_argument("--ib-dev", type=str, default="mlx5_0")
    parser.add_argument("--rdma-mode", type=str, default="loopback",
                        choices=["loopback"])
    parser.add_argument(
        "--seq-lens", type=str, default="32768,65536,131072",
        help="Comma-separated sequence lengths",
    )
    parser.add_argument(
        "--num-tokens", type=str,
        default="64,128,256,512,1024,2048,4096,8192,16384",
        help="Comma-separated num_tokens values (discrete KV entries to read)",
    )
    parser.add_argument("--item-size", type=int, default=656,
                        help="Per-token KV entry size in bytes (MLA latent dim)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save JSON results")

    args = parser.parse_args()

    backends = [b.strip().lower() for b in args.backends.split(",")]
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    num_tokens_list = [int(k) for k in args.num_tokens.split(",")]
    cxl_devs_x2 = [d.strip() for d in args.cxl_devs.split(",")]

    torch.cuda.set_device(args.gpu_id)

    print("=" * 70)
    print("Single-Layer Sparse KV Read Latency Benchmark")
    print("=" * 70)
    print(f"Model params: DeepSeek-V3.2 (MLA FP8, item_size={args.item_size}B)")
    print(f"Backends:     {backends}")
    print(f"Seq lens:     {seq_lens}")
    print(f"Num tokens:   {num_tokens_list}")
    if "cxl" in backends:
        print(f"CXL (x1):     {args.cxl_dev}")
    if "cxl_x2" in backends:
        print(f"CXL (x2):     {cxl_devs_x2}")
    if "rdma" in backends:
        print(f"RDMA:         {args.rdma_mode} mode, device={args.ib_dev}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")

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

    if "cxl_x2" in backends:
        print(f"CXL interleave: will use direct mmap for {len(cxl_devs_x2)} devices: {cxl_devs_x2}")
        for dp in cxl_devs_x2:
            if not os.path.exists(dp):
                print(f"[WARN] CXL device {dp} not found; removing 'cxl_x2' from backends.")
                backends.remove("cxl_x2")
                break

    rdma_ext = None
    if "rdma" in backends:
        print("Loading RDMA extension...")
        rdma_ext = load_rdma_extension()
        if rdma_ext is None:
            print("[WARN] RDMA extension not available; removing 'rdma' from backends.")
            backends.remove("rdma")

    all_results = {}

    for seq_len in seq_lens:
        print(f"\n{'='*70}")
        print(f"  seq_len = {seq_len} ({seq_len // 1024}K)")
        print(f"{'='*70}")

        sweep_results = {}

        for num_tokens in num_tokens_list:
            if num_tokens > seq_len:
                print(f"\n  [SKIP] num_tokens={num_tokens} > seq_len={seq_len}")
                continue

            print_header(seq_len, num_tokens, args.item_size)
            results = run_single_config(
                backends=backends,
                scatter_mod=scatter_mod,
                cxl_ext=cxl_ext,
                rdma_ext=rdma_ext,
                seq_len=seq_len,
                num_tokens=num_tokens,
                item_size=args.item_size,
                cxl_dev=args.cxl_dev,
                cxl_devs_x2=cxl_devs_x2,
                ib_dev=args.ib_dev,
                gpu_id=args.gpu_id,
                warmup=args.warmup,
                iters=args.iters,
            )
            print_comparison(results)
            sweep_results[num_tokens] = results

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
            for nt, res in sr.items():
                serializable[str(sl)][str(nt)] = res
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "backends": backends,
                        "item_size": args.item_size,
                        "seq_lens": seq_lens,
                        "num_tokens": num_tokens_list,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "cxl_dev": args.cxl_dev,
                        "cxl_devs_x2": cxl_devs_x2 if "cxl_x2" in backends else None,
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
