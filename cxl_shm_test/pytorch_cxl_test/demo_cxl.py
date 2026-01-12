#!/usr/bin/env python3
"""Minimal PyTorch demo for copying data between tensors and the mapped CXL region."""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).parent.resolve()


def build_extension():
    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)

    return load(
        name="cxl_mem_ext",
        sources=[
            str(ROOT / "cxl_mem.cpp"),
            str(ROOT / "cxl_mem_pybind.cpp"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-mclflushopt"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=True,
        with_cuda=True
    )


def main():
    ext = build_extension()

    # Adjust the path/size to match your environment.
    ok = ext.cxl_init(dev_path="/dev/dax0.0", map_bytes=32 * 1024 * 1024, register_cuda=torch.cuda.is_available())
    if not ok:
        raise SystemExit("cxl_init failed; check dev_path permissions and size")

    # CPU -> CXL -> CPU
    cpu_src = torch.arange(16, dtype=torch.float32)
    ext.tensor_to_cxl(cpu_src, offset=0)
    cpu_dst = torch.empty_like(cpu_src)
    ext.cxl_to_tensor(cpu_dst, offset=0)
    print("CPU roundtrip match:", torch.allclose(cpu_src, cpu_dst))

    # GPU -> CXL -> GPU (optional)
    if torch.cuda.is_available():
        gpu_src = torch.arange(16, dtype=torch.float32, device="cuda")
        ext.tensor_to_cxl(gpu_src, offset=4096)
        gpu_dst = torch.empty_like(gpu_src)
        ext.cxl_to_tensor(gpu_dst, offset=4096)
        print("GPU roundtrip match:", torch.allclose(gpu_src, gpu_dst))
    else:
        print("CUDA not available; skipped GPU path")

    ext.cxl_close()


if __name__ == "__main__":
    main()
