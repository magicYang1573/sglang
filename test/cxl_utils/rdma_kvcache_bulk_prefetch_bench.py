#!/usr/bin/env python3
"""Standalone RDMA full-KVCache prefetch microbenchmark.

This script does not start SGLang.  It directly uses ``rdma_scatter_ext`` from
``test/cxl_utils`` and simulates the SGLang HiSparse RDMA warm-path layout:

    remote KVCache:  [num_tokens, num_layers, per_layer_kv_bytes]
    landing buffer: same byte size, used as the RDMA READ destination

For each context length, one request transfers:

    context_tokens * num_layers * per_layer_kv_bytes

For each concurrency value, the measured request count is
``concurrency * requests_per_concurrency``.  Requests are issued in batches with
at most ``concurrency`` in flight, and the reported latency is the average over
all measured requests.  Every RDMA context uses the same RNIC.

Build the extension first:

    cd test/cxl_utils && make rdma PYTHON=/path/to/python
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import sysconfig
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import torch
except ImportError:  # Allow --dry-run in lightweight Python environments.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent


def load_rdma_extension():
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    preferred = ROOT / f"rdma_scatter_ext{ext_suffix}"
    candidates = [preferred] if preferred.is_file() else []
    candidates.extend(sorted(ROOT.glob("rdma_scatter_ext*.so")))
    if not candidates:
        raise RuntimeError(
            "rdma_scatter_ext is missing. Build it first:\n"
            f"  cd {ROOT} && make rdma PYTHON={sys.executable}"
        )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import rdma_scatter_ext  # type: ignore[import-not-found]

    return rdma_scatter_ext


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def ctx_tag(tokens: int) -> str:
    if tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    return str(tokens)


def touch_buffer(buf: torch.Tensor, value: int, page_bytes: int = 4096) -> None:
    """Commit pages before MR registration/READ timing."""
    if buf.numel() == 0:
        return
    buf[::page_bytes] = value
    buf[-1] = value


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def summarize(
    request_us: List[float],
    wall_us: List[float],
    total_bytes_per_request: int,
) -> Dict[str, float]:
    avg_req_us = float(np.mean(request_us)) if request_us else 0.0
    avg_wall_us = float(np.mean(wall_us)) if wall_us else 0.0
    p50_req_us = percentile(request_us, 50)
    p99_req_us = percentile(request_us, 99)
    p50_wall_us = percentile(wall_us, 50)
    p99_wall_us = percentile(wall_us, 99)

    per_req_bw = (
        (total_bytes_per_request / 1e9) / (avg_req_us / 1e6)
        if avg_req_us > 0
        else 0.0
    )
    aggregate_bw = (
        (total_bytes_per_request * len(request_us) / 1e9) / (avg_wall_us / 1e6)
        if avg_wall_us > 0
        else 0.0
    )

    return {
        "avg_kvcache_prefetch_ms": round(avg_req_us / 1000.0, 3),
        "p50_kvcache_prefetch_ms": round(p50_req_us / 1000.0, 3),
        "p99_kvcache_prefetch_ms": round(p99_req_us / 1000.0, 3),
        "avg_batch_wall_ms": round(avg_wall_us / 1000.0, 3),
        "p50_batch_wall_ms": round(p50_wall_us / 1000.0, 3),
        "p99_batch_wall_ms": round(p99_wall_us / 1000.0, 3),
        "per_request_bw_gb_s": round(per_req_bw, 3),
        "aggregate_bw_gb_s": round(aggregate_bw, 3),
    }


def init_contexts(
    rdma_ext,
    ib_dev: str,
    remote_buf: torch.Tensor,
    landing_buf: torch.Tensor,
    concurrency: int,
    max_wr: int,
):
    contexts = []
    remote_size = remote_buf.numel() * remote_buf.element_size()
    landing_size = landing_buf.numel() * landing_buf.element_size()
    for _ in range(concurrency):
        ctx = rdma_ext.RDMAContext()
        ctx.init(
            ib_dev,
            landing_buf.data_ptr(),
            landing_size,
            remote_buf.data_ptr(),
            remote_size,
            max_wr,
        )
        contexts.append(ctx)
    return contexts


def destroy_contexts(contexts) -> None:
    for ctx in contexts:
        try:
            ctx.destroy()
        except Exception:
            pass


def run_concurrent_bulk_read(
    contexts,
    total_bytes: int,
    concurrency: int,
    warmup_batches: int,
    measured_batches: int,
) -> Dict[str, object]:
    def run_phase(num_requests: int, phase_name: str) -> Dict[str, object]:
        next_request = 0
        next_lock = threading.Lock()
        start_event = threading.Event()
        request_us = [0.0] * num_requests
        errors: List[str] = []

        def worker(worker_idx: int) -> None:
            nonlocal next_request
            ctx = contexts[worker_idx]
            start_event.wait()
            while True:
                with next_lock:
                    if next_request >= num_requests:
                        return
                    request_idx = next_request
                    next_request += 1
                try:
                    request_us[request_idx] = float(ctx.bulk_read(total_bytes))
                except Exception as exc:
                    errors.append(
                        f"{phase_name}: worker={worker_idx} request={request_idx}: {exc}"
                    )
                    return

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(concurrency)
        ]
        for thread in threads:
            thread.start()

        t0 = time.perf_counter()
        start_event.set()
        for thread in threads:
            thread.join()
        wall_us = (time.perf_counter() - t0) * 1e6

        if errors:
            raise RuntimeError("; ".join(errors[:3]))
        return {"request_us": request_us, "wall_us": wall_us}

    warmup_requests = concurrency * warmup_batches
    if warmup_requests > 0:
        run_phase(warmup_requests, "warmup")

    measured_requests = concurrency * measured_batches
    measured = run_phase(measured_requests, "measured")
    request_us = measured["request_us"]
    wall_us = [measured["wall_us"]]
    return {
        "request_us": request_us,
        "batch_wall_us": wall_us,
        "summary": summarize(request_us, wall_us, total_bytes),
    }


def run_case(
    rdma_ext,
    context_tokens: int,
    concurrency: int,
    ib_dev: str,
    num_layers: int,
    per_layer_kv_bytes: int,
    warmup_batches: int,
    measured_batches: int,
    max_wr: int,
    touch: bool,
    dry_run: bool,
) -> Dict[str, object]:
    bytes_per_token = num_layers * per_layer_kv_bytes
    total_bytes = context_tokens * bytes_per_token
    total_gb = total_bytes / 1e9

    print(
        f"\nctx={ctx_tag(context_tokens):>4s} concurrency={concurrency:>3d} "
        f"requests={concurrency * measured_batches:>4d} "
        f"transfer/request={total_gb:.3f} GB "
        f"({context_tokens} tokens x {num_layers} layers x {per_layer_kv_bytes} B)"
    )

    base = {
        "context_tokens": context_tokens,
        "context_tag": ctx_tag(context_tokens),
        "concurrency": concurrency,
        "num_layers": num_layers,
        "per_layer_kv_bytes": per_layer_kv_bytes,
        "bytes_per_token": bytes_per_token,
        "bytes_per_request": total_bytes,
        "gb_per_request": round(total_gb, 6),
        "requests": concurrency * measured_batches,
        "warmup_requests": concurrency * warmup_batches,
        "ib_dev": ib_dev,
    }
    if dry_run:
        return {**base, "summary": {}}
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for real RDMA runs because the benchmark uses "
            "CPU tensors as MR-backed KVCache buffers. Re-run with the same "
            "Python environment used for SGLang."
        )

    remote_buf = torch.empty(total_bytes, dtype=torch.uint8)
    landing_buf = torch.empty(total_bytes, dtype=torch.uint8)
    if touch:
        touch_buffer(remote_buf, 7)
        touch_buffer(landing_buf, 0)

    contexts = init_contexts(
        rdma_ext=rdma_ext,
        ib_dev=ib_dev,
        remote_buf=remote_buf,
        landing_buf=landing_buf,
        concurrency=concurrency,
        max_wr=max_wr,
    )
    try:
        measured = run_concurrent_bulk_read(
            contexts=contexts,
            total_bytes=total_bytes,
            concurrency=concurrency,
            warmup_batches=warmup_batches,
            measured_batches=measured_batches,
        )
    finally:
        destroy_contexts(contexts)
        del remote_buf
        del landing_buf

    summary = measured["summary"]
    print(
        "  avg_prefetch={avg_kvcache_prefetch_ms:.3f} ms "
        "p50={p50_kvcache_prefetch_ms:.3f} ms "
        "p99={p99_kvcache_prefetch_ms:.3f} ms "
        "agg_bw={aggregate_bw_gb_s:.3f} GB/s".format(**summary)
    )

    return {
        **base,
        "summary": summary,
        "request_us": measured["request_us"],
        "batch_wall_us": measured["batch_wall_us"],
    }


def write_csv(path: Path, cases: List[Dict[str, object]]) -> None:
    fields = [
        "context_tokens",
        "context_tag",
        "concurrency",
        "requests",
        "gb_per_request",
        "avg_kvcache_prefetch_ms",
        "p50_kvcache_prefetch_ms",
        "p99_kvcache_prefetch_ms",
        "avg_batch_wall_ms",
        "aggregate_bw_gb_s",
        "per_request_bw_gb_s",
        "ib_dev",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            summary = case.get("summary", {})
            row = {
                "context_tokens": case["context_tokens"],
                "context_tag": case["context_tag"],
                "concurrency": case["concurrency"],
                "requests": case["requests"],
                "gb_per_request": case["gb_per_request"],
                "ib_dev": case["ib_dev"],
            }
            if isinstance(summary, dict):
                row.update(summary)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone RDMA full-KVCache prefetch concurrency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--contexts",
        default="16384,32768,65536,131072",
        help="Comma-separated context lengths in tokens.",
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8,16,32",
        help="Comma-separated in-flight request counts.",
    )
    parser.add_argument(
        "--ib-dev",
        default="mlx5_0",
        help="Single RNIC device used by every RDMA context.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=61,
        help="Number of KV layers included in the full prefetch.",
    )
    parser.add_argument(
        "--per-layer-kv-bytes",
        type=int,
        default=656,
        help="Per-token per-layer KV bytes. DeepSeek MLA FP8 packed default is 656 B.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Warmup batches; each batch has up to --concurrency requests.",
    )
    parser.add_argument(
        "--requests-per-concurrency",
        type=int,
        default=8,
        help="Measured requests = concurrency * this value.",
    )
    parser.add_argument("--max-wr", type=int, default=4096)
    parser.add_argument(
        "--no-touch",
        action="store_true",
        help="Do not pre-touch allocated buffers before MR registration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matrix and transfer sizes without allocating RDMA buffers.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results_rdma_kvcache_bulk_prefetch"),
    )
    args = parser.parse_args()

    contexts = parse_int_list(args.contexts)
    concurrencies = parse_int_list(args.concurrency)
    if not contexts:
        parser.error("--contexts cannot be empty")
    if not concurrencies:
        parser.error("--concurrency cannot be empty")
    if args.requests_per_concurrency < 1:
        parser.error("--requests-per-concurrency must be >= 1")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be >= 0")

    max_context = max(contexts)
    bytes_per_token = args.num_layers * args.per_layer_kv_bytes
    max_transfer = max_context * bytes_per_token
    print("=" * 78)
    print("Standalone RDMA Full-KVCache Prefetch Benchmark")
    print("=" * 78)
    print(f"contexts:       {contexts}")
    print(f"concurrency:    {concurrencies}")
    print(f"ib_dev:         {args.ib_dev} (single NIC)")
    print(f"KV shape:       [tokens, {args.num_layers} layers, {args.per_layer_kv_bytes} B/layer]")
    print(f"max transfer:   {max_transfer / 1e9:.3f} GB/request")
    print(f"buffer memory:  {2 * max_transfer / 1e9:.3f} GB per case (shared across concurrency)")
    print(
        f"warmup:        {args.warmup_batches} batch(es); "
        f"measured requests = concurrency * {args.requests_per_concurrency}"
    )

    rdma_ext = None if args.dry_run else load_rdma_extension()
    all_cases: List[Dict[str, object]] = []

    for context_tokens in contexts:
        for concurrency in concurrencies:
            case = run_case(
                rdma_ext=rdma_ext,
                context_tokens=context_tokens,
                concurrency=concurrency,
                ib_dev=args.ib_dev,
                num_layers=args.num_layers,
                per_layer_kv_bytes=args.per_layer_kv_bytes,
                warmup_batches=args.warmup_batches,
                measured_batches=args.requests_per_concurrency,
                max_wr=args.max_wr,
                touch=not args.no_touch,
                dry_run=args.dry_run,
            )
            all_cases.append(case)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"rdma_kvcache_bulk_prefetch_{ts}.json"
    csv_path = out_dir / f"rdma_kvcache_bulk_prefetch_{ts}.csv"

    payload = {
        "config": {
            "contexts": contexts,
            "concurrency": concurrencies,
            "ib_dev": args.ib_dev,
            "num_layers": args.num_layers,
            "per_layer_kv_bytes": args.per_layer_kv_bytes,
            "bytes_per_token": bytes_per_token,
            "warmup_batches": args.warmup_batches,
            "requests_per_concurrency": args.requests_per_concurrency,
            "max_wr": args.max_wr,
            "touch": not args.no_touch,
            "dry_run": args.dry_run,
        },
        "cases": all_cases,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    write_csv(csv_path, all_cases)

    print("\nSaved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
