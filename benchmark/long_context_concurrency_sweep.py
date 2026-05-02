#!/usr/bin/env python3
"""
Sweep OpenAI-style / native SGLang serving throughput over multiple max-concurrency levels.

Uses the same request path as `python -m sglang.bench_serving`:
  - Fixed-length random prompts (--dataset-name random, --random-range-ratio 0)
  - Default: 32k input tokens, 4k decode tokens
  - Default: each concurrency step runs ``num_prompts == max_concurrency`` for that step

Example (server on localhost:30000, OpenAI completions API):

  python benchmark/long_context_concurrency_sweep.py \\
    --backend sglang-oai --host 127.0.0.1 --port 30000 \\
    --input-len 32768 --output-len 4096 \\
    --concurrencies 1,2,4,8,16,32

  # Each step uses num_prompts = that step's max_concurrency (e.g. 8 requests @ max_concurrency=8).
  # To use the same count for every step: --num-prompts 32

Native /generate endpoint:

  python benchmark/long_context_concurrency_sweep.py --backend sglang ...

Local ShareGPT JSON (avoid HF download for dataset-name random):

  python benchmark/long_context_concurrency_sweep.py ... \\
    --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json

No dataset file needed (synthetic token ids; use when you want zero downloads):

  python benchmark/long_context_concurrency_sweep.py ... \\
    --dataset-name random-ids
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def _parse_concurrencies(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("concurrencies must list at least one int")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find peak throughput vs max concurrency (long random prompts)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang-oai",
        help="bench_serving backend (e.g. sglang-oai, sglang, vllm). Default: sglang-oai",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host for benchmark client (use 127.0.0.1 for local).",
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="If set, overrides host/port (e.g. http://127.0.0.1:30000).",
    )
    parser.add_argument("--model", type=str, default=None, help="Model id (optional).")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path; defaults to model discovery like bench_serving.",
    )
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        metavar="N",
        help="If set, run N requests for every concurrency step. "
        "Default: each step runs num_prompts equal to that step's max_concurrency.",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated max-concurrency values for --max-concurrency.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        help="bench_serving --dataset-name (e.g. random, random-ids, sharegpt). "
        "Use random-ids to skip ShareGPT JSON entirely.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Local file path passed to bench_serving --dataset-path (ShareGPT JSON "
        "for random/sharegpt, or your JSONL for custom/openai). "
        "When valid, HF download is skipped.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Warmup sequences before each sweep step (long contexts: 1–2 typical).",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=600,
        help="First step only: wait for /v1/models. Later steps use 0 to skip.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Passed to bench_serving (default inf = launch all at t=0).",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Write all per-step JSON records to this file (JSON array).",
    )
    args = parser.parse_args()

    concurrencies = _parse_concurrencies(args.concurrencies)

    rows = []
    for i, mc in enumerate(concurrencies):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            out_path = tmp.name

        try:
            num_prompts = args.num_prompts if args.num_prompts is not None else mc
            cmd: List[str] = [
                sys.executable,
                "-m",
                "sglang.bench_serving",
                "--backend",
                args.backend,
                "--dataset-name",
                args.dataset_name,
                "--random-input-len",
                str(args.input_len),
                "--random-output-len",
                str(args.output_len),
                "--random-range-ratio",
                "0",
                "--num-prompts",
                str(num_prompts),
                "--max-concurrency",
                str(mc),
                "--seed",
                str(args.seed),
                "--warmup-requests",
                str(args.warmup_requests),
                "--request-rate",
                str(args.request_rate),
                "--output-file",
                out_path,
            ]
            if args.base_url:
                cmd.extend(["--base-url", args.base_url])
            else:
                cmd.extend(["--host", args.host, "--port", str(args.port)])
            if args.model:
                cmd.extend(["--model", args.model])
            if args.tokenizer:
                cmd.extend(["--tokenizer", args.tokenizer])
            if args.dataset_path:
                cmd.extend(["--dataset-path", args.dataset_path])

            # Only block on server readiness for the first concurrency.
            ready_timeout = args.ready_check_timeout_sec if i == 0 else 0
            cmd.extend(["--ready-check-timeout-sec", str(ready_timeout)])

            print("\n" + "=" * 72)
            print(f"Running: max_concurrency={mc}, num_prompts={num_prompts}")
            print(" ".join(cmd))
            print("=" * 72 + "\n", flush=True)

            subprocess.run(cmd, check=True)

            with open(out_path, encoding="utf-8") as f:
                line = f.readline().strip()
            if not line:
                raise RuntimeError(f"No JSON output written for concurrency={mc}")
            rec = json.loads(line)
            rec["_sweep_max_concurrency"] = mc
            rec["_sweep_num_prompts"] = num_prompts
            rows.append(rec)

        finally:
            Path(out_path).unlink(missing_ok=True)

    # Summary table
    print("\n" + "=" * 80)
    np_desc = (
        f"num_prompts={args.num_prompts} (fixed)"
        if args.num_prompts is not None
        else "num_prompts = concurrency per step"
    )
    print(
        f"Summary  (input={args.input_len} tok, output={args.output_len} tok, {np_desc})"
    )
    print("=" * 80)
    hdr = f"{'n_req':>5}  {'max_conc':>8}  {'out_tok/s':>12}  {'req/s':>10}  {'in_tok/s':>12}  {'total_tok/s':>12}"
    print(hdr)
    print("-" * len(hdr))

    best_out = -1.0
    best_mc = None
    for rec in rows:
        mc = rec.get("_sweep_max_concurrency")
        npr = int(rec.get("_sweep_num_prompts", 0))
        ot = float(rec.get("output_throughput") or 0)
        rt = float(rec.get("request_throughput") or 0)
        it = float(rec.get("input_throughput") or 0)
        tt = float(rec.get("total_throughput") or 0)
        print(f"{npr:5d}  {mc:8d}  {ot:12.2f}  {rt:10.2f}  {it:12.2f}  {tt:12.2f}")
        if ot > best_out:
            best_out = ot
            best_mc = mc

    print("-" * len(hdr))
    if best_mc is not None:
        print(
            f"Peak output token throughput: {best_out:.2f} tok/s at max_concurrency={best_mc}"
        )
    print("=" * 80 + "\n")

    if args.results_json:
        out_p = Path(args.results_json)
        out_p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote detailed records to {out_p}")


if __name__ == "__main__":
    main()
