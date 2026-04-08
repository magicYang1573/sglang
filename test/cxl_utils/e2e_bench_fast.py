#!/usr/bin/env python3
"""Two-round e2e benchmark with repeated prompts to shorten Round 1 prefill.

Builds ``--num-unique-prompts`` distinct prompts, then expands to
``--num-requests`` total HTTP calls per round by repetition. Later repeats hit
radix cache during Round 1, so full prefill runs only ~once per unique prompt.

Server-side behaviour (e.g. HiSparse RDMA locality per request) is unchanged:
each HTTP request is still one independent Req.

Repeat modes:
  block       — [A]*k, [B]*k, ... (default; e.g. 8 unique × 64 = 512)
  interleaved — A,B,C,...,A,B,C,... cycling until num-requests

Usage:
  python e2e_bench_fast.py \\
    --num-requests 512 --num-unique-prompts 8 --repeat-mode block \\
    --prompt-mode synthetic --target-input-tokens 8192 --output-tokens 1024 \\
    --output results_fast.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Same directory as e2e_bench.py → import shared implementation
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import e2e_bench as eb  # noqa: E402

SGLANG_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SGLANG_ROOT / "python"))


def expand_repeat_prompts(
    unique: List[Tuple[str, int, int]],
    total: int,
    mode: str,
) -> List[Tuple[str, int, int]]:
    """Expand ``unique`` prompts to length ``total`` by repetition.

    Each tuple is (text, prompt_len, output_len); repeated entries share the
    same object references (same strings), matching identical HTTP bodies.
    """
    u = len(unique)
    if u == 0:
        raise ValueError("No unique prompts to expand.")
    if total < u:
        raise ValueError(
            f"--num-requests ({total}) must be >= --num-unique-prompts ({u})."
        )
    if mode == "interleaved":
        return [unique[i % u] for i in range(total)]

    if mode != "block":
        raise ValueError(f"Unknown repeat mode: {mode}")

    base = total // u
    rem = total % u
    out: List[Tuple[str, int, int]] = []
    for i in range(u):
        cnt = base + (1 if i < rem else 0)
        out.extend([unique[i]] * cnt)
    return out


async def main_async(args):
    from transformers import AutoTokenizer

    tok_id = args.tokenizer or args.model
    tok_kw = {"trust_remote_code": True}
    if args.local_files_only:
        tok_kw["local_files_only"] = True

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tok_id, **tok_kw)

    fixed_output = args.output_tokens if args.output_tokens > 0 else None

    if args.prompt_mode == "synthetic":
        if args.output_tokens <= 0:
            raise ValueError("--output-tokens must be > 0 for --prompt-mode synthetic")
        print(
            f"Building {args.num_unique_prompts} unique synthetic prompts "
            f"(~{args.target_input_tokens} tokens in, {args.output_tokens} out)..."
        )
        unique_prompts = eb.load_synthetic_long_prompts(
            tokenizer=tokenizer,
            num_prompts=args.num_unique_prompts,
            target_input_tokens=args.target_input_tokens,
            output_tokens=args.output_tokens,
            seed=args.seed,
        )
    else:
        print(f"Loading {args.num_unique_prompts} unique prompts from ShareGPT...")
        unique_prompts = eb.load_sharegpt_prompts(
            dataset_path=args.dataset_path,
            num_prompts=args.num_unique_prompts,
            tokenizer=tokenizer,
            min_prompt_tokens=args.min_prompt_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            fixed_output_len=fixed_output,
            seed=args.seed,
        )

    round_prompts = expand_repeat_prompts(
        unique_prompts, args.num_requests, args.repeat_mode
    )

    prompt_lens = [p[1] for p in round_prompts]
    output_lens = [p[2] for p in round_prompts]
    print(f"  Unique prompts: {len(unique_prompts)}")
    print(f"  Total requests per round: {len(round_prompts)}  "
          f"(repeat_mode={args.repeat_mode})")
    print(f"  Prompt length: min={min(prompt_lens)} max={max(prompt_lens)} "
          f"mean={int(np.mean(prompt_lens))} tokens")
    print(f"  Output length: min={min(output_lens)} max={max(output_lens)} "
          f"mean={int(np.mean(output_lens))} tokens")
    print(f"\nRound 1 and Round 2 replay the same {len(round_prompts)} requests in order.")
    print(f"Request rate: {args.request_rate} req/s  |  Max concurrency: {args.max_concurrency}")

    all_results: Dict = {}

    print(f"\n{'#'*60}")
    print("  ROUND 1: Cold Start")
    print(f"{'#'*60}")
    r1_results, r1_duration = await eb.run_round(
        url=args.server_url,
        prompts=round_prompts,
        model=args.model,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
    )
    r1_metrics = eb.compute_round_metrics("Round 1: Cold Start", r1_results, r1_duration)
    eb.print_metrics(r1_metrics)
    all_results["round1"] = {
        "metrics": vars(r1_metrics),
        "per_request": eb._per_request_records(r1_results),
    }

    print(f"\nPausing {args.pause_between_rounds}s between rounds...")
    await asyncio.sleep(args.pause_between_rounds)

    print(f"\n{'#'*60}")
    print("  ROUND 2: Warm Start (same sequence)")
    print(f"{'#'*60}")
    r2_results, r2_duration = await eb.run_round(
        url=args.server_url,
        prompts=round_prompts,
        model=args.model,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
    )
    r2_metrics = eb.compute_round_metrics("Round 2: Warm Start", r2_results, r2_duration)
    eb.print_metrics(r2_metrics)
    all_results["round2"] = {
        "metrics": vars(r2_metrics),
        "per_request": eb._per_request_records(r2_results),
    }

    print(f"\n{'='*60}")
    print("  COMPARISON: Round 2 vs Round 1")
    print(f"{'='*60}")
    if r1_metrics.ttft_mean_ms > 0 and r2_metrics.ttft_mean_ms > 0:
        ttft_speedup = r1_metrics.ttft_mean_ms / r2_metrics.ttft_mean_ms
        print(f"  TTFT speedup:      {ttft_speedup:.2f}x"
              f"  ({r1_metrics.ttft_mean_ms:.1f} -> {r2_metrics.ttft_mean_ms:.1f} ms)")
    if r1_metrics.output_throughput_tok_s > 0 and r2_metrics.output_throughput_tok_s > 0:
        tput_ratio = r2_metrics.output_throughput_tok_s / r1_metrics.output_throughput_tok_s
        print(f"  Throughput change: {tput_ratio:.2f}x")
    if r1_metrics.tbt_mean_ms > 0 and r2_metrics.tbt_mean_ms > 0:
        tbt_ratio = r1_metrics.tbt_mean_ms / r2_metrics.tbt_mean_ms
        print(f"  TBT speedup:       {tbt_ratio:.2f}x")
    if r1_metrics.e2e_mean_ms > 0 and r2_metrics.e2e_mean_ms > 0:
        e2e_speedup = r1_metrics.e2e_mean_ms / r2_metrics.e2e_mean_ms
        print(f"  E2E speedup:       {e2e_speedup:.2f}x")

    all_results["config"] = {
        "variant": "e2e_bench_fast",
        "server_url": args.server_url,
        "model": args.model,
        "tokenizer": tok_id,
        "local_files_only": args.local_files_only,
        "prompt_mode": args.prompt_mode,
        "target_input_tokens": args.target_input_tokens,
        "num_requests": args.num_requests,
        "num_unique_prompts": args.num_unique_prompts,
        "repeat_mode": args.repeat_mode,
        "output_tokens": args.output_tokens,
        "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "request_rate": args.request_rate,
        "max_concurrency": args.max_concurrency,
        "seed": args.seed,
        "prompt_stats": {
            "mean_prompt_len": int(np.mean(prompt_lens)),
            "mean_output_len": int(np.mean(output_lens)),
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--server-url", default="http://localhost:30000")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    p.add_argument("--tokenizer", default="", help="Default: same as --model")
    p.add_argument("--local-files-only", action="store_true")

    p.add_argument(
        "--prompt-mode",
        choices=("synthetic", "sharegpt"),
        default="synthetic",
    )
    p.add_argument("--target-input-tokens", type=int, default=8192)
    p.add_argument(
        "--num-requests",
        type=int,
        default=512,
        help="Total HTTP requests per round (both rounds)",
    )
    p.add_argument(
        "--num-unique-prompts",
        type=int,
        default=8,
        help="How many distinct prompts to generate before repeating (e.g. 8 with 512 total → 64× each in block mode)",
    )
    p.add_argument(
        "--repeat-mode",
        choices=("block", "interleaved"),
        default="block",
        help="block: A…A,B…B; interleaved: A,B,C,A,B,C,…",
    )
    p.add_argument("--output-tokens", type=int, default=1024)
    p.add_argument("--min-prompt-tokens", type=int, default=64)
    p.add_argument("--max-prompt-tokens", type=int, default=8192)
    p.add_argument("--request-rate", type=float, default=0.0)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--dataset-path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pause-between-rounds", type=float, default=3.0)
    p.add_argument("--output", type=str, default="e2e_fast_results.json")

    args = p.parse_args()

    if args.num_unique_prompts > args.num_requests:
        p.error("--num-unique-prompts must be <= --num-requests")
    if args.num_unique_prompts < 1:
        p.error("--num-unique-prompts must be >= 1")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
