#!/usr/bin/env python3
"""Fast two-round e2e benchmark for repeated-request prefix-cache experiments.

Round 1 (Warmup):
  Send enough copies of each unique prompt to populate the radix / prefix cache
  on **every DP rank**.  The server's ``dp_size`` is auto-detected via
  ``/server_info``; warmup sends ``num_unique_prompts * dp_size`` requests
  (one copy per unique prompt per DP rank) sequentially so each rank sees at
  least one copy.

Round 2 (Measurement):
  Expand the same unique prompts to ``--num-requests`` total HTTP requests by
  repetition, then replay that full repeated sequence on the warmed cache.

Server-side behaviour (e.g. HiSparse RDMA locality / one-time prefetch) is not
changed: every HTTP request is still one independent Req.

Repeat modes:
  block       — [A]*k, [B]*k, ... (default; e.g. 8 unique x 64 = 512)
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

import aiohttp
import numpy as np

# Same directory as e2e_bench.py -> import shared implementation.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import e2e_bench as eb  # noqa: E402


async def _query_dp_size(server_url: str) -> int:
    """Query dp_size from the SGLang server via /server_info."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url}/server_info",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    info = await resp.json()
                    dp = int(info.get("dp_size", 1))
                    return max(dp, 1)
    except Exception as e:
        print(f"  Warning: failed to query /server_info ({e}), assuming dp_size=1")
    return 1


def _repeat_counts(num_unique: int, total: int) -> List[int]:
    base = total // num_unique
    rem = total % num_unique
    return [base + (1 if i < rem else 0) for i in range(num_unique)]


def expand_repeat_prompts(
    unique: List[Tuple[str, int, int]],
    total: int,
    mode: str,
) -> List[Tuple[str, int, int]]:
    """Expand ``unique`` prompts to length ``total`` by repetition."""
    u = len(unique)
    if u == 0:
        raise ValueError("No unique prompts to expand.")
    if total < u:
        raise ValueError(
            f"--num-requests ({total}) must be >= --num-unique-prompts ({u})."
        )

    counts = _repeat_counts(u, total)

    if mode == "interleaved":
        rem = list(counts)
        out: List[Tuple[str, int, int]] = []
        while any(x > 0 for x in rem):
            for i, prompt in enumerate(unique):
                if rem[i] > 0:
                    out.append(prompt)
                    rem[i] -= 1
        return out

    if mode != "block":
        raise ValueError(f"Unknown repeat mode: {mode}")

    out: List[Tuple[str, int, int]] = []
    for i, prompt in enumerate(unique):
        out.extend([prompt] * counts[i])
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

    # Auto-detect DP size so warmup covers all DP ranks.
    dp_size = await _query_dp_size(args.server_url)
    print(f"  Server dp_size: {dp_size}")

    # Round 1: send enough copies to populate every DP rank's radix cache.
    # With round-robin load balancing, sending N*dp_size sequential requests
    # ensures each rank sees at least N copies.
    warmup_copies = dp_size
    round1_prompts: List[Tuple[str, int, int]] = []
    for prompt in unique_prompts:
        round1_prompts.extend([prompt] * warmup_copies)

    round2_prompts = expand_repeat_prompts(
        unique_prompts, args.num_requests, args.repeat_mode
    )

    round1_prompt_lens = [p[1] for p in round1_prompts]
    round1_output_lens = [p[2] for p in round1_prompts]
    round2_prompt_lens = [p[1] for p in round2_prompts]
    round2_output_lens = [p[2] for p in round2_prompts]

    print(f"  Unique prompts: {len(unique_prompts)}")
    print(
        f"  Round 1 requests: {len(round1_prompts)} "
        f"({len(unique_prompts)} unique x {warmup_copies} copies for {dp_size} DP ranks)"
    )
    print(
        f"  Round 2 requests: {len(round2_prompts)} "
        f"(repeat_mode={args.repeat_mode})"
    )
    print(
        f"  Prompt length: min={min(round2_prompt_lens)} max={max(round2_prompt_lens)} "
        f"mean={int(np.mean(round2_prompt_lens))} tokens"
    )
    print(
        f"  Output length: min={min(round2_output_lens)} max={max(round2_output_lens)} "
        f"mean={int(np.mean(round2_output_lens))} tokens"
    )
    print(
        f"Request rate: {args.request_rate} req/s  |  Max concurrency: {args.max_concurrency}"
    )

    all_results: Dict = {}

    # Warmup: send one-at-a-time sequentially so the round-robin load
    # balancer distributes them evenly across DP ranks.
    print(f"\n{'#'*60}")
    print(f"  ROUND 1: Warmup ({len(round1_prompts)} requests, 1 per DP rank per prompt)")
    print(f"{'#'*60}")
    r1_results, r1_duration = await eb.run_round(
        url=args.server_url,
        prompts=round1_prompts,
        model=args.model,
        request_rate=args.request_rate,
        max_concurrency=1,
    )
    r1_metrics = eb.compute_round_metrics("Round 1: Warmup", r1_results, r1_duration)
    eb.print_metrics(r1_metrics)
    all_results["round1"] = {
        "metrics": vars(r1_metrics),
        "per_request": eb._per_request_records(r1_results),
    }

    print(f"\nPausing {args.pause_between_rounds}s between rounds...")
    await asyncio.sleep(args.pause_between_rounds)

    print(f"\n{'#'*60}")
    print("  ROUND 2: Measurement (repeated requests on warm cache)")
    print(f"{'#'*60}")
    r2_results, r2_duration = await eb.run_round(
        url=args.server_url,
        prompts=round2_prompts,
        model=args.model,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
    )
    r2_metrics = eb.compute_round_metrics(
        "Round 2: Warm Measurement", r2_results, r2_duration
    )
    eb.print_metrics(r2_metrics)
    all_results["round2"] = {
        "metrics": vars(r2_metrics),
        "per_request": eb._per_request_records(r2_results),
    }

    print(f"\n{'='*60}")
    print("  COMPARISON: Round 2 vs Round 1")
    print(f"{'='*60}")
    print(
        "  Note: Round 1 is unique-only warmup, Round 2 is full repeated-load measurement; "
        "throughput is not directly comparable."
    )
    if r1_metrics.ttft_mean_ms > 0 and r2_metrics.ttft_mean_ms > 0:
        ttft_speedup = r1_metrics.ttft_mean_ms / r2_metrics.ttft_mean_ms
        print(
            f"  TTFT speedup:      {ttft_speedup:.2f}x"
            f"  ({r1_metrics.ttft_mean_ms:.1f} -> {r2_metrics.ttft_mean_ms:.1f} ms)"
        )
    if r1_metrics.tbt_mean_ms > 0 and r2_metrics.tbt_mean_ms > 0:
        tbt_ratio = r1_metrics.tbt_mean_ms / r2_metrics.tbt_mean_ms
        print(
            f"  TBT speedup:       {tbt_ratio:.2f}x"
            f"  ({r1_metrics.tbt_mean_ms:.1f} -> {r2_metrics.tbt_mean_ms:.1f} ms)"
        )
    if r1_metrics.e2e_mean_ms > 0 and r2_metrics.e2e_mean_ms > 0:
        e2e_speedup = r1_metrics.e2e_mean_ms / r2_metrics.e2e_mean_ms
        print(
            f"  E2E speedup:       {e2e_speedup:.2f}x"
            f"  ({r1_metrics.e2e_mean_ms:.1f} -> {r2_metrics.e2e_mean_ms:.1f} ms)"
        )

    all_results["config"] = {
        "variant": "e2e_bench_fast",
        "server_url": args.server_url,
        "model": args.model,
        "tokenizer": tok_id,
        "local_files_only": args.local_files_only,
        "prompt_mode": args.prompt_mode,
        "target_input_tokens": args.target_input_tokens,
        "dp_size": dp_size,
        "round1_num_requests": len(round1_prompts),
        "round2_num_requests": len(round2_prompts),
        "num_requests": args.num_requests,
        "num_unique_prompts": args.num_unique_prompts,
        "repeat_mode": args.repeat_mode,
        "output_tokens": args.output_tokens,
        "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "request_rate": args.request_rate,
        "max_concurrency": args.max_concurrency,
        "seed": args.seed,
        "round1_prompt_stats": {
            "mean_prompt_len": int(np.mean(round1_prompt_lens)),
            "mean_output_len": int(np.mean(round1_output_lens)),
        },
        "round2_prompt_stats": {
            "mean_prompt_len": int(np.mean(round2_prompt_lens)),
            "mean_output_len": int(np.mean(round2_output_lens)),
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
        help="Round 2 total HTTP requests (measurement round)",
    )
    p.add_argument(
        "--num-unique-prompts",
        type=int,
        default=8,
        help="Round 1 warmup requests, and unique template count for Round 2 repeats",
    )
    p.add_argument(
        "--repeat-mode",
        choices=("block", "interleaved"),
        default="block",
        help="block: A...A,B...B; interleaved: A,B,C,A,B,C,...",
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
