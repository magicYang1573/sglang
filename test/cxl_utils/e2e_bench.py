#!/usr/bin/env python3
"""Two-round end-to-end benchmark for CXL vs RDMA prefix cache comparison.

Round 1 (Cold):
  Send N requests sampled from the ShareGPT dataset.
  Each request has unique content (different conversation).
  This populates the server's radix prefix cache.
  Measures: TTFT, TBT (inter-token latency), throughput.

Round 2 (Warm):
  Replay the EXACT SAME N requests as Round 1.
  The radix cache now holds the full KV for each request's prefix,
  so prefill is skipped and only extend tokens are computed.
  Measures: TTFT (should drop), TBT, throughput.

Request design:
  - Both rounds use the SAME set of N prompts (same objects, same order)
  - Each of the N prompts is a different ShareGPT user turn — no shared prefix
  - The "prefix" exploited by the cache is each request's own token prefix from Round 1
  - This tests KV-cache read bandwidth (CXL vs RDMA) under realistic workload diversity

Usage:
  python e2e_bench.py \\
    --server-url http://localhost:30000 \\
    --model deepseek-ai/DeepSeek-V3.2 \\
    --num-requests 16 \\
    --output-tokens 128 \\
    --request-rate 4 \\
    --output results.json
"""

import argparse
import asyncio
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np

# Add sglang to path for dataset/tokenizer utilities
SGLANG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SGLANG_ROOT / "python"))


@dataclass
class RequestResult:
    """Per-request timing result."""

    success: bool = False
    prompt_len: int = 0
    output_len: int = 0
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    itl_ms: List[float] = field(default_factory=list)
    error: str = ""


@dataclass
class RoundMetrics:
    """Aggregated metrics for one benchmark round."""

    round_name: str
    num_requests: int
    num_success: int
    duration_s: float
    total_input_tokens: int
    total_output_tokens: int

    # TTFT
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float

    # TBT (inter-token latency, aka time-between-tokens)
    tbt_mean_ms: float
    tbt_p50_ms: float
    tbt_p99_ms: float

    # TPOT (time per output token = (latency - TTFT) / (output_len - 1))
    tpot_mean_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float

    # Throughput
    output_throughput_tok_s: float
    request_throughput_req_s: float

    # E2E latency
    e2e_mean_ms: float
    e2e_p50_ms: float
    e2e_p99_ms: float


def _percentile(arr, p):
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, p))


def compute_round_metrics(
    round_name: str,
    results: List[RequestResult],
    duration_s: float,
) -> RoundMetrics:
    successful = [r for r in results if r.success]
    n_success = len(successful)
    if n_success == 0:
        return RoundMetrics(
            round_name=round_name,
            num_requests=len(results),
            num_success=0,
            duration_s=duration_s,
            total_input_tokens=0,
            total_output_tokens=0,
            ttft_mean_ms=0, ttft_p50_ms=0, ttft_p99_ms=0,
            tbt_mean_ms=0, tbt_p50_ms=0, tbt_p99_ms=0,
            tpot_mean_ms=0, tpot_p50_ms=0, tpot_p99_ms=0,
            output_throughput_tok_s=0,
            request_throughput_req_s=0,
            e2e_mean_ms=0, e2e_p50_ms=0, e2e_p99_ms=0,
        )

    total_input = sum(r.prompt_len for r in successful)
    total_output = sum(r.output_len for r in successful)

    ttfts = [r.ttft_ms for r in successful]
    latencies = [r.latency_ms for r in successful]

    # TBT: flatten all inter-token latencies
    all_itl = []
    for r in successful:
        all_itl.extend(r.itl_ms)

    # TPOT: per-request average decode latency
    tpots = []
    for r in successful:
        if r.output_len > 1:
            tpots.append((r.latency_ms - r.ttft_ms) / (r.output_len - 1))

    return RoundMetrics(
        round_name=round_name,
        num_requests=len(results),
        num_success=n_success,
        duration_s=round(duration_s, 3),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        ttft_mean_ms=round(float(np.mean(ttfts)), 2),
        ttft_p50_ms=round(_percentile(ttfts, 50), 2),
        ttft_p99_ms=round(_percentile(ttfts, 99), 2),
        tbt_mean_ms=round(float(np.mean(all_itl)), 2) if all_itl else 0.0,
        tbt_p50_ms=round(_percentile(all_itl, 50), 2),
        tbt_p99_ms=round(_percentile(all_itl, 99), 2),
        tpot_mean_ms=round(float(np.mean(tpots)), 2) if tpots else 0.0,
        tpot_p50_ms=round(_percentile(tpots, 50), 2),
        tpot_p99_ms=round(_percentile(tpots, 99), 2),
        output_throughput_tok_s=round(total_output / duration_s, 1) if duration_s > 0 else 0,
        request_throughput_req_s=round(n_success / duration_s, 2) if duration_s > 0 else 0,
        e2e_mean_ms=round(float(np.mean(latencies)), 2),
        e2e_p50_ms=round(_percentile(latencies, 50), 2),
        e2e_p99_ms=round(_percentile(latencies, 99), 2),
    )


# ---------------------------------------------------------------------------
#  Dataset loading: sample N diverse prompts from ShareGPT
# ---------------------------------------------------------------------------

def load_sharegpt_prompts(
    dataset_path: str,
    num_prompts: int,
    tokenizer,
    min_prompt_tokens: int = 64,
    max_prompt_tokens: int = 8192,
    fixed_output_len: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, int, int]]:
    """Load N diverse prompts from the ShareGPT dataset.

    Each prompt is a full user turn from a different conversation.
    No shared prefix — each request has entirely independent content.

    Returns a list of (prompt_text, prompt_len, output_len) tuples.
    prompt_len and output_len are in tokens.
    """
    from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json

    SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

    if not dataset_path or not is_file_valid_json(dataset_path):
        dataset_path = download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )

    with open(dataset_path) as f:
        dataset = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(dataset)

    prompts: List[Tuple[str, int, int]] = []
    for data in dataset:
        if len(prompts) >= num_prompts:
            break

        convs = data.get("conversations", data.get("conversation", []))
        if len(convs) < 2:
            continue

        user_msg = convs[0].get("value", "").strip()
        assistant_msg = convs[1].get("value", "").strip()
        if not user_msg or not assistant_msg:
            continue

        prompt_ids = tokenizer.encode(user_msg)
        prompt_len = len(prompt_ids)
        if prompt_len < min_prompt_tokens or prompt_len > max_prompt_tokens:
            continue

        if fixed_output_len is not None:
            output_len = fixed_output_len
        else:
            output_len = max(16, len(tokenizer.encode(assistant_msg)))

        prompts.append((user_msg, prompt_len, output_len))

    if len(prompts) < num_prompts:
        warnings.warn(
            f"Only {len(prompts)} usable prompts found in ShareGPT "
            f"(need {num_prompts}). Consider loosening token range filters.",
            stacklevel=2,
        )

    return prompts


# ---------------------------------------------------------------------------
#  Async request sender with streaming (for TTFT/ITL measurement)
# ---------------------------------------------------------------------------

async def send_request_streaming(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    prompt_len: int,
    max_new_tokens: int,
    model: str,
) -> RequestResult:
    """Send a streaming generation request and measure TTFT + per-token ITL."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }

    result = RequestResult(prompt_len=prompt_len)
    st = time.perf_counter()
    ttft_recorded = False
    most_recent_ts = st

    try:
        async with session.post(
            f"{url}/v1/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result

            output_tokens = 0
            async for chunk_bytes in resp.content:
                chunk_str = chunk_bytes.decode("utf-8").strip()
                if not chunk_str:
                    continue

                # Handle SSE format
                for line in chunk_str.split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        line = line[6:]
                    if not line or line == "[DONE]":
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    text = choices[0].get("text", "")
                    if not text:
                        continue

                    now = time.perf_counter()
                    if not ttft_recorded:
                        result.ttft_ms = (now - st) * 1000.0
                        ttft_recorded = True
                    else:
                        result.itl_ms.append((now - most_recent_ts) * 1000.0)

                    most_recent_ts = now
                    output_tokens += 1

            result.latency_ms = (time.perf_counter() - st) * 1000.0
            result.output_len = output_tokens
            result.success = output_tokens > 0

    except Exception as e:
        result.error = str(e)
        result.latency_ms = (time.perf_counter() - st) * 1000.0

    return result


async def run_round(
    url: str,
    prompts: List[Tuple[str, int, int]],  # (text, prompt_len, output_len)
    model: str,
    request_rate: float,
    max_concurrency: int,
) -> Tuple[List[RequestResult], float]:
    """Run one benchmark round with rate-limited request dispatch.

    Each entry in ``prompts`` carries its own output_len so requests can have
    varying expected output lengths (as sampled from the dataset).
    """
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def rate_limited_send(session, prompt, prompt_len, output_len, delay):
        if delay > 0:
            await asyncio.sleep(delay)
        if semaphore:
            async with semaphore:
                return await send_request_streaming(
                    session, url, prompt, prompt_len, output_len, model,
                )
        return await send_request_streaming(
            session, url, prompt, prompt_len, output_len, model,
        )

    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        t0 = time.perf_counter()
        for i, (prompt, prompt_len, output_len) in enumerate(prompts):
            delay = i / request_rate if request_rate > 0 else 0
            tasks.append(
                asyncio.create_task(
                    rate_limited_send(session, prompt, prompt_len, output_len, delay)
                )
            )

        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t0

    return list(results), duration


def _per_request_records(results: List[RequestResult]) -> List[Dict]:
    """Flatten per-request results to JSON-serialisable dicts."""
    return [
        {
            "success": r.success,
            "prompt_len": r.prompt_len,
            "output_len": r.output_len,
            "ttft_ms": round(r.ttft_ms, 2),
            "latency_ms": round(r.latency_ms, 2),
            "tbt_mean_ms": round(float(np.mean(r.itl_ms)), 2) if r.itl_ms else 0.0,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
#  Main benchmark logic
# ---------------------------------------------------------------------------

def print_metrics(m: RoundMetrics):
    print(f"\n{'='*60}")
    print(f"  {m.round_name}")
    print(f"{'='*60}")
    print(f"  Requests:       {m.num_success}/{m.num_requests} succeeded")
    print(f"  Duration:       {m.duration_s:.1f} s")
    print(f"  Input tokens:   {m.total_input_tokens}")
    print(f"  Output tokens:  {m.total_output_tokens}")
    print()
    print(f"  TTFT    mean={m.ttft_mean_ms:.1f}ms  p50={m.ttft_p50_ms:.1f}ms  p99={m.ttft_p99_ms:.1f}ms")
    print(f"  TBT     mean={m.tbt_mean_ms:.1f}ms  p50={m.tbt_p50_ms:.1f}ms  p99={m.tbt_p99_ms:.1f}ms")
    print(f"  TPOT    mean={m.tpot_mean_ms:.1f}ms  p50={m.tpot_p50_ms:.1f}ms  p99={m.tpot_p99_ms:.1f}ms")
    print(f"  E2E     mean={m.e2e_mean_ms:.1f}ms  p50={m.e2e_p50_ms:.1f}ms  p99={m.e2e_p99_ms:.1f}ms")
    print(f"  Throughput: {m.output_throughput_tok_s:.1f} tok/s  {m.request_throughput_req_s:.2f} req/s")


async def main_async(args):
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load N diverse prompts from ShareGPT — each request is a different conversation.
    # Both rounds will replay the exact same N prompts.
    fixed_output = args.output_tokens if args.output_tokens > 0 else None
    print(f"Loading {args.num_requests} diverse prompts from ShareGPT...")
    sharegpt_prompts = load_sharegpt_prompts(
        dataset_path=args.dataset_path,
        num_prompts=args.num_requests,
        tokenizer=tokenizer,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        fixed_output_len=fixed_output,
        seed=args.seed,
    )

    # prompts for run_round: list of (text, prompt_len); output_len comes from each entry
    # We carry output_len per request for accurate per-request max_tokens
    round_prompts: List[Tuple[str, int, int]] = sharegpt_prompts  # (text, prompt_len, output_len)

    prompt_lens = [p[1] for p in round_prompts]
    output_lens = [p[2] for p in round_prompts]
    print(f"  Loaded {len(round_prompts)} prompts")
    print(f"  Prompt length: min={min(prompt_lens)} max={max(prompt_lens)} "
          f"mean={int(np.mean(prompt_lens))} tokens")
    print(f"  Output length: min={min(output_lens)} max={max(output_lens)} "
          f"mean={int(np.mean(output_lens))} tokens")
    print(f"\nRound 1 and Round 2 use IDENTICAL {len(round_prompts)} requests.")
    print(f"Request rate: {args.request_rate} req/s")
    print(f"Max concurrency: {args.max_concurrency}")

    all_results: Dict = {}

    # --- Round 1: Cold start (no cache) ---
    print(f"\n{'#'*60}")
    print(f"  ROUND 1: Cold Start (populating prefix cache)")
    print(f"{'#'*60}")

    r1_results, r1_duration = await run_round(
        url=args.server_url,
        prompts=round_prompts,
        model=args.model,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
    )
    r1_metrics = compute_round_metrics("Round 1: Cold Start", r1_results, r1_duration)
    print_metrics(r1_metrics)
    all_results["round1"] = {
        "metrics": vars(r1_metrics),
        "per_request": _per_request_records(r1_results),
    }

    # Pause to let KV settle in cache
    pause_s = args.pause_between_rounds
    print(f"\nPausing {pause_s}s between rounds...")
    await asyncio.sleep(pause_s)

    # --- Round 2: Warm start (same prompts, cache hit) ---
    print(f"\n{'#'*60}")
    print(f"  ROUND 2: Warm Start (same requests, prefix cache hit)")
    print(f"{'#'*60}")

    r2_results, r2_duration = await run_round(
        url=args.server_url,
        prompts=round_prompts,
        model=args.model,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
    )
    r2_metrics = compute_round_metrics("Round 2: Warm Start", r2_results, r2_duration)
    print_metrics(r2_metrics)
    all_results["round2"] = {
        "metrics": vars(r2_metrics),
        "per_request": _per_request_records(r2_results),
    }

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("  COMPARISON: Round 2 vs Round 1")
    print(f"{'='*60}")
    if r1_metrics.ttft_mean_ms > 0 and r2_metrics.ttft_mean_ms > 0:
        ttft_speedup = r1_metrics.ttft_mean_ms / r2_metrics.ttft_mean_ms
        print(f"  TTFT speedup:      {ttft_speedup:.2f}x"
              f"  ({r1_metrics.ttft_mean_ms:.1f} -> {r2_metrics.ttft_mean_ms:.1f} ms)")
    if r1_metrics.output_throughput_tok_s > 0 and r2_metrics.output_throughput_tok_s > 0:
        tput_ratio = r2_metrics.output_throughput_tok_s / r1_metrics.output_throughput_tok_s
        print(f"  Throughput change: {tput_ratio:.2f}x"
              f"  ({r1_metrics.output_throughput_tok_s:.0f} -> "
              f"{r2_metrics.output_throughput_tok_s:.0f} tok/s)")
    if r1_metrics.tbt_mean_ms > 0 and r2_metrics.tbt_mean_ms > 0:
        tbt_ratio = r1_metrics.tbt_mean_ms / r2_metrics.tbt_mean_ms
        print(f"  TBT speedup:       {tbt_ratio:.2f}x"
              f"  ({r1_metrics.tbt_mean_ms:.1f} -> {r2_metrics.tbt_mean_ms:.1f} ms)")
    if r1_metrics.e2e_mean_ms > 0 and r2_metrics.e2e_mean_ms > 0:
        e2e_speedup = r1_metrics.e2e_mean_ms / r2_metrics.e2e_mean_ms
        print(f"  E2E speedup:       {e2e_speedup:.2f}x"
              f"  ({r1_metrics.e2e_mean_ms:.1f} -> {r2_metrics.e2e_mean_ms:.1f} ms)")

    # Save config in results
    all_results["config"] = {
        "server_url": args.server_url,
        "model": args.model,
        "num_requests": args.num_requests,
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

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CXL vs RDMA two-round e2e benchmark.\n\n"
        "Both rounds replay the EXACT SAME set of N requests sampled from "
        "the ShareGPT dataset. Each request has unique content (different "
        "conversation). Round 1 populates the prefix cache; Round 2 hits it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server-url", default="http://localhost:30000",
        help="SGLang server URL",
    )
    parser.add_argument(
        "--model", default="deepseek-ai/DeepSeek-V3.2",
        help="Model name (for tokenizer and API)",
    )
    parser.add_argument(
        "--num-requests", type=int, default=16,
        help="Number of requests per round (identical in both rounds)",
    )
    parser.add_argument(
        "--output-tokens", type=int, default=128,
        help="Fixed output token count per request. 0 = use dataset ground truth",
    )
    parser.add_argument(
        "--min-prompt-tokens", type=int, default=64,
        help="Minimum prompt length filter (tokens)",
    )
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=8192,
        help="Maximum prompt length filter (tokens)",
    )
    parser.add_argument(
        "--request-rate", type=float, default=4.0,
        help="Request dispatch rate (req/s). 0 = send all simultaneously",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=0,
        help="Max concurrent in-flight requests. 0 = unlimited",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="",
        help="Path to ShareGPT JSON file. Auto-downloaded from HuggingFace if empty.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for prompt sampling",
    )
    parser.add_argument(
        "--pause-between-rounds", type=float, default=3.0,
        help="Seconds to pause between Round 1 and Round 2",
    )
    parser.add_argument(
        "--output", type=str, default="e2e_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
