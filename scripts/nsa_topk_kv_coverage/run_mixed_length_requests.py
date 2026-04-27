#!/usr/bin/env python3
"""Send mixed-length fixed-token prompts to an SGLang OpenAI completions server.

Why greedy outputs often "loop"
--------------------------------
With ``temperature=0``, ``min_tokens == max_tokens``, and ``ignore_eos=True``, the model
must emit exactly N new tokens. Greedy decoding repeatedly picks the single highest-probability
next token; on long forced continuations this often locks into repeated n-grams (especially
when the prompt is synthetic token soup with no crisp continuation).

Defaults use stronger stochastic sampling plus repetition / frequency / presence penalties
while still requesting N tokens and ``ignore_eos=True`` so decode-length stays N for NSA
logging experiments. Tune flags if text becomes incoherent or over-penalized.
Use ``--greedy-decode`` to restore the old fully-greedy behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List

import aiohttp


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stable_seed_for_rid(rid: str) -> int:
    """32-bit positive int for sampling_seed (reproducible per rid)."""
    return (zlib.crc32(rid.encode("utf-8")) & 0x7FFFFFFF) or 1


async def _send_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    row: Dict[str, Any],
    extra_body: Dict[str, Any],
    sampling: Dict[str, Any],
) -> Dict[str, Any]:
    rid = row["rid"]
    output_len = int(row["output_len"])
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": row["input_ids"],
        "max_tokens": output_len,
        "min_tokens": output_len,
        "stream": True,
        "ignore_eos": True,
        "rid": rid,
    }
    payload.update(sampling)
    payload.update(extra_body)

    start = time.perf_counter()
    generated_text: List[str] = []
    chunk_count = 0
    result: Dict[str, Any] = {
        "rid": rid,
        "target_prompt_len": row["target_prompt_len"],
        "requested_output_len": output_len,
        "success": False,
        "sampling": {k: v for k, v in sampling.items() if k != "seed" or v is not None},
    }

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                result["error"] = await response.text()
                result["status"] = response.status
                return result

            async for chunk_bytes in response.content:
                chunk = chunk_bytes.decode("utf-8").strip()
                if not chunk:
                    continue
                if chunk.startswith("data:"):
                    chunk = chunk[len("data:") :].strip()
                if chunk == "[DONE]":
                    continue
                data = json.loads(chunk)
                text = data["choices"][0].get("text", "")
                if text:
                    generated_text.append(text)
                    chunk_count += 1

        result.update(
            {
                "success": True,
                "latency": time.perf_counter() - start,
                "stream_text_chunks": chunk_count,
                "generated_text": "".join(generated_text),
            }
        )
    except Exception as exc:  # noqa: BLE001 - keep benchmark output self-contained.
        result["error"] = repr(exc)
        result["latency"] = time.perf_counter() - start
    return result


async def _run(args: argparse.Namespace) -> None:
    rows = _load_dataset(args.dataset_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_body = json.loads(args.extra_request_body) if args.extra_request_body else {}
    url = args.base_url.rstrip("/") + "/v1/completions"

    if args.greedy_decode:
        sampling = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "seed": None,
        }
    else:
        sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
        }

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def limited(row: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                samp = dict(sampling)
                if args.greedy_decode:
                    samp.pop("seed", None)
                elif args.seed is not None:
                    samp["seed"] = int(args.seed)
                elif args.per_rid_seed:
                    samp["seed"] = _stable_seed_for_rid(row["rid"])
                else:
                    samp.pop("seed", None)
                return await _send_one(session, url, args.model, row, extra_body, samp)

        tasks = [asyncio.create_task(limited(row)) for row in rows]
        completed = 0
        with open(output_path, "w", encoding="utf-8") as out:
            for task in asyncio.as_completed(tasks):
                result = await task
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {result['rid']} "
                    f"success={result['success']} latency={result.get('latency', 0):.2f}s",
                    flush=True,
                )

    failures = sum(1 for task in tasks if not task.result().get("success"))
    if failures:
        print(f"{failures} requests failed. See {output_path}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Wrote request outputs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60)
    parser.add_argument(
        "--extra-request-body",
        default="",
        help="JSON object merged into every completions request (overrides script defaults).",
    )
    parser.add_argument(
        "--greedy-decode",
        action="store_true",
        help="Use temperature=0 and no penalties (old behavior; more repetitive on long N).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.82,
        help=">0 reduces repetitive loops when min_tokens forces long completions.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus mass; slightly higher widens alternatives vs verbatim loops.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="If >0, limits tail of sampling distribution (SGLang extension).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.22,
        help=">1 penalizes reusing the same token ids (stronger anti-loop).",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.55,
        help="Penalizes tokens proportional to how often they already appeared.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.28,
        help="Penalizes any token that has already appeared at least once.",
    )
    parser.set_defaults(per_rid_seed=True)
    parser.add_argument(
        "--no-per-rid-seed",
        dest="per_rid_seed",
        action="store_false",
        help="Do not set sampling seed (less reproducible across runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed sampling seed for all requests (overrides per-rid seed).",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
