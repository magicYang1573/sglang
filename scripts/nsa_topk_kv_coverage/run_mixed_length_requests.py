#!/usr/bin/env python3
"""Send mixed-length fixed-token prompts to an SGLang OpenAI completions server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
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


async def _send_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    row: Dict[str, Any],
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    rid = row["rid"]
    output_len = int(row["output_len"])
    payload = {
        "model": model,
        "prompt": row["input_ids"],
        "max_tokens": output_len,
        "min_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "rid": rid,
    }
    payload.update(extra_body)

    start = time.perf_counter()
    generated_text = []
    chunk_count = 0
    result: Dict[str, Any] = {
        "rid": rid,
        "target_prompt_len": row["target_prompt_len"],
        "requested_output_len": output_len,
        "success": False,
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

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def limited(row: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await _send_one(session, url, args.model, row, extra_body)

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
        help="JSON object merged into every completions request.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
