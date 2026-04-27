#!/usr/bin/env python3
"""Generate fixed-length ShareGPT-derived prompts for NSA top-k experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer

from sglang.benchmark.datasets.common import SHAREGPT_FILENAME, SHAREGPT_REPO_ID
from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json


def _load_sharegpt(path: str) -> List[Dict[str, Any]]:
    if not is_file_valid_json(path) and path == "":
        path = download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )
    if not path:
        raise ValueError("ShareGPT path is empty and automatic download failed.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_turn_texts(records: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for item in records:
        conv = item.get("conversations", item.get("conversation", []))
        if not isinstance(conv, list):
            continue
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            text = turn.get("value", turn.get("content", ""))
            if isinstance(text, str) and text.strip():
                yield text.strip()


def _format_ctx_label(length: int) -> str:
    if length % 1024 == 0:
        return f"ctx{length // 1024:03d}k"
    return f"ctx{length}"


def _build_prompt_ids(
    token_chunks: List[List[int]],
    separator_ids: List[int],
    target_len: int,
    start_idx: int,
    add_bos_id: Optional[int],
) -> tuple[List[int], int]:
    ids: List[int] = []
    if add_bos_id is not None:
        ids.append(add_bos_id)

    idx = start_idx
    while len(ids) < target_len:
        chunk = token_chunks[idx % len(token_chunks)]
        idx += 1
        if ids and separator_ids:
            ids.extend(separator_ids)
        ids.extend(chunk)

    return ids[:target_len], idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Tokenizer/model path.")
    parser.add_argument("--sharegpt-path", default="", help="ShareGPT JSON path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lengths",
        default="16384,32768,65536,131072",
        help="Comma-separated prompt lengths.",
    )
    parser.add_argument("--requests-per-length", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Insert tokenizer.bos_token_id as the first prompt token when available.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records = _load_sharegpt(args.sharegpt_path)
    texts = list(_iter_turn_texts(records))
    random.shuffle(texts)

    token_chunks = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in texts
        if isinstance(text, str) and text.strip()
    ]
    token_chunks = [chunk for chunk in token_chunks if chunk]
    if not token_chunks:
        raise RuntimeError("No non-empty token chunks were extracted from ShareGPT.")

    separator_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    bos_id = tokenizer.bos_token_id if args.add_bos else None
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    dataset_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.json"
    requests: List[Dict[str, Any]] = []

    chunk_idx = 0
    with open(dataset_path, "w", encoding="utf-8") as f:
        for req_idx in range(args.requests_per_length):
            for length in lengths:
                label = _format_ctx_label(length)
                rid = f"{label}_req{req_idx:03d}"
                input_ids, chunk_idx = _build_prompt_ids(
                    token_chunks=token_chunks,
                    separator_ids=separator_ids,
                    target_len=length,
                    start_idx=chunk_idx,
                    add_bos_id=bos_id,
                )
                if len(input_ids) != length:
                    raise AssertionError(f"{rid} has {len(input_ids)} tokens, expected {length}")

                row = {
                    "rid": rid,
                    "target_prompt_len": length,
                    "output_len": args.output_len,
                    "input_ids": input_ids,
                }
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                requests.append(
                    {
                        "rid": rid,
                        "target_prompt_len": length,
                        "output_len": args.output_len,
                    }
                )

    manifest = {
        "model": args.model,
        "sharegpt_path": args.sharegpt_path,
        "lengths": lengths,
        "requests_per_length": args.requests_per_length,
        "output_len": args.output_len,
        "dataset_path": str(dataset_path),
        "requests": requests,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(requests)} requests to {dataset_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
