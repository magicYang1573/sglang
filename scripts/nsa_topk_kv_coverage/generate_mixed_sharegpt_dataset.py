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


def _encode_turn(tokenizer, text: str, max_length: int) -> List[int]:
    """Encode one turn with truncation so a pathological ShareGPT row cannot exceed model max."""
    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )


def _turn_encode_max_length(tokenizer, user_cap: int) -> int:
    """Clamp by tokenizer.model_max_length when it is a sane finite value."""
    m = getattr(tokenizer, "model_max_length", None)
    if m is None or m <= 0 or m > 10**7:
        return user_cap
    return min(int(m), user_cap)


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
    parser.add_argument(
        "--requests-per-length",
        type=int,
        default=16,
        help="Number of requests per context length (rid suffix req000..).",
    )
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Insert tokenizer.bos_token_id as the first prompt token when available.",
    )
    parser.add_argument(
        "--skip-if-params-match",
        action="store_true",
        help="If dataset.jsonl + manifest.json + build_params.json exist and match, exit without regenerating.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when cached params match (ignored without --skip-if-params-match).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=131072,
        help="Hard cap for prompt token length in experiments (default 128k). "
        "Also used as max_length when encoding each ShareGPT turn for the chunk pool.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    for L in lengths:
        if L > args.max_prompt_length:
            raise ValueError(
                f"Requested prompt length {L} exceeds --max-prompt-length {args.max_prompt_length} (128k cap)."
            )
    build_params: Dict[str, Any] = {
        "model": args.model,
        "sharegpt_path": args.sharegpt_path,
        "lengths": lengths,
        "requests_per_length": args.requests_per_length,
        "output_len": args.output_len,
        "seed": args.seed,
        "add_bos": bool(args.add_bos),
        "max_prompt_length": args.max_prompt_length,
    }
    dataset_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.json"
    build_params_path = output_dir / "build_params.json"

    if (
        args.skip_if_params_match
        and not args.force
        and dataset_path.is_file()
        and manifest_path.is_file()
        and build_params_path.is_file()
    ):
        try:
            cached = json.loads(build_params_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if cached == build_params:
            print(
                f"[generate] Skip: reuse existing dataset in {output_dir} (params match).",
                flush=True,
            )
            return

    print(f"[generate] Loading tokenizer from {args.model!r} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("[generate] Tokenizer loaded.", flush=True)
    turn_encode_cap = _turn_encode_max_length(tokenizer, args.max_prompt_length)
    print(
        f"[generate] Per-turn encode cap: {turn_encode_cap} tokens "
        f"(max_prompt_length={args.max_prompt_length}).",
        flush=True,
    )

    print(f"[generate] Loading ShareGPT from {args.sharegpt_path!r} ...", flush=True)
    records = _load_sharegpt(args.sharegpt_path)
    print(f"[generate] Loaded {len(records)} ShareGPT records.", flush=True)

    texts = list(_iter_turn_texts(records))
    random.shuffle(texts)
    print(f"[generate] Collected {len(texts)} text turns; encoding to token chunks ...", flush=True)

    token_chunks: List[List[int]] = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            token_chunks.append(_encode_turn(tokenizer, text, turn_encode_cap))
        if (i + 1) % 2000 == 0 or i + 1 == len(texts):
            print(
                f"[generate]  token-chunk progress: {i + 1}/{len(texts)} turns scanned, "
                f"{len(token_chunks)} non-empty chunks",
                flush=True,
            )
    token_chunks = [chunk for chunk in token_chunks if chunk]
    if not token_chunks:
        raise RuntimeError("No non-empty token chunks were extracted from ShareGPT.")
    print(f"[generate] Built {len(token_chunks)} non-empty token chunks.", flush=True)

    separator_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    bos_id = tokenizer.bos_token_id if args.add_bos else None

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
    build_params_path.write_text(json.dumps(build_params, indent=2), encoding="utf-8")
    print(f"Wrote {len(requests)} requests to {dataset_path}", flush=True)
    print(f"Wrote manifest to {manifest_path}", flush=True)
    print(f"Wrote build params to {build_params_path}", flush=True)


if __name__ == "__main__":
    main()
