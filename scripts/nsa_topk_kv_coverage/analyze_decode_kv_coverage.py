#!/usr/bin/env python3
"""Analyze decode-time NSA top-k history KV coverage."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List

import numpy as np

_NSA_TOPK_MAGIC = 0x4E534154
_HEADER_FMT = "<II"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _safe_float(value: float) -> str:
    return f"{value:.8f}"


def _read_bin_file(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(_HEADER_SIZE)
        if len(header) < _HEADER_SIZE:
            raise ValueError(f"{path} is too short to contain a header")
        magic, topk_width = struct.unpack(_HEADER_FMT, header)
        if magic != _NSA_TOPK_MAGIC:
            raise ValueError(f"{path} has bad magic 0x{magic:08x}")
        raw = f.read()

    dtype = np.dtype(
        [
            ("mode", np.uint8),
            ("layer_id", np.uint8),
            ("_pad", "V2"),
            ("ctx_len", np.int32),
            ("token_pos", np.int32),
            ("topk", np.int32, (topk_width,)),
        ]
    )
    complete_size = (len(raw) // dtype.itemsize) * dtype.itemsize
    if complete_size == 0:
        return np.empty(0, dtype=dtype)
    return np.frombuffer(raw[:complete_size], dtype=dtype)


def _load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {item["rid"]: item for item in manifest["requests"]}


def _length_label(length: int) -> str:
    if length % 1024 == 0:
        return f"{length // 1024}k"
    return str(length)


def _rid_from_log_path(path: Path) -> str:
    return path.stem


def _infer_prompt_len_from_rid(rid: str) -> int:
    match = re.match(r"ctx(\d+)k_", rid)
    if not match:
        raise ValueError(f"Cannot infer prompt length from rid={rid}")
    return int(match.group(1)) * 1024


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}
    sorted_values = sorted(values)
    p90_idx = min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "p50": median(values),
        "p90": sorted_values[p90_idx],
    }


def analyze(log_dir: Path, manifest_path: Path, output_dir: Path) -> None:
    manifest_by_rid = _load_manifest(manifest_path)
    per_layer_rows: List[Dict[str, Any]] = []
    per_request_rows: List[Dict[str, Any]] = []

    for log_path in sorted(log_dir.glob("*.bin")):
        rid = _rid_from_log_path(log_path)
        meta = manifest_by_rid.get(rid, {})
        prompt_len = int(meta.get("target_prompt_len") or _infer_prompt_len_from_rid(rid))
        output_len = int(meta.get("output_len", 0))
        length_label = _length_label(prompt_len)

        records = _read_bin_file(log_path)
        if len(records) == 0:
            continue
        records = records[records["mode"] == 1]
        if len(records) == 0:
            continue

        request_union: set[int] = set()
        for layer_id in sorted(int(x) for x in np.unique(records["layer_id"])):
            layer_records = records[records["layer_id"] == layer_id]
            flat_topk = layer_records["topk"].reshape(-1)
            valid = flat_topk[(flat_topk >= 0) & (flat_topk < prompt_len)]
            unique = np.unique(valid)
            request_union.update(int(x) for x in unique.tolist())
            avg_valid_topk = (
                float(np.mean(np.sum((layer_records["topk"] >= 0) & (layer_records["topk"] < prompt_len), axis=1)))
                if len(layer_records) > 0
                else 0.0
            )

            per_layer_rows.append(
                {
                    "rid": rid,
                    "length_label": length_label,
                    "target_prompt_len": prompt_len,
                    "output_len": output_len,
                    "layer_id": layer_id,
                    "decode_records": len(layer_records),
                    "unique_history_kv_used": len(unique),
                    "coverage": _safe_float(len(unique) / prompt_len),
                    "avg_history_topk_per_decode": _safe_float(avg_valid_topk),
                    "min_token_pos": int(np.min(layer_records["token_pos"])),
                    "max_token_pos": int(np.max(layer_records["token_pos"])),
                }
            )

        per_request_rows.append(
            {
                "rid": rid,
                "length_label": length_label,
                "target_prompt_len": prompt_len,
                "output_len": output_len,
                "num_layers": len(set(int(x) for x in records["layer_id"])),
                "decode_records": len(records),
                "unique_history_kv_used_any_layer": len(request_union),
                "coverage_any_layer": _safe_float(len(request_union) / prompt_len),
            }
        )

    _write_csv(
        output_dir / "per_request_layer.csv",
        per_layer_rows,
        [
            "rid",
            "length_label",
            "target_prompt_len",
            "output_len",
            "layer_id",
            "decode_records",
            "unique_history_kv_used",
            "coverage",
            "avg_history_topk_per_decode",
            "min_token_pos",
            "max_token_pos",
        ],
    )
    _write_csv(
        output_dir / "per_request.csv",
        per_request_rows,
        [
            "rid",
            "length_label",
            "target_prompt_len",
            "output_len",
            "num_layers",
            "decode_records",
            "unique_history_kv_used_any_layer",
            "coverage_any_layer",
        ],
    )

    by_length_layer: Dict[tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_layer_rows:
        by_length_layer[(int(row["target_prompt_len"]), int(row["layer_id"]))].append(row)

    summary_layer_rows: List[Dict[str, Any]] = []
    for (prompt_len, layer_id), rows in sorted(by_length_layer.items()):
        coverages = [float(row["coverage"]) for row in rows]
        unique_counts = [float(row["unique_history_kv_used"]) for row in rows]
        coverage_stats = _summarize(coverages)
        unique_stats = _summarize(unique_counts)
        summary_layer_rows.append(
            {
                "length_label": _length_label(prompt_len),
                "target_prompt_len": prompt_len,
                "layer_id": layer_id,
                "num_requests": len(rows),
                "mean_unique_history_kv_used": _safe_float(unique_stats["mean"]),
                "mean_coverage": _safe_float(coverage_stats["mean"]),
                "std_coverage": _safe_float(coverage_stats["std"]),
                "p50_coverage": _safe_float(coverage_stats["p50"]),
                "p90_coverage": _safe_float(coverage_stats["p90"]),
            }
        )

    _write_csv(
        output_dir / "summary_by_length_layer.csv",
        summary_layer_rows,
        [
            "length_label",
            "target_prompt_len",
            "layer_id",
            "num_requests",
            "mean_unique_history_kv_used",
            "mean_coverage",
            "std_coverage",
            "p50_coverage",
            "p90_coverage",
        ],
    )

    by_length: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_request_rows:
        by_length[int(row["target_prompt_len"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for prompt_len, rows in sorted(by_length.items()):
        coverages = [float(row["coverage_any_layer"]) for row in rows]
        unique_counts = [float(row["unique_history_kv_used_any_layer"]) for row in rows]
        coverage_stats = _summarize(coverages)
        unique_stats = _summarize(unique_counts)
        summary_rows.append(
            {
                "length_label": _length_label(prompt_len),
                "target_prompt_len": prompt_len,
                "num_requests": len(rows),
                "mean_unique_history_kv_used_any_layer": _safe_float(unique_stats["mean"]),
                "mean_coverage_any_layer": _safe_float(coverage_stats["mean"]),
                "std_coverage_any_layer": _safe_float(coverage_stats["std"]),
                "p50_coverage_any_layer": _safe_float(coverage_stats["p50"]),
                "p90_coverage_any_layer": _safe_float(coverage_stats["p90"]),
            }
        )

    _write_csv(
        output_dir / "summary_by_length.csv",
        summary_rows,
        [
            "length_label",
            "target_prompt_len",
            "num_requests",
            "mean_unique_history_kv_used_any_layer",
            "mean_coverage_any_layer",
            "std_coverage_any_layer",
            "p50_coverage_any_layer",
            "p90_coverage_any_layer",
        ],
    )
    print(f"Wrote analysis CSVs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    analyze(Path(args.log_dir), Path(args.manifest), Path(args.output_dir))


if __name__ == "__main__":
    main()
