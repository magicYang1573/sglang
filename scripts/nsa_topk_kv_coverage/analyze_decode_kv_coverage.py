#!/usr/bin/env python3
"""Analyze decode-time NSA top-k: per-layer ratio of used KV slots vs total prompt KV slots.

Each transformer layer has its own KV tensors; sequence positions are shared indices.
For a fixed prompt length *P*, each layer has *P* prompt-side KV slots (positions 0..P-1).

Metrics (per request, per layer):
- total_prompt_kv_slots: P (denominator for "全部 KV" in prompt range).
- unique_kv_slots_used_decode_union: distinct positions in [0, P) ever appearing in logged
  top-k across all decode steps (union).
- kv_union_used_to_total_prompt: union / P — fraction of prompt KV ever read during decode.
- mean_per_decode_step_used_to_total_prompt: mean over decode steps of
  (count of top-k indices in [0, P) / P). Counts duplicates within one step if any.

Summary files aggregate these per (prompt_length, layer_id) across requests.

Additionally, under ``<output_dir>/by_length/len_<16k|32k|...>/`` the same per-length
slices are written (``per_request_layer.csv``, ``per_request.csv``, ``summary_by_layer.csv``,
``summary.csv``) so each context length can be inspected without filtering the global CSVs.
"""

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
    expected_rids = set(manifest_by_rid)
    found_rids: set[str] = set()
    per_layer_rows: List[Dict[str, Any]] = []
    per_request_rows: List[Dict[str, Any]] = []

    for log_path in sorted(log_dir.glob("*.bin")):
        rid = log_path.stem
        meta = manifest_by_rid.get(rid, {})
        prompt_len = int(meta.get("target_prompt_len") or _infer_prompt_len_from_rid(rid))
        output_len = int(meta.get("output_len", 0))
        length_label = _length_label(prompt_len)
        denom = max(prompt_len, 1)

        records = _read_bin_file(log_path)
        if len(records) == 0:
            continue
        records = records[records["mode"] == 1]
        if len(records) == 0:
            continue
        found_rids.add(rid)

        request_union: set[int] = set()
        layer_decode_steps: List[int] = []
        for layer_id in sorted(int(x) for x in np.unique(records["layer_id"])):
            layer_records = records[records["layer_id"] == layer_id]
            layer_decode_steps.append(len(layer_records))
            in_prompt = (layer_records["topk"] >= 0) & (layer_records["topk"] < prompt_len)
            counts_in_prompt = np.sum(in_prompt, axis=1).astype(np.float64)
            mean_step_ratio = float(np.mean(counts_in_prompt / denom))

            flat_topk = layer_records["topk"].reshape(-1)
            valid = flat_topk[(flat_topk >= 0) & (flat_topk < prompt_len)]
            unique = np.unique(valid)
            union_n = int(len(unique))
            request_union.update(int(x) for x in unique.tolist())
            union_ratio = union_n / denom

            per_layer_rows.append(
                {
                    "rid": rid,
                    "length_label": length_label,
                    "target_prompt_len": prompt_len,
                    "output_len": output_len,
                    "layer_id": layer_id,
                    "total_prompt_kv_slots": prompt_len,
                    "decode_steps": len(layer_records),
                    "unique_kv_slots_used_decode_union": union_n,
                    "kv_union_used_to_total_prompt": _safe_float(union_ratio),
                    "mean_per_decode_step_used_to_total_prompt": _safe_float(mean_step_ratio),
                    "min_token_pos": int(np.min(layer_records["token_pos"])),
                    "max_token_pos": int(np.max(layer_records["token_pos"])),
                }
            )

        union_any = len(request_union)
        expected_layer_steps = output_len if output_len > 0 else None
        layers_with_expected_steps = (
            sum(1 for steps in layer_decode_steps if steps == expected_layer_steps)
            if expected_layer_steps is not None
            else 0
        )
        per_request_rows.append(
            {
                "rid": rid,
                "length_label": length_label,
                "target_prompt_len": prompt_len,
                "output_len": output_len,
                "num_layers": len(set(int(x) for x in records["layer_id"])),
                "decode_steps_total": len(records),
                "min_decode_steps_per_layer": min(layer_decode_steps) if layer_decode_steps else 0,
                "max_decode_steps_per_layer": max(layer_decode_steps) if layer_decode_steps else 0,
                "mean_decode_steps_per_layer": _safe_float(mean(layer_decode_steps))
                if layer_decode_steps
                else _safe_float(0.0),
                "layers_with_expected_decode_steps": layers_with_expected_steps,
                "unique_kv_slots_used_any_layer_union": union_any,
                "kv_union_used_to_total_prompt_any_layer": _safe_float(union_any / denom),
            }
        )

    per_layer_fields = [
        "rid",
        "length_label",
        "target_prompt_len",
        "output_len",
        "layer_id",
        "total_prompt_kv_slots",
        "decode_steps",
        "unique_kv_slots_used_decode_union",
        "kv_union_used_to_total_prompt",
        "mean_per_decode_step_used_to_total_prompt",
        "min_token_pos",
        "max_token_pos",
    ]
    _write_csv(output_dir / "per_request_layer.csv", per_layer_rows, per_layer_fields)

    per_request_fields = [
        "rid",
        "length_label",
        "target_prompt_len",
        "output_len",
        "num_layers",
        "decode_steps_total",
        "min_decode_steps_per_layer",
        "max_decode_steps_per_layer",
        "mean_decode_steps_per_layer",
        "layers_with_expected_decode_steps",
        "unique_kv_slots_used_any_layer_union",
        "kv_union_used_to_total_prompt_any_layer",
    ]
    _write_csv(output_dir / "per_request.csv", per_request_rows, per_request_fields)

    missing_rids = sorted(expected_rids - found_rids)
    extra_rids = sorted(found_rids - expected_rids)
    completeness_rows = [
        {
            "expected_requests": len(expected_rids),
            "found_requests": len(found_rids),
            "missing_requests": len(missing_rids),
            "extra_requests": len(extra_rids),
            "missing_rids": " ".join(missing_rids),
            "extra_rids": " ".join(extra_rids),
        }
    ]
    _write_csv(
        output_dir / "completeness.csv",
        completeness_rows,
        [
            "expected_requests",
            "found_requests",
            "missing_requests",
            "extra_requests",
            "missing_rids",
            "extra_rids",
        ],
    )

    summary_layer_fields = [
        "length_label",
        "target_prompt_len",
        "layer_id",
        "num_requests",
        "mean_unique_kv_slots_used_decode_union",
        "mean_kv_union_used_to_total_prompt",
        "std_kv_union_used_to_total_prompt",
        "p50_kv_union_used_to_total_prompt",
        "p90_kv_union_used_to_total_prompt",
        "mean_per_decode_step_used_to_total_prompt",
    ]
    summary_length_fields = [
        "length_label",
        "target_prompt_len",
        "num_requests",
        "mean_kv_union_used_to_total_prompt_any_layer",
        "std_kv_union_used_to_total_prompt_any_layer",
        "p50_kv_union_used_to_total_prompt_any_layer",
        "p90_kv_union_used_to_total_prompt_any_layer",
    ]

    by_length_layer: Dict[tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_layer_rows:
        by_length_layer[(int(row["target_prompt_len"]), int(row["layer_id"]))].append(row)

    summary_layer_rows: List[Dict[str, Any]] = []
    for (prompt_len, layer_id), rows in sorted(by_length_layer.items()):
        union_ratios = [float(row["kv_union_used_to_total_prompt"]) for row in rows]
        step_ratios = [float(row["mean_per_decode_step_used_to_total_prompt"]) for row in rows]
        u_stats = _summarize([float(row["unique_kv_slots_used_decode_union"]) for row in rows])
        ur_stats = _summarize(union_ratios)
        sr_stats = _summarize(step_ratios)
        summary_layer_rows.append(
            {
                "length_label": _length_label(prompt_len),
                "target_prompt_len": prompt_len,
                "layer_id": layer_id,
                "num_requests": len(rows),
                "mean_unique_kv_slots_used_decode_union": _safe_float(u_stats["mean"]),
                "mean_kv_union_used_to_total_prompt": _safe_float(ur_stats["mean"]),
                "std_kv_union_used_to_total_prompt": _safe_float(ur_stats["std"]),
                "p50_kv_union_used_to_total_prompt": _safe_float(ur_stats["p50"]),
                "p90_kv_union_used_to_total_prompt": _safe_float(ur_stats["p90"]),
                "mean_per_decode_step_used_to_total_prompt": _safe_float(sr_stats["mean"]),
            }
        )

    _write_csv(
        output_dir / "summary_by_length_layer.csv",
        summary_layer_rows,
        summary_layer_fields,
    )

    by_length: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_request_rows:
        by_length[int(row["target_prompt_len"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for prompt_len, rows in sorted(by_length.items()):
        ratios = [float(row["kv_union_used_to_total_prompt_any_layer"]) for row in rows]
        r_stats = _summarize(ratios)
        summary_rows.append(
            {
                "length_label": _length_label(prompt_len),
                "target_prompt_len": prompt_len,
                "num_requests": len(rows),
                "mean_kv_union_used_to_total_prompt_any_layer": _safe_float(r_stats["mean"]),
                "std_kv_union_used_to_total_prompt_any_layer": _safe_float(r_stats["std"]),
                "p50_kv_union_used_to_total_prompt_any_layer": _safe_float(r_stats["p50"]),
                "p90_kv_union_used_to_total_prompt_any_layer": _safe_float(r_stats["p90"]),
            }
        )

    _write_csv(
        output_dir / "summary_by_length.csv",
        summary_rows,
        summary_length_fields,
    )

    by_len_root = output_dir / "by_length"
    for prompt_len in sorted(
        {int(r["target_prompt_len"]) for r in per_request_rows},
        key=int,
    ):
        lab = _length_label(prompt_len)
        sub = by_len_root / f"len_{lab}"
        r_pl = [r for r in per_layer_rows if int(r["target_prompt_len"]) == prompt_len]
        r_pr = [r for r in per_request_rows if int(r["target_prompt_len"]) == prompt_len]
        r_sl = [r for r in summary_layer_rows if int(r["target_prompt_len"]) == prompt_len]
        r_one = [r for r in summary_rows if int(r["target_prompt_len"]) == prompt_len]
        _write_csv(sub / "per_request_layer.csv", r_pl, per_layer_fields)
        _write_csv(sub / "per_request.csv", r_pr, per_request_fields)
        _write_csv(sub / "summary_by_layer.csv", r_sl, summary_layer_fields)
        if r_one:
            _write_csv(sub / "summary.csv", r_one, summary_length_fields)

    print(f"Wrote analysis CSVs to {output_dir}")
    print(f"Per-length slices: {by_len_root}/len_<label>/")
    print(
        "Completeness: "
        f"expected={len(expected_rids)} found={len(found_rids)} "
        f"missing={len(missing_rids)} extra={len(extra_rids)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    analyze(Path(args.log_dir), Path(args.manifest), Path(args.output_dir))


if __name__ == "__main__":
    main()
