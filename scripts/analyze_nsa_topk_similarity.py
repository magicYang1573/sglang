"""Analyze NSA top-k index similarity across consecutive tokens.

For each request log produced by the NSA top-k logger (SGLANG_NSA_TOPK_LOG_DIR),
this script computes how similar the top-k sparse-attention index sets are between
a boundary token (token_pos that is a multiple of 4096) and the k preceding tokens
(k = 1 … 8).

Similarity definition
---------------------
sim(t, window=w) = |topk(t) ∩ union(topk(t-1), …, topk(t-w))| / |topk(t)|

i.e. the fraction of token t's top-k indices that appear in the union of the
preceding w tokens' top-k sets.

Output
------
8 tables (one per window size w = 1 … 8), each with:
  rows    = context-length bucket  (the ctx_len of the boundary token, rounded
            down to the nearest 4096)
  columns = layer_id
  cells   = mean similarity across all requests / boundary positions

Usage
-----
    python scripts/analyze_nsa_topk_similarity.py \\
        --log-dir /tmp/nsa_topk_logs \\
        [--output-dir ./nsa_analysis] \\
        [--boundary 4096] \\
        [--window 8] \\
        [--csv]          # also write tables as CSV files
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import pandas as pd

    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False


# ---------------------------------------------------------------------------
# Core similarity computation
# ---------------------------------------------------------------------------

def _topk_sets_for_req(records: np.ndarray, boundary: int, window: int):
    """Return per-layer list of (ctx_len_bucket, boundary_token_pos, window_sets).

    window_sets[w] = union of topk sets for positions [token_pos-w .. token_pos-1].

    Only boundary tokens (token_pos % boundary == 0, token_pos > 0) that have
    all w preceding tokens present in the log are included.

    Returns
    -------
    list of (layer_id, ctx_bucket, token_pos, topk_set_t, [union_set_1, …, union_set_window])
    """
    results = []

    # Group by layer_id
    for layer_id in np.unique(records["layer_id"]):
        layer_mask = records["layer_id"] == layer_id
        layer_recs = records[layer_mask]

        # Build pos -> topk_set mapping
        pos_to_topk: dict[int, frozenset] = {}
        pos_to_ctx: dict[int, int] = {}
        for rec in layer_recs:
            pos = int(rec["token_pos"])
            pos_to_topk[pos] = frozenset(rec["topk"].tolist())
            pos_to_ctx[pos] = int(rec["ctx_len"])

        # Find all boundary positions that have enough preceding tokens logged
        for pos, topk_t in pos_to_topk.items():
            if pos == 0 or pos % boundary != 0:
                continue

            # Check all preceding window positions are present
            preceding = [pos - j for j in range(1, window + 1)]
            if not all(p in pos_to_topk for p in preceding):
                continue

            ctx_bucket = (pos_to_ctx[pos] // boundary) * boundary

            # Build cumulative union sets for w = 1 … window
            union_sets = []
            cumulative: set = set()
            for p in preceding:  # p = pos-1, pos-2, …, pos-window
                cumulative |= pos_to_topk[p]
                union_sets.append(frozenset(cumulative))

            results.append((int(layer_id), ctx_bucket, pos, topk_t, union_sets))

    return results


def _compute_similarities(log_dir: str, boundary: int, window: int):
    """Iterate all .npz files and accumulate similarity statistics.

    Returns
    -------
    sums  : dict[(window_idx, ctx_bucket, layer_id)] -> float
    counts: dict[(window_idx, ctx_bucket, layer_id)] -> int
    """
    sums: dict = defaultdict(float)
    counts: dict = defaultdict(int)

    npz_files = sorted(Path(log_dir).glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(npz_files)} request log file(s) in {log_dir}")

    for fpath in npz_files:
        try:
            records = np.load(fpath)["records"]
        except Exception as exc:
            print(f"[WARN] Could not load {fpath}: {exc}", file=sys.stderr)
            continue

        if records.size == 0:
            continue

        entries = _topk_sets_for_req(records, boundary, window)
        for layer_id, ctx_bucket, _pos, topk_t, union_sets in entries:
            denom = len(topk_t)
            if denom == 0:
                continue
            for w_idx, union_set in enumerate(union_sets):  # w_idx = 0 → window=1
                overlap = len(topk_t & union_set)
                key = (w_idx, ctx_bucket, layer_id)
                sums[key] += overlap / denom
                counts[key] += 1

    return sums, counts


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def _build_tables(sums, counts, window: int):
    """Return list of (window_size, ctx_buckets_sorted, layers_sorted, 2-D mean array)."""
    all_ctx = sorted({k[1] for k in counts})
    all_layers = sorted({k[2] for k in counts})

    tables = []
    for w_idx in range(window):
        w = w_idx + 1
        arr = np.full((len(all_ctx), len(all_layers)), np.nan)
        for ci, ctx in enumerate(all_ctx):
            for li, layer in enumerate(all_layers):
                key = (w_idx, ctx, layer)
                if counts[key] > 0:
                    arr[ci, li] = sums[key] / counts[key]
        tables.append((w, all_ctx, all_layers, arr))

    return tables


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_ctx(ctx: int) -> str:
    return f"{ctx // 1024}k" if ctx % 1024 == 0 else str(ctx)


def _print_table(w: int, ctx_buckets, layers, arr: np.ndarray):
    header = f"=== Window w={w}: similarity of boundary token t with union(topk(t-1)…topk(t-{w})) ==="
    print(header)

    col_w = 8
    # Header row
    header_label = "ctx\\layer"
    row = f"{header_label:>12}" + "".join(f"{str(l):>{col_w}}" for l in layers)
    print(row)
    print("-" * len(row))

    for ci, ctx in enumerate(ctx_buckets):
        cells = []
        for li in range(len(layers)):
            v = arr[ci, li]
            cells.append(f"{'N/A':>{col_w}}" if np.isnan(v) else f"{v:>{col_w}.4f}")
        print(f"{_fmt_ctx(ctx):>12}" + "".join(cells))
    print()


def _save_csv(output_dir: str, w: int, ctx_buckets, layers, arr: np.ndarray):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"similarity_window_{w}.csv")
    with open(path, "w") as f:
        f.write("ctx_bucket," + ",".join(f"layer_{l}" for l in layers) + "\n")
        for ci, ctx in enumerate(ctx_buckets):
            vals = []
            for li in range(len(layers)):
                v = arr[ci, li]
                vals.append("" if np.isnan(v) else f"{v:.6f}")
            f.write(f"{_fmt_ctx(ctx)}," + ",".join(vals) + "\n")
    print(f"  Saved {path}")


def _save_pandas(output_dir: str, w: int, ctx_buckets, layers, arr: np.ndarray):
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"similarity_window_{w}.csv")
    df = pd.DataFrame(
        arr,
        index=[_fmt_ctx(c) for c in ctx_buckets],
        columns=[f"layer_{l}" for l in layers],
    )
    df.index.name = "ctx_bucket"
    df.to_csv(path, float_format="%.6f")
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NSA top-k similarity across consecutive tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("SGLANG_NSA_TOPK_LOG_DIR", ""),
        help="Directory containing per-request .npz log files "
             "(default: $SGLANG_NSA_TOPK_LOG_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to write CSV result tables (optional).",
    )
    parser.add_argument(
        "--boundary",
        type=int,
        default=4096,
        help="Token position boundary to analyse (default: 4096).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="Number of preceding tokens to compare against (default: 8).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write result tables as CSV files to --output-dir.",
    )
    args = parser.parse_args()

    if not args.log_dir:
        parser.error(
            "Provide --log-dir or set the SGLANG_NSA_TOPK_LOG_DIR environment variable."
        )

    print(f"boundary={args.boundary}  window={args.window}")
    print(f"log_dir : {args.log_dir}\n")

    sums, counts = _compute_similarities(args.log_dir, args.boundary, args.window)

    if not counts:
        print("[ERROR] No qualifying boundary tokens found in the logs.")
        print(
            "  Make sure the logs were generated with matching BOUNDARY/WINDOW settings "
            "and that context lengths exceed the boundary value."
        )
        sys.exit(1)

    tables = _build_tables(sums, counts, args.window)

    for w, ctx_buckets, layers, arr in tables:
        _print_table(w, ctx_buckets, layers, arr)
        if args.csv and args.output_dir:
            if _HAVE_PANDAS:
                _save_pandas(args.output_dir, w, ctx_buckets, layers, arr)
            else:
                _save_csv(args.output_dir, w, ctx_buckets, layers, arr)

    if args.csv and not args.output_dir:
        print("[WARN] --csv specified but --output-dir not set; skipping file output.")


if __name__ == "__main__":
    main()
