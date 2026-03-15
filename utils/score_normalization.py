#!/usr/bin/env python3
"""
JSONL Scores Winsorization + Normalization Tool

Processing steps (for each numeric score):
  1. Winsorization: For each key, calculate the lower and upper bounds independently based on the key's own value percentiles [p_low, p_high].
     Values outside the bounds for that key are clipped to the boundary.
  2. Normalization: Linearly map the clipped value to [0, 1]:
     normalized = (clipped - lb) / (ub - lb)
"""

import argparse
import sys
import os
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    import ujson as json
    _JSON_LIB = "ujson"
except ImportError:
    import json
    _JSON_LIB = "json"


# ─────────────────────────── Argument Parsing ────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="score_normalization.py",
        description="Perform Winsorization clipping + [0,1] normalization for the 'scores' field in a JSONL file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Example usage:
  # Clip and normalize all numeric fields to p5~p95
  python score_normalization.py -i data.jsonl -o out.jsonl --pct-range 5 95

  # Process only specified fields and keep the original values
  python score_normalization.py -i data.jsonl -o out.jsonl --pct-range 1 99 \
      --keys quality relevance --keep-original

  # Flip the perplexity field (lower is better, after normalization use 1 - val)
  python score_normalization.py -i data.jsonl -o out.jsonl --pct-range 5 95 \
      --flip-keys perplexity repetition
        """,
    )

    parser.add_argument("-i", "--input",  required=True, metavar="INPUT",  help="Input JSONL file path")
    parser.add_argument("-o", "--output", required=True, metavar="OUTPUT", help="Output JSONL file path")

    parser.add_argument(
        "--pct-range", nargs=2, type=float, required=True,
        metavar=("LOWER_PCT", "UPPER_PCT"),
        help="Percentile clipping range (0-100), calculate bounds for each key independently, e.g.: --pct-range 5 95",
    )
    parser.add_argument(
        "--keys", nargs="+", default=None, metavar="KEY",
        help="List of keys in the scores field to process. If unspecified, all numeric keys are processed.",
    )
    parser.add_argument(
        "--keep-original", action="store_true",
        help="Keep original values as well, writing to <key>_orig field",
    )
    parser.add_argument(
        "--flip-keys", nargs="+", default=None, metavar="KEY",
        help="List of keys to flip after normalization (use 1 minus normalized value), suitable for metrics where lower is better",
    )
    parser.add_argument(
        "--scores-field", default="scores", metavar="FIELD",
        help="Name of the field containing the scores dictionary, default: scores",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000, metavar="N",
        help="Buffer size for batch writing, default: 50000",
    )

    return parser.parse_args()


# ─────────────────────────── Utility Functions ────────────────────────────

def count_lines_fast(filepath: str) -> int:
    """Quickly count lines in a file (read by block)."""
    count = 0
    with open(filepath, "rb") as f:
        buf = f.read(1 << 23)
        while buf:
            count += buf.count(b"\n")
            buf = f.read(1 << 23)
    return count


def _is_finite_number(val) -> bool:
    """Check if value is a finite number (exclude bool / NaN / Inf)."""
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return val == val and abs(val) != float("inf")
    return False


# ─────────────────────────── First Pass: Collect Values ────────────────────

def collect_values(
    filepath: str,
    scores_field: str,
    target_keys: set | None,
    total_lines: int,
) -> dict[str, np.ndarray]:
    """Scan file and collect all values for each key, for calculating percentiles per key."""
    values_dict: dict[str, list] = defaultdict(list)

    with open(filepath, "r", encoding="utf-8", buffering=1 << 22) as f:
        for line in tqdm(
            f,
            total=total_lines,
            desc="  [1/2] Scanning and collecting values",
            unit="lines",
            ncols=90,
            colour="cyan",
        ):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                scores = record.get(scores_field)
                if not isinstance(scores, dict):
                    continue
                for key, val in scores.items():
                    if target_keys is not None and key not in target_keys:
                        continue
                    if _is_finite_number(val):
                        values_dict[key].append(val)
            except Exception:
                continue

    return {k: np.array(v, dtype=np.float64) for k, v in values_dict.items()}


# ─────────────────────────── Compute Bounds ────────────────────────────

# Table column widths (keep the header and data aligned, avoid issues with Chinese character width)
_W_KEY, _W_NUM, _W_CNT, _W_RATE, _W_FLIP = 26, 12, 10, 9, 5


def compute_bounds(
    values_dict: dict[str, np.ndarray],
    pct_low: float,
    pct_high: float,
) -> dict[str, tuple[float, float]]:
    """
    For each key, calculate lower/upper bounds based on percentiles.
    Returns {key: (lb, ub)}.
    """
    bounds: dict[str, tuple[float, float]] = {}
    print(f"\n  Percentile range: p{pct_low} ~ p{pct_high}  (per key)")
    print(f"  {'key':<{_W_KEY}} {'lb':>{_W_NUM}} {'ub':>{_W_NUM}} {'N':>{_W_CNT}}")
    sep_len_cb = 2 + _W_KEY + _W_NUM * 2 + _W_CNT + 3
    print(f"  {'─' * sep_len_cb}")

    for key in sorted(values_dict):
        vals = values_dict[key]
        lb = float(np.percentile(vals, pct_low))
        ub = float(np.percentile(vals, pct_high))
        bounds[key] = (lb, ub)
        span = ub - lb
        extra = "  ⚠ span=0" if span == 0 else ""
        print(f"  {key:<{_W_KEY}} {lb:>{_W_NUM}.6g} {ub:>{_W_NUM}.6g} {len(vals):>{_W_CNT},}{extra}")

    print(f"  Normalization interval: [0, 1]  (linear mapping)")
    return bounds


# ─────────────────────────── Second Pass: Processing & Output ────────────────────

def process_and_write(
    input_path: str,
    output_path: str,
    bounds: dict[str, tuple[float, float]],
    scores_field: str,
    target_keys: set | None,
    keep_original: bool,
    flip_keys: set,
    total_lines: int,
    batch_size: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    """
    Read line by line, for each numeric score:
      1. Winsorize to [lb, ub]
      2. Min-max normalize to [0, 1]: (clipped - lb) / (ub - lb)
    Batch write results.
    Return (clipped_count, total_count, error_lines).
    """
    clipped_count: dict[str, int] = defaultdict(int)
    total_count:   dict[str, int] = defaultdict(int)
    error_lines = 0
    out_buf: list[str] = []

    with (
        open(input_path,  "r", encoding="utf-8", buffering=1 << 22) as fin,
        open(output_path, "w", encoding="utf-8", buffering=1 << 22) as fout,
    ):
        for line in tqdm(
            fin,
            total=total_lines,
            desc="  [2/2] Winsorize + Normalize",
            unit="lines",
            ncols=90,
            colour="green",
        ):
            stripped = line.rstrip("\n\r")
            if not stripped:
                out_buf.append("\n")
            else:
                try:
                    record = json.loads(stripped)
                    scores = record.get(scores_field)

                    if isinstance(scores, dict):
                        new_scores: dict = {}
                        for key, val in scores.items():
                            should_process = (
                                key in bounds
                                and (target_keys is None or key in target_keys)
                                and _is_finite_number(val)
                            )
                            if should_process:
                                lb, ub = bounds[key]
                                total_count[key] += 1

                                # 1. Winsorize
                                clipped = lb if val < lb else (ub if val > ub else val)
                                if clipped != val:
                                    clipped_count[key] += 1

                                # 2. Normalize to [0, 1]
                                span = ub - lb
                                normalized = (clipped - lb) / span if span != 0 else 0.0

                                # 3. Flip (metrics where lower is better)
                                if key in flip_keys:
                                    normalized = 1.0 - normalized

                                if keep_original:
                                    new_scores[key + "_orig"] = val
                                new_scores[key] = normalized
                            else:
                                new_scores[key] = val

                        record[scores_field] = new_scores

                    out_buf.append(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    error_lines += 1
                    out_buf.append(stripped + "\n")

            if len(out_buf) >= batch_size:
                fout.write("".join(out_buf))
                out_buf.clear()

        if out_buf:
            fout.write("".join(out_buf))

    return clipped_count, total_count, error_lines


# ─────────────────────────── Print Summary ────────────────────────────

def print_summary(
    bounds: dict[str, tuple[float, float]],
    clipped_count: dict[str, int],
    total_count: dict[str, int],
    flip_keys: set,
    error_lines: int,
    elapsed: float,
    total_lines: int,
    output_path: str,
):
    print(f"\n{'═'*72}")
    print(f"  Processing complete! Statistical summary")
    print(f"{'═'*72}")
    print(f"  Total time elapsed : {elapsed:.2f} seconds")
    print(f"  Processing speed   : {total_lines / elapsed:,.0f} lines/sec")
    if error_lines:
        print(f"  ⚠ Error lines      : {error_lines:,} line(s) (written to output as-is)")

    sep_len = 2 + _W_KEY + _W_NUM * 2 + _W_CNT * 2 + _W_RATE + _W_FLIP + 6
    print(f"  {'key':<{_W_KEY}} {'lb':>{_W_NUM}} {'ub':>{_W_NUM}} {'clip':>{_W_CNT}} {'total':>{_W_CNT}} {'rate':>{_W_RATE}} {'flip':>{_W_FLIP}}")
    print(f"  {'─' * sep_len}")
    for key in sorted(bounds):
        lb, ub = bounds[key]
        tot  = total_count.get(key, 0)
        clip = clipped_count.get(key, 0)
        rate = clip / tot * 100 if tot > 0 else 0.0
        rate_str = f"{rate:.2f}%"
        flip_mark = "✓" if key in flip_keys else ""
        print(f"  {key:<{_W_KEY}} {lb:>{_W_NUM}.6g} {ub:>{_W_NUM}.6g} {clip:>{_W_CNT},} {tot:>{_W_CNT},} {rate_str:>{_W_RATE}} {flip_mark:>{_W_FLIP}}")

    print(f"{'═'*72}")
    print(f"  Output file: {output_path}")
    print()


# ─────────────────────────── Entry Point ────────────────────────────────

def main():
    args = parse_args()

    # --- Parameter validation ---
    lo, hi = args.pct_range
    if not (0 <= lo < hi <= 100):
        print(f"[Error] --pct-range must satisfy 0 ≤ lower < upper ≤ 100, now: {lo}, {hi}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"[Error] Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            print(f"  Output directory was created: {out_dir}")
        except Exception as e:
            print(f"[Error] Failed to create output directory: {out_dir}\n{e}", file=sys.stderr)
            sys.exit(1)

    target_keys = set(args.keys) if args.keys else None
    flip_keys   = set(args.flip_keys) if args.flip_keys else set()

    # --- Run info header ---
    print(f"\n{'═'*72}")
    print(f"  JSONL Scores Winsorization + Normalization Tool (JSON: {_JSON_LIB})")
    print(f"{'═'*72}")
    print(f"  Input file     : {args.input}")
    print(f"  Output file    : {args.output}")
    print(f"  Scores field   : {args.scores_field}")
    print(f"  Percentile range: p{lo} ~ p{hi} (per key)")
    if target_keys:
        print(f"  Target fields  : {', '.join(sorted(target_keys))}")
    else:
        print(f"  Target fields  : all numeric fields")
    print(f"  Keep original value: {'yes → <key>_orig' if args.keep_original else 'no'}")
    if flip_keys:
        print(f"  Flipped fields : {', '.join(sorted(flip_keys))}  (after normalization → 1 - val)")

    # --- Count lines ---
    print(f"\n  Counting file lines...", end="", flush=True)
    total_lines = count_lines_fast(args.input)
    print(f" {total_lines:,} lines")

    start = time.perf_counter()

    # --- First pass: collect values ---
    print()
    values_dict = collect_values(args.input, args.scores_field, target_keys, total_lines)

    if not values_dict:
        print(f"\n[Error] No processable numeric key found in field '{args.scores_field}'!", file=sys.stderr)
        sys.exit(1)

    # --- Compute bounds for each key ---
    bounds = compute_bounds(values_dict, lo, hi)

    # --- Second pass: Winsorize + Normalize ---
    print()
    clipped_count, total_count, error_lines = process_and_write(
        args.input, args.output,
        bounds, args.scores_field,
        target_keys, args.keep_original,
        flip_keys,
        total_lines, args.batch_size,
    )

    elapsed = time.perf_counter() - start
    print_summary(bounds, clipped_count, total_count, flip_keys, error_lines, elapsed, total_lines, args.output)


if __name__ == "__main__":
    main()

