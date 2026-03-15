#!/usr/bin/env python3
"""
Merge local scores: use best_weights from each subdirectory to compute weighted sums
and add them as new score keys in the source JSONL file.

New scores are min-max normalized across all data. Original (unnormalized) values
are stored in key_name_orig (e.g., Layer1_c1_orig).

Example:
  Input: data/demo_normalized.jsonl (has scores: {AtheneScore: 0.5, ...})
  Weights dir: results/demo/Layer1/ (contains Layer1_c1/, Layer1_c2/, ...)
  Output: scores get Layer1_c1_orig (raw), Layer1_c1 (normalized 0~1), ...
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge local best_weights: compute weighted score sums per subdirectory and add to JSONL."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSONL file path (source data with scores dict).",
    )
    parser.add_argument(
        "--weights_dir",
        "-w",
        required=True,
        help="Directory containing subdirectories, each with best_weights.json.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSONL file path. If not specified, modify input in-place.",
    )
    parser.add_argument(
        "--score_field",
        "-s",
        default="scores",
        help="Field name containing the scores dict (default: scores).",
    )
    parser.add_argument(
        "--missing_zero",
        action="store_true",
        help="Use 0 for missing score keys (default: skip key in sum).",
    )
    parser.add_argument(
        "--no_count",
        action="store_true",
        help="Skip line count (faster start for huge files; progress bar without total).",
    )
    return parser.parse_args()


def load_all_weights(weights_dir: str) -> List[Tuple[str, Dict[str, float]]]:
    """
    Scan first-level subdirs for best_weights.json.
    Returns: [(subdir_name, weights_dict), ...]
    """
    results: List[Tuple[str, Dict[str, float]]] = []
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    for name in sorted(os.listdir(weights_dir)):
        subdir = os.path.join(weights_dir, name)
        if not os.path.isdir(subdir):
            continue
        path = os.path.join(subdir, "best_weights.json")
        if not os.path.isfile(path):
            print(f"[Skip] No best_weights.json in {subdir}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights = data.get("best_weights", data.get("weights", {}))
        if not isinstance(weights, dict):
            print(f"[Skip] Invalid best_weights in {path}", file=sys.stderr)
            continue
        # Normalize to float
        weights = {k: float(v) for k, v in weights.items()}
        results.append((name, weights))

    return results


def compute_weighted_sum(
    scores: Dict[str, float],
    weights: Dict[str, float],
    missing_zero: bool = False,
) -> float:
    """Compute weighted sum: sum(weights[k] * scores[k]) for k in weights."""
    total = 0.0
    for k, w in weights.items():
        if k in scores:
            total += w * float(scores[k])
        elif missing_zero:
            total += w * 0.0
        # else: skip key
    return total


def _fast_line_count(path: str) -> int:
    """Fast line count using buffered read (avoids Python loop overhead)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _minmax_normalize(x: float, min_val: float, max_val: float, eps: float = 1e-9) -> float:
    """Normalize x to [0, 1] using min-max. Returns 0.5 when min==max."""
    if max_val - min_val < eps:
        return 0.5
    return (x - min_val) / (max_val - min_val)


def run(
    input_path: str,
    weights_dir: str,
    output_path: str | None,
    score_field: str = "scores",
    missing_zero: bool = False,
    no_count: bool = False,
) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    weight_configs = load_all_weights(weights_dir)
    if not weight_configs:
        raise ValueError(f"No best_weights.json found under {weights_dir}")

    subdir_names = [name for name, _ in weight_configs]
    print(f"[Merge] Found {len(weight_configs)} weight configs: {subdir_names}")

    line_count = None
    if not no_count:
        print(f"[Merge] Counting lines...", end=" ", flush=True)
        line_count = _fast_line_count(input_path)
        print(f"{line_count:,} lines")

    # Pass 1: collect min/max for each new score
    print(f"[Merge] Pass 1/2: collecting min/max...")
    min_max: Dict[str, Tuple[float, float]] = {n: (float("inf"), float("-inf")) for n in subdir_names}
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=line_count, desc="Pass1", unit="lines"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            scores = item.get(score_field)
            if not isinstance(scores, dict):
                scores = {}
            for subdir_name, weights in weight_configs:
                ws = compute_weighted_sum(scores, weights, missing_zero=missing_zero)
                lo, hi = min_max[subdir_name]
                min_max[subdir_name] = (min(lo, ws), max(hi, ws))
    for k in subdir_names:
        lo, hi = min_max[k]
        if lo == float("inf"):
            min_max[k] = (0.0, 1.0)  # fallback when no valid data
        print(f"[Merge]   {k}: min={min_max[k][0]:.6f}, max={min_max[k][1]:.6f}")

    in_place = output_path is None or os.path.abspath(output_path) == os.path.abspath(input_path)
    if in_place:
        parent = os.path.dirname(input_path) or "."
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir=parent)
        try:
            os.close(fd)
            out_filepath = tmp_path
        except Exception:
            os.close(fd)
            raise
    else:
        # Ensure the output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    # Pass 2: write with _orig and normalized values
    print(f"[Merge] Pass 2/2: writing with normalization...")
    written = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(
        out_filepath if in_place else output_path, "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin, total=line_count, desc="Pass2", unit="lines"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] Invalid JSON line: {e}", file=sys.stderr)
                continue

            scores = item.get(score_field)
            if not isinstance(scores, dict):
                scores = {}
                item[score_field] = scores

            for subdir_name, weights in weight_configs:
                ws = compute_weighted_sum(scores, weights, missing_zero=missing_zero)
                orig_key = f"{subdir_name}_orig"
                lo, hi = min_max[subdir_name]
                norm_val = _minmax_normalize(ws, lo, hi)
                scores[orig_key] = ws
                scores[subdir_name] = norm_val

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    if in_place:
        os.replace(tmp_path, input_path)
        print(f"[Merge] Done. Updated {written:,} lines in-place: {input_path}")
    else:
        print(f"[Merge] Done. Wrote {written:,} lines to {output_path}")


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    weights_dir = os.path.abspath(args.weights_dir)
    output_path = os.path.abspath(args.output) if args.output else None

    if not os.path.isfile(input_path):
        print(f"[Error] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure that the output directory exists, if output path is specified
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    run(
        input_path=input_path,
        weights_dir=weights_dir,
        output_path=output_path,
        score_field=args.score_field,
        missing_zero=args.missing_zero,
        no_count=args.no_count,
    )


if __name__ == "__main__":
    main()
