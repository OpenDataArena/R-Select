#!/usr/bin/env python3
"""
For each record in a JSONL file, compute the weighted sum of the "scores" field and write the result to the "final_score" field.
Supports loading weights from files such as best_weights.json.

Usage:
    python cluster_aggregation.py -i data/all.jsonl -w results/demo/Layer2/Layer2_c1/best_weights.json
    python cluster_aggregation.py -i data/all.jsonl -w best_weights.json -o out.jsonl
    If --output is not specified, the file will be modified in-place.
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
import time

from tqdm import tqdm


def load_weights(weights_path: str) -> dict:
    """Load weights from a JSON file, supporting the keys 'best_weights' or 'weights'."""
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w = data.get("best_weights", data.get("weights", {}))
    if not isinstance(w, dict):
        raise ValueError(f"Invalid weights format in {weights_path}")
    return {k: float(v) for k, v in w.items()}


def count_lines(filepath: str) -> int:
    """Quickly count file lines (for progress bar total)."""
    count = 0
    with open(filepath, "rb") as f:
        buf_size = 1 << 20  # 1 MB
        buf = f.raw.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = f.raw.read(buf_size)
    return count


def process(input_path: str, output_path: str | None, weights: dict, score_field: str = "scores"):
    inplace = output_path is None
    abs_input = os.path.abspath(input_path)

    print(f"[INFO] Input file: {abs_input}")
    if inplace:
        print("[INFO] Output path not specified, modifying in place.")
    else:
        print(f"[INFO] Output file: {os.path.abspath(output_path)}")

    print("[INFO] Counting total lines...")
    t0 = time.time()
    total = count_lines(abs_input)
    print(f"[INFO] Total lines: {total:,}  (Elapsed {time.time() - t0:.1f}s)")

    if inplace:
        dir_name = os.path.dirname(abs_input)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir=dir_name)
        os.close(tmp_fd)
        final_output = tmp_path
    else:
        output_abs = os.path.abspath(output_path)
        out_dir = os.path.dirname(output_abs)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        final_output = output_path

    weight_keys = list(weights.keys())
    weight_vals = [weights[k] for k in weight_keys]
    missing_warned = False
    error_count = 0

    print("[INFO] Starting processing...")
    t1 = time.time()

    with open(abs_input, "rb") as fin, open(final_output, "wb") as fout:
        for line in tqdm(fin, total=total, desc="Progress", unit="lines", mininterval=0.5):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                error_count += 1
                fout.write(line + b"\n")
                continue

            scores = obj.get(score_field)
            if scores is not None:
                weighted_sum = 0.0
                for k, w in zip(weight_keys, weight_vals):
                    v = scores.get(k)
                    if v is not None:
                        weighted_sum += float(v) * w
                    elif not missing_warned:
                        print(f"[WARN] Missing field: {k}. This and future missing fields will be treated as 0 (this message appears only once).")
                        missing_warned = True
                obj["final_score"] = weighted_sum

            fout.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")

    elapsed = time.time() - t1
    speed = total / elapsed if elapsed > 0 else float("inf")
    print(f"[INFO] Processing complete! Elapsed: {elapsed:.1f}s, Speed: {speed:,.0f} lines/s")

    if error_count > 0:
        print(f"[WARN] {error_count} lines failed to parse and were kept as-is.")

    if inplace:
        shutil.move(tmp_path, abs_input)
        print(f"[INFO] Updated in place: {abs_input}")
    else:
        print(f"[INFO] Output written to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Compute weighted sum of the 'scores' field in JSONL and write to 'final_score' field")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--weights", "-w", required=True, help="Weights file path (e.g., best_weights.json)")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL file path (if not specified, modifies in place)")
    parser.add_argument("--score_field", "-s", default="scores", help="Field name containing scores (default: 'scores')")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.weights):
        print(f"[ERROR] Weights file does not exist: {args.weights}")
        sys.exit(1)

    weights = load_weights(args.weights)
    print(f"[INFO] Loaded {len(weights)} weights: {list(weights.keys())}")

    process(args.input, args.output, weights=weights, score_field=args.score_field)


if __name__ == "__main__":
    main()
