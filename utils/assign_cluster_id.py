"""
Assign cluster_id from cluster_labels.npy to each line of a JSONL file.

Usage:
    python utils/assign_cluster_id.py \
        --input_jsonl data/all.jsonl \
        --labels_path data/cluster_labels.npy \
        --output_jsonl data/all_with_cluster.jsonl
"""

import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

try:
    import orjson

    def loads(s):
        return orjson.loads(s)

    def dumps(obj):
        return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE)

    _USE_ORJSON = True
except ImportError:
    import json

    def loads(s):
        return json.loads(s)

    def dumps(obj):
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    _USE_ORJSON = False


def count_lines(filepath: str) -> int:
    """Fast line count using buffered raw read."""
    count = 0
    buf_size = 1 << 20  # 1 MB
    with open(filepath, "rb") as f:
        buf = f.raw.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = f.raw.read(buf_size)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign cluster_id to each record in a JSONL file."
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--labels_path",
        required=True,
        help="Path to the cluster_labels.npy file.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Path to the output JSONL file. Defaults to overwriting the input file in-place.",
    )
    parser.add_argument(
        "--write_buf_size",
        type=int,
        default=64 * 1024 * 1024,
        help="Write buffer size in bytes (default: 64MB).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    if args.output_jsonl is None:
        args.output_jsonl = args.input_jsonl

    print(f"JSON backend: {'orjson (fast)' if _USE_ORJSON else 'stdlib json'}")
    print(f"Input JSONL : {args.input_jsonl}")
    print(f"Labels file : {args.labels_path}")
    print(f"Output JSONL: {args.output_jsonl}")

    # --- Load cluster labels ---
    print("\n[1/3] Loading cluster labels...")
    labels = np.load(args.labels_path)
    num_labels = len(labels)
    print(f"      Loaded {num_labels:,} labels  (dtype={labels.dtype}, "
          f"unique clusters={len(np.unique(labels))})")

    # --- Count lines for progress bar ---
    print("\n[2/3] Counting lines in input JSONL...")
    num_lines = count_lines(args.input_jsonl)
    print(f"      Total lines: {num_lines:,}")

    if num_lines != num_labels:
        print(f"\n[ERROR] Line count ({num_lines:,}) != label count ({num_labels:,}). Aborting.")
        sys.exit(1)
    print(f"      ✓ Line count matches label count.")

    # --- Process ---
    print(f"\n[3/3] Writing output with cluster_id...")
    tmp_output = args.output_jsonl + ".tmp"
    processed = 0
    errors = 0

    with open(args.input_jsonl, "rb") as fin, \
         open(tmp_output, "wb", buffering=args.write_buf_size) as fout:
        for idx, raw_line in enumerate(tqdm(fin, total=num_lines, unit=" lines",
                                            dynamic_ncols=True, smoothing=0.1)):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = loads(raw_line)
                record["cluster_id"] = int(labels[idx])
                fout.write(dumps(record))
                processed += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n      [WARN] Line {idx}: {e}")

    os.replace(tmp_output, args.output_jsonl)

    elapsed = time.time() - start_time
    speed = processed / elapsed if elapsed > 0 else 0
    print(f"\n--- Done ---")
    print(f"Processed : {processed:,} records")
    if errors > 0:
        print(f"Errors    : {errors:,}")
    print(f"Output    : {args.output_jsonl}")
    print(f"Time      : {elapsed:.1f}s ({speed:,.0f} lines/s)")


if __name__ == "__main__":
    main()
