#!/usr/bin/env python3
"""
Select the top k% of data by score and save to a new file. Supports two modes:
  --per_cluster: select the top k% within each cluster (grouped by cluster_id)
  --global: select the top k% across all data (regardless of cluster)

Two-pass scan, memory efficient. Usage:
    python sampling.py -i data/all.jsonl -o data/sampled.jsonl -k 10 --per_cluster
    python sampling.py -i data/all.jsonl -o data/sampled.jsonl -k 10 --global
"""

import argparse
import json
import math
import os
import sys
import time

from tqdm import tqdm


def count_lines(filepath: str) -> int:
    count = 0
    with open(filepath, "rb") as f:
        buf_size = 1 << 20
        buf = f.raw.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = f.raw.read(buf_size)
    return count


def _run_per_cluster(abs_input: str, abs_output: str, total: int, ratio: float, score_field: str):
    """Select the top k% within each cluster"""
    # First pass: collect the score list of each cluster
    print("[INFO] First pass: collecting score distribution for each cluster...")
    cluster_scores: dict[int, list[float]] = {}
    skip_count = 0
    t1 = time.time()

    with open(abs_input, "rb") as f:
        for line in tqdm(f, total=total, desc="Scanning", unit="lines", mininterval=0.5):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("cluster_id")
            score = obj.get(score_field)
            if cid is None or score is None:
                skip_count += 1
                continue
            cluster_scores.setdefault(cid, []).append(float(score))

    num_clusters = len(cluster_scores)
    total_valid = sum(len(v) for v in cluster_scores.values())
    print(f"[INFO] Valid data: {total_valid:,} lines, {num_clusters} clusters (elapsed {time.time() - t1:.1f}s)")
    if skip_count:
        print(f"[WARN] Skipped {skip_count:,} lines missing cluster_id or {score_field}")

    # Compute threshold and quota for each cluster
    print("[INFO] Calculating score threshold for each cluster...")
    cluster_threshold: dict[int, float] = {}
    cluster_quota: dict[int, int] = {}
    total_selected = 0

    for cid, scores in cluster_scores.items():
        scores.sort(reverse=True)
        n_keep = max(1, math.ceil(len(scores) * ratio))
        total_selected += n_keep
        cluster_threshold[cid] = scores[n_keep - 1]
        cluster_quota[cid] = n_keep

    del cluster_scores

    print(f"[INFO] Expected number of selected samples: {total_selected:,} ({total_selected / total_valid * 100:.2f}% of valid data)")

    sorted_clusters = sorted(cluster_quota.items())
    preview_n = min(20, len(sorted_clusters))
    print(f"[INFO] Preview of top {preview_n} clusters and their quotas:")
    for i, (cid, quota) in enumerate(sorted_clusters):
        if i >= preview_n:
            print(f"       ... total {num_clusters} clusters, omitted others")
            break
        print(f"       cluster {cid}: keep {quota} lines, threshold = {cluster_threshold[cid]:.6f}")

    # Second pass: stream filter and write output
    print(f"\n[INFO] Second pass: filtering and writing output...")
    t2 = time.time()
    out_dir = os.path.dirname(abs_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cluster_written: dict[int, int] = {}
    written = 0

    with open(abs_input, "rb") as fin, open(abs_output, "wb") as fout:
        for line in tqdm(fin, total=total, desc="Filtering", unit="lines", mininterval=0.5):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("cluster_id")
            score = obj.get(score_field)
            if cid is None or score is None:
                continue

            thr = cluster_threshold.get(cid)
            quota = cluster_quota.get(cid, 0)
            already = cluster_written.get(cid, 0)

            if thr is not None and float(score) >= thr and already < quota:
                fout.write(line + b"\n")
                cluster_written[cid] = already + 1
                written += 1

    elapsed = time.time() - t2
    speed = total / elapsed if elapsed > 0 else float("inf")
    print(f"[INFO] Selection complete! {written:,} lines written (elapsed {elapsed:.1f}s, {speed:,.0f} lines/s)")
    print(f"[INFO] Output saved to: {abs_output}")

    under_filled = [(cid, cluster_written.get(cid, 0), cluster_quota[cid])
                    for cid in cluster_quota if cluster_written.get(cid, 0) < cluster_quota[cid]]
    if under_filled:
        print(f"[WARN] {len(under_filled)} clusters did not reach their expected quota (possibly because data with identical scores appeared further down in the file)")


def _run_global(abs_input: str, abs_output: str, total: int, ratio: float, score_field: str):
    """Select the top k% of all data (not separated by cluster)"""
    # First pass: collect all scores
    print("[INFO] First pass: collecting score distribution...")
    all_scores: list[float] = []
    skip_count = 0
    t1 = time.time()

    with open(abs_input, "rb") as f:
        for line in tqdm(f, total=total, desc="Scanning", unit="lines", mininterval=0.5):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = obj.get(score_field)
            if score is None:
                skip_count += 1
                continue
            all_scores.append(float(score))

    total_valid = len(all_scores)
    print(f"[INFO] Valid data: {total_valid:,} lines (elapsed {time.time() - t1:.1f}s)")
    if skip_count:
        print(f"[WARN] Skipped {skip_count:,} lines missing {score_field}")

    # Compute global threshold
    all_scores.sort(reverse=True)
    n_keep = max(1, math.ceil(total_valid * ratio))
    global_threshold = all_scores[n_keep - 1]
    total_selected = n_keep
    del all_scores

    print(f"[INFO] Expected number of selected samples: {total_selected:,} (top {ratio*100:.2f}%)")
    print(f"[INFO] Global threshold: {global_threshold:.6f}")

    # Second pass: stream filter and write output (use quota to handle duplicate scores)
    print(f"\n[INFO] Second pass: filtering and writing output...")
    t2 = time.time()
    out_dir = os.path.dirname(abs_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = 0

    with open(abs_input, "rb") as fin, open(abs_output, "wb") as fout:
        for line in tqdm(fin, total=total, desc="Filtering", unit="lines", mininterval=0.5):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = obj.get(score_field)
            if score is None:
                continue
            if float(score) >= global_threshold and written < total_selected:
                fout.write(line + b"\n")
                written += 1

    elapsed = time.time() - t2
    speed = total / elapsed if elapsed > 0 else float("inf")
    print(f"[INFO] Selection complete! {written:,} lines written (elapsed {elapsed:.1f}s, {speed:,.0f} lines/s)")
    print(f"[INFO] Output saved to: {abs_output}")


def main():
    parser = argparse.ArgumentParser(description="Select the top k%% of data by score")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--top_k", "-k", type=float, required=True,
                        help="Keep the top k%% with the highest score (e.g. 10 means top 10%%)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--per_cluster", action="store_true",
                       help="Select top k%% within each cluster (grouped by cluster_id)")
    group.add_argument("--global", dest="global_mode", action="store_true",
                       help="Select top k%% among all data (no cluster distinction)")
    parser.add_argument("--score_field", "-s", default="final_score",
                        help="Field containing the score (default: score or final_score)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)
    if not 0 < args.top_k <= 100:
        print(f"[ERROR] top_k should be in (0, 100], current value: {args.top_k}")
        sys.exit(1)

    abs_input = os.path.abspath(args.input)
    abs_output = os.path.abspath(args.output)
    ratio = args.top_k / 100.0
    score_field = args.score_field
    per_cluster = args.per_cluster

    print(f"[INFO] Input file: {abs_input}")
    print(f"[INFO] Output file: {abs_output}")
    print(f"[INFO] Selection ratio: top {args.top_k}%")
    print(f"[INFO] Mode: {'Select top k% within each cluster' if per_cluster else 'Select top k% in all data'}")

    # Count total lines
    print("[INFO] Counting total number of lines...")
    t0 = time.time()
    total = count_lines(abs_input)
    print(f"[INFO] Total lines: {total:,} (elapsed {time.time() - t0:.1f}s)")

    if per_cluster:
        _run_per_cluster(abs_input, abs_output, total, ratio, score_field)
    else:
        _run_global(abs_input, abs_output, total, ratio, score_field)


if __name__ == "__main__":
    main()

