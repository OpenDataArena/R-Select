#!/usr/bin/env python3
"""
Precompute the score index of a data pool to accelerate Optuna optimization.

For 3.3M records:
- The original JSONL may exceed 10+ GB
- The index file contains only the scores, about 200-500 MB
- Loading speed is improved by 10-50x

Usage:
    # 1. Precompute the index (run once only)
    python precompute_index.py \
        --pool_jsonl /path/to/large_pool.jsonl \
        --output_index /path/to/pool_index.pkl \
        --score_field scores

    # 2. Use the index to run Optuna
    srun -p raise python precompute_index.py --pool_jsonl /mnt/dhwfile/raise/data-leaderboard/data/gaoxin/BUDA_ODA/data/merged_data_refined.jsonl --output_index /mnt/dhwfile/raise/data-leaderboard/data/gaoxin/BUDA_ODA/data/kdd26_merged_data_refined_pool_index.pkl
"""

import argparse
import json
import pickle
import os
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm


def count_lines(filepath: str) -> int:
    """Quickly count the number of lines in a file"""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


def extract_scores(item: Dict, score_field: str) -> Dict[str, float]:
    """Extract scores from a data item"""
    scores = item.get(score_field, None)
    if isinstance(scores, dict):
        return scores
    source = item.get("source", None)
    if isinstance(source, dict):
        return source
    raise ValueError(f"Cannot find score dict in field '{score_field}' or 'source'.")


def build_index(
    pool_jsonl: str,
    score_field: str,
    score_keys: List[str] = None,
) -> Tuple[List[Tuple[List[float], int]], List[str], Dict]:
    """
    Build the score index for the data pool.

    Returns:
        index: List of (score_vec, line_offset) - score vector and file offset for each data entry
        score_keys: list of score dimension names
        metadata: metadata (such as the total line count)
    """
    print(f"Counting lines in file...")
    total_lines = count_lines(pool_jsonl)
    print(f"Total lines in file: {total_lines:,}")
    
    index = []
    inferred_keys = None
    
    print(f"Building the index...")
    with open(pool_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Building index")):
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            scores = extract_scores(item, score_field)
            
            # Infer score_keys from the first line if not given
            if inferred_keys is None:
                inferred_keys = sorted(scores.keys()) if score_keys is None else score_keys
                print(f"Found {len(inferred_keys)} score dimensions: {inferred_keys[:5]}...")
            
            # Build score vector
            score_vec = [float(scores.get(k, 0.0)) for k in inferred_keys]
            index.append((score_vec, line_idx))
    
    metadata = {
        "pool_jsonl": os.path.abspath(pool_jsonl),
        "total_lines": total_lines,
        "indexed_lines": len(index),
        "score_field": score_field,
    }
    
    return index, inferred_keys, metadata


def save_index(
    index: List[Tuple[List[float], int]],
    score_keys: List[str],
    metadata: Dict,
    output_path: str,
) -> None:
    """Save the index to a file"""
    data = {
        "index": index,
        "score_keys": score_keys,
        "metadata": metadata,
        "version": "1.0",
    }
    
    print(f"Saving index to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Index file size: {file_size:.1f} MB")


def load_index(index_path: str) -> Tuple[List[Tuple[List[float], int]], List[str], Dict]:
    """Load index file"""
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["score_keys"], data["metadata"]


def main():
    parser = argparse.ArgumentParser(description="Precompute data pool score index")
    parser.add_argument("--pool_jsonl", required=True, help="Original data pool JSONL file")
    parser.add_argument("--output_index", required=True, help="Output index file path (.pkl)")
    parser.add_argument("--score_field", default="scores", help="Score field name")
    parser.add_argument("--score_keys", default="", help="Specify score keys (comma separated), if empty auto-infer")
    args = parser.parse_args()
    
    score_keys = [k.strip() for k in args.score_keys.split(",") if k.strip()] if args.score_keys else None
    
    # Build index
    index, inferred_keys, metadata = build_index(
        args.pool_jsonl,
        args.score_field,
        score_keys,
    )
    
    # Save index
    save_index(index, inferred_keys, metadata, args.output_index)
    
    print(f"\n✅ Index building complete!")
    print(f"   Total data items: {metadata['indexed_lines']:,}")
    print(f"   Score dimensions: {len(inferred_keys)}")
    print(f"   Index file: {args.output_index}")
    print(f"\nUsage example:")
    print(f"   python optuna_sft_selector.py \\")
    print(f"       --pool_jsonl {args.pool_jsonl} \\")
    print(f"       --pool_index {args.output_index} \\")
    print(f"       ...")


if __name__ == "__main__":
    main()
