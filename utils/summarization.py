#!/usr/bin/env python3
"""
Aggregate multi-layer weights and output:
1. hierarchy: Layer hierarchy (which child clusters/scores are in each layer's cluster)
2. weights_by_cluster: Raw weights of each cluster (e.g., Layer2 to Layer1, Layer1 to leaf scores)
3. final_leaf_weights: Final leaf score weight proportions after top-down propagation

Usage:
    python summarization.py -i results/demo -o results/demo/weight_summary.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set


LAYER_CLUSTER_PATTERN = re.compile(r"^Layer\d+_c\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-layer weights and output the hierarchy structure and weights for each layer"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Root directory, e.g. results/demo, which contains Layer1/, Layer2/, ...",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file path",
    )
    return parser.parse_args()


def discover_layers(root: str) -> Dict[str, Dict[str, float]]:
    """
    Scan all Layer*/*/best_weights.json files under the directory.
    
    Returns:
        cluster_weights: { "Layer1_c1": {"AtheneScore": 0.4, ...}, "Layer2_c1": {"Layer1_c1": 0.12, ...}, ... }
    """
    cluster_weights: Dict[str, Dict[str, float]] = {}
    root_path = Path(root)

    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    for layer_dir in sorted(root_path.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("Layer"):
            continue
        for cluster_dir in sorted(layer_dir.iterdir()):
            if not cluster_dir.is_dir():
                continue
            best_path = cluster_dir / "best_weights.json"
            if not best_path.is_file():
                continue
            cluster_name = cluster_dir.name
            if not LAYER_CLUSTER_PATTERN.match(cluster_name):
                continue
            try:
                with open(best_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                weights = data.get("best_weights", data.get("weights", {}))
                if isinstance(weights, dict):
                    cluster_weights[cluster_name] = {k: float(v) for k, v in weights.items()}
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[Warning] Skipped invalid file {best_path}: {e}")

    return cluster_weights


def get_layer_number(name: str) -> int:
    m = re.match(r"Layer(\d+)_c\d+", name)
    return int(m.group(1)) if m else 0


def build_hierarchy(cluster_weights: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build the hierarchy: for each layer, specify which clusters or leaf scores each cluster contains.
    
    Returns:
        { "Layer2": { "Layer2_c1": ["Layer1_c1", "Layer1_c2", ...] },
          "Layer1": { "Layer1_c1": ["AtheneScore", "CleanlinessScore", ...], ... } }
    """
    hierarchy: Dict[str, Dict[str, List[str]]] = {}

    for cluster_name, weights in cluster_weights.items():
        layer_num = get_layer_number(cluster_name)
        layer_key = f"Layer{layer_num}"
        if layer_key not in hierarchy:
            hierarchy[layer_key] = {}

        children = sorted(weights.keys())
        hierarchy[layer_key][cluster_name] = children

    # Sort layers in order
    layers = sorted(hierarchy.keys(), key=lambda x: int(re.search(r"\d+", x).group()))
    return {k: hierarchy[k] for k in layers}


def compute_leaf_weights(
    cluster_name: str,
    cluster_weights: Dict[str, Dict[str, float]],
    cache: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Recursively compute the actual leaf score weights for a cluster (by top-down propagation)."""
    if cluster_name in cache:
        return cache[cluster_name]

    weights = cluster_weights.get(cluster_name, {})
    leaf: Dict[str, float] = {}

    for key, w in weights.items():
        if LAYER_CLUSTER_PATTERN.match(key):
            sub_leaf = compute_leaf_weights(key, cluster_weights, cache)
            for s, v in sub_leaf.items():
                leaf[s] = leaf.get(s, 0.0) + w * v
        else:
            leaf[key] = leaf.get(key, 0.0) + w

    cache[cluster_name] = leaf
    return leaf


def run(root: str, output_path: str) -> None:
    print(f"[Summary] Scanning directory: {root}")
    cluster_weights = discover_layers(root)

    if not cluster_weights:
        print("[Error] No best_weights.json file found")
        return

    layers = sorted(set(get_layer_number(c) for c in cluster_weights))
    print(f"[Summary] Discovered {len(cluster_weights)} clusters, hierarchy: Layer{layers[0]} ~ Layer{layers[-1]}")

    # 1. hierarchy: Layer split
    hierarchy = build_hierarchy(cluster_weights)

    # 2. weights_by_cluster: original weights of each cluster (from best_weights.json)
    weights_by_cluster: Dict[str, Dict[str, float]] = {}
    for cluster_name in sorted(cluster_weights.keys()):
        weights_by_cluster[cluster_name] = dict(sorted(cluster_weights[cluster_name].items()))

    # 3. final_leaf_weights: Aggregate top-layer clusters' propagated leaf weights
    top_layer_num = max(layers)
    top_clusters = [c for c in cluster_weights if get_layer_number(c) == top_layer_num]

    cache: Dict[str, Dict[str, float]] = {}
    top_leaf_weights: Dict[str, Dict[str, float]] = {}
    for cluster in sorted(top_clusters):
        top_leaf_weights[cluster] = compute_leaf_weights(cluster, cluster_weights, cache)

    all_leaf_keys = sorted(set().union(*(v.keys() for v in top_leaf_weights.values())))
    final_leaf_weights: Dict[str, float] = {}
    for key in all_leaf_keys:
        vals = [top_leaf_weights[c].get(key, 0.0) for c in top_clusters]
        final_leaf_weights[key] = sum(vals) / len(vals) if vals else 0.0

    result = {
        "hierarchy": hierarchy,
        "weights_by_cluster": weights_by_cluster,
        "final_leaf_weights": final_leaf_weights,
        "meta": {
            "root": os.path.abspath(root),
        },
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Summary] Saved to: {output_path}")
    print(f"[Summary] Number of layers: {len(hierarchy)}, total leaf scores: {len(all_leaf_keys)}")


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    run(root, output_path)


if __name__ == "__main__":
    main()
