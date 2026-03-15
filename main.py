#!/usr/bin/env python3
import argparse
import copy
import heapq
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import optuna
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna-based SFT data selection with LlamaFactory proxy training."
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args()


_CONFIG_DEFAULTS = {
    "score_field": "scores",
    "seed": 42,
    "num_train_epochs": 1.0,
    "eval_strategy": "epoch",
    "per_device_eval_batch_size": 0,
    "nproc_per_node": 1,
    "master_port": 0,
    "cuda_visible_devices": "",
    "llama_factory_dir": "",
    "max_trials_in_dir": 1000000,
    "early_stop_patience": 0,
    "early_stop_delta": 0.0,
    "cache_pool_in_memory": False,
    "pool_index": "",
    "warmup_trials_json": "",
    "enqueue_weights_json": "",
    "running_timeout_hours": 24.0,
    "n_startup_trials": 0,
    "normalization_method": "softmax",
    "suggest_range": "-5,5",
    "tpe_multivariate": False,
    "tpe_n_ei_candidates": 500,
    "tpe_gamma_percent": 0.1,
    "tpe_gamma_max": 25,
}

_CONFIG_REQUIRED = [
    "pool_jsonl", "val_jsonl", "top_k", "base_train_yaml",
    "run_dir", "study_name", "storage", "n_trials", "score_keys",
]

_CONFIG_TYPES: Dict[str, type] = {
    "top_k": int,
    "seed": int,
    "num_train_epochs": float,
    "per_device_eval_batch_size": int,
    "nproc_per_node": int,
    "master_port": int,
    "max_trials_in_dir": int,
    "early_stop_patience": int,
    "early_stop_delta": float,
    "n_trials": int,
    "running_timeout_hours": float,
    "n_startup_trials": int,
    "tpe_n_ei_candidates": int,
    "tpe_gamma_percent": float,
    "tpe_gamma_max": int,
    "cache_pool_in_memory": bool,
    "tpe_multivariate": bool,
}


def load_config(config_path: str) -> argparse.Namespace:
    """Load configuration from a YAML file, apply defaults and type coercion."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    missing = [k for k in _CONFIG_REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}: {missing}")

    merged = {**_CONFIG_DEFAULTS, **cfg}

    for key, target_type in _CONFIG_TYPES.items():
        if key in merged:
            merged[key] = target_type(merged[key])

    if merged["normalization_method"] not in ("softmax", "linear"):
        raise ValueError(
            f"normalization_method must be 'softmax' or 'linear', got '{merged['normalization_method']}'"
        )

    # score_keys: support YAML list or comma-separated string (required, non-empty)
    sk = merged["score_keys"]
    if isinstance(sk, list):
        merged["score_keys"] = ", ".join(str(k).strip() for k in sk if str(k).strip())
    else:
        merged["score_keys"] = str(sk).strip() if sk else ""
    if not merged["score_keys"]:
        raise ValueError(
            f"score_keys is required and cannot be empty in {config_path}. "
            "Provide a non-empty list or comma-separated string."
        )

    # suggest_range: support YAML list [min, max] or string "min,max"
    sr = merged["suggest_range"]
    if isinstance(sr, list) and len(sr) == 2:
        merged["suggest_range"] = f"{sr[0]},{sr[1]}"
    else:
        merged["suggest_range"] = str(sr)

    return argparse.Namespace(**merged)


def set_global_seed(seed: int) -> None:
    random.seed(seed)


def parse_suggest_range(range_str: str) -> Tuple[float, float]:
    """
    Parse suggest_range string to (min, max) tuple.
    
    Args:
        range_str: Format 'min,max', e.g., '-5,5' or '-10,10'
    
    Returns:
        (min_val, max_val) tuple
    """
    parts = range_str.strip().split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid suggest_range format: '{range_str}'. Expected 'min,max' (e.g., '-5,5').")
    try:
        min_val = float(parts[0].strip())
        max_val = float(parts[1].strip())
    except ValueError:
        raise ValueError(f"Invalid suggest_range values: '{range_str}'. Expected numeric values.")
    if min_val >= max_val:
        raise ValueError(f"Invalid suggest_range: min ({min_val}) must be less than max ({max_val}).")
    return min_val, max_val


def load_jsonl_first(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty jsonl file: {path}")


def extract_scores(item: Dict, score_field: str) -> Dict[str, float]:
    scores = item.get(score_field, None)
    if isinstance(scores, dict):
        return scores
    source = item.get("source", None)
    if isinstance(source, dict):
        return source
    raise ValueError(f"Cannot find score dict in field '{score_field}' or 'source'.")


def infer_score_keys(pool_jsonl: str, score_field: str, user_keys: str) -> List[str]:
    if user_keys.strip():
        return [k.strip() for k in user_keys.split(",") if k.strip()]
    first = load_jsonl_first(pool_jsonl)
    scores = extract_scores(first, score_field)
    return sorted(scores.keys())


def softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [pow(2.718281828459045, x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def linear_normalize(xs: List[float], min_val: float = -5.0, epsilon: float = 1e-8) -> List[float]:
    """
    Linear normalization with min-shift.
    
    Formula: w_i = (x_i - min_val + ε) / Σ(x_j - min_val + ε)
    
    Args:
        xs: List of values to normalize
        min_val: The search lower bound (typically the lower bound of suggest_range)
        epsilon: Small value to prevent zero weights
    
    Returns:
        Normalized weights that sum to 1
    """
    if not xs:
        return []
    shifted = [x - min_val + epsilon for x in xs]
    s = sum(shifted)
    return [v / s for v in shifted]


def normalize_weights(
    xs: List[float], 
    method: str = "softmax", 
    min_val: float = -5.0,
    epsilon: float = 1e-8,
) -> List[float]:
    """
    Normalize weights using the specified method.
    
    Args:
        xs: List of z values to normalize
        method: 'softmax' or 'linear'
        min_val: The search lower bound (used for linear normalization)
        epsilon: Small value to prevent zero weights (used for linear normalization)
    
    Returns:
        Normalized weights that sum to 1
    """
    if method == "softmax":
        return softmax(xs)
    elif method == "linear":
        return linear_normalize(xs, min_val=min_val, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def iter_pool_items(pool_jsonl: str) -> Iterable[Dict]:
    with open(pool_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_weighted_score_from_vec(score_vec: List[float], weights: List[float]) -> float:
    total = 0.0
    for s, w in zip(score_vec, weights):
        total += w * s
    return total


def compute_score_vec(scores: Dict[str, float], keys: List[str]) -> List[float]:
    vec = []
    for k in keys:
        if k not in scores:
            raise ValueError(f"Missing score key '{k}' in item scores.")
        vec.append(float(scores[k]))
    return vec


def select_top_k(
    pool_jsonl: str,
    score_field: str,
    keys: List[str],
    weights: List[float],
    top_k: int,
    pool_cache: List[Tuple[List[float], Dict]] = None,
) -> List[Dict]:
    t0 = time.time()
    heap: List[Tuple[float, int, Dict]] = []
    if pool_cache is None:
        iterable = ((compute_score_vec(extract_scores(item, score_field), keys), item) for item in iter_pool_items(pool_jsonl))
        source = "file"
    else:
        iterable = pool_cache
        source = "cache"
    for idx, (score_vec, item) in enumerate(iterable):
        score = compute_weighted_score_from_vec(score_vec, weights)
        if len(heap) < top_k:
            heapq.heappush(heap, (score, idx, item))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, idx, item))
    heap.sort(key=lambda x: (-x[0], x[1]))
    selected = []
    for score, _, item in heap:
        new_item = dict(item)
        new_item["_weighted_score"] = score
        selected.append(new_item)
    elapsed = time.time() - t0
    print(f"[TopK] select_top_k(source={source}): {elapsed:.2f}s")
    return selected


def load_pool_index(index_path: str) -> Tuple[List[Tuple[List[float], int]], List[str], Dict]:
    """
    Load precomputed pool index.

    Returns:
        index: List of (score_vec, line_idx) - each item's score vector and line number.
        score_keys: List of score dimension names.
        metadata: Metadata (may include byte_offsets).
    """
    import pickle
    t0 = time.time()
    print(f"[Index] Loading precomputed index from {index_path}...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    
    index = data["index"]
    score_keys = data["score_keys"]
    metadata = data["metadata"]
    
    # Check if byte_offsets exist (for fast random access)
    has_offsets = "byte_offsets" in data
    elapsed = time.time() - t0
    print(f"[Index] Loaded {len(index):,} items, {len(score_keys)} score dimensions, byte_offsets: {has_offsets} ({elapsed:.2f}s)")
    return index, score_keys, metadata


def build_byte_offset_index(pool_jsonl: str) -> List[int]:
    """
    Build byte offset index for a JSONL file for O(1) random access.

    Returns: List[int], offsets[i] = starting byte position of line i.
    """
    t0 = time.time()
    print(f"[Index] Building byte offset index for {pool_jsonl}...")
    offsets = []
    with open(pool_jsonl, "rb") as f:  # Binary mode for accurate byte positions
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(offset)
    elapsed = time.time() - t0
    print(f"[Index] Built byte offset index: {len(offsets):,} lines ({elapsed:.2f}s)")
    return offsets


def read_lines_by_offsets(
    pool_jsonl: str,
    byte_offsets: List[int],
    target_line_indices: List[int],
) -> Dict[int, str]:
    """
    Quickly read specified lines using byte offsets, O(k) complexity instead of O(n).

    Args:
        pool_jsonl: Path to the JSONL file.
        byte_offsets: Byte offset index.
        target_line_indices: List of line numbers to read.

    Returns:
        Dict[line_idx, line_content]
    """
    result = {}
    with open(pool_jsonl, "rb") as f:
        for line_idx in target_line_indices:
            f.seek(byte_offsets[line_idx])
            line = f.readline().decode("utf-8").strip()
            if line:
                result[line_idx] = line
    return result


def select_top_k_with_index(
    pool_jsonl: str,
    pool_index: List[Tuple[List[float], int]],
    weights: List[float],
    top_k: int,
    byte_offsets: List[int] = None,
    score_matrix = None,  # numpy.ndarray, optional
    line_indices = None,  # numpy.ndarray, optional
) -> List[Dict]:
    """
    Quickly select Top-K data using a precomputed index.

    1. Use the index (only score_vec) to compute weighted scores, select top-k line numbers.
    2. Only read the full data for those lines.

    If score_matrix/line_indices (NumPy) are provided, Step 1 is vectorized (very fast).
    If byte_offsets is provided, Step 2 uses O(k) random access.
    Otherwise Step 2 is O(n) sequential scan (slower, but compatible with older index files).
    """
    # Step 1: Compute top-k line numbers
    t1 = time.time()
    if score_matrix is not None and line_indices is not None:
        # 🚀 NumPy vectorized path (very fast: 3.3M items < 0.5 sec)
        import numpy as np
        weights_arr = np.array(weights, dtype=np.float32)
        scores = score_matrix @ weights_arr  # Matrix multiplication, scores for all rows
        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]  # O(n) find top-k
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]  # sort by descending score

        target_line_indices = line_indices[top_k_idx].tolist()
        line_idx_to_score = {int(line_indices[i]): float(scores[i]) for i in top_k_idx}
        step1_method = "numpy"
    else:
        # 🐢 Pure Python path (slower, compatible with old index)
        heap: List[Tuple[float, int]] = []  # (score, line_idx)

        for score_vec, line_idx in pool_index:
            score = compute_weighted_score_from_vec(score_vec, weights)
            if len(heap) < top_k:
                heapq.heappush(heap, (score, line_idx))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, line_idx))

        # Get top-k line numbers and their scores
        top_k_items = [(score, line_idx) for score, line_idx in heap]
        top_k_items.sort(key=lambda x: -x[0])  # descending score

        line_idx_to_score = {line_idx: score for score, line_idx in top_k_items}
        target_line_indices = list(line_idx_to_score.keys())
        step1_method = "python"
    t1_elapsed = time.time() - t1

    # Step 2: Read selected lines
    t2 = time.time()
    if byte_offsets is not None:
        # 🚀 Fast path: O(k) random access
        lines_map = read_lines_by_offsets(pool_jsonl, byte_offsets, target_line_indices)
        selected = []
        for line_idx in target_line_indices:
            line = lines_map.get(line_idx)
            if line:
                item = json.loads(line)
                item["_weighted_score"] = line_idx_to_score[line_idx]
                selected.append(item)
        step2_method = "random_access"
    else:
        # 🐢 Slow path: O(n) sequential scan (compatible with old index)
        target_set = set(target_line_indices)
        selected = []
        with open(pool_jsonl, "r", encoding="utf-8") as f:
            for current_line_idx, line in enumerate(f):
                if current_line_idx in target_set:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        item["_weighted_score"] = line_idx_to_score[current_line_idx]
                        selected.append(item)

                        # Early stop: if all target lines have been found
                        if len(selected) >= top_k:
                            break
        step2_method = "sequential_scan"
    t2_elapsed = time.time() - t2

    print(f"[TopK] Step1({step1_method}): {t1_elapsed:.2f}s, Step2({step2_method}): {t2_elapsed:.2f}s, total: {t1_elapsed + t2_elapsed:.2f}s")

    # Sort by score
    selected.sort(key=lambda x: -x["_weighted_score"])
    return selected


def write_jsonl(path: str, items: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_dataset_info(
    dataset_dir: str,
    train_name: str,
    train_file: str,
    val_name: str,
    val_file: str,
) -> None:
    info = {
        train_name: {
            "file_name": train_file,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        val_name: {
            "file_name": val_file,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
    }
    with open(os.path.join(dataset_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def find_free_port() -> int:
    """Find an available TCP port on localhost for torchrun rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def build_train_yaml(
    base_yaml_path: str,
    out_yaml_path: str,
    base_cfg: Dict,
    dataset_dir: str,
    train_dataset: str,
    eval_dataset: str,
    output_dir: str,
    num_train_epochs: float,
    eval_strategy: str,
    per_device_eval_batch_size: int,
) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset_dir"] = dataset_dir
    cfg["dataset"] = train_dataset
    cfg["eval_dataset"] = eval_dataset
    cfg["do_train"] = True
    cfg["do_eval"] = True
    cfg["eval_strategy"] = eval_strategy
    cfg["num_train_epochs"] = float(num_train_epochs)
    cfg["output_dir"] = output_dir
    cfg["overwrite_output_dir"] = True
    # Proxy training: don't save model checkpoints, only need eval_loss
    cfg["save_strategy"] = "no"
    cfg["save_only_model"] = False
    if per_device_eval_batch_size > 0:
        cfg["per_device_eval_batch_size"] = per_device_eval_batch_size
    else:
        if "per_device_eval_batch_size" not in cfg and "per_device_train_batch_size" in cfg:
            cfg["per_device_eval_batch_size"] = cfg["per_device_train_batch_size"]
    save_yaml(out_yaml_path, cfg)


def get_trial_hf_cache_dir(trial_id: int, timestamp: str = None) -> str:
    """
    Get the HuggingFace datasets cache directory for a specific trial.
    
    Args:
        trial_id: Trial number
        timestamp: Optional timestamp string. If None, uses current time with microsecond precision.
    
    Returns:
        Cache directory path: {HF_HOME}/datasets/trial_{trial_id}_{timestamp}/
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    if timestamp is None:
        # Use microsecond precision to avoid conflicts when multiple trials start in the same second
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(hf_home, "datasets", f"trial_{trial_id}_{timestamp}")


def cleanup_trial_hf_cache(trial_id: int, cache_dir: str = None) -> None:
    """
    Clean up HuggingFace datasets cache for a specific trial.
    Each trial uses its own cache directory to avoid conflicts and allow precise cleanup.
    
    Args:
        trial_id: Trial number
        cache_dir: Specific cache directory to clean. If None, searches for all matching directories.
    """
    if cache_dir is not None:
        # Clean specific directory
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"[Cleanup] Removed HF datasets cache: {cache_dir}")
            except OSError as e:
                print(f"[Warning] Failed to clean HF cache for trial {trial_id}: {e}")
    else:
        # Search for all directories matching trial_{trial_id}_*
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        datasets_dir = os.path.join(hf_home, "datasets")
        if not os.path.exists(datasets_dir):
            return
        
        prefix = f"trial_{trial_id}_"
        cleaned_count = 0
        for item in os.listdir(datasets_dir):
            if item.startswith(prefix):
                cache_path = os.path.join(datasets_dir, item)
                try:
                    shutil.rmtree(cache_path)
                    print(f"[Cleanup] Removed HF datasets cache: {cache_path}")
                    cleaned_count += 1
                except OSError as e:
                    print(f"[Warning] Failed to clean HF cache {cache_path}: {e}")
        
        if cleaned_count == 0:
            print(f"[Cleanup] No HF datasets cache found for trial {trial_id}")


def run_llama_factory(
    llama_factory_dir: str,
    train_yaml_path: str,
    nproc_per_node: int,
    cuda_visible_devices: str,
    force_torchrun: bool,
    log_path: str,
    master_port: int = 0,
    trial_id: int = 0,
) -> str:
    """
    Run LlamaFactory training.

    NOTE: Do NOT wrap llamafactory.cli with external torchrun!
    LlamaFactory's cli.py internally calls torchrun when it detects multi-GPU training
    (via deepspeed config or CUDA_VISIBLE_DEVICES). Wrapping it again causes nested
    torchrun and port conflicts (EADDRINUSE).

    Instead, we:
    1. Always call `python -m llamafactory.cli train ...` (single process entry).
    2. Set CUDA_VISIBLE_DEVICES to control which GPUs are visible.
    3. Set MASTER_PORT env var so LlamaFactory's internal torchrun uses a specific port.
    4. Retry with different ports if EADDRINUSE is detected.
    5. Set HF_DATASETS_CACHE per trial to enable precise cache cleanup.
    
    Returns:
        The HF_DATASETS_CACHE directory path used for this trial.
    """
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if force_torchrun:
        env["FORCE_TORCHRUN"] = "1"
    src_path = os.path.join(llama_factory_dir, "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    
    # Set per-trial HF_DATASETS_CACHE with timestamp to enable precise cleanup after each trial
    trial_cache_dir = get_trial_hf_cache_dir(trial_id)
    os.makedirs(trial_cache_dir, exist_ok=True)
    env["HF_DATASETS_CACHE"] = trial_cache_dir

    def _run_once(port: int) -> int:
        run_env = env.copy()
        # Set MASTER_PORT so LlamaFactory's internal torchrun uses this port
        run_env["MASTER_PORT"] = str(port)
        run_env["MASTER_ADDR"] = "127.0.0.1"
        cmd = [sys.executable, "-m", "llamafactory.cli", "train", train_yaml_path]
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"[optuna_sft_selector] MASTER_PORT={port} cmd: {' '.join(cmd)}\n")
            log_f.flush()
            proc = subprocess.run(
                cmd, cwd=llama_factory_dir, env=run_env, stdout=log_f, stderr=subprocess.STDOUT
            )
        return int(proc.returncode)

    # Retry with different ports if EADDRINUSE is detected (multi-GPU case).
    # For single GPU, port conflicts are less likely but we still handle them.
    max_port_tries = 20 if int(master_port) <= 0 else 1
    last_rc = 1
    last_port = None
    for _ in range(max_port_tries):
        port = int(master_port) if int(master_port) > 0 else find_free_port()
        last_port = port
        last_rc = _run_once(port)
        if last_rc == 0:
            return trial_cache_dir

        # If it's a port conflict, try again with a new port (only in auto mode).
        if int(master_port) <= 0:
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_txt = f.read()
                if "EADDRINUSE" in log_txt or "address already in use" in log_txt:
                    continue
            except OSError:
                pass

        break

    raise RuntimeError(
        f"LlamaFactory training failed (last MASTER_PORT={last_port}, see {log_path})."
    )


def read_eval_loss(output_dir: str) -> float:
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"trainer_state.json not found in {output_dir}")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    log_history = state.get("log_history", [])
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    if not eval_losses:
        raise ValueError("No eval_loss found in trainer_state.json")
    return float(eval_losses[-1])


def cleanup_output_dir(output_dir: str) -> None:
    """
    Remove large model files from output_dir after reading eval_loss.
    Keep only trainer_state.json and small metadata files for reference.
    """
    if not os.path.exists(output_dir):
        return

    # Files to delete (model weights and large files)
    patterns_to_delete = [
        "model.safetensors",
        "model*.safetensors",
        "pytorch_model.bin",
        "pytorch_model*.bin",
        "model.safetensors.index.json",
        "tokenizer.json",  # Large tokenizer file
        "merges.txt",      # Tokenizer merges
        "vocab.json",      # Tokenizer vocab
    ]

    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if not os.path.isfile(filepath):
            continue

        # Delete model weight files
        if any(filename == p or filename.startswith(p.replace("*", "")) for p in patterns_to_delete):
            try:
                os.remove(filepath)
            except OSError:
                pass

        # Delete any .safetensors or .bin files (model weights)
        if filename.endswith(".safetensors") or filename.endswith(".bin"):
            try:
                os.remove(filepath)
            except OSError:
                pass


def cleanup_trial_data(dataset_dir: str) -> None:
    """
    Remove trial data directory after training is complete.
    The data (train.jsonl, dataset_info.json) is only used once during proxy training
    and is no longer needed afterwards.
    """
    if not os.path.exists(dataset_dir):
        return
    
    try:
        shutil.rmtree(dataset_dir)
        print(f"[Cleanup] Removed trial data directory: {dataset_dir}")
    except OSError as e:
        print(f"[Warning] Failed to clean trial data directory {dataset_dir}: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_trial_history(study: optuna.Study, score_keys: List[str], output_path: str) -> None:
    """
    Plot trial history: weights for each score key and eval_loss over trials.
    Saves the plot as a PDF file.
    """
    # Collect data from completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return

    # Sort by trial number
    trials.sort(key=lambda t: t.number)

    trial_numbers = [t.number for t in trials]
    losses = [t.value for t in trials]

    # Extract weights for each score key
    weights_by_key: Dict[str, List[float]] = {k: [] for k in score_keys}
    for t in trials:
        weights = t.user_attrs.get("weights", {})
        for k in score_keys:
            weights_by_key[k].append(weights.get(k, 0.0))

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top plot: weights for each score key
    ax1 = axes[0]
    colors = plt.cm.tab10.colors
    for i, k in enumerate(score_keys):
        color = colors[i % len(colors)]
        ax1.plot(trial_numbers, weights_by_key[k], marker="o", markersize=3, label=k, color=color, linewidth=1.5)
    ax1.set_ylabel("Weight", fontsize=12)
    ax1.set_title("Score Key Weights over Trials", fontsize=14)
    # Place legend outside the plot (to the right) to avoid obscuring lines
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, borderaxespad=0)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: eval_loss
    ax2 = axes[1]
    ax2.plot(trial_numbers, losses, marker="o", markersize=4, color="crimson", linewidth=1.5, label="eval_loss")

    # Mark best trial
    best_idx = losses.index(min(losses))
    best_trial_num = trial_numbers[best_idx]
    best_loss = losses[best_idx]
    ax2.scatter([best_trial_num], [best_loss], color="gold", s=150, zorder=5, edgecolors="black", linewidths=1.5, label=f"Best (trial {best_trial_num}, loss={best_loss:.4f})")

    ax2.set_xlabel("Trial Number", fontsize=12)
    ax2.set_ylabel("Eval Loss", fontsize=12)
    ax2.set_title("Eval Loss over Trials", fontsize=14)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Adjust layout to make room for the legend outside the plot
    plt.tight_layout()
    fig.subplots_adjust(right=0.78)  # Leave space on right for legend
    plt.savefig(output_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def weights_to_z_params(
    weights: Dict[str, float], 
    score_keys: List[str],
    normalization_method: str = "softmax",
    suggest_min: float = -5.0,
    suggest_max: float = 5.0,
) -> Dict[str, float]:
    """
    Convert weight dict to z parameters for Optuna.
    
    For softmax: Since weights = softmax(z), we use z = log(weights) as approximation.
    For linear: Since weights = (z - min + ε) / Σ, we use z = weights * scale + min.
    
    The actual z values don't need to be exact since both methods have degrees of freedom.
    """
    import math
    params = {}
    
    if normalization_method == "softmax":
        for k in score_keys:
            w = weights.get(k, 1.0 / len(score_keys))
            # Avoid log(0) by clamping to small value
            w = max(w, 1e-10)
            z = math.log(w)
            # Clamp to suggest range
            z = max(suggest_min, min(suggest_max, z))
            params[f"z_{k}"] = z
    elif normalization_method == "linear":
        # For linear normalization: w_i = (z_i - min + ε) / Σ(z_j - min + ε)
        # Inverse: z_i ≈ w_i * (max - min) + min (approximate)
        range_size = suggest_max - suggest_min
        for k in score_keys:
            w = weights.get(k, 1.0 / len(score_keys))
            z = w * range_size + suggest_min
            # Clamp to suggest range
            z = max(suggest_min, min(suggest_max, z))
            params[f"z_{k}"] = z
    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")
    
    return params


def load_warmup_trials(json_path: str, score_keys: List[str]) -> List[Dict]:
    """
    Load prior trials from JSON file.
    
    Expected format:
    [
        {"weights": {"key1": 0.5, "key2": 0.3, ...}, "eval_loss": 1.8},
        {"weights": {"key1": 0.6, "key2": 0.2, ...}, "eval_loss": 1.7},
        ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"warmup_trials_json must be a list, got {type(data)}")
    
    trials = []
    for item in data:
        if "weights" not in item or "eval_loss" not in item:
            print(f"[Warning] Skipping invalid warmup trial (missing weights or eval_loss): {item}")
            continue
        trials.append(item)
    
    return trials


def z_params_to_signature(
    params: Dict[str, float], 
    score_keys: List[str],
    normalization_method: str = "softmax",
    suggest_min: float = -5.0,
) -> tuple:
    """Compute weight signature from z parameters (for comparing trials)."""
    zs = [params.get(f"z_{k}", 0.0) for k in score_keys]
    weights = normalize_weights(zs, method=normalization_method, min_val=suggest_min)
    return tuple(round(w, 4) for w in weights)


def get_trial_signatures_by_state(
    study: optuna.Study, 
    score_keys: List[str],
    running_timeout_hours: float = 24.0,
    normalization_method: str = "softmax",
    suggest_min: float = -5.0,
) -> Tuple[set, set, set, set]:
    """
    Get parameter signatures of trials, sorted by state.

    Args:
        study: Optuna study
        score_keys: List of score dimension names
        running_timeout_hours: RUNNING trials longer than this are considered zombies
        normalization_method: Normalization method ('softmax' or 'linear')
        suggest_min: Search lower bound (used for linear normalization)

    Returns:
        completed_signatures: Completed trials (skip)
        failed_signatures: Failed trials (should be retried)
        waiting_signatures: Waiting trials (skip)
        zombie_signatures: Zombie trials (RUNNING over timeout, should retry)
    """
    from datetime import datetime, timedelta
    
    completed_signatures = set()
    failed_signatures = set()
    waiting_signatures = set()
    zombie_signatures = set()
    
    now = datetime.now()
    timeout_threshold = timedelta(hours=running_timeout_hours)
    
    for trial in study.trials:
        # Get params: for WAITING, params are in system_attrs['fixed_params']
        # For other states, params are in trial.params
        if trial.state == optuna.trial.TrialState.WAITING:
            # WAITING state trial: params stored in system_attrs['fixed_params']
            params = trial.system_attrs.get('fixed_params', {})
        else:
            params = trial.params
        
        # Compute signature
        try:
            if not params:
                continue
            signature = z_params_to_signature(
                params, score_keys, 
                normalization_method=normalization_method, 
                suggest_min=suggest_min
            )
        except Exception:
            continue
        
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed_signatures.add(signature)
        elif trial.state == optuna.trial.TrialState.FAIL:
            failed_signatures.add(signature)
        elif trial.state == optuna.trial.TrialState.WAITING:
            waiting_signatures.add(signature)
        elif trial.state == optuna.trial.TrialState.RUNNING:
            # Check for zombie trial (running too long)
            if trial.datetime_start:
                running_time = now - trial.datetime_start
                if running_time > timeout_threshold:
                    # Zombie trial, needs retry
                    zombie_signatures.add(signature)
                    print(f"[Warning] Trial {trial.number} has been RUNNING for {running_time}, marking as zombie")
                # else: running normally, skip
    
    return completed_signatures, failed_signatures, waiting_signatures, zombie_signatures


def weights_to_signature(weights: Dict[str, float], score_keys: List[str]) -> tuple:
    """Convert weights to a signature (for comparison)."""
    # Normalize weights first
    total = sum(weights.values())
    if total > 0:
        normalized = {k: weights.get(k, 0) / total for k in score_keys}
    else:
        normalized = {k: 1.0 / len(score_keys) for k in score_keys}
    return tuple(round(normalized.get(k, 0), 4) for k in score_keys)


def inject_warmup_trials(
    study: optuna.Study,
    warmup_trials: List[Dict],
    score_keys: List[str],
    normalization_method: str = "softmax",
    suggest_min: float = -5.0,
    suggest_max: float = 5.0,
) -> int:
    """
    Inject prior trials into the study using study.add_trial().
    These trials are marked as COMPLETE and will influence the TPE sampler.
    
    IMPORTANT: This function checks which trials have already been injected
    and only injects new ones. This allows safe resumption after interruption.
    
    Returns the number of successfully injected trials.
    """
    # Get signatures of already injected warmup trials
    existing_warmup_signatures = set()
    for t in study.trials:
        if t.user_attrs.get("warmup", False):
            weights = t.user_attrs.get("weights", {})
            if weights:
                signature = weights_to_signature(weights, score_keys)
                existing_warmup_signatures.add(signature)
    
    print(f"[Warmup] Found {len(existing_warmup_signatures)} already injected warmup trials")
    
    injected = 0
    skipped = 0
    for item in warmup_trials:
        weights = item["weights"]
        eval_loss = item["eval_loss"]
        
        # Check if already injected
        signature = weights_to_signature(weights, score_keys)
        if signature in existing_warmup_signatures:
            skipped += 1
            continue
        
        # Convert weights to z parameters
        params = weights_to_z_params(
            weights, score_keys,
            normalization_method=normalization_method,
            suggest_min=suggest_min,
            suggest_max=suggest_max,
        )
        
        # Create distributions for the parameters
        distributions = {f"z_{k}": optuna.distributions.FloatDistribution(suggest_min, suggest_max) for k in score_keys}
        
        # Create a FrozenTrial
        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            values=[eval_loss],
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={
                "weights": weights,
                "eval_loss": eval_loss,
                "warmup": True,  # Mark as warmup trial
            },
        )
        
        try:
            study.add_trial(trial)
            injected += 1
        except Exception as e:
            print(f"[Warning] Failed to inject warmup trial: {e}")
    
    if skipped > 0:
        print(f"[Warmup] Skipped {skipped} already injected trials")
    
    return injected


def enqueue_weight_configs(
    study: optuna.Study,
    json_path: str,
    score_keys: List[str],
    running_timeout_hours: float = 24.0,
    normalization_method: str = "softmax",
    suggest_min: float = -5.0,
    suggest_max: float = 5.0,
) -> int:
    """
    Add weight configurations from a JSON file into the Optuna queue.

    Logic (to ensure no repeats or omissions):
    - COMPLETE: skip (already finished)
    - WAITING: skip (already queued)
    - FAIL: retry enqueue (failed before, need to retry)
    - RUNNING (timeout): retry enqueue (zombie trial, worker may have crashed)
    - RUNNING (no timeout): skip (currently running)
    
    Args:
        study: Optuna study
        json_path: Path to the weights JSON file
        score_keys: List of score keys
        running_timeout_hours: RUNNING longer than this is considered zombie (default 24h)
        normalization_method: Normalization method ('softmax' or 'linear')
        suggest_min: Lower bound for search
        suggest_max: Upper bound for search
    
    Returns:
        The total number enqueued + retried
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"enqueue_weights_json must be a list, got {type(data)}")
    
    # Get signature sets by state
    (completed_signatures, 
     failed_signatures, 
     waiting_signatures, 
     zombie_signatures) = get_trial_signatures_by_state(
        study, score_keys, running_timeout_hours,
        normalization_method=normalization_method,
        suggest_min=suggest_min,
    )
    
    # Get signatures of currently running (not zombies)
    from datetime import datetime, timedelta
    running_signatures = set()
    now = datetime.now()
    timeout_threshold = timedelta(hours=running_timeout_hours)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            if trial.datetime_start:
                running_time = now - trial.datetime_start
                if running_time <= timeout_threshold:
                    # Still running normally
                    try:
                        signature = z_params_to_signature(
                            trial.params, score_keys,
                            normalization_method=normalization_method,
                            suggest_min=suggest_min,
                        )
                        running_signatures.add(signature)
                    except Exception:
                        pass
    
    print(f"[Enqueue] Study status: {len(completed_signatures)} completed, "
          f"{len(failed_signatures)} failed, {len(waiting_signatures)} waiting, "
          f"{len(running_signatures)} running, {len(zombie_signatures)} zombie")
    
    enqueued = 0
    skipped_completed = 0
    skipped_waiting = 0
    skipped_running = 0
    retried_failed = 0
    retried_zombie = 0
    
    for weights in data:
        if not isinstance(weights, dict):
            print(f"[Warning] Skipping invalid weight config: {weights}")
            continue
        
        signature = weights_to_signature(weights, score_keys)
        
        # 1. Completed -> skip
        if signature in completed_signatures:
            skipped_completed += 1
            continue
        
        # 2. Already queued -> skip
        if signature in waiting_signatures:
            skipped_waiting += 1
            continue
        
        # 3. Running (not zombie) -> skip
        if signature in running_signatures:
            skipped_running += 1
            continue
        
        # 4. Fail or zombie -> retry
        is_retry_failed = signature in failed_signatures
        is_retry_zombie = signature in zombie_signatures
        
        params = weights_to_z_params(
            weights, score_keys,
            normalization_method=normalization_method,
            suggest_min=suggest_min,
            suggest_max=suggest_max,
        )
        
        try:
            study.enqueue_trial(params)
            if is_retry_failed:
                retried_failed += 1
            elif is_retry_zombie:
                retried_zombie += 1
            else:
                enqueued += 1
        except Exception as e:
            print(f"[Warning] Failed to enqueue weight config: {e}")
    
    print(f"[Enqueue] Results: "
          f"skipped {skipped_completed} completed, "
          f"{skipped_waiting} waiting, "
          f"{skipped_running} running | "
          f"retrying {retried_failed} failed + {retried_zombie} zombie | "
          f"new {enqueued}")
    
    return enqueued + retried_failed + retried_zombie


def main() -> None:
    cli_args = parse_args()
    args = load_config(cli_args.config)
    set_global_seed(args.seed)

    # Parse suggest_range
    suggest_min, suggest_max = parse_suggest_range(args.suggest_range)
    normalization_method = args.normalization_method
    
    print(f"[Config] Normalization method: {normalization_method}")
    print(f"[Config] Suggest range: [{suggest_min}, {suggest_max}]")

    pool_jsonl = os.path.abspath(args.pool_jsonl)
    val_jsonl = os.path.abspath(args.val_jsonl)
    run_dir = os.path.abspath(args.run_dir)
    base_train_yaml = os.path.abspath(args.base_train_yaml)
    llama_factory_dir = (
        os.path.abspath(args.llama_factory_dir)
        if args.llama_factory_dir
        else os.path.abspath(os.path.join(os.getcwd(), "LlamaFactory"))
    )

    ensure_dir(run_dir)
    trial_root = os.path.join(run_dir, "trials")
    ensure_dir(trial_root)

    # Load precomputed index if available (recommended for large datasets)
    pool_index = None
    pool_index_keys = None
    byte_offsets = None
    score_matrix = None
    line_indices_arr = None
    score_key_indices = None  # Indices of selected score keys in pool_index_keys
    if args.pool_index and os.path.exists(args.pool_index):
        pool_index, pool_index_keys, index_metadata = load_pool_index(args.pool_index)
        print(f"[Index] Loaded precomputed index with {len(pool_index_keys)} score dimensions: {pool_index_keys}")
        
        # Check if user specified --score_keys to use a subset
        if args.score_keys.strip():
            user_keys = [k.strip() for k in args.score_keys.split(",") if k.strip()]
            # Validate that all user keys exist in pool_index_keys
            missing_keys = [k for k in user_keys if k not in pool_index_keys]
            if missing_keys:
                raise ValueError(
                    f"Score keys {missing_keys} not found in pool_index. "
                    f"Available keys: {pool_index_keys}"
                )
            # Build index mapping: which columns to use from score_matrix
            score_key_indices = [pool_index_keys.index(k) for k in user_keys]
            score_keys = user_keys
            print(f"[Index] Using user-specified subset: {score_keys} (indices: {score_key_indices})")
        else:
            score_keys = pool_index_keys
            score_key_indices = list(range(len(pool_index_keys)))
            print(f"[Index] Using all {len(score_keys)} score dimensions from index")
        
        # Build byte offset index for O(k) random access (one-time cost, all trials benefit after)
        byte_offsets = build_byte_offset_index(pool_jsonl)
        
        # Build NumPy matrix for vectorized computation (one-time cost, very fast Step 1 for every trial)
        try:
            import numpy as np
            t_np = time.time()
            print(f"[Index] Building NumPy score matrix for vectorized computation...")
            full_score_matrix = np.array([item[0] for item in pool_index], dtype=np.float32)
            line_indices_arr = np.array([item[1] for item in pool_index], dtype=np.int64)
            # Select only the columns for the specified score keys
            score_matrix = full_score_matrix[:, score_key_indices]
            t_np_elapsed = time.time() - t_np
            print(f"[Index] NumPy matrix shape: {score_matrix.shape}, dtype: {score_matrix.dtype} ({t_np_elapsed:.2f}s)")
        except ImportError:
            print("[Warning] NumPy not available, using pure Python (slower)")
            # Also need to filter pool_index for pure Python path
            pool_index = [
                ([score_vec[i] for i in score_key_indices], line_idx)
                for score_vec, line_idx in pool_index
            ]
    else:
        score_keys = infer_score_keys(pool_jsonl, args.score_field, args.score_keys)
    
    if len(score_keys) == 0:
        raise ValueError("Score keys are empty.")

    base_cfg = load_yaml(base_train_yaml)
    force_torchrun = bool(base_cfg.get("deepspeed"))

    # Legacy cache mode (not recommended for large datasets)
    pool_cache = None
    if args.cache_pool_in_memory and pool_index is None:
        t_cache = time.time()
        print("[Warning] cache_pool_in_memory is slow for large datasets. Consider using --pool_index instead.")
        pool_cache = []
        for item in iter_pool_items(pool_jsonl):
            scores = extract_scores(item, args.score_field)
            pool_cache.append((compute_score_vec(scores, score_keys), item))
        t_cache_elapsed = time.time() - t_cache
        print(f"[Cache] Loaded {len(pool_cache):,} items into memory ({t_cache_elapsed:.2f}s)")

    # Custom gamma function: controls the number of good trials used in TPE sampler
    def custom_gamma(n: int) -> int:
        """Custom gamma function for TPE sampler."""
        import math
        return min(int(math.ceil(args.tpe_gamma_percent * n)), args.tpe_gamma_max)
    
    sampler = optuna.samplers.TPESampler(
        seed=args.seed, 
        n_startup_trials=args.n_startup_trials,
        multivariate=args.tpe_multivariate,  # Enable multivariate mode (consider param dependencies)
        n_ei_candidates=args.tpe_n_ei_candidates,  # Number of EI candidates
        gamma=custom_gamma,  # Custom gamma function, controls good trials used
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    # Warm-start: inject prior trials with known eval_loss
    if args.warmup_trials_json and os.path.exists(args.warmup_trials_json):
        t_warmup = time.time()
        warmup_trials = load_warmup_trials(args.warmup_trials_json, score_keys)
        injected = inject_warmup_trials(
            study, warmup_trials, score_keys,
            normalization_method=normalization_method,
            suggest_min=suggest_min,
            suggest_max=suggest_max,
        )
        t_warmup_elapsed = time.time() - t_warmup
        print(f"[Warm-start] Injected {injected} prior trials from {args.warmup_trials_json} ({t_warmup_elapsed:.2f}s)")

    # Enqueue: add weight configurations to try first (will be evaluated)
    if args.enqueue_weights_json and os.path.exists(args.enqueue_weights_json):
        t_enqueue = time.time()
        enqueued = enqueue_weight_configs(
            study, 
            args.enqueue_weights_json, 
            score_keys,
            running_timeout_hours=args.running_timeout_hours,
            normalization_method=normalization_method,
            suggest_min=suggest_min,
            suggest_max=suggest_max,
        )
        t_enqueue_elapsed = time.time() - t_enqueue
        print(f"[Warm-start] Enqueued {enqueued} weight configurations from {args.enqueue_weights_json} ({t_enqueue_elapsed:.2f}s)")

    best_value = None
    no_improve = 0
    plot_pdf_path = os.path.join(run_dir, "trial_history.pdf")

    def early_stop_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_value, no_improve
        if args.early_stop_patience <= 0:
            return
        if trial.value is None:
            return
        if best_value is None or trial.value < best_value - args.early_stop_delta:
            best_value = trial.value
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                study.stop()

    summary_path = os.path.join(run_dir, "best_weights.json")

    def save_best_weights(study: optuna.Study) -> None:
        """Save current best weights to JSON file."""
        try:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                return
            best = study.best_trial
            best_weights = best.user_attrs.get("weights", {})
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "study_name": args.study_name,
                        "best_value": best.value,
                        "best_weights": best_weights,
                        "best_trial": best.number,
                        "total_completed_trials": len(completed_trials),
                        "normalization_method": normalization_method,
                        "suggest_range": [suggest_min, suggest_max],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"[Warning] Failed to save best_weights.json: {e}")

    def plot_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Update the trial history plot and best_weights.json after each completed trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Update best_weights.json
            save_best_weights(study)
            # Update plot
            try:
                plot_trial_history(study, score_keys, plot_pdf_path)
            except Exception as e:
                print(f"[Warning] Failed to update plot: {e}")

    def objective(trial: optuna.Trial) -> float:
        trial_start_time = time.time()
        t_suggest_start = time.time()
        zs = [trial.suggest_float(f"z_{k}", suggest_min, suggest_max) for k in score_keys]
        t_suggest_elapsed = time.time() - t_suggest_start
        if trial.number == 0:
            print(f"[TPE] Parameter suggestion time: {t_suggest_elapsed:.4f}s (n_ei_candidates={args.tpe_n_ei_candidates})")
        weights = normalize_weights(zs, method=normalization_method, min_val=suggest_min)
        
        # Save weights immediately so they're available even for RUNNING trials
        trial.set_user_attr("weights", {k: float(w) for k, w in zip(score_keys, weights)})

        trial_id = trial.number
        if trial_id > args.max_trials_in_dir:
            raise RuntimeError("Exceeded max_trials_in_dir safety limit.")

        trial_dir = os.path.join(trial_root, f"trial_{trial_id:05d}")
        dataset_dir = os.path.join(trial_dir, "data")
        output_dir = os.path.join(trial_dir, "output")
        train_yaml_path = os.path.join(trial_dir, "train.yaml")
        log_path = os.path.join(trial_dir, "train.log")
        ensure_dir(dataset_dir)

        train_data_path = os.path.join(dataset_dir, "train.jsonl")
        train_dataset_name = "optuna_train"
        val_dataset_name = "optuna_val"

        print(f"\n{'='*60}")
        print(f"[Trial {trial_id}] Starting...")
        
        # Use index-based selection if available (much faster for large datasets)
        t_select = time.time()
        if pool_index is not None:
            selected = select_top_k_with_index(
                pool_jsonl,
                pool_index,
                weights,
                args.top_k,
                byte_offsets=byte_offsets,
                score_matrix=score_matrix,
                line_indices=line_indices_arr,
            )
        else:
            selected = select_top_k(
                pool_jsonl,
                args.score_field,
                score_keys,
                weights,
                args.top_k,
                pool_cache=pool_cache,
            )
        t_select_elapsed = time.time() - t_select
        
        t_write = time.time()
        write_jsonl(train_data_path, selected)
        write_dataset_info(dataset_dir, train_dataset_name, "train.jsonl", val_dataset_name, val_jsonl)
        t_write_elapsed = time.time() - t_write
        print(f"[Trial {trial_id}] Data selection: {t_select_elapsed:.2f}s, Write data: {t_write_elapsed:.2f}s")

        build_train_yaml(
            base_train_yaml,
            train_yaml_path,
            base_cfg=base_cfg,
            dataset_dir=dataset_dir,
            train_dataset=train_dataset_name,
            eval_dataset=val_dataset_name,
            output_dir=output_dir,
            num_train_epochs=args.num_train_epochs,
            eval_strategy=args.eval_strategy,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
        )

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        t_train = time.time()
        trial_cache_dir = run_llama_factory(
            llama_factory_dir=llama_factory_dir,
            train_yaml_path=train_yaml_path,
            nproc_per_node=args.nproc_per_node,
            cuda_visible_devices=args.cuda_visible_devices,
            force_torchrun=force_torchrun,
            log_path=log_path,
            master_port=args.master_port,
            trial_id=trial_id,
        )
        t_train_elapsed = time.time() - t_train

        loss = read_eval_loss(output_dir)

        # Clean up model files to save disk space (only keep trainer_state.json)
        t_cleanup = time.time()
        cleanup_output_dir(output_dir)
        # Clean up HuggingFace datasets cache for this trial (with timestamp)
        cleanup_trial_hf_cache(trial_id, cache_dir=trial_cache_dir)
        # Clean up trial data directory (train.jsonl, dataset_info.json) - no longer needed
        cleanup_trial_data(dataset_dir)
        t_cleanup_elapsed = time.time() - t_cleanup

        trial.set_user_attr("weights", {k: float(w) for k, w in zip(score_keys, weights)})
        trial.set_user_attr("train_data_path", train_data_path)
        trial.set_user_attr("output_dir", output_dir)
        trial.set_user_attr("eval_loss", float(loss))
        
        trial_total_time = time.time() - trial_start_time
        print(f"[Trial {trial_id}] Training: {t_train_elapsed:.2f}s, Cleanup: {t_cleanup_elapsed:.2f}s")
        print(f"[Trial {trial_id}] Total: {trial_total_time:.2f}s, eval_loss: {loss:.6f}")
        print(f"{'='*60}\n")
        
        return loss

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1, callbacks=[early_stop_callback, plot_callback])

    # Final save (in case callbacks missed anything)
    save_best_weights(study)
    plot_trial_history(study, score_keys, plot_pdf_path)

    best = study.best_trial
    print(f"Best loss: {best.value}")
    print(f"Best weights saved to: {summary_path}")
    print(f"Trial history plot saved to: {plot_pdf_path}")


if __name__ == "__main__":
    main()
