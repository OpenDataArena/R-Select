# R-Select: SFT Data Selection Pipeline

**R-Select** is a hierarchical score weight optimization pipeline based on [Optuna](https://github.com/optuna/optuna) and [LlamaFactory](https://github.com/hiyouga/LlamaFactory). It selects high-quality subsets from large-scale data pools for SFT (Supervised Fine-Tuning) training. The framework supports an arbitrary number of layers, cluster divisions, and metrics.

---

## Method Overview

**Core idea**: Given multiple scoring metrics (e.g., IFD, PPL, Length, etc.), the pipeline uses the **proxy model** (a small model fine-tuned on the selected subset) validation loss as the feedback signal. **Optuna TPE** searches for the optimal weights of each metric so that the weighted score `weighted_score = Σ(weight_i × score_i)` yields top-k data with the lowest loss on the validation set.

**Hierarchical optimization** is supported: metrics are first clustered into groups by correlation (Layer1), and each group optimizes its weights independently; then Layer2 optimizes the weights of each group, reducing search space and improving stability. Finally, data is weighted and sampled according to the optimized weights.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Format Requirements](#data-format-requirements)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [main.py Usage](#mainpy-usage)
- [Warm-start](#warm-start)
- [Detailed Steps](#detailed-steps)
- [Utility Scripts](#utility-scripts)
- [FAQ](#faq)
- [Project Structure](#project-structure)
- [References](#references)

---

## Environment Setup

It is recommended to run this project in the `rselect` conda environment.

### Installation

```bash
# 1. Clone the repository
git clone --recurse-submodules https://github.com/OpenDataArena/R-Select.git
cd R-Select

# 2. Create and activate the environment
conda create -n rselect python=3.10 -y
conda activate rselect

# 3. Install PyTorch (choose according to your CUDA version; example for CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install LlamaFactory and core dependencies (LlamaFactory is included in the project)
cd LlamaFactory
pip install -e .
pip install deepspeed==0.16.9
pip install optuna==4.7.0
conda install -c conda-forge pandas=2.3.3
pip install seaborn==0.13.2
pip install nltk==3.9.1
pip install datasketch==1.7.0
pip install faiss-gpu==1.7.2
pip install numpy==1.26.3
pip install lexicalrichness==0.5.1
pip install spacy==3.8.8
pip install fasttext==0.9.3
pip install prettytable==3.16.0

# 5. Optional: vLLM required for embed scripts (data clustering); skip if not used
pip install vllm==0.8.5.post1
```

### Notes

- `main.py` performs proxy training via LlamaFactory; ensure transformers, peft, deepspeed, etc. are compatible with `LlamaFactory/requirements.txt`
- If you encounter version conflicts, refer to `LlamaFactory/requirements.txt` and install dependencies manually

### Run

```bash
conda activate rselect
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
```

---

## Data Format Requirements

### pool_jsonl (Data Pool)

Raw data must be in JSONL format, one JSON object per line, with at least the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier |
| `instruction` | Instruction/question |
| `input` | Input (can be empty) |
| `output` | Answer/output |
| `scores` | Score dictionary; keys are metric names |

### val_jsonl (Validation Set)

The validation set in the example code is sampled from the following five benchmarks:
- https://huggingface.co/datasets/bigcode/bigcodebench
- https://huggingface.co/datasets/Idavidrein/gpqa
- https://huggingface.co/datasets/google-research-datasets/mbpp
- https://huggingface.co/datasets/cais/mmlu
- https://huggingface.co/datasets/KbsdJames/Omni-MATH

Users can choose or construct a validation set appropriate for their downstream task.

The validation file (val_jsonl) **must follow the Alpaca format**. Each sample must include the three keys `instruction`, `input`, and `output` for LlamaFactory to construct SFT samples and compute eval loss:

```json
{"instruction": "Task description", "input": "Additional input (can be empty string)", "output": "Expected answer"}
```

### pool_jsonl Example

```json
{
  "id": 1,
  "instruction": "xxx",
  "input": "",
  "output": "xxx",
  "scores": {
    "AtheneScore": 0.75,
    "CleanlinessScore": 0.82,
    "PPLScore": 12.5,
    ...
  }
}
```

### Demo Data Description

The `data/demo.jsonl` used in the example is based on the [alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) dataset, using the **alpaca_gpt4** subset with 30 scoring metrics from [OpenDataArena/ODA-scored-data-2603](https://huggingface.co/datasets/OpenDataArena/ODA-scored-data-2603). For metric definitions and calculation methods, refer to the ODA-scored-data-2603 dataset documentation.

---

## Pipeline Overview

```
Raw data → Normalization → [Data clustering] → Index generation → [Metrics clustering] → Layer1 optimization → Aggregation + Normalization → Layer2 optimization → Final weighting → Sampling
```

- **[]** indicates optional steps
- **Normalization**: Windsorization and [0,1] normalization of `scores`
- **Data clustering**: For diversity sampling (top-k% within each cluster)
- **Metrics clustering**: Grouping correlated metrics for hierarchical optimization
- **Optimization**: Optuna searches for weights of each score to minimize proxy model eval loss

> **Note**: The demo uses ~30 metrics, two-layer optimization, and 6 Layer1 clusters as an example only. The framework supports any number of metrics, layers, and cluster divisions; write configs as needed.

---

## Quick Start

For the demo, the full workflow is in `run.sh`. Summary of steps:

```bash
# 1. Normalization
python utils/score_normalization.py -i data/demo.jsonl -o data/demo_normalized.jsonl \
  --pct-range 5 95 --keep-original --flip-keys PPL Normalized_Loss

# 2. Generate index (accelerate optimization)
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl

# 3. Run Layer1 optimization (6 clusters can run in parallel)
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
# ... Layer1_c2 ~ Layer1_c6

# 4. Aggregate Layer1 results and write new scores (in-place)
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1

# 5. Regenerate index and run Layer2
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl
python main.py --config configs/demo/Layer2/Layer2_c1.yaml

# 6. Compute final scores and sample
python utils/cluster_aggregation.py -i data/demo_normalized.jsonl \
  -w results/demo/Layer2/Layer2_c1/best_weights.json
python utils/sampling.py -i data/demo_normalized.jsonl -o data/sampled.jsonl \
  -k 10 --per_cluster -s final_score
```

---

## Detailed Steps

### 1. Normalization

Clip `scores` by percentile and normalize to [0,1]:

```bash
python utils/score_normalization.py \
  -i data/demo.jsonl \
  -o data/demo_normalized.jsonl \
  --pct-range 5 95 \
  --keep-original \
  --flip-keys PPL Normalized_Loss
```

- `--pct-range`: Clip bounds (e.g., 5~95 truncates beyond p5/p95)
- `--keep-original`: Keep original values in `*_orig` fields
- `--flip-keys`: For "lower is better" metrics, subtract from 1 after normalization

### 2. Metrics Clustering (Optional, for hierarchical optimization)

Cluster correlated scores to define Layer1 clusters:

```bash
python utils/metrics_clustering.py \
  -i data/demo_normalized.jsonl \
  -o data/metrics_clustering/cluster_results.txt \
  --sample_size 10000 \
  --n_clusters 6 \
  --scores data/scores.txt \
  --use_absolute_corr
```

If `--scores` is not specified, all scores are clustered; otherwise only the given scores are used.

> **Note**: Cluster division for hierarchical optimization should follow clustering results. `configs/demo` is for demonstration only, splitting 30 metrics into 6 clusters evenly without strictly following clustering; in practice, write Layer1 configs based on `metrics_clustering.py` output.

### 3. Data Clustering (Optional, for diversity sampling)

Extract embeddings, cluster, then assign `cluster_id` to each sample:

```bash
# Extract embeddings
python utils/embedding.py \
  --embedder_model /path/to/embedding_model \
  --input_path data/demo_normalized.jsonl \
  --output_path data/samples_clustering/demo_normalized_embeddings.npy \
  --fields instruction input \
  --tensor_parallel_size 8

# Clustering
python utils/samples_clustering.py --input_path data/samples_clustering/demo_normalized_embeddings.npy \
  --output_dir data/samples_clustering --opt_k 50

# Assign cluster_id
python utils/assign_cluster_id.py \
  --input_jsonl data/demo_normalized.jsonl \
  --labels_path data/samples_clustering/cluster_labels.npy
```

### 4. Generate Index File

Precompute score index to speed up Optuna optimization (recommended for millions of samples):

```bash
python utils/precompute_index.py \
  --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl
```

### 5. Optuna Optimization (main.py)

Each config corresponds to one cluster in one layer; configs can run in parallel or sequentially:

```bash
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
python main.py --config configs/demo/Layer1/Layer1_c2.yaml
# ... Layer1_c3 ~ Layer1_c6
```

Outputs include `trial_history.pdf`, `best_weights.json`, etc. Early stopping, warmup, and enqueue are supported.

### 6. Inter-layer Aggregation (global_aggregation)

Aggregate `best_weights` from each Layer1 cluster, write new scores (e.g., Layer1_c1, Layer1_c2, ...), and apply min-max normalization. Input file is modified in-place when `-o` is not specified. Regenerate index after aggregation:

```bash
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1

# After inter-layer aggregation, regenerate index
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl --output_index data/demo_normalized_index.pkl
```

### 7. Layer2 Aggregation and Optimization

After Layer1 aggregation, the data will contain new weighted scores for each Layer1 cluster (Layer1_c1, Layer1_c2, ...). Layer2 optimization follows. Layer2 configs are similar to Layer1; each config typically corresponds to a higher-level cluster. Run Optuna again to obtain Layer2_c1 `best_weights`:

```bash
python main.py --config configs/demo/Layer2/Layer2_c1.yaml
```

After Layer2 optimization, `trial_history.pdf` and `best_weights.json` are produced, yielding combined weights over the previous layer (Layer1_c1, Layer1_c2, ...).

### 8. Final Weighting and Sampling

Use the top-layer cluster's `best_weights` to compute `final_score`, then sample by top-k%:

```bash
# Compute final_score (weights apply to Layer1_c1, etc. in scores)
python utils/cluster_aggregation.py \
  -i data/demo_normalized.jsonl \
  -w results/demo/Layer2/Layer2_c1/best_weights.json

# Diversity sampling: top k% within each cluster
python utils/sampling.py \
  -i data/demo_normalized.jsonl \
  -o results/demo/demo_diversity_sampled.jsonl \
  -k 10 --per_cluster -s final_score

# Global sampling: top k% over all data
python utils/sampling.py \
  -i data/demo_normalized.jsonl \
  -o results/demo/demo_global_sampled.jsonl \
  -k 10 --global -s final_score
```

---

## Utility Scripts

| Script | Function |
|--------|----------|
| `score_normalization.py` | Windsorization + [0,1] normalization |
| `metrics_clustering.py` | Cluster scores by Pearson correlation |
| `embedding.py` | Extract text embeddings |
| `samples_clustering.py` | K-Means data clustering |
| `assign_cluster_id.py` | Assign cluster_id to samples |
| `precompute_index.py` | Precompute pool index |
| `global_aggregation.py` | Aggregate Layer scores with best_weights |
| `cluster_aggregation.py` | Compute final_score from weight file |
| `sampling.py` | Sample top-k% by score (per_cluster / global) |
| `summarization.py` | Summarize multi-layer weights and hierarchy |
| `trials_analysis.py` | Read Optuna trial details |

---

## main.py Usage

### Basic Usage

```bash
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
```

### Proxy Model Configuration

Proxy model training configuration is an important part of R-Select. This demo uses [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) as the proxy model. Training parameters and DeepSpeed settings are in `configs/demo/demo_proxy.yaml`, referenced via `base_train_yaml`. Users can modify this file according to their needs (GPU memory, data size, epochs, etc.).

### Data Format Requirements

**pool_jsonl**: Data pool; must include a `scores` dictionary (or the field specified by `score_field`).

**val_jsonl**: Validation set; **must be in Alpaca format** with at least `instruction`, `input`, and `output`:

```json
{"instruction": "Question or task description", "input": "Additional input (can be empty)", "output": "Expected answer"}
```

LlamaFactory uses these fields to construct SFT samples. If column names differ, configure column mapping in the dataset section of `base_train_yaml`.

### Config Parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `pool_jsonl` | ✓ | str | Path to pool JSONL |
| `val_jsonl` | ✓ | str | Path to validation JSONL (Alpaca format) |
| `top_k` | ✓ | int | Number of samples selected per trial by weighted score |
| `base_train_yaml` | ✓ | str | LlamaFactory training config template |
| `run_dir` | ✓ | str | Output directory for current run (trials, best_weights, etc.) |
| `study_name` | ✓ | str | Optuna study name |
| `storage` | ✓ | str | Optuna storage URL, e.g., `sqlite:///path/to/study.db` |
| `n_trials` | ✓ | int | Maximum number of trials |
| `score_keys` | ✓ | list/str | Score keys for weighting. Layer1: raw metrics; Layer2: Layer1 cluster names |
| `score_field` | | str | Field containing scores; default `scores` |
| `pool_index` | | str | Path to precomputed index; recommended for large pools |
| `seed` | | int | Random seed |
| `num_train_epochs` | | float | Proxy training epochs |
| `eval_strategy` | | str | Eval strategy, e.g., `epoch` |
| `per_device_eval_batch_size` | | int | Eval batch size; 0 means same as train |
| `cuda_visible_devices` | | str | Visible GPUs, e.g., `0,1,2,3` |
| `nproc_per_node` | | int | GPUs per node for distributed training |
| `master_port` | | int | Distributed port; 0 for auto |
| `llama_factory_dir` | | str | LlamaFactory root; empty to use bundled LlamaFactory |
| `normalization_method` | | str | Weight normalization: `softmax` or `linear` |
| `suggest_range` | | str/list | Optuna search range, e.g., `-5,5` or `[0, 1]` |
| `n_startup_trials` | | int | TPE random exploration trials |
| `tpe_multivariate` | | bool | Enable multivariate TPE |
| `tpe_n_ei_candidates` | | int | EI candidate count |
| `tpe_gamma_percent` | | float | Gamma proportion |
| `tpe_gamma_max` | | int | Gamma upper bound |
| `early_stop_patience` | | int | Early stopping patience; 0 to disable |
| `early_stop_delta` | | float | Early stopping minimum improvement |
| `warmup_trials_json` | | str | Warm-start: path to existing trials to inject |
| `enqueue_weights_json` | | str | Warm-start: path to weight queue for priority evaluation |
| `cache_pool_in_memory` | | bool | Cache pool in memory (for small data) |
| `max_trials_in_dir` | | int | Safety limit for max trials per directory |
| `running_timeout_hours` | | float | Hours before RUNNING trial is considered zombie |

---

## Warm-start

Two warm-start modes are supported to accelerate convergence or prioritize specific configurations.

### Recommended: Warm Trajectory Generation Covering the Search Space

For warm-start, use **Dirichlet distribution** together with **boundary configurations** to generate warm trajectories that adequately cover the weight search space:

- **Dirichlet distribution**: `Dir(α,...,α)` samples on the simplex. Smaller α yields sparser (more peaked) weights; α=1 is uniform; α>1 is more concentrated at the center. Use multiple α values (e.g., 0.1, 0.5, 1.0, 2.0) for the same `score_keys` to sample diverse weight sets.
- **Boundary configurations**: Add simplex vertices and boundary cases, e.g., one-hot (one dimension 1, others 0), uniform (1/k), etc., so corners and the uniform solution are covered.

Warmup or enqueue configs generated this way better guide TPE to explore the full feasible region.

### 1. warmup_trials_json

Inject **completed** (weights, eval_loss) as prior; **no retraining**. TPE uses this to adjust sampling.

**JSON format**: Array of `{weights, eval_loss}`:

```json
[
  {"weights": {"AtheneScore": 0.2, "CleanlinessScore": 0.3, ...}, "eval_loss": 6.83},
  {"weights": {"AtheneScore": 0.1, "CleanlinessScore": 0.4, ...}, "eval_loss": 6.92}
]
```

- `weights`: Keys must match `score_keys`
- `eval_loss`: Validation loss for this configuration

**Example**: `configs/demo/warmup_trials_example.json`

### 2. enqueue_weights_json

Add weight configurations to the **evaluation queue**; they are **actually trained** and executed before TPE-sampled trials.

**JSON format**: Array of **pure weight dicts** (no `weights` wrapper, no `eval_loss`):

```json
[
  {"AtheneScore": 0.2, "CleanlinessScore": 0.2, "ComplexityLaJScore": 0.2, ...},
  {"AtheneScore": 0.133, "CleanlinessScore": 0.338, ...}
]
```

**Example**: `configs/demo/enqueue_weights_example.json`

### Comparison

| Aspect | warmup_trials_json | enqueue_weights_json |
|--------|-------------------|----------------------|
| Training | No (prior only) | Yes |
| Element format | `{weights, eval_loss}` | Pure weight dict |
| Typical use | Transfer existing best results | Prioritize specific weight combinations |

See `configs/demo/warmstart_README.md` for details.

---

## FAQ

**Q: How to determine Layer1 cluster division?**  
A: Use `metrics_clustering.py` to cluster scores by correlation; each cluster maps to one YAML config.

**Q: Will warmup repeat when running the same study multiple times?**  
A: No. The script deduplicates by weight signature; already injected trials are skipped.

**Q: What if data has no cluster_id?**  
A: Skip data clustering and use `--global` mode for global top-k% sampling.

**Q: How to view aggregated weights across layers?**  
A: Use `summarization.py`:
```bash
python utils/summarization.py -i results/demo -o results/demo/weight_summary.json
```

---

## Project Structure

```
sft_data_selection_pipeline/
├── main.py                 # Optuna optimization main program
├── run.sh                  # Example pipeline script
├── readme.md
├── configs/
│   └── demo/
│       ├── demo_proxy.yaml # LlamaFactory proxy training config
│       ├── Layer1/         # Layer1 cluster configs
│       └── Layer2/         # Layer2 configs
├── utils/                  # Utility scripts
│   ├── score_normalization.py
│   ├── metrics_clustering.py
│   ├── embedding.py
│   ├── samples_clustering.py
│   ├── assign_cluster_id.py
│   ├── precompute_index.py
│   ├── global_aggregation.py
│   ├── cluster_aggregation.py
│   ├── sampling.py
│   ├── summarization.py
│   └── trials_analysis.py
├── data/                   # Data directory
└── results/                # Optimization results
    └── demo/
        ├── Layer1/
        ├── Layer2/
        └── weight_summary.json
```

---

## References

This project is built on the following open-source frameworks; acknowledgments:

- **[LlamaFactory](https://github.com/hiyouga/LlamaFactory)** — Unified efficient fine-tuning framework for LLMs/VLMs, supporting 100+ models (ACL 2024). Used for proxy model SFT training and eval loss computation.
- **[Optuna](https://github.com/optuna/optuna)** — Hyperparameter optimization framework. Used for TPE (Tree-structured Parzen Estimator) sampler to search optimal weights for scoring metrics.
