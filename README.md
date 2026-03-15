# R-Select: SFT Data Selection Pipeline

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

**R-Select** is a hierarchical score-weight optimization pipeline built on [Optuna](https://github.com/optuna/optuna) and [LlamaFactory](https://github.com/hiyouga/LlamaFactory), for selecting high-quality subsets from large data pools for SFT (Supervised Fine-Tuning). The framework supports arbitrary numbers of layers, cluster partitions, and metrics.

---

## Method Overview

**Core idea**: Given multiple scoring metrics (e.g., IFD, PPL, Length, etc.), the pipeline uses the **proxy model** (a small model fine-tuned on the selected subset) validation loss as the feedback signal. It searches for optimal weights of each metric via **Optuna TPE**, so that the weighted score `weighted_score = Σ(weight_i × score_i)` selects top-k data with the lowest validation loss.

**Hierarchical optimization**: First cluster metrics by correlation into groups (Layer1) and optimize weights within each group independently; then optimize group weights in Layer2 to reduce search space and improve stability. Finally, weight and sample data according to the optimized weights.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Format Requirements](#data-format-requirements)
- [Data Scoring](#data-scoring)
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
cd ..
conda install -c conda-forge pandas=2.3.3
pip install -r requirements.txt

# 5. Optional: embed script (data clustering) requires vLLM; skip if not used
pip install vllm==0.8.5.post1
```

### Notes

- `main.py` performs proxy training via LlamaFactory; ensure transformers, peft, deepspeed, etc. are compatible with `LlamaFactory/requirements.txt`
- If you encounter version conflicts, refer to `LlamaFactory/requirements.txt` and install dependencies accordingly

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

The validation set in the example code comes from the following five benchmarks:
- https://huggingface.co/datasets/bigcode/bigcodebench
- https://huggingface.co/datasets/Idavidrein/gpqa
- https://huggingface.co/datasets/google-research-datasets/mbpp
- https://huggingface.co/datasets/cais/mmlu
- https://huggingface.co/datasets/KbsdJames/Omni-MATH

You can choose or build a suitable validation set according to your downstream task.

The validation file (val_jsonl) **must follow the Alpaca format**. Each sample must contain the three keys `instruction`, `input`, and `output`, so that LlamaFactory can build SFT samples and compute eval loss:

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

The example `data/demo.jsonl` is based on the [alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) dataset, using the **alpaca_gpt4** subset, with 30 scoring metrics from [OpenDataArena/ODA-scored-data-2603](https://huggingface.co/datasets/OpenDataArena/ODA-scored-data-2603). For metric definitions and computation methods, refer to the ODA-scored-data-2603 dataset documentation.

---

## Data Scoring

R-Select requires multi-dimensional quality scores for each data sample before optimization. The project includes [ODA-DataScorer](ODA-DataScorer/) as a submodule, providing **60+ scorers**:

| Category | Description | Examples |
|----------|-------------|----------|
| **Model-based** | Neural model-based scoring (reward models, classifiers, etc.) | AtheneScorer, PPLScorer, DeitaQScorer, IFDScorer, ... |
| **Heuristic** | Statistical and rule-based metrics | TokenLengthScorer, MtldScorer, VocdDScorer, CompressRatioScorer, ... |

> For the full scorer list and parameter descriptions, see [ODA-DataScorer/README.md](ODA-DataScorer/README.md).

### Step 1: Define Scoring Metrics

List the metric names to use in `scores.txt` (one per line). For example, `configs/demo/scores.txt` defines the 30 metrics used in the demo:

```
AtheneRM
Cleanliness
LLM_as_Judge_Complexity
Compress_Ratio
Deita_Complexity
Deita_Quality
...
```

These names will be used as keys in the data pool `scores` dictionary and referenced in R-Select configs.

### Step 2: Install ODA-DataScorer

No separate installation is required; ODA-DataScorer dependencies are covered in the "Environment Setup" section above. If you encounter missing dependencies or installation issues later, refer to the README and requirements.txt under the ODA-DataScorer directory and install or troubleshoot as needed.

### Step 3: Prepare Input Data

ODA-DataScorer input is JSONL and must include at least `instruction` and `output` fields:

```json
{"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."}
{"instruction": "Explain quantum entanglement.", "input": "", "output": "Quantum entanglement is a phenomenon..."}
```

### Step 4: Configure and Run Scorer

Create an ODA-DataScorer YAML config according to your defined metrics. Examples for each category:

**Model-based scorers** — run under `ODA-DataScorer/model_based/`:

```yaml
# ODA-DataScorer/model_based/configs/MyScoring.yaml
input_path: ../../data/my_data.jsonl
output_path: results/MyScoring/
num_gpu: 4
num_gpu_per_job: 1
scorers:
- name: AtheneScorer
  model: Nexusflow/Athene-RM-8B
  batch_size: 8
  max_length: 4096
- name: PPLScorer
  model: meta-llama/Llama-3.1-8B
  batch_size: 8
  max_length: 4096
- name: DeitaQScorer
  model: hkust-nlp/deita-quality-scorer
  batch_size: 8
  max_length: 2048
```

```bash
cd ODA-DataScorer/model_based
python main.py --config configs/MyScoring.yaml
```

**Heuristic scorers** — run under `ODA-DataScorer/heuristic/`:

```yaml
# ODA-DataScorer/heuristic/configs/MyHeuristic.yaml
input_path: ../../data/my_data.jsonl
output_path: results/MyHeuristic/
num_gpu: 0
num_gpu_per_job: 0
scorers:
- name: TokenLengthScorer
  encoder: o200k_base
  fields: [instruction, input, output]
  max_workers: 128
- name: MtldScorer
  fields: [output]
  max_workers: 128
- name: CompressRatioScorer
  fields: [output]
  max_workers: 128
```

```bash
cd ODA-DataScorer/heuristic
python main.py --config configs/MyHeuristic.yaml
```

### Step 5: Merge Scores into the Data Pool

**Score output location**: Under each ODA-DataScorer module’s output directory there will be `pointwise_scores.jsonl` (e.g., model-based: `ODA-DataScorer/model_based/results/<task_name>/pointwise_scores.jsonl`; heuristic similarly under its results directory).

**ODA-DataScorer output format**: One record per line with `id` and nested `scores`, e.g.:

```json
{"id": 0, "scores": {"AtheneScorer": {"score": 4.5}, "PPLScorer": {"score": 12.34}}}
```

**R-Select required format**: Merge scores back into the original pool so each sample is one JSONL line and `scores` is a flat key → value dictionary, e.g.:

```json
{"id": 0, "instruction": "...", "input": "", "output": "...", "scores": {"AtheneScorer": 4.5, "PPLScorer": 12.34}}
```

So you need to: read from `pointwise_scores.jsonl`, flatten nested `{"score": value}` to a single value, merge by `id` into the original data, and produce a pool file with `id`, `instruction`, `input`, `output`, and `scores`. If you score with other tools or custom score names, ensure **scores.txt**, **R-Select input data**, and **score names in the R-Select optimization config** are consistent.

> For more detailed scorer documentation, config options, and advanced usage (data parallelism, resume, etc.), see [ODA-DataScorer README](ODA-DataScorer/README.md).

---

## Pipeline Overview

```
Scoring (ODA-DataScorer) → Merge into pool → Normalization → [Data clustering] → Index generation → [Metric clustering] → Layer1 optimization → Aggregation + normalization → Layer2 optimization → Final weighting → Sampling
```

- **Scoring**: Use ODA-DataScorer to compute multi-dimensional scores for each sample (see [Data Scoring](#data-scoring))
- **[]** denotes optional steps
- **Normalization**: Windsorize and [0,1] normalize `scores`
- **Data clustering**: For diversity sampling (top k% per cluster)
- **Metric clustering**: Group related metrics for hierarchical optimization
- **Optimization**: Optuna searches score weights to minimize proxy model eval loss

> **Note**: The demo uses ~30 metrics, two optimization layers, and 6 Layer1 clusters. The framework supports any number of metrics, layers, and cluster partitions; write configs as needed.

---

## Quick Start

The full demo flow is in `run.sh`. Step overview:

```bash
# 0. Scoring (skip if data already has scores, e.g. data/demo.jsonl)
#    Use ODA-DataScorer to compute scores, then merge into pool JSONL
#    See "Data Scoring" section for details

# 1. Normalization
python utils/score_normalization.py -i data/demo.jsonl -o data/demo_normalized.jsonl \
  --pct-range 5 95 --keep-original --flip-keys PPL Normalized_Loss

# 2. Generate index (speed up optimization)
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

# 6. Compute final score and sample
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

- `--pct-range`: Clip bounds (e.g., 5~95 means truncate beyond p5/p95)
- `--keep-original`: Keep original values in `*_orig` fields
- `--flip-keys`: For "lower is better" metrics, use 1 - normalized after normalization

### 2. Metric Clustering (Optional, for Hierarchical Optimization)

Cluster scores by correlation to define Layer1 clusters:

```bash
python utils/metrics_clustering.py \
  -i data/demo_normalized.jsonl \
  -o data/metrics_clustering/cluster_results.txt \
  --sample_size 10000 \
  --n_clusters 6 \
  --scores data/scores.txt \
  --use_absolute_corr
```

If `--scores` is not specified, all scores are clustered; otherwise only the given scores.

> **Note**: Cluster assignment for hierarchical optimization should follow clustering results. `configs/demo` is for demonstration only and splits 30 metrics evenly into 6 clusters without strict clustering; in practice, write Layer1 configs based on `metrics_clustering.py` output.

### 3. Data Clustering (Optional, for Diversity Sampling)

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

Precompute score index to speed up Optuna optimization (recommended for million-scale samples):

```bash
python utils/precompute_index.py \
  --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl
```

### 5. Optuna Optimization (main.py)

Each config corresponds to one cluster in one layer; can be run in parallel or sequentially:

```bash
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
python main.py --config configs/demo/Layer1/Layer1_c2.yaml
# ... Layer1_c3 ~ Layer1_c6
```

Outputs include `trial_history.pdf`, `best_weights.json`, etc. Supports early stopping, warmup, and enqueue.

### 6. Inter-layer Aggregation (global_aggregation)

Aggregate `best_weights` from each Layer1 cluster, write new scores (e.g., Layer1_c1, Layer1_c2, etc.), and min-max normalize. When `-o` is not specified, the input file is modified in place. Regenerate index after aggregation:

```bash
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1

# Regenerate index after inter-layer aggregation
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl --output_index data/demo_normalized_index.pkl
```

### 7. Layer2 Aggregation and Optimization

After Layer1 aggregation, data will contain new weighted scores from each Layer1 cluster (Layer1_c1, Layer1_c2, etc.). Then run Layer2 optimization. Layer2 config is similar to Layer1; each config usually corresponds to a higher-level cluster. Run Optuna again to get Layer2_c1 `best_weights`:

```bash
python main.py --config configs/demo/Layer2/Layer2_c1.yaml
```

After Layer2 optimization, `trial_history.pdf` and `best_weights.json` are produced, giving the weighted combination of the previous layer (Layer1_c1, Layer1_c2, etc.).

### 8. Final Weighting and Sampling

Use the top-level cluster’s `best_weights` to compute `final_score`, then sample by top-k%:

```bash
# Compute final_score (weights apply to Layer1_c1, etc. in scores)
python utils/cluster_aggregation.py \
  -i data/demo_normalized.jsonl \
  -w results/demo/Layer2/Layer2_c1/best_weights.json

# Diversity sampling: top k% per cluster
python utils/sampling.py \
  -i data/demo_normalized.jsonl \
  -o results/demo/demo_diversity_sampled.jsonl \
  -k 10 --per_cluster -s final_score

# Global sampling: global top k%
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

Proxy model training config is an important part of R-Select. This demo uses [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) as the proxy model. Training parameters and DeepSpeed config are in `configs/demo/demo_proxy.yaml`, referenced via `base_train_yaml`. You can modify this file according to GPU memory, data scale, epochs, etc.

### Data Format Requirements

**pool_jsonl**: Data pool; must contain a `scores` dictionary (or the field specified by `score_field`).

**val_jsonl**: Validation set; **must be in Alpaca format**, with at least `instruction`, `input`, `output`:

```json
{"instruction": "Question or task description", "input": "Additional input (can be empty)", "output": "Expected answer"}
```

LlamaFactory uses these fields to build SFT samples. If column names differ, configure column mapping in the dataset section of `base_train_yaml`.

### Configuration Parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `pool_jsonl` | ✓ | str | Path to pool JSONL |
| `val_jsonl` | ✓ | str | Path to validation JSONL (Alpaca format) |
| `top_k` | ✓ | int | Number of samples selected per trial by weighted score |
| `base_train_yaml` | ✓ | str | LlamaFactory training config template |
| `run_dir` | ✓ | str | Output directory for current run (trials, best_weights, etc.) |
| `study_name` | ✓ | str | Optuna study name |
| `storage` | ✓ | str | Optuna storage URL, e.g. `sqlite:///path/to/study.db` |
| `n_trials` | ✓ | int | Maximum number of trials |
| `score_keys` | ✓ | list/str | Score keys used in weighting. Layer1: raw metrics; Layer2: Layer1 cluster names |
| `score_field` | | str | Field containing scores; default `scores` |
| `pool_index` | | str | Precomputed index path; recommended for large pools |
| `seed` | | int | Random seed |
| `num_train_epochs` | | float | Proxy training epochs |
| `eval_strategy` | | str | Evaluation strategy, e.g. `epoch` |
| `per_device_eval_batch_size` | | int | Eval batch size; 0 means same as training |
| `cuda_visible_devices` | | str | Visible GPUs, e.g. `0,1,2,3` |
| `nproc_per_node` | | int | GPUs per node for distributed training |
| `master_port` | | int | Distributed port; 0 for auto |
| `llama_factory_dir` | | str | LlamaFactory root; empty to use in-project LlamaFactory |
| `normalization_method` | | str | Weight normalization: `softmax` or `linear` |
| `suggest_range` | | str/list | Optuna search range, e.g. `-5,5` or `[0, 1]` |
| `n_startup_trials` | | int | TPE random exploration trials |
| `tpe_multivariate` | | bool | Enable multivariate TPE |
| `tpe_n_ei_candidates` | | int | EI candidate count |
| `tpe_gamma_percent` | | float | Gamma fraction |
| `tpe_gamma_max` | | int | Gamma upper bound |
| `early_stop_patience` | | int | Early stopping patience; 0 to disable |
| `early_stop_delta` | | float | Early stopping minimum improvement |
| `warmup_trials_json` | | str | Warm-start: path to existing trials to inject |
| `enqueue_weights_json` | | str | Warm-start: path to weight queue to evaluate first |
| `cache_pool_in_memory` | | bool | Cache pool in memory (for small data) |
| `max_trials_in_dir` | | int | Safety limit on max trials per directory |
| `running_timeout_hours` | | float | Hours after which RUNNING trial is treated as zombie |

---

## Warm-start

Two warm-start modes are supported to speed up convergence or prioritize specific configurations.

### Recommended: Warm Trajectory Generation Covering the Search Space

Use **Dirichlet distribution** with **boundary configurations** to generate warm trajectories that cover the weight search space:

- **Dirichlet distribution**: Sample from `Dir(α,...,α)` on the simplex. Smaller α gives sparser weights (sharper peaks); α=1 is uniform; α>1 is more concentrated at the center. Using several α (e.g., 0.1, 0.5, 1.0, 2.0) for the same `score_keys` yields diverse weight samples.
- **Boundary configurations**: Add simplex vertices and boundary cases, e.g. one-hot (one dimension 1, rest 0), uniform (1/k), etc., to cover corners and the uniform solution.

Warmup or enqueue configs generated this way better guide TPE to explore the feasible region.

### 1. warmup_trials_json

Inject **completed** (weights, eval_loss) as prior; **no retraining**. TPE uses them to adjust sampling.

**JSON format**: Array of `{weights, eval_loss}`:

```json
[
  {"weights": {"AtheneScore": 0.2, "CleanlinessScore": 0.3, ...}, "eval_loss": 6.83},
  {"weights": {"AtheneScore": 0.1, "CleanlinessScore": 0.4, ...}, "eval_loss": 6.92}
]
```

- `weights`: Keys must match `score_keys`
- `eval_loss`: Validation loss for that configuration

**Example**: `configs/demo/warmup_trials_example.json`

### 2. enqueue_weights_json

Add weight configurations to the **evaluation queue**; they **are actually trained** and run before TPE-sampled trials.

**JSON format**: Array of **plain weight dictionaries** (no `weights` wrapper, no `eval_loss`):

```json
[
  {"AtheneScore": 0.2, "CleanlinessScore": 0.2, "ComplexityLaJScore": 0.2, ...},
  {"AtheneScore": 0.133, "CleanlinessScore": 0.338, ...}
]
```

**Example**: `configs/demo/enqueue_weights_example.json`

### Comparison

| Aspect | warmup_trials_json | enqueue_weights_json |
|--------|--------------------|----------------------|
| Training | No (prior only) | Yes |
| Element format | `{weights, eval_loss}` | Plain weight dict |
| Typical use | Transfer existing best results | Prioritize specific weight combinations |

See `configs/demo/warmstart_README.md` for details.

---

## FAQ

**Q: How do I determine Layer1 cluster assignment?**  
A: Use `metrics_clustering.py` to cluster scores by correlation; each cluster corresponds to one YAML config.

**Q: Will warmup repeat when running the same study multiple times?**  
A: No. The script deduplicates by weight signature; already injected trials are skipped.

**Q: Data has no cluster_id, what to do?**  
A: Skip data clustering and use `--global` mode for global top-k% sampling.

**Q: How do I view weights aggregated across layers?**  
A: Use `summarization.py`:
```bash
python utils/summarization.py -i results/demo -o results/demo/weight_summary.json
```

---

## Project Structure

```
R-Select/
├── main.py                 # Optuna optimization main program
├── run.sh                  # Example pipeline script
├── requirements.txt        # R-Select dependencies
├── readme.md / readme_en.md
├── ODA-DataScorer/         # Data scoring submodule (60+ scorers)
│   ├── model_based/        #   Neural model scorers
│   └── heuristic/          #   Statistical/rule-based scorers
├── LlamaFactory/           # SFT training framework (submodule)
├── configs/
│   └── demo/
│       ├── demo_proxy.yaml # LlamaFactory proxy training config
│       ├── scores.txt      # Metric names used in this demo
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
│   └── scores.txt          # Default metric names
└── results/                # Optimization results
    └── demo/
        ├── Layer1/
        ├── Layer2/
        └── weight_summary.json
```

---

## References

This project builds on the following open-source frameworks; thanks to:

- **[ODA-DataScorer](https://github.com/OpenDataArena/ODA-DataScorer)** — Multi-dimensional data scoring toolkit with 60+ scorers (model-based, heuristic). Used to compute quality/complexity/diversity scores for each data sample before optimization.
- **[LlamaFactory](https://github.com/hiyouga/LlamaFactory)** — Unified efficient fine-tuning framework for LLMs/VLMs, supporting 100+ models (ACL 2024). Used for proxy model SFT training and eval loss computation.
- **[Optuna](https://github.com/optuna/optuna)** — Hyperparameter optimization framework. Used for TPE (Tree-structured Parzen Estimator) sampler to search for optimal weights of scoring metrics.
