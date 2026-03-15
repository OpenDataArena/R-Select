# R-Select：SFT 数据选择流水线

**R-Select** 是基于 [Optuna](https://github.com/optuna/optuna) 与 [LlamaFactory](https://github.com/hiyouga/LlamaFactory) 的层次化分数权重优化流水线，用于从大规模数据池中选出高质量子集用于 SFT（Supervised Fine-Tuning）训练。框架支持任意层数、任意 cluster 划分与指标数量。

---

## 方法简介

核心思路：给定多个打分指标（如 IFD、PPL、Length 等），通过 **proxy 模型**（小模型在选定子集上 SFT）的验证 loss 作为反馈信号，用 **Optuna TPE** 搜索各指标的最优权重，使得加权分数 `weighted_score = Σ(weight_i × score_i)` 选出的 top-k 数据在验证集上 loss 最低。支持 **层次化优化**：先将指标按相关性聚类为若干组（Layer1），每组独立优化权重；再在 Layer2 中优化各组的权重，从而降低搜索空间、提升稳定性。最终根据优化得到的权重对数据加权打分并采样。

---

## 目录

- [环境配置](#环境配置)
- [数据格式要求](#数据格式要求)
- [流水线概览](#流水线概览)
- [快速开始](#快速开始)
- [main.py 使用说明](#mainpy-使用说明)
- [Warm-start](#warm-start)
- [详细步骤](#详细步骤)
- [工具脚本说明](#工具脚本说明)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [参考文献](#参考文献)

---

## 环境配置

推荐使用 `rselect` conda 环境运行本项目。

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/GX-XinGao/R-Select.git
cd R-Select

# 2. 创建并激活环境
conda create -n rselect python=3.10 -y
conda activate rselect

# 3. 安装 PyTorch（按 CUDA 版本选择，示例为 CUDA 12.4）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 LlamaFactory 及核心依赖（LlamaFactory 已包含于项目内）
cd LlamaFactory
pip install -e .
pip install deepspeed==0.16.9
pip install optuna==4.7.0
conda install -c conda-forge pandas=2.3.3
pip install seaborn==0.13.2

# 5. 可选：embed 脚本（数据聚类）需要 vLLM，不用时可跳过
pip install vllm==0.8.5.post1
```

### 说明

- `main.py` 通过 LlamaFactory 进行 proxy 训练，需保证 transformers、peft、deepspeed 等与 `LlamaFactory/requirements.txt` 兼容
- 若遇版本冲突，可参考 `LlamaFactory/requirements.txt` 逐项安装

### 运行

```bash
conda activate rselect
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
```

---

## 数据格式要求

### pool_jsonl（数据池）

原始数据需整理为 JSONL 格式，每行一个 JSON 对象，包含至少以下字段：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识 |
| `instruction` | 指令/问题 |
| `input` | 输入（可为空） |
| `output` | 回答/输出 |
| `scores` | 打分字典，键名为各评分指标 |

### val_jsonl（验证集）

示例代码中的验证集数据由以下五个 benchmark 采样制作：
- https://huggingface.co/datasets/bigcode/bigcodebench
- https://huggingface.co/datasets/Idavidrein/gpqa
- https://huggingface.co/datasets/google-research-datasets/mbpp
- https://huggingface.co/datasets/cais/mmlu
- https://huggingface.co/datasets/KbsdJames/Omni-MATH

用户可根据自身下游任务场景，自行选择或构建合适的验证集。

验证集文件（val_jsonl）**必须满足 Alpaca 格式**，即每个样本需包含 `instruction`、`input`、`output` 三个键，供 LlamaFactory 构造 SFT 样本并计算 eval loss：

```json
{"instruction": "任务描述", "input": "附加输入（可为空字符串）", "output": "期望回答"}
```


### pool_jsonl 示例

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

### Demo 示例数据说明

示例代码中使用的 `data/demo.jsonl` 基于 [alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) 数据集，并采用 [OpenDataArena/ODA-scored-data-2603](https://huggingface.co/datasets/OpenDataArena/ODA-scored-data-2603) 中 **alpaca_gpt4** 子集的带 30 个打分指标的版本。各指标的含义与计算方法可参考 ODA-scored-data-2603 数据集的说明。

---

## 流水线概览

```
原始数据 → 归一化 → [数据聚类] → 生成索引 → [指标聚类] → Layer1 优化 → 聚合+归一化 → Layer2 优化 → 最终加权 → 采样
```

- **[]** 表示可选步骤
- **归一化**：对 `scores` 进行温莎化与 [0,1] 归一化
- **数据聚类**：用于多样化采样（在每个 cluster 内取 top-k%）
- **指标聚类**：将相关指标分组，用于层次优化
- **优化**：Optuna 搜索各 score 的权重，使 proxy 模型 eval loss 最小

> **说明**：本仓库 demo 使用约 30 个指标、两层优化、第一层 6 个 cluster，仅为示例。框架支持任意指标数、任意层数与 cluster 划分，按需编写 config 即可。

---

## 快速开始

以 demo 为例，完整流程见 `run.sh`。简要步骤：

```bash
# 1. 归一化
python utils/score_normalization.py -i data/demo.jsonl -o data/demo_normalized.jsonl \
  --pct-range 5 95 --keep-original --flip-keys PPL Normalized_Loss

# 2. 生成索引（加速优化）
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl

# 3. 运行 Layer1 优化（6 个 cluster 可并行）
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
# ... Layer1_c2 ~ Layer1_c6

# 4. 聚合 Layer1 结果，写入新 score（原地修改）
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1

# 5. 重新生成索引后运行 Layer2
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl
python main.py --config configs/demo/Layer2/Layer2_c1.yaml

# 6. 计算最终分数并采样
python utils/cluster_aggregation.py -i data/demo_normalized.jsonl \
  -w results/demo/Layer2/Layer2_c1/best_weights.json
python utils/sampling.py -i data/demo_normalized.jsonl -o data/sampled.jsonl \
  -k 10 --per_cluster -s final_score
```

---

## 详细步骤

### 1. 归一化

对 `scores` 中数值按百分位裁剪并归一化到 [0,1]：

```bash
python utils/score_normalization.py \
  -i data/demo.jsonl \
  -o data/demo_normalized.jsonl \
  --pct-range 5 95 \
  --keep-original \
  --flip-keys PPL Normalized_Loss
```

- `--pct-range`：裁剪上下界（如 5~95 表示超出 p5/p95 的截断）
- `--keep-original`：保留原始值到 `*_orig` 字段
- `--flip-keys`：对「越小越好」的指标归一化后用 1 减

### 2. 指标聚类（可选，用于层次优化）

将相关 score 聚类，用于划分 Layer1 的多个 cluster：

```bash
python utils/metrics_clustering.py \
  -i data/demo_normalized.jsonl \
  -o data/metrics_clustering/cluster_results.txt \
  --sample_size 10000 \
  --n_clusters 6 \
  --scores data/scores.txt \
  --use_absolute_corr
```

不指定 `--scores` 时默认对所有 score 聚类；指定时只对给定 score 聚类。

> **说明**：层次优化的 cluster 划分应依据聚类结果。`configs/demo` 仅为演示，将 30 个指标均分为 6 簇，未严格按聚类结果划分；实际使用时请根据 `metrics_clustering.py` 输出编写 Layer1 配置。

### 3. 数据聚类（可选，用于多样化采样）

先提取 embedding，再聚类，最后为每条数据分配 `cluster_id`：

```bash
# 提取 embedding
python utils/embedding.py \
  --embedder_model /path/to/embedding_model \
  --input_path data/demo_normalized.jsonl \
  --output_path data/samples_clustering/demo_normalized_embeddings.npy \
  --fields instruction input \
  --tensor_parallel_size 8

# 聚类
python utils/samples_clustering.py --input_path data/samples_clustering/demo_normalized_embeddings.npy \
  --output_dir data/samples_clustering --opt_k 50

# 分配 cluster_id
python utils/assign_cluster_id.py \
  --input_jsonl data/demo_normalized.jsonl \
  --labels_path data/samples_clustering/cluster_labels.npy
```

### 4. 生成索引文件

预计算 score 索引，加速 Optuna 优化（百万级数据推荐）：

```bash
python utils/precompute_index.py \
  --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl
```

### 5. Optuna 优化（main.py）

每个 config 对应一个 Layer 下的一个 cluster，可并行或串行运行：

```bash
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
python main.py --config configs/demo/Layer1/Layer1_c2.yaml
# ... Layer1_c3 ~ Layer1_c6
```

运行时会生成 `trial_history.pdf`、`best_weights.json` 等。支持早停、warmup、enqueue 等。

### 6. 层级间聚合（global_aggregation）

将 Layer1 各 cluster 的 `best_weights` 加权聚合，写入新的 score（如 Layer1_c1、Layer1_c2…），并进行 min-max 归一化。不指定 `-o` 时原地修改输入文件。聚合后需重新生成索引以加速后续优化：

```bash
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1

# 层间聚合后，需重新进行索引预计算
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl --output_index data/demo_normalized_index.pkl
```



### 7. Layer2 聚合与优化

经过 Layer1 层级聚合后，会在数据中新增各 Layer1 cluster 的加权分数（如 Layer1_c1、Layer1_c2...）。此时，进入 Layer2 的优化过程。Layer2 的配置方式与 Layer1 类似，每个 config 通常对应聚合后的一个更高层 cluster。再次通过 Optuna 优化，得到 Layer2_c1 的 `best_weights`：

```bash
python main.py --config configs/demo/Layer2/Layer2_c1.yaml
```

Layer2 优化完成后，也会生成对应的 `trial_history.pdf`、`best_weights.json` 文件，形成针对上一层（如 Layer1_c1、Layer1_c2...）结果的综合权重。

### 8. 最终加权与采样

用顶层 cluster 的 `best_weights` 计算 `final_score`，再按 top-k% 采样：

```bash
# 计算 final_score（权重针对 scores 中的 Layer1_c1 等键）
python utils/cluster_aggregation.py \
  -i data/demo_normalized.jsonl \
  -w results/demo/Layer2/Layer2_c1/best_weights.json

# 多样性采样：每个 cluster 内取 top k%
python utils/sampling.py \
  -i data/demo_normalized.jsonl \
  -o results/demo/demo_diversity_sampled.jsonl \
  -k 10 --per_cluster -s final_score

# 全局采样：全量数据取 top k%
python utils/sampling.py \
  -i data/demo_normalized.jsonl \
  -o results/demo/demo_global_sampled.jsonl \
  -k 10 --global -s final_score
```

---

## 工具脚本说明

| 脚本 | 功能 |
|------|------|
| `score_normalization.py` | 温莎化 + [0,1] 归一化 |
| `metrics_clustering.py` | 按 Pearson 相关对 score 聚类 |
| `embedding.py` | 提取文本 embedding |
| `samples_clustering.py` | K-Means 数据聚类 |
| `assign_cluster_id.py` | 为数据分配 cluster_id |
| `precompute_index.py` | 预计算 pool 索引 |
| `global_aggregation.py` | 用 best_weights 聚合 Layer 分数 |
| `cluster_aggregation.py` | 用权重文件计算 final_score |
| `sampling.py` | 按 score 取 top-k%（per_cluster / global） |
| `summarization.py` | 汇总多层权重与层次结构 |
| `trials_analysis.py` | 读取 Optuna trials 详情 |

---

## main.py 使用说明

### 基本用法

```bash
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
```

### Proxy 模型配置

R-Select 中 proxy 模型的训练配置是重要一环。本 demo 使用 [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) 作为 proxy 模型，具体训练参数与 DeepSpeed 等设置在 `configs/demo/demo_proxy.yaml` 中，通过 `base_train_yaml` 引用。用户可根据自身需求（显存、数据量、训练轮数等）修改该配置文件。

### 数据格式要求

**pool_jsonl**：数据池，需包含 `scores` 字典（或通过 `score_field` 指定字段）。

**val_jsonl**：验证集，**必须是 Alpaca 格式**，至少包含 `instruction`、`input`、`output` 三个键：

```json
{"instruction": "问题或任务描述", "input": "附加输入（可为空）", "output": "期望回答"}
```

LlamaFactory 会据此构造 SFT 样本；若字段名不同，需在 `base_train_yaml` 的 dataset 配置中做列映射。

### 配置参数详解

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `pool_jsonl` | ✓ | str | 数据池 JSONL 路径 |
| `val_jsonl` | ✓ | str | 验证集 JSONL 路径（Alpaca 格式） |
| `top_k` | ✓ | int | 每次 trial 从池中按加权分选出的样本数 |
| `base_train_yaml` | ✓ | str | LlamaFactory 训练配置模板 |
| `run_dir` | ✓ | str | 当前任务输出目录（trials、best_weights 等） |
| `study_name` | ✓ | str | Optuna study 名称 |
| `storage` | ✓ | str | Optuna 存储地址，如 `sqlite:///path/to/study.db` |
| `n_trials` | ✓ | int | 最大 trial 数 |
| `score_keys` | ✓ | list/str | 参与加权的 score 键名。Layer1 为原始指标；Layer2 为 Layer1 cluster 名 |
| `score_field` | | str | scores 所在字段，默认 `scores` |
| `pool_index` | | str | 预计算索引路径，推荐大池使用 |
| `seed` | | int | 随机种子 |
| `num_train_epochs` | | float | proxy 训练 epoch 数 |
| `eval_strategy` | | str | 验证策略，如 `epoch` |
| `per_device_eval_batch_size` | | int | 验证 batch 大小，0 表示与 train 相同 |
| `cuda_visible_devices` | | str | 可见 GPU，如 `0,1,2,3` |
| `nproc_per_node` | | int | 每节点 GPU 数量，用于分布式训练 |
| `master_port` | | int | 分布式端口，0 表示自动 |
| `llama_factory_dir` | | str | LlamaFactory 根目录，空则用项目内 LlamaFactory |
| `normalization_method` | | str | 权重归一化：`softmax` 或 `linear` |
| `suggest_range` | | str/list | Optuna 搜索范围，如 `-5,5` 或 `[0, 1]` |
| `n_startup_trials` | | int | TPE 随机探索 trial 数 |
| `tpe_multivariate` | | bool | 是否启用多变量 TPE |
| `tpe_n_ei_candidates` | | int | EI 候选数 |
| `tpe_gamma_percent` | | float | gamma 比例 |
| `tpe_gamma_max` | | int | gamma 上限 |
| `early_stop_patience` | | int | 早停 patience，0 表示不早停 |
| `early_stop_delta` | | float | 早停最小改善量 |
| `warmup_trials_json` | | str | Warm-start：已有 trial 注入路径 |
| `enqueue_weights_json` | | str | Warm-start：待优先评估的权重队列路径 |
| `cache_pool_in_memory` | | bool | 是否将池缓存到内存（小数据可用） |
| `max_trials_in_dir` | | int | 单目录最大 trial 数安全限制 |
| `running_timeout_hours` | | float | RUNNING 超时视为僵尸的小时数 |

---

## Warm-start

支持两种 Warm-start 方式，用于加速收敛或优先尝试指定配置。

### 推荐：覆盖搜索空间的 Warm 轨迹生成

进行 Warm-start 时，建议使用 **狄利克雷分布（Dirichlet）** 配合 **边界分布** 来生成 Warm 轨迹，以充分覆盖权重搜索空间：

- **狄利克雷分布**：`Dir(α,...,α)` 在 simplex 上采样。α 越小越稀疏（接近单峰），α=1 为均匀，α>1 更集中于中心。可对同一 `score_keys` 采用多种 α（如 0.1、0.5、1.0、2.0）采样多组权重。
- **边界分布**：补充 simplex 顶点与边界的配置，例如 one-hot（某一维为 1、其余为 0）、均匀（1/k）等，保证角点与均匀解被覆盖。

这样生成的 warmup 或 enqueue 配置能更好地引导 TPE 探索整个可行域。

### 1. warmup_trials_json

将**已完成**的 (weights, eval_loss) 作为先验注入，**不重新训练**，TPE 会据此调整采样。

**JSON 格式**：数组，每项为 `{weights, eval_loss}`：

```json
[
  {"weights": {"AtheneScore": 0.2, "CleanlinessScore": 0.3, ...}, "eval_loss": 6.83},
  {"weights": {"AtheneScore": 0.1, "CleanlinessScore": 0.4, ...}, "eval_loss": 6.92}
]
```

- `weights`：键名与 `score_keys` 一致
- `eval_loss`：该配置下的验证 loss

**示例**：`configs/demo/warmup_trials_example.json`

### 2. enqueue_weights_json

将权重配置加入**待评估队列**，会**实际训练**并优先于 TPE 采样执行。

**JSON 格式**：数组，每项为**纯权重字典**（无 `weights` 外壳、无 `eval_loss`）：

```json
[
  {"AtheneScore": 0.2, "CleanlinessScore": 0.2, "ComplexityLaJScore": 0.2, ...},
  {"AtheneScore": 0.133, "CleanlinessScore": 0.338, ...}
]
```

**示例**：`configs/demo/enqueue_weights_example.json`

### 对比

| 维度 | warmup_trials_json | enqueue_weights_json |
|------|-------------------|----------------------|
| 是否训练 | 否（仅作先验） | 是 |
| 元素格式 | `{weights, eval_loss}` | 纯权重 dict |
| 典型用途 | 迁移已有最优结果 | 优先尝试指定权重组合 |

详见 `configs/demo/warmstart_README.md`。

---

## 常见问题

**Q: 如何确定 Layer1 的 cluster 划分？**  
A: 使用 `metrics_clustering.py` 对 score 做相关性聚类，每个 cluster 对应一个 YAML 配置。

**Q: 多次在同一 study 上运行会重复 warmup 吗？**  
A: 不会，脚本会按权重签名去重，已注入的会跳过。

**Q: 数据没有 cluster_id 怎么办？**  
A: 可跳过数据聚类，使用 `--global` 模式做全局 top-k% 采样。

**Q: 如何查看各层权重的汇总？**  
A: 使用 `summarization.py`：
```bash
python utils/summarization.py -i results/demo -o results/demo/weight_summary.json
```

---

## 项目结构

```
sft_data_selection_pipeline/
├── main.py                 # Optuna 优化主程序
├── run.sh                  # 示例流水线脚本
├── readme.md
├── configs/
│   └── demo/
│       ├── demo_proxy.yaml # LlamaFactory 代理训练配置
│       ├── Layer1/         # Layer1 各 cluster 配置
│       └── Layer2/         # Layer2 配置
├── utils/                  # 工具脚本
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
├── data/                   # 数据目录
└── results/                # 优化结果
    └── demo/
        ├── Layer1/
        ├── Layer2/
        └── weight_summary.json
```

---

## 参考文献

本项目基于以下开源框架构建，在此致谢：

- **[LlamaFactory](https://github.com/hiyouga/LlamaFactory)** — 统一的 LLM/VLM 高效微调框架，支持 100+ 模型（ACL 2024）。本项目用于 proxy 模型的 SFT 训练及 eval loss 计算。
- **[Optuna](https://github.com/optuna/optuna)** — 超参数优化框架。本项目使用其 TPE（Tree-structured Parzen Estimator）采样器搜索各评分指标的最优权重。
