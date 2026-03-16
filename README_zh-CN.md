# R-Select: SFT 数据选择流水线

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

**R-Select** 是基于 [Optuna](https://github.com/optuna/optuna) 和 [LlamaFactory](https://github.com/hiyouga/LlamaFactory) 的层次化分数权重优化流水线，用于从大规模数据池中选出高质量子集用于 SFT（监督微调）训练。框架支持任意层数、聚类划分和指标数量。

---

## 方法概述

**核心思路**：给定多个打分指标（如 IFD、PPL、Length 等），流水线以 **proxy 模型**（在所选子集上微调的小模型）的验证 loss 作为反馈信号，通过 **Optuna TPE** 搜索各指标的最优权重，使加权分数 `weighted_score = Σ(weight_i × score_i)` 选出的 top-k 数据在验证集上 loss 最低。

**层次化优化**：先将指标按相关性聚类成组（Layer1），各组独立优化权重；再在 Layer2 中优化各组权重，缩小搜索空间并提升稳定性。最后按优化后的权重对数据进行加权与采样。

---

## 目录

- [环境设置](#环境设置)
- [数据格式要求](#数据格式要求)
- [数据打分](#数据打分)
- [流水线概述](#流水线概述)
- [快速开始](#快速开始)
- [main.py 用法](#mainpy-用法)
- [Warm-start](#warm-start)
- [详细步骤](#详细步骤)
- [工具脚本](#工具脚本)
- [FAQ](#faq)
- [项目结构](#项目结构)
- [参考文献](#参考文献)

---

## 环境设置

建议在 `rselect` conda 环境中运行本项目。

### 安装

```bash
# 1. 克隆仓库
git clone --recurse-submodules https://github.com/OpenDataArena/R-Select.git
cd R-Select

# 2. 创建并激活环境
conda create -n rselect python=3.10 -y
conda activate rselect

# 3. 安装 PyTorch（按 CUDA 版本选择，示例为 CUDA 12.4）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 LlamaFactory 及核心依赖（LlamaFactory 已包含于项目内）
conda install -c conda-forge pandas=2.3.3 pyarrow=22.0.0 av=16.0.1 sentencepiece=0.2.1 tiktoken=0.12.0 -y
cd LlamaFactory
pip install -e .
cd ..
pip install -r requirements.txt

# 5. 可选：embed 脚本（数据聚类）需要 vLLM，不用时可跳过
pip install vllm==0.8.5.post1
pip install numpy==1.26.3 # vllm 安装会导致 numpy 版本过高，需要手动回退到 1.26.3
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

原始数据须为 JSONL 格式，每行一个 JSON 对象，至少包含以下字段：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识符 |
| `instruction` | 指令/问题 |
| `input` | 输入（可为空） |
| `output` | 答案/输出 |
| `scores` | 分数字典，键为指标名 |

### val_jsonl（验证集）

示例代码中的验证集来自以下五个基准：
- [bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench)
- [gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
- [mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp)
- [mmlu](https://huggingface.co/datasets/cais/mmlu)
- [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH)

用户可根据下游任务选择或构建合适的验证集。

验证文件（val_jsonl）**必须遵循 Alpaca 格式**。每条样本须包含 `instruction`、`input`、`output` 三个键，以便 LlamaFactory 构建 SFT 样本并计算 eval loss：

```json
{"instruction": "任务描述", "input": "额外输入（可为空字符串）", "output": "期望答案"}
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

### Demo 数据说明

示例中的 `data/demo.jsonl` 基于 [alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) 数据集，使用 **alpaca_gpt4** 子集，包含来自 [OpenDataArena/ODA-scored-data-2603](https://huggingface.co/datasets/OpenDataArena/ODA-scored-data-2603) 的 30 个打分指标。指标定义与计算方法请参考 ODA-scored-data-2603 数据集文档。

---

## 数据打分
在进行优化前，R-Select 需为每条数据样本打上多维度的质量分数，打分方法完全灵活，无论是使用自定义脚本、人工打分还是直接采用现有带分数的数据集（如 [OpenDataArena/ODA-scored-data-2603](https://huggingface.co/datasets/OpenDataArena/ODA-scored-data-2603)），只要结果格式与下游流程兼容即可。

**可选：官方打分工具推荐**  
如果你需要自动、大规模地为数据打分，我们也提供了一个功能强大的评分工具 [ODA-DataScorer](ODA-DataScorer/)，集成于本项目。该工具支持 **60+ 个 scorer**，**80+ 个指标**，覆盖多维度质量、复杂度、启发式特征等：

| 类别 | 说明 | 示例 |
|------|------|------|
| **Model-based** | 基于神经模型的打分（奖励模型、分类器等） | AtheneScorer, PPLScorer, DeitaQScorer, IFDScorer, ... |
| **Heuristic** | 统计与规则类指标 | TokenLengthScorer, MtldScorer, VocdDScorer, CompressRatioScorer, ... |

> scorer 的完整列表及参数说明详见 [ODA-DataScorer/README.md](ODA-DataScorer/README.md)。


### 步骤 1：安装 ODA-DataScorer

无需额外单独安装依赖，已在前文“环境设置”部分涵盖了 ODA-DataScorer 依赖的安装。如果后续遇到依赖缺失或安装问题，请参考 ODA-DataScorer 目录下的 README 文档和 requirements.txt，按需补充安装和排查。

### 步骤 2：准备输入数据

ODA-DataScorer 的输入为 JSONL，至少需包含 `instruction` 和 `output` 字段：

```json
{"instruction": "法国的首都是哪里？", "input": "", "output": "法国的首都是巴黎。"}
{"instruction": "解释量子纠缠。", "input": "", "output": "量子纠缠是一种现象..."}
```

### 步骤 3：配置并运行 Scorer

你可以根据自己的需求选择需要使用哪些 scorer 及其参数，具体可参考 ODA-DataScorer 的 README.md 以及[数据评分文档](https://opendataarena-tool.readthedocs.io/en/latest/)查看所有官方内置 scorer 以及各个参数的说明。ODA-DataScorer 是一个开放且灵活的打分框架，也支持用户自定义 scorer。

**Model-based scorers** — 在 `ODA-DataScorer/model_based/` 下运行：

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

**Heuristic scorers** — 在 `ODA-DataScorer/heuristic/` 下运行：

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

### 步骤 4：将分数合并到数据池

**打分结果位置**：ODA-DataScorer 各模块的输出目录下会有 `pointwise_scores.jsonl`（例如 model-based 在 `ODA-DataScorer/model_based/results/<任务名>/pointwise_scores.jsonl`，heuristic 也一致，在其结果目录下）。

**ODA-DataScorer 输出格式**：每行一条记录，包含 `id` 和嵌套的 `scores`，例如：

```json
{"id": 0, "scores": {"AtheneScorer": {"score": 4.5}, "PPLScorer": {"score": 12.34}}}
```

**合并说明与建议**：

- 合并分数至原始数据池时，要将打分结果合并回原始数据（以 `id` 对齐），对于每个 scorer 可能会输出多个指标——请根据你的实际需要选择哪些指标要合并和后续使用。
- 合并后样例格式（R-Select 需要的输入格式）为：

```json
{"id": 0, "instruction": "...", "input": "", "output": "...", "scores": {"AtheneScorer": 4.5, "PPLScorer": 12.34}}
```

- **注意：项目未提供统一脚本进行这一处理，因每个 scorer 输出的指标种类和命名不一定完全一致。请根据你实际使用的 scorer 和指标，自行处理合并与扁平化。**
- 等你决定要用哪些指标后，将所有你需要参与优化与采样的分数字段名写进`scores.txt`，每行一个。例如 `configs/demo/scores.txt` 定义了 demo 使用的 30 个指标：

```
AtheneRM
Cleanliness
LLM_as_Judge_Complexity
Compress_Ratio
Deita_Complexity
Deita_Quality
...
```

scores.txt 内容应与数据池 JSONL 文件中的 `scores` 字段的键一一对应，并与 R-Select 的配置一致。


> 更详细的 scorer 文档、配置选项及高级用法（数据并行、resume 等）请参阅 [ODA-DataScorer README](ODA-DataScorer/README.md)。


---

## 流水线概述

```
打分（ODA-DataScorer）→ 合并到 pool → 归一化 → [数据聚类] → 索引生成 → [指标聚类] → Layer1 优化 → 聚合 + 归一化 → Layer2 优化 → 最终加权 → 采样
```

- **打分**：使用 ODA-DataScorer 为每条样本计算多维分数（见 [使用 ODA-DataScorer 进行数据打分](#使用-oda-datascorer-进行数据打分)）
- **[]** 表示可选步骤
- **归一化**：对 `scores` 进行温莎化及 [0,1] 归一化
- **数据聚类**：用于多样性采样（每个 cluster 内取 top k%）
- **指标聚类**：将相关指标分组，用于层次化优化
- **优化**：Optuna 搜索各分数权重，使 proxy 模型 eval loss 最小

> **说明**：Demo 以约 30 个指标、两层优化、6 个 Layer1 cluster 为例，框架支持任意数量的指标、层数和聚类划分，按需编写配置即可。

---

## 快速开始

Demo 的完整流程见 `run.sh`。步骤概览：

```bash
# 0. 打分（若数据已有分数如 data/demo.jsonl 可跳过）
#    使用 ODA-DataScorer 计算分数，再合并到 pool JSONL
#    详见「使用 ODA-DataScorer 进行数据打分」章节

# 1. 归一化
python utils/score_normalization.py -i data/demo.jsonl -o data/demo_normalized.jsonl \
  --pct-range 5 95 --keep-original --flip-keys PPL Normalized_Loss

# 2. 生成索引（加速优化）
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl

# 3. 运行 Layer1 优化（6 个 cluster 可并行）
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
# ... Layer1_c2 ~ Layer1_c6

# 4. 聚合 Layer1 结果并写入新分数（原地修改）
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1

# 5. 重新生成索引并运行 Layer2
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

按百分位裁剪 `scores` 并归一化到 [0,1]：

```bash
python utils/score_normalization.py \
  -i data/demo.jsonl \
  -o data/demo_normalized.jsonl \
  --pct-range 5 95 \
  --keep-original \
  --flip-keys PPL Normalized_Loss
```

- `--pct-range`：裁剪边界（如 5~95 表示超出 p5/p95 的截断）
- `--keep-original`：在 `*_orig` 字段中保留原始值
- `--flip-keys`：对「越小越好」的指标，归一化后做 1 - normalized

### 2. 指标聚类（可选，用于层次化优化）

按相关性对分数聚类，定义 Layer1 cluster：

```bash
python utils/metrics_clustering.py \
  -i data/demo_normalized.jsonl \
  -o data/metrics_clustering/cluster_results.txt \
  --sample_size 10000 \
  --n_clusters 6 \
  --scores data/scores.txt \
  --use_absolute_corr
```

若不指定 `--scores`，则对所有分数聚类；否则仅对给定分数聚类。

> **说明**：层次化优化的 cluster 划分应参考聚类结果。`configs/demo` 仅为演示，将 30 个指标均匀分为 6 个 cluster，未严格按聚类；实际使用时应根据 `metrics_clustering.py` 输出编写 Layer1 配置。

### 3. 数据聚类（可选，用于多样性采样）

提取 embedding、聚类，再为每条样本分配 `cluster_id`：

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

预计算分数索引以加速 Optuna 优化（百万级样本推荐）：

```bash
python utils/precompute_index.py \
  --pool_jsonl data/demo_normalized.jsonl \
  --output_index data/demo_normalized_index.pkl
```

### 5. Optuna 优化（main.py）

每个配置对应一层中的一个 cluster，可并行或顺序运行：

```bash
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
python main.py --config configs/demo/Layer1/Layer1_c2.yaml
# ... Layer1_c3 ~ Layer1_c6
```

输出包括 `trial_history.pdf`、`best_weights.json` 等。支持早停、warmup 和 enqueue。

### 6. 层间聚合（global_aggregation）

聚合各 Layer1 cluster 的 `best_weights`，写入新分数（如 Layer1_c1、Layer1_c2 等），并做 min-max 归一化。未指定 `-o` 时原地修改输入文件。聚合后需重新生成索引：

```bash
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1

# 层间聚合后重新生成索引
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl --output_index data/demo_normalized_index.pkl
```

### 7. Layer2 聚合与优化

Layer1 聚合后，数据将包含各 Layer1 cluster 的新加权分数（Layer1_c1、Layer1_c2 等）。随后进行 Layer2 优化。Layer2 配置与 Layer1 类似，每个配置通常对应更高层的 cluster。再次运行 Optuna 得到 Layer2_c1 的 `best_weights`：

```bash
python main.py --config configs/demo/Layer2/Layer2_c1.yaml
```

Layer2 优化完成后，会生成 `trial_history.pdf` 和 `best_weights.json`，得到对上一层（Layer1_c1、Layer1_c2 等）的加权组合。

### 8. 最终加权与采样

使用顶层 cluster 的 `best_weights` 计算 `final_score`，再按 top-k% 采样：

```bash
# 计算 final_score（权重作用于 scores 中的 Layer1_c1 等）
python utils/cluster_aggregation.py \
  -i data/demo_normalized.jsonl \
  -w results/demo/Layer2/Layer2_c1/best_weights.json

# 多样性采样：每个 cluster 内取 top k%
python utils/sampling.py \
  -i data/demo_normalized.jsonl \
  -o results/demo/demo_diversity_sampled.jsonl \
  -k 10 --per_cluster -s final_score

# 全局采样：全量 top k%
python utils/sampling.py \
  -i data/demo_normalized.jsonl \
  -o results/demo/demo_global_sampled.jsonl \
  -k 10 --global -s final_score
```

---

## 工具脚本

| 脚本 | 功能 |
|------|------|
| `score_normalization.py` | 温莎化 + [0,1] 归一化 |
| `metrics_clustering.py` | 按 Pearson 相关聚类分数 |
| `embedding.py` | 提取文本 embedding |
| `samples_clustering.py` | K-Means 数据聚类 |
| `assign_cluster_id.py` | 为样本分配 cluster_id |
| `precompute_index.py` | 预计算 pool 索引 |
| `global_aggregation.py` | 用 best_weights 聚合 Layer 分数 |
| `cluster_aggregation.py` | 从权重文件计算 final_score |
| `sampling.py` | 按分数采样 top-k%（per_cluster / global） |
| `summarization.py` | 汇总多层权重与层次结构 |
| `trials_analysis.py` | 读取 Optuna trial 详情 |

---

## main.py 用法

### 基本用法

```bash
python main.py --config configs/demo/Layer1/Layer1_c1.yaml
```

### Proxy 模型配置

Proxy 模型训练配置是 R-Select 的重要部分。本 demo 使用 [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) 作为 proxy 模型。训练参数与 DeepSpeed 配置在 `configs/demo/demo_proxy.yaml` 中，通过 `base_train_yaml` 引用。用户可根据 GPU 显存、数据规模、epoch 等修改该文件。

### 数据格式要求

**pool_jsonl**：数据池，须包含 `scores` 字典（或 `score_field` 指定的字段）。

**val_jsonl**：验证集，**须为 Alpaca 格式**，至少包含 `instruction`、`input`、`output`：

```json
{"instruction": "问题或任务描述", "input": "额外输入（可为空）", "output": "期望答案"}
```

LlamaFactory 使用这些字段构建 SFT 样本。若列名不同，请在 `base_train_yaml` 的 dataset 部分配置列映射。

### 配置参数

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `pool_jsonl` | ✓ | str | pool JSONL 路径 |
| `val_jsonl` | ✓ | str | 验证集 JSONL 路径（Alpaca 格式） |
| `top_k` | ✓ | int | 每次 trial 按加权分数选出的样本数 |
| `base_train_yaml` | ✓ | str | LlamaFactory 训练配置模板 |
| `run_dir` | ✓ | str | 当前运行的输出目录（trials、best_weights 等） |
| `study_name` | ✓ | str | Optuna study 名称 |
| `storage` | ✓ | str | Optuna 存储 URL，如 `sqlite:///path/to/study.db` |
| `n_trials` | ✓ | int | 最大 trial 数 |
| `score_keys` | ✓ | list/str | 参与加权的分数键。Layer1：原始指标；Layer2：Layer1 cluster 名 |
| `score_field` | | str | 分数所在字段，默认 `scores` |
| `pool_index` | | str | 预计算索引路径，大 pool 推荐使用 |
| `seed` | | int | 随机种子 |
| `num_train_epochs` | | float | Proxy 训练 epoch 数 |
| `eval_strategy` | | str | 评估策略，如 `epoch` |
| `per_device_eval_batch_size` | | int | 评估 batch 大小，0 表示与训练相同 |
| `cuda_visible_devices` | | str | 可见 GPU，如 `0,1,2,3` |
| `nproc_per_node` | | int | 分布式训练每节点 GPU 数 |
| `master_port` | | int | 分布式端口，0 表示自动 |
| `llama_factory_dir` | | str | LlamaFactory 根目录，空则使用项目内 LlamaFactory |
| `normalization_method` | | str | 权重归一化：`softmax` 或 `linear` |
| `suggest_range` | | str/list | Optuna 搜索范围，如 `-5,5` 或 `[0, 1]` |
| `n_startup_trials` | | int | TPE 随机探索 trial 数 |
| `tpe_multivariate` | | bool | 启用多元 TPE |
| `tpe_n_ei_candidates` | | int | EI 候选数 |
| `tpe_gamma_percent` | | float | Gamma 比例 |
| `tpe_gamma_max` | | int | Gamma 上限 |
| `early_stop_patience` | | int | 早停 patience，0 表示禁用 |
| `early_stop_delta` | | float | 早停最小改进量 |
| `warmup_trials_json` | | str | Warm-start：要注入的已有 trials 路径 |
| `enqueue_weights_json` | | str | Warm-start：优先评估的权重队列路径 |
| `cache_pool_in_memory` | | bool | 将 pool 缓存到内存（小数据时） |
| `max_trials_in_dir` | | int | 每目录最大 trial 数安全限制 |
| `running_timeout_hours` | | float | RUNNING trial 视为僵尸的小时数 |

---

## Warm-start

支持两种 warm-start 模式，用于加速收敛或优先评估特定配置。

### 推荐：覆盖搜索空间的 Warm 轨迹生成

使用 **Dirichlet 分布** 结合 **边界配置** 生成 warm 轨迹，覆盖权重搜索空间：

- **Dirichlet 分布**：在单纯形上对 `Dir(α,...,α)` 采样。α 越小权重越稀疏（峰更尖）；α=1 为均匀；α>1 更集中于中心。对同一 `score_keys` 使用多个 α（如 0.1、0.5、1.0、2.0）可采样多样权重。
- **边界配置**：加入单纯形顶点和边界情况，如 one-hot（一维为 1 其余为 0）、均匀（1/k）等，覆盖各角与均匀解。

这样生成的 warmup 或 enqueue 配置能更好地引导 TPE 探索可行域。

### 1. warmup_trials_json

注入 **已完成** 的 (weights, eval_loss) 作为先验；**不重新训练**。TPE 用它调整采样。

**JSON 格式**：`{weights, eval_loss}` 数组：

```json
[
  {"weights": {"AtheneScore": 0.2, "CleanlinessScore": 0.3, ...}, "eval_loss": 6.83},
  {"weights": {"AtheneScore": 0.1, "CleanlinessScore": 0.4, ...}, "eval_loss": 6.92}
]
```

- `weights`：键须与 `score_keys` 一致
- `eval_loss`：该配置的验证 loss

**示例**：`configs/demo/warmup_trials_example.json`

### 2. enqueue_weights_json

将权重配置加入 **评估队列**；会 **实际训练**，在 TPE 采样的 trial 之前执行。

**JSON 格式**：**纯权重字典** 数组（无 `weights` 包装，无 `eval_loss`）：

```json
[
  {"AtheneScore": 0.2, "CleanlinessScore": 0.2, "ComplexityLaJScore": 0.2, ...},
  {"AtheneScore": 0.133, "CleanlinessScore": 0.338, ...}
]
```

**示例**：`configs/demo/enqueue_weights_example.json`

### 对比

| 方面 | warmup_trials_json | enqueue_weights_json |
|------|-------------------|----------------------|
| 训练 | 否（仅先验） | 是 |
| 元素格式 | `{weights, eval_loss}` | 纯权重字典 |
| 典型用途 | 迁移已有最佳结果 | 优先评估特定权重组合 |

详见 `configs/demo/warmstart_README.md`。

---

## FAQ

**Q: 如何确定 Layer1 cluster 划分？**  
A: 使用 `metrics_clustering.py` 按相关性对分数聚类，每个 cluster 对应一个 YAML 配置。

**Q: 多次运行同一 study 时 warmup 会重复吗？**  
A: 不会。脚本按权重签名去重，已注入的 trial 会被跳过。

**Q: 数据没有 cluster_id 怎么办？**  
A: 跳过数据聚类，使用 `--global` 模式进行全局 top-k% 采样。

**Q: 如何查看跨层聚合的权重？**  
A: 使用 `summarization.py`：
```bash
python utils/summarization.py -i results/demo -o results/demo/weight_summary.json
```

---

## 项目结构

```
R-Select/
├── main.py                 # Optuna 优化主程序
├── run.sh                  # 示例流水线脚本
├── requirements.txt        # R-Select 依赖
├── readme.md / readme_en.md
├── ODA-DataScorer/         # 数据打分子模块（60+ scorers）
│   ├── model_based/        #   神经模型 scorers
│   └── heuristic/          #   统计/规则类 scorers
├── LlamaFactory/           # SFT 训练框架（子模块）
├── configs/
│   └── demo/
│       ├── demo_proxy.yaml # LlamaFactory proxy 训练配置
│       ├── scores.txt      # 本 demo 使用的指标名
│       ├── Layer1/         # Layer1 cluster 配置
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
│   └── scores.txt          # 默认指标名
└── results/                # 优化结果
    └── demo/
        ├── Layer1/
        ├── Layer2/
        └── weight_summary.json
```

---

## 参考文献

本项目基于以下开源框架，致谢：

- **[ODA-DataScorer](https://github.com/OpenDataArena/ODA-DataScorer)** — 多维数据打分工具包，含 60+ scorers（model-based、heuristic）。用于在优化前为每条数据样本计算质量/复杂度/多样性分数。
- **[LlamaFactory](https://github.com/hiyouga/LlamaFactory)** — 统一的 LLM/VLM 高效微调框架，支持 100+ 模型（ACL 2024）。用于 proxy 模型 SFT 训练与 eval loss 计算。
- **[Optuna](https://github.com/optuna/optuna)** — 超参数优化框架。用于 TPE（Tree-structured Parzen Estimator）采样器搜索打分指标的最优权重。
