# 归一化脚本
python utils/score_normalization.py -i data/demo.jsonl -o data/demo_normalized.jsonl --pct-range 5 95 --keep-original --flip-keys PPLScore NormLossScore

# 指标聚类 （可选，如果进行多层优化）
python utils/metrics_clustering.py -i data/demo_normalized.jsonl -o data/metrics_cluster_results.txt --sample_size 10000 --scores "AtheneScore,CleanlinessScore,CompressRatioScore" --n_clusters 5

# 数据聚类（可选）
srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-49  --gres=gpu:0 python utils/embedding.py --embedder_model '/share/wulijun/tangzinan/models/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af' --input_path data/demo_normalized.jsonl --output_path data --fields "instruction" "input" --tensor_parallel_size 8

python utils/samples_clustering.py --input_path data/demo_normalized_embeddings.npy --output_dir data --opt_k 50 

python utils/assign_cluster_id.py --input_jsonl data/demo_normalized.jsonl --labels_path data/cluster_labels.npy

# 生成索引文件
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl --output_index data/demo_normalized_index.pkl

# 具体优化

## 第一层
srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-52  --gres=gpu:0 python main.py --config configs/demo/Layer1/Layer1_c1.yaml

srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-52  --gres=gpu:0 python main.py --config configs/demo/Layer1/Layer1_c2.yaml

srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-52  --gres=gpu:0 python main.py --config configs/demo/Layer1/Layer1_c3.yaml

srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-52  --gres=gpu:0 python main.py --config configs/demo/Layer1/Layer1_c4.yaml

srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-52  --gres=gpu:0 python main.py --config configs/demo/Layer1/Layer1_c5.yaml

srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-52  --gres=gpu:0 python main.py --config configs/demo/Layer1/Layer1_c6.yaml

## 第二层
### 聚合数据并进行归一化
python utils/global_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer1
### 重新生成索引文件
python utils/precompute_index.py --pool_jsonl data/demo_normalized.jsonl --output_index data/demo_normalized_index.pkl

srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-52  --gres=gpu:0 python main.py --config configs/demo/Layer2/Layer2_c1.yaml

# 最终聚合分数，并进行采样
python utils/cluster_aggregation.py -i data/demo_normalized.jsonl -w results/demo/Layer2/Layer2_c1/best_weights.json

# 多样性采样
python utils/sampling.py -i data/demo_normalized.jsonl -o data/demo_diversity_sampled.jsonl -k 10 --per_cluster

# 全局直接采样
python utils/sampling.py -i data/demo_normalized.jsonl -o data/demo_global_sampled.jsonl -k 10 --global

# 查看optuna优化轨迹具体信息
python utils/trials_analysis.py --storage sqlite:///results/demo/Layer1/Layer1_c1/study.db --study_name Layer1_c1 --output results/demo/Layer1/Layer1_c1/trials.json

# 查看各指标详细权重
python utils/summarization.py --input results/demo/ --output results/demo/weight_summary.json



srun -p raise --quotatype=spot -w SH-IDC1-10-140-37-124 --gres=gpu:0  python main.py --config configs/demo/Layer2/Layer2_c1.yaml