#!/usr/bin/env python3
"""
Cluster all scores based on the Pearson correlation coefficient, grouping together scores with high correlations.

Usage:
    # Cluster all scores (default if --scores is not specified)
    python metrics_clustering.py \
        -i data/demo_normalized.jsonl \
        -o cluster_results.txt \
        --sample_size 10000 \
        --n_clusters 5

    # Specify which scores to cluster (comma separated)
    python metrics_clustering.py \
        -i data/demo_normalized.jsonl \
        -o cluster_results.txt \
        --scores "AtheneScore,CleanlinessScore,CompressRatioScore" \
        --n_clusters 3

    # Read score list from file (one score per line)
    python metrics_clustering.py \
        -i data/demo_normalized.jsonl \
        -o cluster_results.txt \
        --scores score_list.txt

    # Specify displayed names for heatmap/dendrogram
    python metrics_clustering.py \
        -i data/demo_normalized.jsonl \
        -o cluster_results.txt \
        --score_names score_display_names.json \
        --plot_heatmap
"""

import argparse
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

# Set font (add Chinese fonts here if needed)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def count_lines(file_path):
    """Quickly count lines in a file (use `wc` command for speed if available)"""
    import subprocess
    print(f"📊 Counting lines in file ...")
    try:
        result = subprocess.run(
            ['wc', '-l', file_path],
            capture_output=True,
            text=True,
            check=True
        )
        count = int(result.stdout.split()[0])
        print(f"✅ Total lines in file: {count:,}")
        return count
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # Fall back to Python if wc fails
        print(f"⚠️  wc command not available, using Python to count lines ...")
        with open(file_path, 'rb') as f:
            count = sum(1 for _ in tqdm(f, desc="Counting lines", unit="lines"))
        return count


def sample_lines(file_path, n_samples, total_lines=None):
    """Randomly sample N lines from a file (efficient implementation)"""
    if total_lines is None:
        total_lines = count_lines(file_path)
    
    print(f"\n🎲 Randomly sampling {n_samples:,} rows from {total_lines:,} total lines ...")
    
    # If requested samples are close to total, just read all
    if n_samples >= total_lines * 0.9:
        print("Sample size is close to total, reading all data ...")
        sampled_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Reading data", unit="lines"):
                try:
                    data = json.loads(line.strip())
                    sampled_data.append(data)
                except json.JSONDecodeError:
                    continue
        return sampled_data
    
    # Efficient reservoir sampling strategy
    selected_indices = set(random.sample(range(total_lines), min(n_samples, total_lines)))
    
    sampled_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=total_lines, desc="Reading data", unit="lines")):
            if idx in selected_indices:
                try:
                    data = json.loads(line.strip())
                    sampled_data.append(data)
                except json.JSONDecodeError:
                    continue
    
    print(f"✅ Successfully sampled {len(sampled_data)} rows.")
    return sampled_data


def parse_score_display_names(names_arg):
    """
    Parse mapping of score names to display names (for heatmap/dendrogram labels)

    Args:
        names_arg: File path. File can be:
            - JSON object: {"ScoreKey": "DisplayName", ...}
            - Text file: each line as "key\\tDisplayName", "key=DisplayName" or "key,DisplayName"

    Returns:
        dict: score key -> display name; if None given, returns empty dict.
    """
    if names_arg is None:
        return {}
    path = Path(names_arg)
    if not path.exists() or not path.is_file():
        print(f"⚠️  --score_names file does not exist, using original score names: {names_arg}")
        return {}
    text = path.read_text(encoding='utf-8').strip()
    mapping = {}
    if text.startswith('{'):
        try:
            mapping = json.loads(text)
            mapping = {str(k).strip(): str(v).strip() for k, v in mapping.items()}
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error for display names, fallback to original names: {e}")
            return {}
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sep in ('\t', '=', ','):
                if sep in line:
                    key, _, val = line.partition(sep)
                    mapping[key.strip()] = val.strip()
                    break
    print(f"✅ Loaded {len(mapping)} score display names")
    return mapping


def parse_score_list(scores_arg):
    """
    Parse score list argument

    Args:
        scores_arg: One of:
            - Path to a file (one score name per line)
            - Comma separated score names string
            - None (use all scores)

    Returns:
        Score list, or None if using all scores.
    """
    if scores_arg is None:
        return None
    
    # If contains comma, treat as comma-separated list (don't treat as file path)
    if ',' in scores_arg:
        scores = [s.strip() for s in scores_arg.split(',') if s.strip()]
        print(f"✅ Parsed {len(scores)} scores from argument")
        return scores
    
    # Otherwise treat as file path (only when no comma)
    score_path = Path(scores_arg)
    if score_path.exists() and score_path.is_file():
        print(f"📄 Reading score list from file: {score_path}")
        with open(score_path, 'r', encoding='utf-8') as f:
            scores = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(scores)} scores from file")
        return scores
    else:
        # Argument is a single score or file does not exist
        scores = [scores_arg.strip()] if scores_arg.strip() else []
        if scores:
            print(f"✅ Parsed {len(scores)} scores from argument")
        return scores


def extract_scores(data_list, selected_scores=None, score_field='scores'):
    """
    Extract score field from the data

    Args:
        data_list: List of data dicts
        selected_scores: List of scores to extract, or None for all
        score_field: name of the scores field (default 'scores')

    Returns:
        all_scores: list of all score dicts
        score_keys: sorted list of score keys
    """
    print(f"\n📈 Extracting scores (field name: {score_field}) ...")
    if selected_scores is not None:
        print(f"   Using {len(selected_scores)} specified scores for clustering")
    else:
        print(f"   No --scores specified, clustering all scores")
    
    all_scores = []
    score_keys = set()
    
    for item in tqdm(data_list, desc="Extracting scores", unit="records"):
        field_val = item.get(score_field)
        if isinstance(field_val, dict):
            scores = field_val
            all_scores.append(scores)
            score_keys.update(scores.keys())
        else:
            continue
    
    if not all_scores:
        print(f"❌ No valid data found (check for field {score_field})")
        return [], []
    
    # If a score list is given, filter accordingly
    if selected_scores is not None:
        # Which scores exist in the data
        available_scores = set(score_keys)
        requested_scores = set(selected_scores)
        
        # Intersection and missing sets
        found_scores = available_scores & requested_scores
        missing_scores = requested_scores - available_scores
        
        if missing_scores:
            print(f"⚠️  Warning: The following {len(missing_scores)} score(s) do not exist in the data:")
            for score in sorted(missing_scores):
                print(f"     - {score}")
        
        if not found_scores:
            print(f"❌ Error: None of the specified scores are in the data")
            return [], []
        
        score_keys = sorted(found_scores)
        print(f"✅ Found {len(score_keys)} valid score dimensions (from {len(available_scores)} total dimensions)")
    else:
        score_keys = sorted(score_keys)
        print(f"✅ Found {len(score_keys)} distinct score dimensions")
    
    print(f"   Number of valid rows: {len(all_scores)}")
    
    return all_scores, score_keys


def build_score_dataframe(all_scores, score_keys):
    """Build DataFrame"""
    print(f"\n🔨 Building DataFrame ...")
    
    # Convert to dict of column-lists
    data_dict = {}
    for key in tqdm(score_keys, desc="Processing dimensions"):
        data_dict[key] = [scores.get(key, np.nan) for scores in all_scores]
    
    df = pd.DataFrame(data_dict)
    
    print(f"✅ DataFrame shape: {df.shape}")
    print(f"   Missing count summary:")
    missing = df.isnull().sum()
    for key, count in missing[missing > 0].items():
        print(f"     {key}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def calculate_correlation(df):
    """Compute Pearson correlation matrix"""
    print(f"\n🔍 Calculating Pearson correlation matrix ...")
    
    # Remove columns that are all NaN
    df_clean = df.dropna(axis=1, how='all')
    
    # Pearson correlation
    corr_matrix = df_clean.corr(method='pearson')
    
    print(f"✅ Correlation matrix completed")
    print(f"   Matrix shape: {corr_matrix.shape}")
    print(f"   Number of valid dimensions: {len(corr_matrix.columns)}")
    
    return corr_matrix


def correlation_to_distance(corr_matrix, use_absolute=True):
    """
    Convert correlation matrix to distance matrix

    Args:
        corr_matrix: the correlation matrix
        use_absolute: If True, use 1 - |correlation|, otherwise 1 - correlation

    Returns:
        Distance matrix as ndarray
    """
    if use_absolute:
        # Use |correlation|: higher absolute correlation means closer
        dist_matrix = 1 - np.abs(corr_matrix.values)
    else:
        # Only positive correlation: higher positive, closer
        dist_matrix = 1 - corr_matrix.values
    
    # Ensure diagonal is zero (self-distance)
    np.fill_diagonal(dist_matrix, 0)
    
    # Ensure non-negative
    dist_matrix[dist_matrix < 0] = 0
    
    return dist_matrix


def perform_clustering(corr_matrix, n_clusters=None, method='ward', use_absolute_corr=True):
    """
    Perform hierarchical clustering on the correlation matrix

    Args:
        corr_matrix: correlation matrix
        n_clusters: number of clusters (if None, use automatic threshold)
        method: linkage method, one of:
            - 'ward': minimize within-cluster sum of squares, gives compact/similar-sized clusters (recommended for score clustering)
            - 'complete': use maximum distance (complete linkage), gives compact/well-separated clusters
            - 'average': use average linkage, balances robustness and compactness
            - 'single': use minimum distance (single linkage), may chain clusters
        use_absolute_corr: whether to use absolute correlation for clustering

    Returns:
        linkage_matrix, cluster_labels, order
    """
    print(f"\n🔗 Performing hierarchical clustering ...")
    print(f"   Linkage method: {method}")
    print(f"   Using absolute correlation: {use_absolute_corr}")
    
    # Convert to distance matrix
    dist_matrix = correlation_to_distance(corr_matrix, use_absolute=use_absolute_corr)
    
    # Condense (for scipy) to 1D
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Compute hierarchical clusters
    linkage_matrix = linkage(condensed_dist, method=method)
    
    # Leaf order for dendrogram display
    order = leaves_list(linkage_matrix)
    
    # Assign cluster labels
    if n_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        print(f"   Number of clusters: {n_clusters}")
    else:
        # Use default threshold at 70th percentile of linkage distances
        threshold = np.percentile(linkage_matrix[:, 2], 70)
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        n_clusters = len(np.unique(cluster_labels))
        print(f"   Automatic number of clusters: {n_clusters}")
    
    print(f"✅ Clustering complete, total clusters: {n_clusters}")
    
    return linkage_matrix, cluster_labels, order


def print_cluster_results(score_names, cluster_labels, corr_matrix):
    """Prints clustering results"""
    print(f"\n📊 Cluster results:")
    print(f"{'='*80}")
    
    # Group by cluster label
    clusters = {}
    for idx, (name, label) in enumerate(zip(score_names, cluster_labels)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((name, idx))
    
    # Print per-cluster
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        print(f"\nCluster {cluster_id} ({len(members)} scores):")
        
        # Compute mean correlation within-cluster
        if len(members) > 1:
            cluster_indices = [idx for _, idx in members]
            cluster_corr = corr_matrix.iloc[cluster_indices, cluster_indices]
            # Use only upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(cluster_corr, dtype=bool), k=1)
            avg_corr = cluster_corr.values[mask].mean()
            print(f"  Mean within-cluster correlation: {avg_corr:.4f}")
        
        for name, _ in members:
            print(f"    - {name}")
    
    print(f"{'='*80}\n")
    
    return clusters


def save_cluster_results(clusters, score_names, output_path):
    """Save clustering results to file"""
    print(f"\n💾 Saving cluster results to: {output_path}")
    
    out_path_obj = Path(output_path)
    out_dir = out_path_obj.parent
    if not out_dir.exists():
        print(f"📝 Output directory '{out_dir}' does not exist. Creating it.")
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Score Clustering Results Based on Pearson Correlation\n")
        f.write("="*80 + "\n\n")
        
        for cluster_id in sorted(clusters.keys()):
            members = clusters[cluster_id]
            f.write(f"Cluster {cluster_id} ({len(members)} scores):\n")
            for name, _ in members:
                f.write(f"  - {name}\n")
            f.write("\n")
    
    print(f"✅ Results saved.")


def plot_dendrogram(linkage_matrix, score_names, output_path, figsize=(20, 12)):
    """Plot dendrogram"""
    print(f"\n🎨 Generating dendrogram ...")
    
    out_path_obj = Path(output_path)
    out_dir = out_path_obj.parent
    if not out_dir.exists():
        print(f"📝 Output directory '{out_dir}' does not exist. Creating it.")
        out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    
    # Draw
    dendrogram(
        linkage_matrix,
        labels=score_names,
        leaf_rotation=90,
        leaf_font_size=8,
        orientation='top'
    )
    
    plt.title('Hierarchical Clustering Dendrogram\n(Based on Pearson Correlation)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Score Names', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✅ Dendrogram saved to: {output_path}")
    
    plt.close()


def plot_clustered_correlation(corr_matrix, cluster_labels, score_names, output_path, figsize=(20, 16)):
    """Plot clustered correlation heatmap"""
    print(f"\n🎨 Drawing clustered correlation heatmap ...")

    out_path_obj = Path(output_path)
    out_dir = out_path_obj.parent
    if not out_dir.exists():
        print(f"📝 Output directory '{out_dir}' does not exist. Creating it.")
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by cluster label first, then internally alphabetically
    cluster_order = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        # Sort inside each cluster by name (alphabetically)
        cluster_indices = sorted(cluster_indices, key=lambda i: score_names[i])
        cluster_order.extend(cluster_indices)
    
    # Reorder correlation matrix and use (possibly mapped) score names for axis
    corr_ordered = corr_matrix.iloc[cluster_order, cluster_order].copy()
    labels_ordered = [score_names[i] for i in cluster_order]
    corr_ordered.index = labels_ordered
    corr_ordered.columns = labels_ordered
    
    # Draw heatmap
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Mark cluster boundaries
    cluster_boundaries = []
    current_cluster = cluster_labels[cluster_order[0]]
    for i, idx in enumerate(cluster_order):
        if cluster_labels[idx] != current_cluster:
            cluster_boundaries.append(i)
            current_cluster = cluster_labels[idx]
    
    sns.heatmap(
        corr_ordered,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
        annot_kws={'size': 7}
    )
    
    # Draw cluster boundary lines
    for boundary in cluster_boundaries:
        ax.axhline(y=boundary, color='black', linewidth=2)
        ax.axvline(x=boundary, color='black', linewidth=2)
    
    ax.set_title(
        f'Score Correlation Matrix (Clustered)\n({len(corr_matrix.columns)} dimensions, {len(np.unique(cluster_labels))} clusters)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✅ Clustered heatmap saved to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cluster all scores based on Pearson correlation coefficient"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSONL file (e.g., data/demo_normalized.jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        default="cluster_results.txt",
        help="Output results file path (default: cluster_results.txt)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of samples to read (default: 10000)"
    )
    parser.add_argument(
        "--total_lines",
        type=int,
        default=None,
        help="Total line count for the file (if known, skips counting step)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Cluster count (if unspecified, automatically determined)"
    )
    parser.add_argument(
        "--method",
        choices=['ward', 'complete', 'average', 'single'],
        default='ward',
        help="Hierarchical clustering linkage method (default: ward). "
             "ward: minimizes within-cluster variance, compact/similar-sized clusters (recommended); "
             "complete: maximum distance (well-separated clusters); "
             "average: average linkage; "
             "single: minimum distance, may chain clusters"
    )
    parser.add_argument(
        "--use_absolute_corr",
        action='store_true',
        help="Use absolute correlation for clustering (groups highly correlated scores together, regardless of sign)"
    )
    parser.add_argument(
        "--plot_dendrogram",
        action='store_true',
        help="Draw dendrogram"
    )
    parser.add_argument(
        "--plot_heatmap",
        action='store_true',
        help="Draw clustered correlation heatmap"
    )
    parser.add_argument(
        "--scores",
        type=str,
        default=None,
        help="List of scores to cluster. Can be: 1) path to file (one score name per line); 2) comma separated string ('score1,score2,...'). If unspecified, clusters all scores."
    )
    parser.add_argument(
        "--score_names",
        type=str,
        default=None,
        help="Optional. File specifying display names to use for scores in heatmap/dendrogram: "
             "1) JSON object {\"ScoreKey\": \"DisplayName\", ...}; 2) text file lines of key\\tDisplayName or key=DisplayName"
    )
    parser.add_argument(
        "--score_field", "-s",
        type=str,
        default="scores",
        help="Scores field name (default: scores)"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse selected scores and display name mapping
    selected_scores = parse_score_list(args.scores)
    score_name_map = parse_score_display_names(args.score_names)
    
    print("="*80)
    print("Score Clustering Based on Pearson Correlation")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Random seed: {args.seed}")
    print(f"Output file: {args.output}")
    print(f"Number of clusters: {args.n_clusters if args.n_clusters else 'auto'}")
    print(f"Linkage method: {args.method}")
    print(f"Use absolute correlation: {args.use_absolute_corr}")
    if selected_scores is not None:
        print(f"Scores specified: {len(selected_scores)}")
    else:
        print(f"Scores specified: all (will cluster every score key in the data)")
    print(f"Scores field: {args.score_field}")
    print("="*80)
    
    # Check input exists
    if not Path(args.input).exists():
        print(f"❌ Error: file not found: {args.input}")
        return
    
    # Step 1: Sample data
    sampled_data = sample_lines(args.input, args.sample_size, args.total_lines)
    
    if not sampled_data:
        print("❌ Error: failed to sample any data")
        return
    
    # Step 2: Extract scores
    all_scores, score_keys = extract_scores(
        sampled_data, selected_scores, score_field=args.score_field
    )
    
    if not all_scores:
        print("❌ Error: No valid score data found")
        return
    
    # Step 3: Build DataFrame
    df = build_score_dataframe(all_scores, score_keys)
    
    # Step 4: Pearson correlation
    corr_matrix = calculate_correlation(df)
    
    # Step 5: Clustering
    linkage_matrix, cluster_labels, order = perform_clustering(
        corr_matrix, 
        n_clusters=args.n_clusters,
        method=args.method,
        use_absolute_corr=args.use_absolute_corr
    )
    
    # Step 6: Print and save result
    clusters = print_cluster_results(score_keys, cluster_labels, corr_matrix)
    
    output_path = Path(args.output)

    # Ensure output directory exists before saving results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_cluster_results(clusters, score_keys, str(output_path))
    
    # Use mapped display names if provided
    score_labels = [score_name_map.get(k, k) for k in score_keys]
    
    # Step 7: Plotting
    base_name = output_path.stem
    output_dir = output_path.parent

    # Ensure output directory exists for plots as well (redundancy for safety)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot_dendrogram or (not args.plot_dendrogram and not args.plot_heatmap):
        # By default, plot the dendrogram
        dendrogram_path = output_dir / f"{base_name}_dendrogram.pdf"
        plot_dendrogram(linkage_matrix, score_labels, str(dendrogram_path))
    
    if args.plot_heatmap or (not args.plot_dendrogram and not args.plot_heatmap):
        # By default, plot the heatmap as well
        heatmap_path = output_dir / f"{base_name}_heatmap.pdf"
        plot_clustered_correlation(corr_matrix, cluster_labels, score_labels, str(heatmap_path))
    
    print("\n" + "="*80)
    print("✅ Clustering analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
