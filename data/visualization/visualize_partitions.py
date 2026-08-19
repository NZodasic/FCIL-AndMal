"""Visualization for FL data partitions.

Creates heatmaps and bar charts to analyze partition distribution.

"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize FL data partitions'
    )
    parser.add_argument(
        '--base_dir', type=str, default='./fl_data_partitions',
        help='Base directory containing partitions'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for plots (default: same as base_dir)'
    )
    return parser.parse_args()


def load_partition_data(base_dir: str) -> Dict:
    """Load partition data from directory.

    Args:
        base_dir: Base directory containing partitions.

    Returns:
        Dictionary with partition information.
    """
    base_path = Path(base_dir)

    # Find all partition directories
    partitions = []
    for feature_type_dir in base_path.glob('*'):
        if not feature_type_dir.is_dir():
            continue
        for n_clients_dir in feature_type_dir.glob('*clients'):
            metadata_path = n_clients_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                partitions.append({
                    'path': n_clients_dir,
                    'feature_type': feature_type_dir.name,
                    'n_clients': metadata['n_clients'],
                    'metadata': metadata
                })

    return partitions


def create_partition_matrix(metadata: Dict) -> pd.DataFrame:
    """Create client x (task, label) matrix.

    Args:
        metadata: Partition metadata.

    Returns:
        DataFrame with sample counts.
    """
    # Build matrix
    rows = []

    for client_key, client_data in metadata['clients'].items():
        client_id = int(client_key.split('_')[1])

        for task_id_str, label_dist in client_data['label_distribution'].items():
            task_id = int(task_id_str)

            for label, count in label_dist.items():
                rows.append({
                    'client_id': client_id,
                    'task_id': task_id,
                    'label': label,
                    'count': count
                })

    df = pd.DataFrame(rows)

    # Create pivot table: clients x (task, label)
    df['task_label'] = df['task_id'].astype(str) + '_' + df['label']
    matrix = df.pivot_table(
        index='client_id',
        columns='task_label',
        values='count',
        fill_value=0
    )

    return matrix


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: str,
    log_scale: bool = True,
    figsize: tuple = (16, 10)
) -> None:
    """Plot heatmap of partition distribution.

    Args:
        matrix: Partition matrix (clients x task_labels).
        output_path: Output file path.
        log_scale: Whether to use log scale.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Apply log scale if requested
    plot_matrix = matrix.copy()
    if log_scale:
        plot_matrix = np.log1p(plot_matrix)

    # Plot heatmap
    sns.heatmap(
        plot_matrix,
        cmap='YlOrRd',
        cbar_kws={'label': 'log(samples + 1)' if log_scale else 'samples'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_xlabel('Task_Label')
    ax.set_ylabel('Client ID')
    ax.set_title(f'Partition Distribution ({"Log Scale" if log_scale else "Linear"})')

    # Add task separators
    task_labels = matrix.columns.tolist()
    current_task = None
    separator_positions = []

    for i, tl in enumerate(task_labels):
        task = tl.split('_')[0]
        if task != current_task:
            if current_task is not None:
                separator_positions.append(i)
            current_task = task

    for pos in separator_positions:
        ax.axvline(x=pos, color='blue', linewidth=2, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to {output_path}")


def plot_class_distribution(
    metadata: Dict,
    output_path: str,
    figsize: tuple = (12, 6)
) -> None:
    """Plot bar chart of class distribution.

    Args:
        metadata: Partition metadata.
        output_path: Output file path.
        figsize: Figure size.
    """
    # Aggregate counts per label
    label_counts = {}

    for client_data in metadata['clients'].values():
        for task_dist in client_data['label_distribution'].values():
            for label, count in task_dist.items():
                label_counts[label] = label_counts.get(label, 0) + count

    # Sort by task order
    sorted_labels = []
    for task_id, labels in metadata['task_label_map'].items():
        for label in labels:
            if label not in sorted_labels:
                sorted_labels.append(label)

    counts = [label_counts.get(l, 0) for l in sorted_labels]

    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(sorted_labels)), counts, color='steelblue')

    # Color by task
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    task_colors = {}
    for task_id, labels in metadata['task_label_map'].items():
        for label in labels:
            task_colors[label] = colors[task_id]

    for i, label in enumerate(sorted_labels):
        bars[i].set_color(task_colors.get(label, 'gray'))

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples (log scale)')
    ax.set_yscale('log')
    ax.set_title('Class Distribution Across All Clients')
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')

    # Add legend for tasks
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=f'Task {i}')
        for i in range(5)
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Class distribution plot saved to {output_path}")


def visualize_partition(partition_info: Dict, output_dir: Path) -> None:
    """Visualize a single partition.

    Args:
        partition_info: Partition information dictionary.
        output_dir: Output directory for plots.
    """
    metadata = partition_info['metadata']
    feature_type = partition_info['feature_type']
    n_clients = partition_info['n_clients']

    print(f"\nVisualizing {feature_type} / {n_clients} clients")

    # Create partition matrix
    matrix = create_partition_matrix(metadata)

    # Plot heatmaps
    plot_heatmap(
        matrix,
        str(output_dir / f'{feature_type}_{n_clients}clients_heatmap_log.png'),
        log_scale=True
    )

    if n_clients <= 25:
        plot_heatmap(
            matrix,
            str(output_dir / f'{feature_type}_{n_clients}clients_heatmap_linear.png'),
            log_scale=False
        )

    # Plot class distribution
    plot_class_distribution(
        metadata,
        str(output_dir / f'{feature_type}_{n_clients}clients_class_dist.png')
    )


def main():
    """Main entry point."""
    args = parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading partitions from {base_dir}...")
    partitions = load_partition_data(str(base_dir))

    print(f"Found {len(partitions)} partition configurations")

    for partition in partitions:
        visualize_partition(partition, output_dir)

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
