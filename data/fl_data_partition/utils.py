"""Utility functions for FL data partitioning."""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_metadata(scenario_dir: str) -> Dict:
    """Load partition metadata.

    Args:
        scenario_dir: Directory containing partitions.

    Returns:
        Metadata dictionary.
    """
    metadata_path = Path(scenario_dir) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_partition_statistics(scenario_dir: str) -> pd.DataFrame:
    """Get statistics about partitions.

    Args:
        scenario_dir: Directory containing partitions.

    Returns:
        DataFrame with statistics.
    """
    metadata = load_metadata(scenario_dir)
    stats = []

    for client_key, client_data in metadata['clients'].items():
        client_id = int(client_key.split('_')[1])

        for task_id_str, n_samples in client_data['n_samples'].items():
            task_id = int(task_id_str)
            label_dist = client_data['label_distribution'].get(task_id_str, {})

            row = {
                'client_id': client_id,
                'task_id': task_id,
                'n_samples': n_samples,
                'n_labels': len(label_dist),
            }

            for label, count in label_dist.items():
                row[f'count_{label}'] = count

            stats.append(row)

    return pd.DataFrame(stats)


def check_partition_quality(scenario_dir: str) -> Dict:
    """Check quality of partitions.

    Args:
        scenario_dir: Directory containing partitions.

    Returns:
        Dictionary with quality metrics.
    """
    metadata = load_metadata(scenario_dir)
    issues = []
    metrics = {
        'total_clients': len(metadata['clients']),
        'total_tasks': len(metadata['task_label_map']),
        'issues': issues
    }

    # Check each client has data in at least one task
    for client_key, client_data in metadata['clients'].items():
        if not client_data['n_samples']:
            issues.append(f"{client_key} has no data in any task")

    # Check label coverage
    all_labels = set()
    for labels in metadata['task_label_map'].values():
        all_labels.update(labels)

    for label in all_labels:
        found = False
        for client_data in metadata['clients'].values():
            for task_dist in client_data['label_distribution'].values():
                if label in task_dist:
                    found = True
                    break
            if found:
                break

        if not found:
            issues.append(f"Label '{label}' not found in any client")

    metrics['n_issues'] = len(issues)
    return metrics
