"""FL Data Partitioner for non-IID distribution.

Implements Dirichlet-based label skew partitioning for federated learning.

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from config import ScenarioConfig, get_client_counts, MIN_SAMPLES_PER_LABEL_CLIENT


class FLDataPartitioner:
    """Partition data for federated learning with non-IID distribution.

    Uses Dirichlet distribution to create label skew across clients.
    """

    def __init__(self, config: ScenarioConfig):
        """Initialize partitioner.

        Args:
            config: Scenario configuration.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def partition_data(
        self,
        df: pd.DataFrame,
        task_label_map: Dict[int, List[str]]
    ) -> Dict[int, Dict[int, pd.DataFrame]]:
        """Partition data into tasks and clients.

        Args:
            df: Input DataFrame with features and labels.
            task_label_map: Mapping from task_id to list of labels.

        Returns:
            Nested dictionary: task_id -> client_id -> DataFrame.
        """
        partitions = {}

        # Get client counts per task
        client_counts = get_client_counts(self.config.n_clients)

        for task_id, labels in task_label_map.items():
            print(f"\nPartitioning Task {task_id}: {labels}")
            n_active_clients = client_counts[task_id]

            # Filter data for this task
            task_df = df[df[self.config.label_col].isin(labels)].copy()

            # Partition among active clients
            task_partitions = self._partition_task(
                task_df, labels, n_active_clients, task_id
            )

            partitions[task_id] = task_partitions

        return partitions

    def _partition_task(
        self,
        df: pd.DataFrame,
        labels: List[str],
        n_clients: int,
        task_id: int
    ) -> Dict[int, pd.DataFrame]:
        """Partition a single task among clients.

        Args:
            df: Task DataFrame.
            labels: Labels in this task.
            n_clients: Number of active clients.
            task_id: Task ID.

        Returns:
            Dictionary mapping client_id to DataFrame.
        """
        # Calculate max trainable clients per label
        label_pools = {}
        max_trainable = n_clients

        for label in labels:
            pool_df = df[df[self.config.label_col] == label]
            pool_size = len(pool_df)
            max_for_label = min(n_clients, pool_size // MIN_SAMPLES_PER_LABEL_CLIENT)
            max_trainable = min(max_trainable, max_for_label)
            label_pools[label] = pool_df

        print(f"  Max trainable clients: {max_trainable}")

        # Identify cover label (largest pool)
        cover_label = max(label_pools.keys(), key=lambda l: len(label_pools[l]))
        print(f"  Cover label: {cover_label}")

        # Assign labels to clients
        client_labels = self._assign_labels_to_clients(
            labels, cover_label, n_clients, max_trainable
        )

        # Distribute samples using Dirichlet
        client_data = {cid: [] for cid in range(n_clients)}

        for label in labels:
            pool_df = label_pools[label]
            pool_size = len(pool_df)

            # Get clients assigned this label
            assigned_clients = [cid for cid, lbls in client_labels.items() if label in lbls]
            k = len(assigned_clients)

            if k == 0:
                continue

            # Dirichlet split
            proportions = self._dirichlet_split(pool_size, k, task_id, label)

            # Assign samples
            indices = pool_df.index.tolist()
            self.rng.shuffle(indices)

            start = 0
            for i, cid in enumerate(assigned_clients):
                n_samples = proportions[i]
                end = start + n_samples
                selected_indices = indices[start:end]
                client_data[cid].append(pool_df.loc[selected_indices])
                start = end

        # Concatenate and shuffle each client's data
        result = {}
        for cid in range(n_clients):
            if client_data[cid]:
                client_df = pd.concat(client_data[cid], ignore_index=True)
                client_df = client_df.sample(frac=1, random_state=self._get_slot_seed(task_id, cid))
                result[cid] = client_df
            else:
                result[cid] = pd.DataFrame()

        return result

    def _assign_labels_to_clients(
        self,
        labels: List[str],
        cover_label: str,
        n_clients: int,
        max_trainable: int
    ) -> Dict[int, List[str]]:
        """Assign labels to clients creating label skew.

        Args:
            labels: All labels in task.
            cover_label: Label with largest pool.
            n_clients: Total clients.
            max_trainable: Maximum trainable clients.

        Returns:
            Dictionary mapping client_id to list of assigned labels.
        """
        client_labels = {cid: [] for cid in range(n_clients)}

        # Cover label goes to all clients
        for cid in range(n_clients):
            client_labels[cid].append(cover_label)

        # Other labels assigned to subset of clients
        other_labels = [l for l in labels if l != cover_label]

        for label in other_labels:
            # Random number of clients between max_trainable//2 and max_trainable
            min_k = max(1, max_trainable // 2)
            k = self.rng.integers(min_k, max_trainable + 1)

            # Randomly select clients
            selected = self.rng.choice(n_clients, size=k, replace=False)
            for cid in selected:
                client_labels[cid].append(label)

        return client_labels

    def _dirichlet_split(
        self,
        pool_size: int,
        k: int,
        task_id: int,
        label: str
    ) -> List[int]:
        """Split samples among k clients using Dirichlet distribution.

        Args:
            pool_size: Total number of samples.
            k: Number of clients.
            task_id: Task ID for seeding.
            label: Label name.

        Returns:
            List of sample counts for each client.
        """
        # Minimum samples per client
        min_per_client = min(MIN_SAMPLES_PER_LABEL_CLIENT, pool_size // k)
        remaining = pool_size - min_per_client * k

        if remaining <= 0:
            # Equal split if not enough for minimum
            base = pool_size // k
            extra = pool_size % k
            counts = [base + 1 if i < extra else base for i in range(k)]
            return counts

        # Dirichlet distribution for remaining samples
        alpha = [self.config.dirichlet_alpha] * k
        proportions = self.rng.dirichlet(alpha)

        # Convert to counts
        counts = [min_per_client + int(p * remaining) for p in proportions]

        # Adjust for rounding errors
        diff = pool_size - sum(counts)
        if diff > 0:
            # Add to clients with smallest counts
            indices = np.argsort(counts)
            for i in range(diff):
                counts[indices[i % k]] += 1
        elif diff < 0:
            # Remove from clients with largest counts
            indices = np.argsort(counts)[::-1]
            for i in range(-diff):
                if counts[indices[i % k]] > min_per_client:
                    counts[indices[i % k]] -= 1

        return counts

    def _get_slot_seed(self, task_id: int, client_id: int) -> int:
        """Get deterministic seed for a task-client slot.

        Args:
            task_id: Task ID.
            client_id: Client ID.

        Returns:
            Seed value.
        """
        return self.config.seed ^ (task_id * 1000 + client_id)


def save_partitions(
    partitions: Dict[int, Dict[int, pd.DataFrame]],
    output_dir: str,
    config: ScenarioConfig,
    task_label_map: Dict[int, List[str]]
) -> None:
    """Save partitions to disk.

    Args:
        partitions: Nested dictionary of partitions.
        output_dir: Output directory.
        config: Scenario configuration.
        task_label_map: Task to labels mapping.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create label to ID mapping
    all_labels = []
    for labels in task_label_map.values():
        all_labels.extend(labels)
    all_labels = sorted(set(all_labels))
    label2id = {label: idx for idx, label in enumerate(all_labels)}

    # Save each task
    metadata = {
        'n_clients': config.n_clients,
        'feature_type': config.feature_type,
        'dirichlet_alpha': config.dirichlet_alpha,
        'seed': config.seed,
        'label2id': label2id,
        'task_label_map': task_label_map,
        'clients': {}
    }

    partition_table_data = []

    for task_id, task_partitions in partitions.items():
        task_dir = output_path / f'task_{task_id}'
        task_dir.mkdir(exist_ok=True)

        for client_id, client_df in task_partitions.items():
            if len(client_df) == 0:
                continue

            # Save as parquet (preferred) or csv
            parquet_path = task_dir / f'client_{client_id:02d}.parquet'
            client_df.to_parquet(parquet_path, index=False)

            # Record metadata
            client_key = f'client_{client_id:02d}'
            if client_key not in metadata['clients']:
                metadata['clients'][client_key] = {
                    'n_samples': {},
                    'label_distribution': {}
                }

            n_samples = len(client_df)
            metadata['clients'][client_key]['n_samples'][task_id] = n_samples

            # Label distribution
            label_counts = client_df[config.label_col].value_counts().to_dict()
            metadata['clients'][client_key]['label_distribution'][task_id] = label_counts

            # Partition table row
            row = {
                'client_id': client_id,
                'task_id': task_id,
                'total_samples': n_samples
            }
            for label, count in label_counts.items():
                row[f'label_{label}'] = count
            partition_table_data.append(row)

    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save partition table
    partition_df = pd.DataFrame(partition_table_data)
    partition_df.to_csv(output_path / 'partition_table.csv', index=False)

    # Save task label map
    task_map_df = pd.DataFrame([
        {'task_id': tid, 'labels': ','.join(labels)}
        for tid, labels in task_label_map.items()
    ])
    task_map_df.to_csv(output_path / 'task_label_map.csv', index=False)

    print(f"\nPartitions saved to {output_path}")
