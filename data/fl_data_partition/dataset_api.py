"""Dataset API for FL experiments.

Provides convenient data loading for training and evaluation.

"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data.schema import get_feature_columns


class FLTaskDataset(Dataset):
    """PyTorch Dataset for a specific task and client."""

    def __init__(
        self,
        scenario_dir: str,
        task_id: int,
        client_id: int,
        label_col: str = 'label'
    ):
        """Initialize dataset.

        Args:
            scenario_dir: Directory containing partitions.
            task_id: Task ID (0-4).
            client_id: Client ID.
            label_col: Name of label column.
        """
        self.scenario_dir = Path(scenario_dir)
        self.task_id = task_id
        self.client_id = client_id
        self.label_col = label_col

        # Load metadata
        metadata_path = self.scenario_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.label2id = self.metadata['label2id']

        # Load data
        self.df = self._load_data()
        self.feature_cols = self._get_feature_columns()

    def _load_data(self) -> pd.DataFrame:
        """Load data for this task and client."""
        task_dir = self.scenario_dir / f'task_{self.task_id}'

        # Try parquet first, then csv
        parquet_path = task_dir / f'client_{self.client_id:02d}.parquet'
        csv_path = task_dir / f'client_{self.client_id:02d}.csv'

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No data file found for task {self.task_id}, client {self.client_id}")

    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns (exclude label and metadata)."""
        return get_feature_columns(self.df, [self.label_col])

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, label_id).
        """
        row = self.df.iloc[idx]

        # Extract features
        features = row[self.feature_cols].values.astype(np.float32)

        # Map label to ID
        label = row[self.label_col]
        label_id = self.label2id.get(label, -1)

        return torch.tensor(features), label_id

    def as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all data as numpy arrays.

        Returns:
            Tuple of (X, y) arrays.
        """
        X = self.df[self.feature_cols].values.astype(np.float32)
        y = self.df[self.label_col].map(self.label2id).values.astype(np.int64)
        return X, y

    def as_dataloader(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Create PyTorch DataLoader.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            num_workers: Number of worker processes.

        Returns:
            PyTorch DataLoader.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in this dataset.

        Returns:
            Dictionary mapping label to count.
        """
        return self.df[self.label_col].value_counts().to_dict()


def get_participating_clients(scenario_dir: str, task_id: int) -> List[int]:
    """Get list of clients that have data for a specific task.

    Args:
        scenario_dir: Directory containing partitions.
        task_id: Task ID (0-4).

    Returns:
        List of client IDs.
    """
    scenario_path = Path(scenario_dir)
    task_dir = scenario_path / f'task_{task_id}'

    if not task_dir.exists():
        return []

    clients = []
    for f in task_dir.glob('client_*.parquet'):
        client_id = int(f.stem.split('_')[1])
        clients.append(client_id)

    return sorted(clients)


def recommend_batch_size(scenario_dir: str, min_batch_size: int = 256) -> int:
    """Recommend batch size based on smallest client dataset.

    Formula: max(min_client_samples // 10, min_batch_size)

    Args:
        scenario_dir: Directory containing partitions.
        min_batch_size: Minimum batch size.

    Returns:
        Recommended batch size.
    """
    scenario_path = Path(scenario_dir)

    min_samples = float('inf')

    # Check all tasks and clients
    for task_dir in scenario_path.glob('task_*'):
        for client_file in task_dir.glob('client_*.parquet'):
            df = pd.read_parquet(client_file)
            min_samples = min(min_samples, len(df))

    if min_samples == float('inf'):
        return min_batch_size

    recommended = max(min_samples // 10, min_batch_size)
    return recommended


def load_task_data(
    scenario_dir: str,
    task_id: int,
    client_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """Load data for a task, optionally filtering by clients.

    Args:
        scenario_dir: Directory containing partitions.
        task_id: Task ID (0-4).
        client_ids: Optional list of client IDs to include.

    Returns:
        Concatenated DataFrame.
    """
    scenario_path = Path(scenario_dir)
    task_dir = scenario_path / f'task_{task_id}'

    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    dfs = []

    for client_file in task_dir.glob('client_*.parquet'):
        client_id = int(client_file.stem.split('_')[1])

        if client_ids is not None and client_id not in client_ids:
            continue

        df = pd.read_parquet(client_file)
        df['client_id'] = client_id
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def get_label_mapping(scenario_dir: str) -> Dict[str, int]:
    """Get label to ID mapping.

    Args:
        scenario_dir: Directory containing partitions.

    Returns:
        Dictionary mapping label to ID.
    """
    metadata_path = Path(scenario_dir) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['label2id']


def get_num_classes(scenario_dir: str) -> int:
    """Get total number of classes.

    Args:
        scenario_dir: Directory containing partitions.

    Returns:
        Number of classes.
    """
    label2id = get_label_mapping(scenario_dir)
    return len(label2id)
