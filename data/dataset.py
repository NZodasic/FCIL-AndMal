"""
PyTorch Dataset and Data Loading Utilities for Federated Class-Incremental Learning.
Provides FLTaskDataset, recommended batch sizing, memory-efficient tensor conversion,
and held-out test loaders.
"""

import os
import json
import glob
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import LABEL2ID, ID2LABEL, ALL_LABELS


class TabularMalwareDataset(Dataset):
    """PyTorch Dataset wrapper for in-memory numerical feature tensors and integer labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class FLTaskDataset:
    """
    Reader for client-specific task partition files.
    Extracts numeric features, standardizes class IDs, and constructs PyTorch DataLoaders.
    """

    def __init__(
        self,
        scenario_dir: str,
        task_id: int,
        client_id: int,
        label_col: str = "label",
        ignore_cols: Optional[List[str]] = None
    ):
        self.scenario_dir = scenario_dir
        self.task_id = task_id
        self.client_id = client_id
        self.label_col = label_col
        self.ignore_cols = ignore_cols or ["Sample_ID", "reboot_phase", "label"]

        self.file_path = self._find_client_file()
        self.df = self._load_data()

    def _find_client_file(self) -> str:
        task_dir = os.path.join(self.scenario_dir, f"task_{self.task_id}")
        parquet_file = os.path.join(task_dir, f"client_{self.client_id:02d}.parquet")
        csv_file = os.path.join(task_dir, f"client_{self.client_id:02d}.csv")

        if os.path.isfile(parquet_file):
            return parquet_file
        elif os.path.isfile(csv_file):
            return csv_file
        else:
            raise FileNotFoundError(f"Partition file not found for client {self.client_id} in {task_dir}")

    def _load_data(self) -> pd.DataFrame:
        if self.file_path.endswith(".parquet"):
            return pd.read_parquet(self.file_path)
        return pd.read_csv(self.file_path, low_memory=False)

    def as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract numerical features and mapped class IDs as numpy arrays.
        """
        if len(self.df) == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        feature_cols = [c for c in self.df.columns if c not in self.ignore_cols]
        # Filter for numeric columns
        X = self.df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
        y_raw = self.df[self.label_col].values
        y = np.array([LABEL2ID.get(lbl, -1) for lbl in y_raw], dtype=np.int64)

        return X, y

    def as_dataloader(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> DataLoader:
        """Construct standard PyTorch DataLoader."""
        X, y = self.as_numpy()
        if len(y) == 0:
            return DataLoader(TabularMalwareDataset(np.zeros((0, 1)), np.zeros(0)), batch_size=batch_size)
        
        ds = TabularMalwareDataset(X, y)
        actual_batch_size = min(batch_size, len(ds)) if len(ds) > 0 else batch_size
        return DataLoader(ds, batch_size=actual_batch_size, shuffle=shuffle, drop_last=drop_last)


def get_participating_clients(scenario_dir: str, task_id: int) -> List[int]:
    """
    Retrieve list of active client IDs with valid partition files for task_id.
    """
    task_dir = os.path.join(scenario_dir, f"task_{task_id}")
    if not os.path.isdir(task_dir):
        return []

    files = glob.glob(os.path.join(task_dir, "client_*.parquet")) + glob.glob(os.path.join(task_dir, "client_*.csv"))
    client_ids = set()
    for f in files:
        basename = os.path.basename(f)
        try:
            cid_str = basename.replace("client_", "").replace(".parquet", "").replace(".csv", "")
            client_ids.add(int(cid_str))
        except ValueError:
            pass
    return sorted(list(client_ids))


def recommend_batch_size(scenario_dir: str, min_batch_size: int = 256) -> int:
    """
    Compute recommended batch size based on smallest client sample pool:
    recommendation = max(min_samples_of_smallest_client // 10, min_batch_size)
    """
    meta_path = os.path.join(scenario_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            clients = meta.get("clients", {})
            sample_counts = [info["total_samples"] for info in clients.values() if "total_samples" in info]
            if sample_counts:
                min_client = min(sample_counts)
                return max(min_client // 10, min_batch_size)
    return min_batch_size


def load_heldout_test_set(
    prepared_data_dir: str,
    feature_type: str = "dynamic",
    ignore_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load central held-out test split for global multi-task evaluation.
    """
    ignore_cols = ignore_cols or ["Sample_ID", "reboot_phase", "label"]
    test_dir = os.path.join(prepared_data_dir, feature_type)
    parquet_path = os.path.join(test_dir, "test.parquet")
    csv_path = os.path.join(test_dir, "test.csv")

    if os.path.isfile(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(f"Held-out test set not found in {test_dir} (Checked test.parquet and test.csv)")

    feature_cols = [c for c in df.columns if c not in ignore_cols]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    y_raw = df["label"].values
    y = np.array([LABEL2ID.get(lbl, -1) for lbl in y_raw], dtype=np.int64)

    return X, y
