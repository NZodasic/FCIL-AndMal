"""
Stage 2 Partitioning: Dirichlet Non-IID Label Skew & Progressive Client Scaling.
Implements the 5-task FCIL partitioning strategy with dynamic client participation,
cover-label distribution, minimum sample bounds, and metadata generation.
"""

import os
import argparse
import json
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from config import (
    TASK_LABEL_MAP,
    ALL_LABELS,
    LABEL2ID,
    ScenarioConfig,
)
from utils.seed import get_slot_seed, set_seed


class FCILDataPartitioner:
    """
    Partitions prepared tabular Android malware datasets into federated client slices
    across 5 incremental tasks under realistic Dirichlet non-IID label skew.
    """

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.min_samples_slot = config.min_samples_per_label_client  # 30

    def partition_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Partition a prepared DataFrame into task and client directories with metadata.
        """
        set_seed(self.config.seed)
        scenario_dir = self.config.get_scenario_dir()
        os.makedirs(scenario_dir, exist_ok=True)
        print(f"\n[Stage 2] Partitioning Scenario: {self.config.feature_type} | Clients: {self.config.n_clients} | Alpha: {self.config.dirichlet_alpha}")
        print(f"  Target directory: {scenario_dir}")

        # Ensure label column is clean string
        df["label"] = df["label"].astype(str)
        available_labels = set(df["label"].unique())

        partition_matrix = np.zeros((self.config.n_clients, len(ALL_LABELS)), dtype=int)
        label_to_col_idx = {name: idx for idx, name in enumerate(ALL_LABELS)}

        task_label_records = []
        summary_records = []
        client_task_counts: Dict[str, Dict[str, int]] = {f"client_{c:02d}": {} for c in range(self.config.n_clients)}

        # Iterate through the 5 tasks
        for task_id in range(5):
            task_labels = TASK_LABEL_MAP[task_id]
            n_active_clients = self.config.get_active_client_count(task_id)
            active_client_ids = list(range(n_active_clients))
            task_dir = os.path.join(scenario_dir, f"task_{task_id}")
            os.makedirs(task_dir, exist_ok=True)

            print(f"\n  --- Task {task_id + 1}/5 (Labels: {task_labels}) | Active Clients: {n_active_clients}/{self.config.n_clients} ---")
            task_label_records.append({
                "task_id": task_id,
                "active_clients": n_active_clients,
                "labels": ",".join(task_labels)
            })

            # Identify the cover label (label with largest sample pool in this task)
            pool_sizes = {lbl: int(np.sum(df["label"] == lbl)) for lbl in task_labels}
            cover_label = max(pool_sizes, key=pool_sizes.get)
            print(f"    Task {task_id}: Cover label is '{cover_label}' (Pool size: {pool_sizes[cover_label]})")

            # Collect client slices for this task
            client_data_slices: Dict[int, List[pd.DataFrame]] = {cid: [] for cid in active_client_ids}

            for label in task_labels:
                label_sub_df = df[df["label"] == label].copy()
                pool_size = len(label_sub_df)

                if pool_size == 0:
                    print(f"    [Warning] Pool size for {label} is 0 in dataset!")
                    continue

                # Determine maximum trainable clients for this label
                max_k_possible = min(n_active_clients, max(1, pool_size // self.min_samples_slot))

                if label == cover_label:
                    # Cover label is assigned to ALL active clients (up to max_k_possible)
                    selected_clients = active_client_ids[:max_k_possible]
                else:
                    # Other labels assigned to random subset k in [ceil(max_k/2), max_k]
                    min_k = max(1, math.ceil(max_k_possible / 2))
                    k = np.random.randint(min_k, max_k_possible + 1)
                    selected_clients = np.random.choice(active_client_ids, size=k, replace=False).tolist()
                    selected_clients.sort()

                k_selected = len(selected_clients)
                # Split pool_size samples among selected_clients using Dirichlet
                client_allocations = self._dirichlet_split(
                    total_samples=pool_size,
                    k_clients=k_selected,
                    alpha=self.config.dirichlet_alpha
                )

                # Shuffle label samples deterministically
                indices = np.arange(pool_size)
                np.random.shuffle(indices)

                start_idx = 0
                allocated_sample_counts = []
                for cid, n_alloc in zip(selected_clients, client_allocations):
                    if n_alloc <= 0:
                        continue
                    end_idx = start_idx + n_alloc
                    assigned_indices = indices[start_idx:end_idx]
                    start_idx = end_idx

                    client_slice = label_sub_df.iloc[assigned_indices].copy()
                    
                    # Deterministic slot shuffle
                    slot_seed = get_slot_seed(self.config.seed, task_id, cid)
                    client_slice = client_slice.sample(frac=1.0, random_state=slot_seed).reset_index(drop=True)
                    
                    client_data_slices[cid].append(client_slice)
                    partition_matrix[cid, label_to_col_idx[label]] += len(client_slice)
                    allocated_sample_counts.append(len(client_slice))

                summary_records.append({
                    "task_id": task_id,
                    "label": label,
                    "is_cover": (label == cover_label),
                    "pool_size": pool_size,
                    "assigned_total": int(np.sum(allocated_sample_counts)),
                    "selected_clients": k_selected,
                    "min_per_client": int(np.min(allocated_sample_counts)) if allocated_sample_counts else 0,
                    "max_per_client": int(np.max(allocated_sample_counts)) if allocated_sample_counts else 0,
                    "status": "OK" if allocated_sample_counts and np.min(allocated_sample_counts) >= self.min_samples_slot else "LOW_SAMPLES"
                })

            # Save per-client partition files for this task
            for cid in active_client_ids:
                if len(client_data_slices[cid]) > 0:
                    client_df = pd.concat(client_data_slices[cid], ignore_index=True)
                else:
                    client_df = pd.DataFrame(columns=df.columns)

                c_key = f"client_{cid:02d}"
                client_task_counts[c_key][f"task_{task_id}"] = len(client_df)

                parquet_path = os.path.join(task_dir, f"client_{cid:02d}.parquet")
                csv_path = os.path.join(task_dir, f"client_{cid:02d}.csv")
                try:
                    client_df.to_parquet(parquet_path, index=False)
                except Exception:
                    pass
                client_df.to_csv(csv_path, index=False)

        # 3. Export Partition Table & Metadata
        self._export_metadata(
            scenario_dir=scenario_dir,
            partition_matrix=partition_matrix,
            task_label_records=task_label_records,
            summary_records=summary_records,
            client_task_counts=client_task_counts
        )
        print(f"\n[Stage 2] Partitioning successfully generated in: {scenario_dir}")
        return {"scenario_dir": scenario_dir, "partition_matrix": partition_matrix}

    def _dirichlet_split(self, total_samples: int, k_clients: int, alpha: float) -> List[int]:
        """
        Split total_samples across k_clients using Dirichlet proportion with guaranteed minimum floors.
        """
        if k_clients <= 1:
            return [total_samples]

        # Allocate base floor
        base_floor = min(self.min_samples_slot, total_samples // k_clients)
        remaining = total_samples - (base_floor * k_clients)

        if remaining <= 0:
            return [total_samples // k_clients] * k_clients

        proportions = np.random.dirichlet(np.repeat(alpha, k_clients))
        extra_alloc = np.floor(proportions * remaining).astype(int)
        
        # Distribute rounding remainder
        rem_diff = remaining - np.sum(extra_alloc)
        for i in range(rem_diff):
            extra_alloc[i % k_clients] += 1

        final_alloc = (base_floor + extra_alloc).tolist()
        return final_alloc

    def _export_metadata(
        self,
        scenario_dir: str,
        partition_matrix: np.ndarray,
        task_label_records: List[Dict[str, Any]],
        summary_records: List[Dict[str, Any]],
        client_task_counts: Dict[str, Dict[str, int]]
    ) -> None:
        """Save metadata.json, partition_table.csv, task_label_map.csv, label_client_summary.csv."""
        # 1. partition_table.csv
        col_names = [f"{lbl}" for lbl in ALL_LABELS]
        client_index = [f"client_{c:02d}" for c in range(self.config.n_clients)]
        part_df = pd.DataFrame(partition_matrix, index=client_index, columns=col_names)
        part_df.to_csv(os.path.join(scenario_dir, "partition_table.csv"))

        # 2. task_label_map.csv
        pd.DataFrame(task_label_records).to_csv(os.path.join(scenario_dir, "task_label_map.csv"), index=False)

        # 3. label_client_summary.csv
        pd.DataFrame(summary_records).to_csv(os.path.join(scenario_dir, "label_client_summary.csv"), index=False)

        # 4. metadata.json
        client_distributions = {}
        for c in range(self.config.n_clients):
            dist = {ALL_LABELS[j]: int(partition_matrix[c, j]) for j in range(len(ALL_LABELS)) if partition_matrix[c, j] > 0}
            client_distributions[f"client_{c:02d}"] = {
                "total_samples": int(np.sum(partition_matrix[c, :])),
                "label_distribution": dist,
                "task_samples": client_task_counts[f"client_{c:02d}"]
            }

        meta = {
            "feature_type": self.config.feature_type,
            "n_clients": self.config.n_clients,
            "dirichlet_alpha": self.config.dirichlet_alpha,
            "seed": self.config.seed,
            "label2id": LABEL2ID,
            "all_labels": ALL_LABELS,
            "task_label_map": TASK_LABEL_MAP,
            "task_client_counts": {t: self.config.get_active_client_count(t) for t in range(5)},
            "clients": client_distributions,
        }
        with open(os.path.join(scenario_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)


def run_scenario(df: pd.DataFrame, config: ScenarioConfig) -> Dict[str, Any]:
    """Top-level functional API to generate partition scenario."""
    partitioner = FCILDataPartitioner(config)
    return partitioner.partition_dataframe(df)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Dirichlet Partitioning for FCIL")
    parser.add_argument("--dataset", type=str, default="./prepared_data/dynamic/train.parquet")
    parser.add_argument("--static_dataset", type=str, default="./prepared_data/static/train.parquet")
    parser.add_argument("--dynamic_dataset", type=str, default="./prepared_data/dynamic/train.parquet")
    parser.add_argument("--feature_type", type=str, choices=["static", "dynamic", "fused"], default="dynamic")
    parser.add_argument("--n_clients", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./fl_data_partitions")

    args = parser.parse_args()

    # Load dataset
    if args.feature_type == "fused":
        print(f"Loading static ({args.static_dataset}) and dynamic ({args.dynamic_dataset})...")
        s_df = pd.read_parquet(args.static_dataset) if args.static_dataset.endswith(".parquet") else pd.read_csv(args.static_dataset)
        d_df = pd.read_parquet(args.dynamic_dataset) if args.dynamic_dataset.endswith(".parquet") else pd.read_csv(args.dynamic_dataset)
        dyn_cols = [c for c in d_df.columns if c not in ["label", "reboot_phase"]]
        df = pd.merge(s_df, d_df[dyn_cols], on="Sample_ID", how="inner")
    else:
        ds_path = args.dataset
        df = pd.read_parquet(ds_path) if ds_path.endswith(".parquet") else pd.read_csv(ds_path)

    for n_cl in args.n_clients:
        cfg = ScenarioConfig(
            feature_type=args.feature_type,
            n_clients=n_cl,
            dirichlet_alpha=args.dirichlet_alpha,
            seed=args.seed,
            partition_output_dir=args.output_dir
        )
        run_scenario(df, cfg)


if __name__ == "__main__":
    main()
