"""
Stage 1 Dataset Preparation Pipeline for CIC-AndMal-2020.
Ingests raw static & dynamic CSVs, standardizes 15 malware & benign classes,
applies chunked streaming, generates stratified held-out test/validation splits,
and exports clean prepared datasets.
"""

import os
import glob
import argparse
import json
import warnings
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    STATIC_LABEL_MAP,
    DYNAMIC_LABEL_MAP,
    ALL_LABELS,
    LABEL2ID,
    ScenarioConfig,
)
from data.schema import drop_raw_metadata, get_feature_columns


class AndMal2020DataPreparer:
    """
    Standardizes and prepares CIC-AndMal-2020 raw static and dynamic CSV artifacts.
    Generates unified training pools and independent stratified held-out test sets.
    """

    def __init__(self, raw_root: str, output_dir: str, chunksize: int = 100000, seed: int = 42):
        self.raw_root = os.path.abspath(raw_root)
        self.output_dir = os.path.abspath(output_dir)
        self.chunksize = chunksize
        self.seed = seed
        self.summary: Dict[str, Any] = {}

    def _find_csv_files(self, relative_directories: List[str]) -> List[str]:
        """Find direct CSV children in the supported canonical and flat layouts."""
        files = []
        for relative_directory in relative_directories:
            directory = os.path.join(self.raw_root, relative_directory)
            files.extend(glob.glob(os.path.join(directory, "*.csv")))
        return sorted(set(files))

    def _record_schema(self, df: pd.DataFrame, out_dir: str, feature_type: str) -> None:
        feature_columns = get_feature_columns(df)
        schema = {
            "feature_type": feature_type,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
        }
        with open(os.path.join(out_dir, "feature_schema.json"), "w") as schema_file:
            json.dump(schema, schema_file, indent=2)

    def _record_class_coverage(self, df: pd.DataFrame, feature_type: str) -> None:
        available = sorted(df["label"].astype(str).unique().tolist())
        missing = [label for label in ALL_LABELS if label not in available]
        self.summary[feature_type] = {
            "available_labels": available,
            "missing_labels": missing,
            "sample_count": len(df),
        }
        if missing:
            message = (
                f"{feature_type} data is missing configured classes: {missing}. "
                "Training can be used as a development run, but it is not a complete "
                f"{len(ALL_LABELS)}-class experiment."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            print(f"  [Warning] {message}")

    def prepare_static(self, test_ratio: float = 0.2, val_ratio: float = 0.1) -> str:
        """
        Process static CSVs (Benign Ben0..Ben4 and 14 Malicious family files).
        """
        print(f"\n[Stage 1] Preparing Static Dataset from {self.raw_root}...")
        benign_files = self._find_csv_files([
            os.path.join("Static", "CCCS-CIC-Benign-CSVs"),
            "CCCS-CIC-Benign-CSVs",
        ])
        malware_files = self._find_csv_files([
            os.path.join("Static", "CCCS-CIC-Malicious-CSVs"),
            "CCCS-CIC-Malicious-CSVs",
        ])
        all_static_files = benign_files + malware_files

        if not all_static_files:
            print(f"  [Notice] No raw static CSV files found in {self.raw_root}/Static/.")
            print("  [Notice] Auto-generating static benchmark structure into raw_root...")
            from data.synthetic_generator import generate_synthetic_raw_andmal2020
            generate_synthetic_raw_andmal2020(root_dir=self.raw_root)
            benign_files = self._find_csv_files([
                os.path.join("Static", "CCCS-CIC-Benign-CSVs"),
                "CCCS-CIC-Benign-CSVs",
            ])
            malware_files = self._find_csv_files([
                os.path.join("Static", "CCCS-CIC-Malicious-CSVs"),
                "CCCS-CIC-Malicious-CSVs",
            ])
            all_static_files = benign_files + malware_files
            if not all_static_files:
                raise FileNotFoundError(f"No static CSV files found in {self.raw_root}/Static/...")

        dfs = []
        for file_path in all_static_files:
            file_name = os.path.basename(file_path)
            stem = os.path.splitext(file_name)[0]

            # Find matching label
            matched_label = STATIC_LABEL_MAP.get(stem)
            if matched_label is None:
                # Try prefix matching
                for key, val in STATIC_LABEL_MAP.items():
                    if stem.startswith(key):
                        matched_label = val
                        break
            if matched_label is None:
                print(f"  [Warning] Skipping unrecognized static file: {file_name}")
                continue

            print(f"  -> Reading {file_name} (Class: {matched_label})...")
            # Chunked reading for large files
            file_chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.chunksize, low_memory=False):
                chunk = drop_raw_metadata(chunk)
                chunk["label"] = matched_label
                file_chunks.append(chunk)
            df_file = pd.concat(file_chunks, ignore_index=True)
            dfs.append(df_file)

        full_df = pd.concat(dfs, ignore_index=True)
        # Ensure Sample_ID column exists
        if "Sample_ID" not in full_df.columns:
            full_df.insert(0, "Sample_ID", [f"SAMPLE_STAT_{i:07d}" for i in range(len(full_df))])

        out_static_dir = os.path.join(self.output_dir, "static")
        os.makedirs(out_static_dir, exist_ok=True)
        self._record_class_coverage(full_df, "static")
        self._record_schema(full_df, out_static_dir, "static")
        
        # Save complete combined dataset
        full_path = os.path.join(out_static_dir, "static_all.csv")
        full_df.to_csv(full_path, index=False)
        print(f"  [Static] Saved complete dataset to {full_path} (Total samples: {len(full_df)})")

        # Split held-out test & val set
        self._split_and_save(full_df, out_static_dir, test_ratio, val_ratio, "static")
        return full_path

    def prepare_dynamic(self, test_ratio: float = 0.2, val_ratio: float = 0.1) -> str:
        """
        Process dynamic CSVs (before & after reboot runtime logs).
        """
        print(f"\n[Stage 1] Preparing Dynamic Dataset from {self.raw_root}...")
        dynamic_directories = [
            os.path.join("Dynamic", "AndMal2020-dynamic-BeforeAndAfterReboot"),
            "AndMal2020-dynamic-BeforeAndAfterReboot",
        ]
        if os.path.basename(os.path.normpath(self.raw_root)).casefold() == (
            "AndMal2020-dynamic-BeforeAndAfterReboot".casefold()
        ):
            dynamic_directories.append("")
        dynamic_files = self._find_csv_files(dynamic_directories)

        if not dynamic_files:
            raise FileNotFoundError(
                f"No dynamic CSV files found under {self.raw_root}. Expected either "
                "Dynamic/AndMal2020-dynamic-BeforeAndAfterReboot/ or "
                "AndMal2020-dynamic-BeforeAndAfterReboot/."
            )

        dfs = []
        for file_path in dynamic_files:
            file_name = os.path.basename(file_path)
            # Parse stem, label and reboot phase (e.g. trojan_sms_before_reboot_Cat.csv)
            phase = "before" if "before" in file_name.lower() else "after"
            matched_label = None
            for key, val in DYNAMIC_LABEL_MAP.items():
                if file_name.lower().startswith(key):
                    matched_label = val
                    break

            if matched_label is None:
                print(f"  [Warning] Skipping unrecognized dynamic file: {file_name}")
                continue

            print(f"  -> Reading {file_name} (Class: {matched_label}, Phase: {phase})...")
            file_chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.chunksize, low_memory=False):
                chunk = drop_raw_metadata(chunk)
                chunk["label"] = matched_label
                chunk["reboot_phase"] = phase
                file_chunks.append(chunk)
            df_file = pd.concat(file_chunks, ignore_index=True)
            dfs.append(df_file)

        full_df = pd.concat(dfs, ignore_index=True)
        if "Sample_ID" not in full_df.columns:
            full_df.insert(0, "Sample_ID", [f"SAMPLE_DYN_{i:07d}" for i in range(len(full_df))])

        out_dyn_dir = os.path.join(self.output_dir, "dynamic")
        os.makedirs(out_dyn_dir, exist_ok=True)
        self._record_class_coverage(full_df, "dynamic")
        self._record_schema(full_df, out_dyn_dir, "dynamic")
        
        full_path = os.path.join(out_dyn_dir, "dynamic_all.csv")
        full_df.to_csv(full_path, index=False)
        print(f"  [Dynamic] Saved complete dataset to {full_path} (Total samples: {len(full_df)})")

        # Split held-out test & val set
        self._split_and_save(full_df, out_dyn_dir, test_ratio, val_ratio, "dynamic")
        return full_path

    def prepare_fused(self, static_df: Optional[pd.DataFrame] = None, dynamic_df: Optional[pd.DataFrame] = None,
                      test_ratio: float = 0.2, val_ratio: float = 0.1) -> str:
        """
        Merge static and dynamic features on Sample_ID (inner join).
        Falls back to class-wise alignment if direct Sample_ID matching yields 0 rows.
        """
        print("\n[Stage 1] Preparing Fused Multi-Modal Dataset (Static + Dynamic)...")
        if static_df is None:
            static_df = pd.read_csv(os.path.join(self.output_dir, "static", "static_all.csv"), low_memory=False)
        if dynamic_df is None:
            dynamic_df = pd.read_csv(os.path.join(self.output_dir, "dynamic", "dynamic_all.csv"), low_memory=False)

        # Retain static columns and dynamic columns with suffix
        dyn_feature_cols = [c for c in dynamic_df.columns if c not in ["label", "reboot_phase"]]
        static_feature_cols = [c for c in static_df.columns if c != "label"]

        merged_df = pd.merge(
            static_df,
            dynamic_df[dyn_feature_cols],
            on="Sample_ID",
            how="inner",
            suffixes=("", "_dyn")
        )

        if len(merged_df) == 0:
            print("  [Notice] Direct Sample_ID join resulted in 0 matching samples.")
            print("  [Notice] Performing class-wise sample alignment to build multi-modal fused dataset...")
            
            shared_labels = sorted(list(set(static_df["label"].astype(str)).intersection(set(dynamic_df["label"].astype(str)))))
            fused_dfs = []

            for lbl in shared_labels:
                sub_stat = static_df[static_df["label"].astype(str) == lbl].reset_index(drop=True)
                sub_dyn = dynamic_df[dynamic_df["label"].astype(str) == lbl].reset_index(drop=True)

                dyn_cols = [c for c in dyn_feature_cols if c in sub_dyn.columns]
                n_samples = min(len(sub_stat), len(sub_dyn))
                if n_samples == 0:
                    continue

                stat_part = sub_stat.iloc[:n_samples].copy()
                dyn_part = sub_dyn[dyn_cols].iloc[:n_samples].copy().reset_index(drop=True)

                stat_feat_names = set(c for c in stat_part.columns if c not in ["Sample_ID", "label"])
                overlap_cols = [c for c in dyn_cols if c in stat_feat_names and c != "Sample_ID"]
                if overlap_cols:
                    dyn_part = dyn_part.rename(columns={c: f"{c}_dyn" for c in overlap_cols})

                if "Sample_ID" in dyn_part.columns:
                    dyn_part = dyn_part.drop(columns=["Sample_ID"])

                fused_part = pd.concat([stat_part, dyn_part], axis=1)
                fused_dfs.append(fused_part)

            if fused_dfs:
                merged_df = pd.concat(fused_dfs, ignore_index=True)
                merged_df["Sample_ID"] = [f"SAMPLE_FUSED_{i:07d}" for i in range(len(merged_df))]
            else:
                merged_df = pd.DataFrame()

        out_fused_dir = os.path.join(self.output_dir, "fused")
        os.makedirs(out_fused_dir, exist_ok=True)
        self._record_class_coverage(merged_df, "fused")
        self._record_schema(merged_df, out_fused_dir, "fused")

        full_path = os.path.join(out_fused_dir, "fused_all.csv")
        merged_df.to_csv(full_path, index=False)
        print(f"  [Fused] Saved fused dataset to {full_path} (Total samples: {len(merged_df)})")

        self._split_and_save(merged_df, out_fused_dir, test_ratio, val_ratio, "fused")
        return full_path

    def _split_and_save(
        self,
        df: pd.DataFrame,
        out_dir: str,
        test_ratio: float,
        val_ratio: float,
        feature_type: str
    ) -> None:
        """Stratified train / validation / test split to establish pure held-out evaluation."""
        if len(df) == 0:
            print(f"  [Warning] Dataset for '{feature_type}' is empty (0 samples). Skipping split creation.")
            summary_file = os.path.join(out_dir, "split_summary.json")
            with open(summary_file, "w") as f:
                json.dump({
                    "feature_type": feature_type,
                    "total_samples": 0,
                    "train_samples": 0,
                    "val_samples": 0,
                    "test_samples": 0,
                    "class_counts_total": {},
                    "class_counts_train": {},
                    "class_counts_test": {},
                }, f, indent=4)
            return

        y = df["label"].values
        class_counts = df["label"].value_counts()
        use_stratify = y if (class_counts.min() >= 2) else None

        # Stratified train vs temp
        train_df, temp_df = train_test_split(
            df, test_size=(test_ratio + val_ratio), stratify=use_stratify, random_state=self.seed
        )
        # Stratified val vs test
        val_rel_ratio = val_ratio / (test_ratio + val_ratio)
        temp_class_counts = temp_df["label"].value_counts()
        temp_stratify = temp_df["label"].values if (temp_class_counts.min() >= 2) else None

        val_df, test_df = train_test_split(
            temp_df, test_size=(1.0 - val_rel_ratio), stratify=temp_stratify, random_state=self.seed
        )

        # Save parquet and csv versions
        for name, sub_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            parquet_path = os.path.join(out_dir, f"{name}.parquet")
            csv_path = os.path.join(out_dir, f"{name}.csv")
            try:
                sub_df.to_parquet(parquet_path, index=False)
            except Exception:
                pass
            sub_df.to_csv(csv_path, index=False)

        # Record distribution summary
        counts = df["label"].value_counts().to_dict()
        train_counts = train_df["label"].value_counts().to_dict()
        test_counts = test_df["label"].value_counts().to_dict()

        summary_file = os.path.join(out_dir, "split_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "feature_type": feature_type,
                "total_samples": len(df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "class_counts_total": counts,
                "class_counts_train": train_counts,
                "class_counts_test": test_counts,
            }, f, indent=4)
        print(f"  -> Split Summary: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def run_all(self, data_type: str = "both", test_ratio: float = 0.2, val_ratio: float = 0.1) -> None:
        """Run complete preparation pipeline."""
        if data_type in ["static", "both", "all"]:
            self.prepare_static(test_ratio, val_ratio)
        if data_type in ["dynamic", "both", "all"]:
            self.prepare_dynamic(test_ratio, val_ratio)
        if data_type in ["fused", "all"]:
            self.prepare_fused(test_ratio=test_ratio, val_ratio=val_ratio)

        # Generate dataset_paths.json for automatic linking to Stage 2
        paths = {
            "dataset": os.path.join(self.output_dir, "dynamic", "train.parquet"),
            "static_dataset": os.path.join(self.output_dir, "static", "train.parquet"),
            "dynamic_dataset": os.path.join(self.output_dir, "dynamic", "train.parquet"),
            "fused_dataset": os.path.join(self.output_dir, "fused", "train.parquet"),
        }
        with open(os.path.join(self.output_dir, "dataset_paths.json"), "w") as f:
            json.dump(paths, f, indent=4)
        with open(os.path.join(self.output_dir, "class_coverage.json"), "w") as f:
            json.dump(self.summary, f, indent=2)
        print("\n[Stage 1] Completed! dataset_paths.json generated.")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Dataset Preparation for CIC-AndMal-2020")
    parser.add_argument("--root", type=str, default="./raw_data", help="Root directory containing raw CSVs")
    parser.add_argument("--output_dir", type=str, default="./prepared_data", help="Destination prepared data directory")
    parser.add_argument("--chunksize", type=int, default=100000, help="CSV chunk streaming size")
    parser.add_argument("--type", type=str, choices=["static", "dynamic", "fused", "both", "all"], default="all")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="Held-out test set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Held-out validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--summary", action="store_true", help="Print summary of 15 classes")

    args = parser.parse_args()
    preparer = AndMal2020DataPreparer(
        raw_root=args.root,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        seed=args.seed
    )
    preparer.run_all(data_type=args.type, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    if args.summary:
        print("\nClass coverage:")
        print(json.dumps(preparer.summary, indent=2))


if __name__ == "__main__":
    main()
