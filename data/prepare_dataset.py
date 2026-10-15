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
import shutil
from typing import Any, Dict, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    STATIC_LABEL_MAP,
    DYNAMIC_LABEL_MAP,
    ALL_LABELS,
    LABEL2ID,
    ScenarioConfig,
)
from data.schema import drop_raw_metadata, get_feature_columns
from data.scaling import fit_transform_splits


class AndMal2020DataPreparer:
    """
    Standardizes and prepares CIC-AndMal-2020 raw static and dynamic CSV artifacts.
    Generates unified training pools and independent stratified held-out test sets.
    """

    def __init__(
        self,
        raw_root: str,
        output_dir: str,
        chunksize: int = 100000,
        seed: int = 42,
        strict_class_coverage: bool = False,
        data_provenance: str = "user_supplied",
        allow_classwise_alignment: bool = False,
    ):
        self.raw_root = os.path.abspath(raw_root)
        self.output_dir = os.path.abspath(output_dir)
        self.chunksize = chunksize
        self.seed = seed
        self.strict_class_coverage = strict_class_coverage
        self.data_provenance = data_provenance
        self.allow_classwise_alignment = allow_classwise_alignment
        self.summary: Dict[str, Any] = {}

    def _find_csv_files(self, relative_directories: List[str]) -> List[str]:
        """Find direct CSV children in the supported canonical and flat layouts."""
        files = []
        for relative_directory in relative_directories:
            directory = os.path.join(self.raw_root, relative_directory)
            files.extend(glob.glob(os.path.join(directory, "*.csv")))
        return sorted(set(files))

    @staticmethod
    def _validate_file_schema(
        expected_features: Optional[List[str]],
        df: pd.DataFrame,
        file_path: str,
    ) -> List[str]:
        """Require every source file in one modality to use the same feature set."""
        features = get_feature_columns(df)
        if expected_features is None:
            return features
        if set(features) != set(expected_features):
            missing = [column for column in expected_features if column not in features]
            unexpected = [column for column in features if column not in expected_features]
            raise ValueError(
                f"Feature schema mismatch in {file_path}: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}, expected_count={len(expected_features)}, "
                f"found_count={len(features)}"
            )
        return expected_features

    @staticmethod
    def _align_chunk_columns(chunk: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
        """Align chunk columns so feature columns match expected canonical order."""
        features = get_feature_columns(chunk)
        if features == expected_features:
            return chunk
        meta_cols = [c for c in chunk.columns if c not in expected_features]
        return chunk[expected_features + meta_cols]

    @staticmethod
    def _detect_csv_has_header(file_path: str) -> Tuple[bool, int]:
        """Inspect the first line to detect if the CSV has a column header and determine column count."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
        if not first_line:
            return True, 0
        tokens = [t.strip().strip('"').strip("'") for t in first_line.split(",")]
        col_count = len(tokens)
        if col_count > 1:
            try:
                float(tokens[1])
                # If the second token parses as a number, row 0 is data, not a header row
                return False, col_count
            except ValueError:
                return True, col_count
        return True, col_count

    def _record_schema(self, df: pd.DataFrame, out_dir: str, feature_type: str) -> None:
        feature_columns = get_feature_columns(df)
        self._record_schema_from_columns(feature_columns, out_dir, feature_type)

    def _record_schema_from_columns(
        self,
        feature_columns: List[str],
        out_dir: str,
        feature_type: str,
        static_dim: Optional[int] = None,
        dynamic_dim: Optional[int] = None,
    ) -> None:
        schema = {
            "feature_type": feature_type,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
        }
        if static_dim is not None:
            schema["static_dim"] = static_dim
        if dynamic_dim is not None:
            schema["dynamic_dim"] = dynamic_dim
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
            if self.strict_class_coverage:
                raise ValueError(message)
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
            raise FileNotFoundError(
                f"No static CSV files found under {self.raw_root}. "
                "Provide the real CIC-AndMal files or explicitly generate a "
                "development dataset before running preparation."
            )

        out_static_dir = os.path.join(self.output_dir, "static")
        os.makedirs(out_static_dir, exist_ok=True)
        temp_cache_dir = os.path.join(out_static_dir, "_temp_cache")
        os.makedirs(temp_cache_dir, exist_ok=True)

        meta_chunks = []
        expected_features: Optional[List[str]] = None
        sample_counter = 0

        try:
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

                file_pq_path = os.path.join(temp_cache_dir, f"{stem}.parquet")
                if os.path.isfile(file_pq_path):
                    print(f"  -> Using cached {stem}.parquet (Class: {matched_label})...")
                    pq_file = pq.ParquetFile(file_pq_path)
                    if expected_features is None:
                        expected_features = [
                            col for col in pq_file.schema.names
                            if col not in ["Sample_ID", "label", "Split_Group_ID"]
                        ]
                    meta_cols = ["Sample_ID", "label"]
                    if "Split_Group_ID" in pq_file.schema.names:
                        meta_cols.append("Split_Group_ID")
                    meta_chunks.append(pd.read_parquet(file_pq_path, columns=meta_cols))
                    continue

                print(f"  -> Reading {file_name} (Class: {matched_label})...")
                has_header, col_count = self._detect_csv_has_header(file_path)
                read_kwargs: Dict[str, Any] = {"chunksize": self.chunksize, "low_memory": False}
                if not has_header:
                    names = ["Hash"] + [f"static_feat_{i}" for i in range(col_count - 1)]
                    dtypes = {f"static_feat_{i}": np.float32 for i in range(col_count - 1)}
                    dtypes["Hash"] = str
                    read_kwargs["names"] = names
                    read_kwargs["header"] = None
                    read_kwargs["dtype"] = dtypes

                file_chunks = []
                for chunk in pd.read_csv(file_path, **read_kwargs):
                    chunk = drop_raw_metadata(chunk)
                    expected_features = self._validate_file_schema(
                        expected_features, chunk, file_path
                    )
                    chunk = self._align_chunk_columns(chunk, expected_features)
                    chunk["label"] = matched_label
                    if "Sample_ID" not in chunk.columns:
                        n_rows = len(chunk)
                        chunk.insert(
                            0,
                            "Sample_ID",
                            [f"SAMPLE_STAT_{sample_counter + i:07d}" for i in range(n_rows)],
                        )
                        sample_counter += n_rows
                    file_chunks.append(chunk)

                df_file = pd.concat(file_chunks, ignore_index=True)
                # Cache each file as parquet to prevent OOM
                file_pq_path = os.path.join(temp_cache_dir, f"{stem}.parquet")
                df_file.to_parquet(file_pq_path, index=False)
                meta_cols = ["Sample_ID", "label"]
                if "Split_Group_ID" in df_file.columns:
                    meta_cols.append("Split_Group_ID")
                meta_chunks.append(df_file[meta_cols].copy())
                del df_file
                del file_chunks

            meta_df = pd.concat(meta_chunks, ignore_index=True)
            self._record_class_coverage(meta_df, "static")
            self._record_schema_from_columns(expected_features, out_static_dir, "static")

            # Stratified train / validation / test split
            val_rel_ratio = val_ratio / (test_ratio + val_ratio)
            group_column = "Split_Group_ID" if "Split_Group_ID" in meta_df.columns else "Sample_ID"

            groups = (
                meta_df[[group_column, "label"]]
                .drop_duplicates(subset=[group_column])
                .reset_index(drop=True)
            )
            group_counts = groups["label"].value_counts()
            group_stratify = groups["label"] if int(group_counts.min()) >= 2 else None
            train_groups, temp_groups = train_test_split(
                groups,
                test_size=(test_ratio + val_ratio),
                stratify=group_stratify,
                random_state=self.seed,
            )
            temp_counts = temp_groups["label"].value_counts()
            temp_stratify = temp_groups["label"] if int(temp_counts.min()) >= 2 else None
            val_groups, test_groups = train_test_split(
                temp_groups,
                test_size=(1.0 - val_rel_ratio),
                stratify=temp_stratify,
                random_state=self.seed,
            )

            train_group_set = set(train_groups[group_column])
            val_group_set = set(val_groups[group_column])
            test_group_set = set(test_groups[group_column])

            train_id_set = set(meta_df[meta_df[group_column].isin(train_group_set)]["Sample_ID"])
            val_id_set = set(meta_df[meta_df[group_column].isin(val_group_set)]["Sample_ID"])
            test_id_set = set(meta_df[meta_df[group_column].isin(test_group_set)]["Sample_ID"])

            # Fit StandardScaler incrementally on train samples
            print("  [Static] Fitting StandardScaler on training split...")
            scaler = StandardScaler()
            for temp_f in sorted(glob.glob(os.path.join(temp_cache_dir, "*.parquet"))):
                df_temp = pd.read_parquet(temp_f)
                train_slice = df_temp[df_temp["Sample_ID"].isin(train_id_set)]
                if len(train_slice) > 0:
                    scaler.partial_fit(
                        train_slice[expected_features].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
                    )
                del df_temp

            joblib.dump(scaler, os.path.join(out_static_dir, "scaler.joblib"))
            with open(os.path.join(out_static_dir, "scaler_metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "transform": "sklearn.preprocessing.StandardScaler",
                        "fit_split": "train",
                        "feature_columns": expected_features,
                        "feature_count": len(expected_features),
                        "train_imputation": "column_mean_fit_on_train",
                        "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(),
                    },
                    handle,
                    indent=2,
                )

            # Transform and write splits
            print("  [Static] Standardizing and streaming splits to disk...")
            writers = {}
            csv_first = {"train": True, "val": True, "test": True, "static_all": True}
            split_files = {
                "train": (os.path.join(out_static_dir, "train.parquet"), os.path.join(out_static_dir, "train.csv")),
                "val": (os.path.join(out_static_dir, "val.parquet"), os.path.join(out_static_dir, "val.csv")),
                "test": (os.path.join(out_static_dir, "test.parquet"), os.path.join(out_static_dir, "test.csv")),
                "static_all": (os.path.join(out_static_dir, "static_all.parquet"), os.path.join(out_static_dir, "static_all.csv")),
            }

            for p_pq, p_csv in split_files.values():
                if os.path.isfile(p_pq):
                    os.remove(p_pq)
                if os.path.isfile(p_csv):
                    os.remove(p_csv)

            write_csv = len(expected_features) < 500
            def append_split_chunk(name: str, chunk_df: pd.DataFrame):
                if len(chunk_df) == 0:
                    return
                p_pq, p_csv = split_files[name]
                table = pa.Table.from_pandas(chunk_df)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(p_pq, table.schema)
                writers[name].write_table(table)
                if write_csv:
                    chunk_df.to_csv(p_csv, mode="a", header=csv_first[name], index=False)
                    csv_first[name] = False

            train_count = 0
            val_count = 0
            test_count = 0
            train_class_counts: Dict[str, int] = {}
            test_class_counts: Dict[str, int] = {}

            for temp_f in sorted(glob.glob(os.path.join(temp_cache_dir, "*.parquet"))):
                df_temp = pd.read_parquet(temp_f)
                scaled_mat = scaler.transform(
                    df_temp[expected_features].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
                ).astype(np.float32, copy=False)
                df_scaled = pd.DataFrame(scaled_mat, columns=expected_features, index=df_temp.index)
                meta_cols = [c for c in df_temp.columns if c not in expected_features]
                df_transformed = pd.concat([df_temp[meta_cols], df_scaled], axis=1)

                train_part = df_transformed[df_transformed["Sample_ID"].isin(train_id_set)]
                val_part = df_transformed[df_transformed["Sample_ID"].isin(val_id_set)]
                test_part = df_transformed[df_transformed["Sample_ID"].isin(test_id_set)]

                train_count += len(train_part)
                val_count += len(val_part)
                test_count += len(test_part)

                for lbl, c in train_part["label"].value_counts().items():
                    train_class_counts[lbl] = train_class_counts.get(lbl, 0) + int(c)
                for lbl, c in test_part["label"].value_counts().items():
                    test_class_counts[lbl] = test_class_counts.get(lbl, 0) + int(c)

                append_split_chunk("train", train_part)
                append_split_chunk("val", val_part)
                append_split_chunk("test", test_part)
                append_split_chunk("static_all", df_transformed)

                del df_temp
                del df_transformed

            for w in writers.values():
                w.close()

            summary_file = os.path.join(out_static_dir, "split_summary.json")
            with open(summary_file, "w") as f:
                json.dump({
                    "feature_type": "static",
                    "total_samples": len(meta_df),
                    "train_samples": train_count,
                    "val_samples": val_count,
                    "test_samples": test_count,
                    "class_counts_total": meta_df["label"].value_counts().to_dict(),
                    "class_counts_train": train_class_counts,
                    "class_counts_test": test_class_counts,
                    "split_group_column": group_column,
                    "group_disjoint": True,
                }, f, indent=4)
            print(f"  -> Split Summary: Train={train_count}, Val={val_count}, Test={test_count}")

            full_path = split_files["static_all"][0] if not write_csv else split_files["static_all"][1]
            print(f"  [Static] Saved complete dataset to {full_path} (Total samples: {len(meta_df)})")
            return full_path

        finally:
            if os.path.isdir(temp_cache_dir):
                shutil.rmtree(temp_cache_dir, ignore_errors=True)

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

        # When real uppercase files exist, exclude synthetic lowercase CSVs
        uppercase_files = [f for f in dynamic_files if os.path.basename(f)[0].isupper()]
        if uppercase_files:
            print(f"  [Stage 1] Real uppercase telemetry detected ({len(uppercase_files)} files). Excluding synthetic lowercase CSVs.")
            dynamic_files = uppercase_files

        dfs = []
        expected_features: Optional[List[str]] = None
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
                expected_features = self._validate_file_schema(
                    expected_features, chunk, file_path
                )
                chunk = self._align_chunk_columns(chunk, expected_features)
                chunk["label"] = matched_label
                chunk["reboot_phase"] = phase
                file_chunks.append(chunk)
            df_file = pd.concat(file_chunks, ignore_index=True)
            dfs.append(df_file)

        full_df = pd.concat(dfs, ignore_index=True).copy()
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

    def prepare_fused(
        self,
        static_df: Optional[pd.DataFrame] = None,
        dynamic_df: Optional[pd.DataFrame] = None,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        allow_classwise_alignment: bool = False,
    ) -> str:
        """Merge static and dynamic features on a genuine shared Sample_ID.

        Class-conditioned row pairing is disabled by default because it creates
        synthetic multimodal examples from unrelated applications.
        """
        print("\n[Stage 1] Preparing Fused Multi-Modal Dataset (Static + Dynamic)...")
        if static_df is None:
            static_parquet = os.path.join(self.output_dir, "static", "static_all.parquet")
            if os.path.isfile(static_parquet):
                static_df = pd.read_parquet(static_parquet)
            else:
                static_df = pd.read_csv(os.path.join(self.output_dir, "static", "static_all.csv"), low_memory=False)
        if dynamic_df is None:
            dynamic_parquet = os.path.join(self.output_dir, "dynamic", "dynamic_all.parquet")
            if os.path.isfile(dynamic_parquet):
                dynamic_df = pd.read_parquet(dynamic_parquet)
            else:
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
            if not allow_classwise_alignment and not self.allow_classwise_alignment:
                raise ValueError(
                    "Static and dynamic Sample_ID values do not overlap. Refusing "
                    "class-wise row pairing because it is not valid sample-level "
                    "multimodal fusion."
                )
            warnings.warn(
                "Class-wise fused alignment is synthetic and development-only.",
                RuntimeWarning,
                stacklevel=2,
            )
            print("  [Development] Performing class-wise synthetic alignment...")

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
                merged_df = pd.concat(fused_dfs, ignore_index=True).copy()
                merged_df["Sample_ID"] = [f"SAMPLE_FUSED_{i:07d}" for i in range(len(merged_df))]
            else:
                merged_df = pd.DataFrame()

        out_fused_dir = os.path.join(self.output_dir, "fused")
        os.makedirs(out_fused_dir, exist_ok=True)
        self._record_class_coverage(merged_df, "fused")
        fused_feat_cols = get_feature_columns(merged_df)
        stat_dim = len([c for c in fused_feat_cols if c.startswith("static_feat_")])
        if stat_dim == 0:
            stat_feat_names = set(c for c in static_df.columns if c not in ["Sample_ID", "Split_Group_ID", "label"])
            stat_dim = len(stat_feat_names)
        dyn_dim = len(fused_feat_cols) - stat_dim
        self._record_schema_from_columns(
            fused_feat_cols, out_fused_dir, "fused",
            static_dim=stat_dim, dynamic_dim=dyn_dim
        )

        parquet_path = os.path.join(out_fused_dir, "fused_all.parquet")
        csv_path = os.path.join(out_fused_dir, "fused_all.csv")
        merged_df.to_parquet(parquet_path, index=False)
        if len(merged_df.columns) < 500:
            merged_df.to_csv(csv_path, index=False)
            full_path = csv_path
        else:
            full_path = parquet_path
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
        val_rel_ratio = val_ratio / (test_ratio + val_ratio)

        group_column = None
        if "Split_Group_ID" in df.columns:
            group_column = "Split_Group_ID"
        elif "Sample_ID" in df.columns and df["Sample_ID"].duplicated().any():
            group_column = "Sample_ID"

        if group_column is not None:
            group_labels = df.groupby(group_column)["label"].nunique()
            if int(group_labels.max()) > 1:
                raise ValueError(
                    f"Split group column {group_column} maps one identity to multiple labels"
                )
            groups = (
                df[[group_column, "label"]]
                .drop_duplicates(subset=[group_column])
                .reset_index(drop=True)
            )
            group_counts = groups["label"].value_counts()
            group_stratify = (
                groups["label"] if int(group_counts.min()) >= 2 else None
            )
            train_groups, temp_groups = train_test_split(
                groups,
                test_size=(test_ratio + val_ratio),
                stratify=group_stratify,
                random_state=self.seed,
            )
            temp_counts = temp_groups["label"].value_counts()
            temp_stratify = (
                temp_groups["label"] if int(temp_counts.min()) >= 2 else None
            )
            val_groups, test_groups = train_test_split(
                temp_groups,
                test_size=(1.0 - val_rel_ratio),
                stratify=temp_stratify,
                random_state=self.seed,
            )
            train_df = df[df[group_column].isin(train_groups[group_column])].copy()
            val_df = df[df[group_column].isin(val_groups[group_column])].copy()
            test_df = df[df[group_column].isin(test_groups[group_column])].copy()
        else:
            use_stratify = y if (class_counts.min() >= 2) else None
            train_df, temp_df = train_test_split(
                df,
                test_size=(test_ratio + val_ratio),
                stratify=use_stratify,
                random_state=self.seed,
            )
            temp_class_counts = temp_df["label"].value_counts()
            temp_stratify = (
                temp_df["label"].values if (temp_class_counts.min() >= 2) else None
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1.0 - val_rel_ratio),
                stratify=temp_stratify,
                random_state=self.seed,
            )

        # Fit preprocessing only on train, then apply the same transform to all splits.
        # This prevents held-out statistics from entering training or client data.
        train_df, val_df, test_df = fit_transform_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=out_dir,
        )

        # Save parquet and csv versions
        for name, sub_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            parquet_path = os.path.join(out_dir, f"{name}.parquet")
            csv_path = os.path.join(out_dir, f"{name}.csv")
            try:
                sub_df.to_parquet(parquet_path, index=False)
            except Exception:
                pass
            if len(sub_df.columns) < 500 or not os.path.isfile(parquet_path):
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
                "split_group_column": group_column,
                "group_disjoint": group_column is not None,
            }, f, indent=4)
        print(f"  -> Split Summary: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def run_all(self, data_type: str = "both", test_ratio: float = 0.2, val_ratio: float = 0.1) -> None:
        """Run complete preparation pipeline."""
        if data_type in ["static", "both", "all"]:
            self.prepare_static(test_ratio, val_ratio)
        if data_type in ["dynamic", "both", "all"]:
            self.prepare_dynamic(test_ratio, val_ratio)
        if data_type in ["fused", "all"]:
            self.prepare_fused(
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                allow_classwise_alignment=(self.allow_classwise_alignment or not self.strict_class_coverage),
            )

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
        with open(os.path.join(self.output_dir, "data_provenance.json"), "w") as f:
            json.dump(
                {
                    "source": self.data_provenance,
                    "raw_root": self.raw_root,
                    "seed": self.seed,
                    "strict_class_coverage": self.strict_class_coverage,
                    "scaling": "train-fitted StandardScaler per feature type",
                },
                f,
                indent=2,
            )
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
    parser.add_argument(
        "--strict_class_coverage",
        action="store_true",
        help="Fail preparation if any configured benchmark class is absent",
    )
    parser.add_argument(
        "--allow_classwise_alignment",
        action="store_true",
        help="Allow class-wise pairing when static and dynamic Sample_IDs do not overlap",
    )
    parser.add_argument(
        "--data_provenance",
        choices=["user_supplied", "synthetic_development"],
        default="user_supplied",
        help="Provenance label persisted with prepared artifacts",
    )

    args = parser.parse_args()
    preparer = AndMal2020DataPreparer(
        raw_root=args.root,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        seed=args.seed,
        strict_class_coverage=args.strict_class_coverage,
        data_provenance=args.data_provenance,
        allow_classwise_alignment=args.allow_classwise_alignment,
    )
    preparer.run_all(data_type=args.type, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    if args.summary:
        print("\nClass coverage:")
        print(json.dumps(preparer.summary, indent=2))


if __name__ == "__main__":
    main()
