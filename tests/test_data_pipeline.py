"""
Unit and Integration Tests for Stage 1 Data Preparation and Stage 2 Partitioning.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

from config import ScenarioConfig, TASK_LABEL_MAP, ALL_LABELS
from data.synthetic_generator import (
    detect_synthetic_prepared_dir,
    detect_synthetic_raw_dir,
    generate_synthetic_raw_andmal2020,
)
from data.prepare_dataset import AndMal2020DataPreparer
from data.partition import FCILDataPartitioner
from data.dataset import FLTaskDataset, get_participating_clients, recommend_batch_size
from data.schema import get_feature_columns


class TestDataPipeline(unittest.TestCase):
    """Test full data lifecycle from raw CSV generation to client partition loading."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./tmp_test_data"
        cls.raw_dir = os.path.join(cls.test_dir, "raw")
        cls.prep_dir = os.path.join(cls.test_dir, "prepared")
        cls.part_dir = os.path.join(cls.test_dir, "partitions")

        os.makedirs(cls.test_dir, exist_ok=True)
        # Generate small synthetic raw dataset
        generate_synthetic_raw_andmal2020(
            root_dir=cls.raw_dir,
            samples_per_class=60,
            static_dim=50,
            dynamic_dim=20,
            seed=42
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_01_prepare_dataset_all_15_classes(self):
        """Verify Stage 1 creates prepared files with all 15 classes and held-out test splits."""
        preparer = AndMal2020DataPreparer(
            raw_root=self.raw_dir,
            output_dir=self.prep_dir,
            seed=42,
            data_provenance="synthetic_development",
        )
        preparer.run_all(data_type="all", test_ratio=0.2, val_ratio=0.1)

        # Verify dynamic files
        dyn_train = os.path.join(self.prep_dir, "dynamic", "train.parquet")
        dyn_test = os.path.join(self.prep_dir, "dynamic", "test.parquet")
        self.assertTrue(os.path.isfile(dyn_train) or os.path.isfile(os.path.join(self.prep_dir, "dynamic", "train.csv")))
        self.assertTrue(os.path.isfile(dyn_test) or os.path.isfile(os.path.join(self.prep_dir, "dynamic", "test.csv")))

    def test_02_partition_dynamic_clients_and_progressive_scaling(self):
        """Verify Stage 2 partitions into 5 tasks with progressive client scaling (12->14->16->18->20)."""
        train_path = os.path.join(self.prep_dir, "dynamic", "train.csv")
        df = pd.read_csv(train_path)

        cfg = ScenarioConfig(
            feature_type="dynamic",
            n_clients=20,
            dirichlet_alpha=0.5,
            seed=42,
            min_samples_per_label_client=5,
            partition_output_dir=self.part_dir
        )
        partitioner = FCILDataPartitioner(cfg)
        result = partitioner.partition_dataframe(df)
        scenario_dir = result["scenario_dir"]

        # Check progressive participation
        expected_active = [12, 14, 16, 18, 20]
        for t in range(5):
            active = get_participating_clients(scenario_dir, t)
            self.assertEqual(len(active), expected_active[t])

        # Verify metadata files exist
        self.assertTrue(os.path.isfile(os.path.join(scenario_dir, "partition_table.csv")))
        self.assertTrue(os.path.isfile(os.path.join(scenario_dir, "metadata.json")))
        self.assertTrue(os.path.isfile(os.path.join(scenario_dir, "task_label_map.csv")))

    def test_03_fl_task_dataset_loader(self):
        """Verify FLTaskDataset loads client data and constructs PyTorch DataLoader."""
        scenario_dir = os.path.join(self.part_dir, "dynamic", "20clients")
        ds = FLTaskDataset(scenario_dir=scenario_dir, task_id=0, client_id=0)
        X, y = ds.as_numpy()
        self.assertGreater(len(y), 0)
        self.assertEqual(X.shape[0], len(y))

        loader = ds.as_dataloader(batch_size=16)
        bx, by = next(iter(loader))
        self.assertGreater(bx.size(0), 0)

    def test_04_flat_dynamic_layout_drops_source_metadata(self):
        flat_root = os.path.join(self.test_dir, "flat_raw")
        dynamic_dir = os.path.join(
            flat_root, "AndMal2020-dynamic-BeforeAndAfterReboot"
        )
        flat_prepared = os.path.join(self.test_dir, "flat_prepared")
        os.makedirs(dynamic_dir, exist_ok=True)
        source = pd.DataFrame({
            "dyn_0": np.arange(20),
            "dyn_1": np.arange(20) + 1,
            "Hash": [f"hash-{index}" for index in range(20)],
            "Category": ["Adware"] * 20,
            "Family": ["family"] * 20,
        })
        source.to_csv(
            os.path.join(dynamic_dir, "Adware_before_reboot_Cat.csv"),
            index=False,
        )

        preparer = AndMal2020DataPreparer(flat_root, flat_prepared, seed=42)
        with self.assertWarnsRegex(RuntimeWarning, "missing configured classes"):
            preparer.prepare_dynamic(test_ratio=0.2, val_ratio=0.1)

        prepared = pd.read_csv(
            os.path.join(flat_prepared, "dynamic", "dynamic_all.csv")
        )
        self.assertEqual(get_feature_columns(prepared), ["dyn_0", "dyn_1"])
        self.assertNotIn("Hash", prepared.columns)
        with open(
            os.path.join(flat_prepared, "dynamic", "feature_schema.json")
        ) as schema_file:
            schema = json.load(schema_file)
        self.assertEqual(schema["feature_count"], 2)

    def test_05_fused_fallback_class_wise_alignment(self):
        """Verify prepare_fused falls back to class-wise alignment when Sample_IDs do not overlap."""
        static_df = pd.DataFrame({
            "Sample_ID": [f"SAMPLE_STAT_{i:04d}" for i in range(10)],
            "label": ["Adware"] * 5 + ["Trojan"] * 5,
            "stat_feat_1": np.random.randn(10),
            "stat_feat_2": np.random.randn(10),
        })
        dynamic_df = pd.DataFrame({
            "Sample_ID": [f"SAMPLE_DYN_{i:04d}" for i in range(8)],
            "label": ["Adware"] * 4 + ["Trojan"] * 4,
            "dyn_feat_1": np.random.randn(8),
            "dyn_feat_2": np.random.randn(8),
            "reboot_phase": ["before"] * 8
        })

        fused_prep_dir = os.path.join(self.test_dir, "fused_fallback_prep")
        os.makedirs(fused_prep_dir, exist_ok=True)
        preparer = AndMal2020DataPreparer(self.raw_dir, fused_prep_dir, seed=42)

        fused_path = preparer.prepare_fused(
            static_df=static_df,
            dynamic_df=dynamic_df,
            test_ratio=0.2,
            val_ratio=0.1,
            allow_classwise_alignment=True,
        )
        self.assertTrue(os.path.isfile(fused_path))

        fused_df = pd.read_csv(fused_path)
        self.assertEqual(len(fused_df), 8)
        self.assertIn("stat_feat_1", fused_df.columns)
        self.assertIn("dyn_feat_1", fused_df.columns)
        self.assertIn("Sample_ID", fused_df.columns)

    def test_06_partition_fused_with_unmatched_ids(self):
        """Verify partition script loads fused dataset via direct file path or class-wise fallback."""
        fused_prep_dir = os.path.join(self.test_dir, "fused_partition_test")
        os.makedirs(os.path.join(fused_prep_dir, "static"), exist_ok=True)
        os.makedirs(os.path.join(fused_prep_dir, "dynamic"), exist_ok=True)

        s_df = pd.DataFrame({
            "Sample_ID": [f"SAMPLE_STAT_{i:04d}" for i in range(100)],
            "label": ["Benign"] * 50 + ["Adware"] * 50,
            "stat_f1": np.random.randn(100)
        })
        d_df = pd.DataFrame({
            "Sample_ID": [f"SAMPLE_DYN_{i:04d}" for i in range(100)],
            "label": ["Benign"] * 50 + ["Adware"] * 50,
            "dyn_f1": np.random.randn(100)
        })
        s_path = os.path.join(fused_prep_dir, "static", "train.parquet")
        d_path = os.path.join(fused_prep_dir, "dynamic", "train.parquet")
        s_df.to_parquet(s_path, index=False)
        d_df.to_parquet(d_path, index=False)

        # Run partition via main command line args mock logic
        import sys
        old_argv = sys.argv
        sys.argv = [
            "data.partition",
            "--static_dataset", s_path,
            "--dynamic_dataset", d_path,
            "--feature_type", "fused",
            "--n_clients", "20",
            "--output_dir", os.path.join(fused_prep_dir, "partitions")
        ]
        from data.partition import main as partition_main
        partition_main()
        sys.argv = old_argv

        part_dir = os.path.join(fused_prep_dir, "partitions", "fused", "20clients")
        self.assertTrue(os.path.isfile(os.path.join(part_dir, "metadata.json")))


    def test_07_scaler_is_persisted_and_fit_on_train_split(self):
        """Prepared features are transformed with a train-fitted scaler."""
        scaler_path = os.path.join(self.prep_dir, "dynamic", "scaler.joblib")
        metadata_path = os.path.join(self.prep_dir, "dynamic", "scaler_metadata.json")
        self.assertTrue(os.path.isfile(scaler_path))
        self.assertTrue(os.path.isfile(metadata_path))

        train = pd.read_csv(os.path.join(self.prep_dir, "dynamic", "train.csv"))
        test = pd.read_csv(os.path.join(self.prep_dir, "dynamic", "test.csv"))
        feature_cols = get_feature_columns(train)
        self.assertEqual(feature_cols, json.load(open(metadata_path))["feature_columns"])
        train_mean = train[feature_cols].mean().abs().max()
        self.assertLess(float(train_mean), 1e-4)
    def test_08_run_stage1_cmd_and_allow_incomplete_benchmark(self):
        """Verify run_stage1 properly adds or omits --strict_class_coverage."""
        from pathlib import Path
        from unittest.mock import patch
        from run_all import run_stage1

        with patch("subprocess.run") as mock_run:
            # Case 1: allow_incomplete_benchmark=False -> must include --strict_class_coverage once
            run_stage1(
                raw_dir=Path(self.raw_dir),
                prepared_dir=Path(self.prep_dir),
                seed=42,
                dry=False,
                feature_type="dynamic",
                allow_incomplete_benchmark=False,
            )
            self.assertTrue(mock_run.called)
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd.count("--strict_class_coverage"), 1)
            self.assertEqual(cmd.count("--type"), 1)
            self.assertIn("dynamic", cmd)

        with patch("subprocess.run") as mock_run:
            # Case 2: allow_incomplete_benchmark=True -> must NOT include --strict_class_coverage
            run_stage1(
                raw_dir=Path(self.raw_dir),
                prepared_dir=Path(self.prep_dir),
                seed=42,
                dry=False,
                feature_type="dynamic",
                allow_incomplete_benchmark=True,
            )
            self.assertTrue(mock_run.called)
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd.count("--strict_class_coverage"), 0)
            self.assertEqual(cmd.count("--type"), 1)
            self.assertIn("dynamic", cmd)

    def test_09_partition_k50_small_pool_produces_no_empty_clients(self):
        """Verify K=50 partitioning with pool_size=490 and min_samples=30 leaves zero empty client shards."""
        import tempfile
        rows = []
        for lbl in ALL_LABELS:
            n_rows = 978 if lbl == "Riskware" else 490
            for i in range(n_rows):
                rows.append({"f0": float(i % 7), "f1": float(i % 5), "label": lbl})
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ScenarioConfig(
                feature_type="dynamic",
                n_clients=50,
                dirichlet_alpha=0.5,
                seed=123,
                min_samples_per_label_client=30,
                partition_output_dir=tmp_dir,
            )
            result = FCILDataPartitioner(cfg).partition_dataframe(df)
            scenario_dir = result["scenario_dir"]

            expected_active = [30, 35, 40, 45, 50]
            for task_id in range(5):
                active = get_participating_clients(scenario_dir, task_id)
                self.assertEqual(len(active), expected_active[task_id])
                for cid in active:
                    c_df = pd.read_parquet(
                        os.path.join(scenario_dir, f"task_{task_id}", f"client_{cid:02d}.parquet")
                    )
                    self.assertGreater(
                        len(c_df),
                        0,
                        f"Empty shard for task_{task_id} client_{cid:02d}",
                    )

    def test_10_rejects_synthetic_static_without_opt_in(self):
        """Reject synthetic static CSVs under user_supplied provenance even without SYNTHETIC_DATASET.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_copy = os.path.join(tmp_dir, "raw")
            prep_out = os.path.join(tmp_dir, "prep")
            shutil.copytree(self.raw_dir, raw_copy)
            marker = os.path.join(raw_copy, "SYNTHETIC_DATASET.json")
            if os.path.isfile(marker):
                os.remove(marker)

            preparer = AndMal2020DataPreparer(
                raw_root=raw_copy,
                output_dir=prep_out,
                seed=42,
                data_provenance="user_supplied",
            )
            with self.assertRaises(ValueError):
                preparer.prepare_static(test_ratio=0.2, val_ratio=0.1)

    def test_11_rejects_synthetic_dynamic_without_opt_in(self):
        """Reject synthetic dynamic CSVs under user_supplied provenance even without SYNTHETIC_DATASET.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_copy = os.path.join(tmp_dir, "raw")
            prep_out = os.path.join(tmp_dir, "prep")
            shutil.copytree(self.raw_dir, raw_copy)
            marker = os.path.join(raw_copy, "SYNTHETIC_DATASET.json")
            if os.path.isfile(marker):
                os.remove(marker)

            preparer = AndMal2020DataPreparer(
                raw_root=raw_copy,
                output_dir=prep_out,
                seed=42,
                data_provenance="user_supplied",
            )
            with self.assertRaises(ValueError):
                preparer.prepare_dynamic(test_ratio=0.2, val_ratio=0.1)

    def test_12_dynamic_excludes_synthetic_when_real_csvs_present(self):
        """Exclude synthetic dynamic CSVs automatically when real telemetry CSVs are also present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = os.path.join(tmp_dir, "raw")
            dyn_dir = os.path.join(raw_dir, "Dynamic", "AndMal2020-dynamic-BeforeAndAfterReboot")
            prep_out = os.path.join(tmp_dir, "prep")
            os.makedirs(dyn_dir, exist_ok=True)

            syn_df = pd.DataFrame({
                "Sample_ID": [f"ADWARE_{i:05d}" for i in range(20)],
                "dyn_metric_0": np.arange(20, dtype=float),
                "dyn_metric_1": np.arange(20, dtype=float) + 1.0,
            })
            syn_df.to_csv(os.path.join(dyn_dir, "adware_before_reboot_Cat.csv"), index=False)

            real_df = pd.DataFrame({
                "Memory_PssTotal": np.arange(20, dtype=float) + 100.0,
                "Memory_PssClean": np.arange(20, dtype=float) + 10.0,
                "Category": ["Adware"] * 20,
            })
            real_df.to_csv(os.path.join(dyn_dir, "Adware_after_reboot_Cat.csv"), index=False)

            preparer = AndMal2020DataPreparer(
                raw_root=raw_dir,
                output_dir=prep_out,
                seed=42,
                data_provenance="user_supplied",
            )
            with self.assertWarnsRegex(RuntimeWarning, "missing configured classes"):
                preparer.prepare_dynamic(test_ratio=0.2, val_ratio=0.1)

            out_df = pd.read_csv(os.path.join(prep_out, "dynamic", "dynamic_all.csv"))
            self.assertEqual(get_feature_columns(out_df), ["Memory_PssTotal", "Memory_PssClean"])
            self.assertEqual(len(out_df), 20)

    def test_13_detects_synthetic_prepared_dir(self):
        """Detect synthetic prepared directories by provenance and schema/Sample_IDs, while allowing real splits."""
        for mod in ("dynamic", "static", "fused"):
            is_syn, _ = detect_synthetic_prepared_dir(self.prep_dir, mod)
            self.assertTrue(is_syn, f"Expected synthetic detection for {mod}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prep_copy = os.path.join(tmp_dir, "prep_spoofed")
            shutil.copytree(self.prep_dir, prep_copy)
            with open(os.path.join(prep_copy, "data_provenance.json"), "w") as handle:
                json.dump({"source": "user_supplied"}, handle)
            for mod in ("dynamic", "static", "fused"):
                is_syn, _ = detect_synthetic_prepared_dir(prep_copy, mod)
                self.assertTrue(is_syn, f"Expected schema/Sample_ID detection for spoofed {mod}")

            fused_only = os.path.join(tmp_dir, "fused_only")
            os.makedirs(os.path.join(fused_only, "fused"), exist_ok=True)
            with open(os.path.join(fused_only, "fused", "feature_schema.json"), "w") as handle:
                json.dump(
                    {
                        "feature_type": "fused",
                        "feature_count": 441,
                        "static_dim": 300,
                        "dynamic_dim": 141,
                        "feature_columns": ["static_feat_0", "static_feat_1", "static_feat_2"],
                    },
                    handle,
                )
            is_syn_f, _ = detect_synthetic_prepared_dir(fused_only, "fused")
            self.assertTrue(is_syn_f)

            real_prep = os.path.join(tmp_dir, "real_prep")
            os.makedirs(os.path.join(real_prep, "dynamic"), exist_ok=True)
            with open(os.path.join(real_prep, "data_provenance.json"), "w") as handle:
                json.dump({"source": "user_supplied"}, handle)
            with open(os.path.join(real_prep, "dynamic", "feature_schema.json"), "w") as handle:
                json.dump(
                    {
                        "feature_type": "dynamic",
                        "feature_count": 2,
                        "feature_columns": ["Memory_PssTotal", "Memory_PssClean"],
                    },
                    handle,
                )
            pd.DataFrame({
                "Sample_ID": ["SAMPLE_DYN_0000001", "SAMPLE_DYN_0000002"],
                "Memory_PssTotal": [0.1, 0.2],
                "Memory_PssClean": [0.3, 0.4],
                "label": ["Adware", "Trojan"],
            }).to_parquet(os.path.join(real_prep, "dynamic", "test.parquet"), index=False)
            is_syn_real, reasons_real = detect_synthetic_prepared_dir(real_prep, "dynamic")
            self.assertFalse(is_syn_real, f"Unexpected synthetic flag on real data: {reasons_real}")

    def test_14_synthetic_generator_refuses_dataset_dir(self):
        """Refuse generating synthetic files in a Dataset dir or over real CSVs."""
        is_syn_raw, _ = detect_synthetic_raw_dir(self.raw_dir, feature_type="all")
        self.assertTrue(is_syn_raw)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                generate_synthetic_raw_andmal2020(
                    root_dir=os.path.join(tmp_dir, "Dataset"),
                    samples_per_class=20,
                    seed=42,
                )

            real_dir = os.path.join(tmp_dir, "custom_real")
            dyn_dir = os.path.join(real_dir, "Dynamic", "AndMal2020-dynamic-BeforeAndAfterReboot")
            os.makedirs(dyn_dir, exist_ok=True)
            pd.DataFrame({
                "Memory_PssTotal": [100, 200],
                "Memory_PssClean": [10, 20],
            }).to_csv(os.path.join(dyn_dir, "Adware_before_reboot_Cat.csv"), index=False)
            is_syn_real_raw, _ = detect_synthetic_raw_dir(real_dir, feature_type="dynamic")
            self.assertFalse(is_syn_real_raw)
            with self.assertRaises(ValueError):
                generate_synthetic_raw_andmal2020(
                    root_dir=real_dir,
                    samples_per_class=20,
                    seed=42,
                )

    def test_15_run_all_rejects_generate_synthetic_and_synthetic_dirs(self):
        """run_all.py forbids --generate_synthetic and rejects synthetic raw/prepared directories."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res = subprocess.run(
            [sys.executable, "run_all.py", "--generate_synthetic", "--dry_run"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)

        from run_all import ensure_dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_prep = Path(tmp_dir) / "prep_missing"
            tmp_part = Path(tmp_dir) / "part_missing"
            with self.assertRaises(RuntimeError):
                ensure_dataset(
                    Path(self.raw_dir),
                    tmp_prep,
                    tmp_part,
                    "dynamic",
                    [20],
                    generate_synthetic=False,
                    skip=False,
                    dry=True,
                )

            tmp_empty_raw = Path(tmp_dir) / "empty_raw"
            tmp_empty_raw.mkdir(parents=True, exist_ok=True)
            with self.assertRaises(RuntimeError):
                ensure_dataset(
                    tmp_empty_raw,
                    Path(self.prep_dir),
                    tmp_part,
                    "dynamic",
                    [20],
                    generate_synthetic=False,
                    skip=False,
                    dry=True,
                )


if __name__ == "__main__":
    unittest.main()

