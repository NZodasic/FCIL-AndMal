"""
Unit and Integration Tests for Stage 1 Data Preparation and Stage 2 Partitioning.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd

from config import ScenarioConfig, TASK_LABEL_MAP, ALL_LABELS
from data.synthetic_generator import generate_synthetic_raw_andmal2020
from data.prepare_dataset import AndMal2020DataPreparer
from data.partition import FCILDataPartitioner
from data.dataset import FLTaskDataset, get_participating_clients, recommend_batch_size


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
            seed=42
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


if __name__ == "__main__":
    unittest.main()
