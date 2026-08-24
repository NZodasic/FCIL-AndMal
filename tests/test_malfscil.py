"""Tests for the paper-aligned MalFSCIL implementation."""

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ExperimentConfig, FLConfig, ILConfig, ModelConfig, ScenarioConfig
from data.dataset import TabularMalwareDataset
from data.few_shot import build_few_shot_session
from methods import build_il_method
from methods.malfscil import MalFSCILMethod
from models.fcil_model import FCILNet
from models.malfscil import PrototypeGraphAttention
from utils.metrics import compute_fscil_session_metrics


class TestFewShotSession(unittest.TestCase):
    def test_selects_exact_k_support_examples_per_class(self):
        X = np.arange(72, dtype=np.float32).reshape(12, 6)
        y = np.repeat(np.array([3, 4, 5]), 4)
        session = build_few_shot_session(
            X,
            y,
            k_shot=2,
            query_per_class=3,
            mask_probability=0.25,
            expected_classes=[3, 4, 5],
            seed=7,
        )

        self.assertEqual(len(session.support_y), 6)
        self.assertEqual(len(session.query_y), 9)
        for class_id in (3, 4, 5):
            self.assertEqual(np.sum(session.support_y == class_id), 2)
            self.assertEqual(np.sum(session.query_y == class_id), 3)
            support_rows = session.support_X[session.support_y == class_id]
            for query_row in session.query_X[session.query_y == class_id]:
                self.assertTrue(
                    any(
                        np.all((query_row == support_row) | (query_row == 0.0))
                        for support_row in support_rows
                    )
                )

    def test_rejects_a_class_with_too_few_examples(self):
        with self.assertRaisesRegex(ValueError, "5 are required"):
            build_few_shot_session(
                np.ones((6, 4), dtype=np.float32),
                np.array([0, 0, 0, 1, 1, 1]),
                k_shot=5,
                expected_classes=[0, 1],
            )

    def test_paper_session_metrics(self):
        summary = compute_fscil_session_metrics(
            [{"accuracy": 0.9}, {"accuracy": 0.8}, {"accuracy": 0.7}]
        )
        self.assertAlmostEqual(summary["performance_degradation"], 0.2)
        self.assertAlmostEqual(summary["average_accuracy"], 0.8)
        self.assertAlmostEqual(summary["area_under_time"], 0.8)


class TestMalFSCILMethod(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = FCILNet(
            ModelConfig(
                backbone_type="mlp",
                input_dim=8,
                latent_dim=6,
                classes_per_task=3,
                num_total_classes=6,
                dropout_rate=0.0,
            )
        )
        self.method = MalFSCILMethod(
            graph_attention_dim=4,
            arc_scale=10.0,
            arc_margin=0.2,
        )
        self.criterion = nn.CrossEntropyLoss()

    def test_base_session_trains_vae_and_feature_extractor(self):
        X = torch.randn(12, 8)
        y = torch.tensor([0, 1, 2] * 4)
        loader = DataLoader(
            TabularMalwareDataset(X.numpy(), y.numpy()), batch_size=6
        )
        self.method.before_task(0, self.model, loader)

        loss, metrics = self.method.compute_loss(
            self.model, X, y, self.criterion, task_id=0
        )
        loss.backward()

        self.assertIn("reconstruction_loss", metrics)
        self.assertIn("kl_loss", metrics)
        self.assertTrue(
            any(parameter.grad is not None for parameter in self.model.backbone.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in self.method.vae.parameters())
        )

    def test_incremental_session_freezes_features_and_evolves_prototypes(self):
        base_X = torch.randn(12, 8)
        base_y = torch.tensor([0, 1, 2] * 4)
        base_loader = DataLoader(
            TabularMalwareDataset(base_X.numpy(), base_y.numpy()), batch_size=6
        )
        self.method.before_task(0, self.model, base_loader)
        self.method.after_task(0, self.model, base_loader)

        self.model.expand_classes(3)
        support_X = torch.randn(15, 8)
        support_y = torch.tensor([3, 4, 5] * 5)
        support_loader = DataLoader(
            TabularMalwareDataset(support_X.numpy(), support_y.numpy()), batch_size=5
        )
        self.method.before_task(1, self.model, support_loader)
        self.model.zero_grad(set_to_none=True)

        loss, metrics = self.method.compute_loss(
            self.model,
            support_X,
            support_y,
            self.criterion,
            task_id=1,
        )
        loss.backward()

        self.assertIn("arc_loss", metrics)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.model.backbone.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in self.model.backbone.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.method.prototype_graph.parameters()
            )
        )
        self.method.after_task(1, self.model, support_loader)
        self.assertEqual(self.model(torch.randn(2, 8)).shape, (2, 6))

    def test_graph_attention_preserves_prototype_shape(self):
        graph = PrototypeGraphAttention(feature_dim=6, attention_dim=4)
        evolved, attention = graph(torch.randn(5, 6))
        self.assertEqual(evolved.shape, (5, 6))
        self.assertEqual(attention.shape, (5, 5))
        self.assertTrue(torch.allclose(attention.sum(dim=1), torch.ones(5)))

    def test_legacy_method_name_resolves_to_malfscil(self):
        config = ILConfig(method_name="malfsil")
        self.assertEqual(config.method_name, "malfscil")
        self.assertIsInstance(build_il_method(config), MalFSCILMethod)

    def test_experiment_config_serializes_fscil_protocol(self):
        config = ExperimentConfig(
            scenario=ScenarioConfig(),
            model=ModelConfig(),
            fl=FLConfig(),
            il=ILConfig(method_name="malfscil", fscil_k_shot=7),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            config.save_json(str(path))
            with path.open() as config_file:
                saved = json.load(config_file)
        self.assertEqual(saved["il"]["method_name"], "malfscil")
        self.assertEqual(saved["il"]["fscil_k_shot"], 7)


if __name__ == "__main__":
    unittest.main()
