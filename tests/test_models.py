"""
Unit Tests for Neural Network Backbones and Dynamic Incremental Classifiers.
"""

import unittest
import torch
import torch.nn as nn

from config import ModelConfig
from models.backbones import (
    MLPBackbone,
    CNN1DBackbone,
    TCNBackbone,
    HybridTCNCNNBackbone,
    FusedMultiModalBackbone
)
from models.classifier import DynamicIncrementalClassifier
from models.fcil_model import FCILNet
from models.model_utils import count_parameters, get_model_summary


class TestModels(unittest.TestCase):
    """Test backbone representations and dynamic class expansion."""

    def test_mlp_backbone_forward(self):
        bb = MLPBackbone(input_dim=141, hidden_dims=[64, 32], latent_dim=16)
        x = torch.randn(8, 141)
        out = bb(x)
        self.assertEqual(out.shape, (8, 16))

    def test_cnn1d_backbone_forward(self):
        bb = CNN1DBackbone(input_dim=141, channels=[16, 32], latent_dim=16)
        x = torch.randn(8, 141)
        out = bb(x)
        self.assertEqual(out.shape, (8, 16))

    def test_tcn_backbone_forward(self):
        bb = TCNBackbone(input_dim=141, num_channels=[16, 32], latent_dim=16)
        x = torch.randn(8, 141)
        out = bb(x)
        self.assertEqual(out.shape, (8, 16))

    def test_fused_backbone_forward(self):
        bb = FusedMultiModalBackbone(static_dim=50, dynamic_dim=20, latent_dim=16)
        x = torch.randn(8, 70)  # 50 static + 20 dynamic
        out = bb(x)
        self.assertEqual(out.shape, (8, 16))

    def test_hybrid_tcn_cnn_backbone_forward(self):
        bb = HybridTCNCNNBackbone(input_dim=141, cnn_channels=[32, 64], tcn_channels=[64, 128], latent_dim=16)
        x = torch.randn(8, 141)
        out = bb(x)
        self.assertEqual(out.shape, (8, 16))

    def test_dynamic_incremental_classifier_expansion(self):
        clf = DynamicIncrementalClassifier(in_features=16, initial_classes=3, max_classes=15)
        self.assertEqual(clf.current_classes, 3)

        x = torch.randn(4, 16)
        logits = clf(x)
        self.assertEqual(logits.shape, (4, 3))

        # Expand to 6 classes (Task 1)
        clf.expand_classes(3)
        self.assertEqual(clf.current_classes, 6)
        logits = clf(x)
        self.assertEqual(logits.shape, (4, 6))

        # Expand to 15 classes (Task 4)
        clf.expand_classes(9)
        self.assertEqual(clf.current_classes, 15)
        logits = clf(x)
        self.assertEqual(logits.shape, (4, 15))

    def test_fcil_net_end_to_end(self):
        cfg = ModelConfig(backbone_type="cnn1d", input_dim=50, latent_dim=16, classes_per_task=3, num_total_classes=15)
        model = FCILNet(cfg)
        self.assertEqual(model.current_classes, 3)

        x = torch.randn(4, 50)
        logits, feat = model(x, return_features=True)
        self.assertEqual(logits.shape, (4, 3))
        self.assertEqual(feat.shape, (4, 16))

        # Parameter summary check
        tot, trainable = count_parameters(model)
        self.assertGreater(tot, 0)
        self.assertEqual(tot, trainable)


if __name__ == "__main__":
    unittest.main()
