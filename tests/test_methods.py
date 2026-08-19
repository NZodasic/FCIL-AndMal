"""
Unit Tests for Class-Incremental Learning Methods.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from models.fcil_model import FCILNet
from data.dataset import TabularMalwareDataset
from methods import (
    FineTuneMethod,
    JointCumulativeMethod,
    EWCMethod,
    LwFMethod,
    ReplayHerdingMethod,
    SPCILMethod,
    MALFSILMethod,
)


class TestILMethods(unittest.TestCase):
    """Test continual learning methods loss computation and state transitions."""

    def setUp(self):
        self.cfg = ModelConfig(backbone_type="mlp", input_dim=20, latent_dim=16, classes_per_task=3, num_total_classes=15)
        self.model = FCILNet(self.cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.x = torch.randn(8, 20)
        self.y = torch.randint(0, 3, (8,))
        self.loader = DataLoader(TabularMalwareDataset(self.x.numpy(), self.y.numpy()), batch_size=4)

    def test_finetune_loss(self):
        method = FineTuneMethod()
        loss, d = method.compute_loss(self.model, self.x, self.y, self.criterion, task_id=0)
        self.assertGreater(loss.item(), 0.0)

    def test_ewc_pipeline(self):
        method = EWCMethod(ewc_lambda=100.0)
        loss0, _ = method.compute_loss(self.model, self.x, self.y, self.criterion, task_id=0)
        method.after_task(task_id=0, model=self.model, train_loader=self.loader)
        self.assertTrue(len(method.fisher_dict) > 0)

        # Task 1
        self.model.expand_classes(3)
        y1 = torch.randint(0, 6, (8,))
        loss1, d1 = method.compute_loss(self.model, self.x, y1, self.criterion, task_id=1)
        self.assertIn("ewc_loss", d1)

    def test_lwf_pipeline(self):
        method = LwFMethod(temperature=2.0, alpha=1.0)
        method.before_task(task_id=1, model=self.model)
        self.model.expand_classes(3)
        y1 = torch.randint(0, 6, (8,))
        loss1, d1 = method.compute_loss(self.model, self.x, y1, self.criterion, task_id=1)
        self.assertIn("distill_loss", d1)

    def test_replay_herding_pipeline(self):
        method = ReplayHerdingMethod(buffer_size_per_class=2, use_herding=True)
        method.after_task(task_id=0, model=self.model, train_loader=self.loader)
        self.assertTrue(len(method.buffer) > 0)

        self.model.expand_classes(3)
        y1 = torch.randint(0, 6, (8,))
        loss1, d1 = method.compute_loss(self.model, self.x, y1, self.criterion, task_id=1)
        self.assertGreater(loss1.item(), 0.0)

    def test_spcil_pipeline(self):
        method = SPCILMethod(lambda_init=0.5, lambda_step=0.1)
        loss, d = method.compute_loss(self.model, self.x, self.y, self.criterion, task_id=0)
        self.assertIn("pacing_lambda", d)

    def test_malfsil_full_pipeline(self):
        method = MALFSILMethod(buffer_size_per_class=2, distill_weight=1.0, proto_weight=0.5)
        # Compute prototypes
        local_protos = method.compute_local_prototypes(self.model, self.loader)
        self.assertTrue(len(local_protos) > 0)

        # Set global prototypes
        method.set_global_prototypes({k: v[0] for k, v in local_protos.items()})
        method.after_task(task_id=0, model=self.model, train_loader=self.loader)

        self.model.expand_classes(3)
        method.before_task(task_id=1, model=self.model)
        y1 = torch.randint(0, 6, (8,))
        loss1, d1 = method.compute_loss(self.model, self.x, y1, self.criterion, task_id=1)
        self.assertIn("distill_loss", d1)
        self.assertIn("proto_loss", d1)


if __name__ == "__main__":
    unittest.main()
