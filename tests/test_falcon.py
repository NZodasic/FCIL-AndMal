"""
Unit Tests for Falcon Framework Modules.
Tests:
1. FocalLoss: gradient computation, attenuation of easy samples, tensor alpha support.
2. BalancedBatchSampler: quota distribution, minority class replacement, batch sizes.
3. CNNBiLSTMAttentionBackbone: forward pass, temporal self-attention, B=1 safe batchnorm.
4. KLDriftDetector: sliding-window KL computation, reference update, drift triggering.
5. FalconMethod: multi-task progression, multi-teacher distillation, replay buffer.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from training.losses import FocalLoss
from data.sampler import BalancedBatchSampler, build_weighted_random_sampler
from models.backbones import CNNBiLSTMAttentionBackbone, build_backbone
from models.fcil_model import FCILNet
from config import ModelConfig, ILConfig
from federated.drift.detectors import KLDriftDetector, build_drift_detector
from methods.falcon import FalconMethod
from methods import build_il_method
from federated.il_strategy_adapter import ILMethodStrategyAdapter


class TestFocalLoss:
    def test_focal_loss_forward_and_backward(self):
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
        logits = torch.randn(8, 4, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

        loss = loss_fn(logits, targets)
        assert loss.item() > 0.0
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_focal_loss_attenuates_easy_examples(self):
        gamma = 2.0
        focal_fn = FocalLoss(alpha=1.0, gamma=gamma, reduction="none")

        # Well-classified logits with non-zero probability error
        easy_logits = torch.tensor([[3.0, -1.0, -1.0]], requires_grad=False)
        targets = torch.tensor([0])

        ce_loss = F.cross_entropy(easy_logits, targets)
        focal_loss = focal_fn(easy_logits, targets)

        # Focal loss must be strictly smaller than CE for easy samples
        assert focal_loss.item() < ce_loss.item()
        assert focal_loss.item() > 0.0

    def test_focal_loss_class_weights_tensor(self):
        class_weights = torch.tensor([0.1, 0.9])
        loss_fn = FocalLoss(alpha=class_weights, gamma=1.0, reduction="none")

        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        targets = torch.tensor([0, 1])
        losses = loss_fn(logits, targets)
        assert len(losses) == 2


class TestBalancedBatchSampler:
    def test_quota_and_batch_composition(self):
        # 100 samples total: 90 class 0, 10 class 1 (severe 9:1 imbalance)
        labels = np.array([0] * 90 + [1] * 10)
        batch_size = 10
        sampler = BalancedBatchSampler(labels=labels, batch_size=batch_size, seed=42)

        batches = list(iter(sampler))
        assert len(batches) > 0

        # Each batch should have 5 of class 0 and 5 of class 1
        for b in batches:
            assert len(b) == batch_size
            b_labels = labels[b]
            counts = np.bincount(b_labels, minlength=2)
            assert counts[0] == 5
            assert counts[1] == 5

    def test_replacement_for_extremely_scarce_class(self):
        # Only 2 samples of class 2, with batch size 12 across 3 classes (quota 4 each)
        labels = np.array([0] * 50 + [1] * 50 + [2] * 2)
        batch_size = 12
        sampler = BalancedBatchSampler(labels=labels, batch_size=batch_size, seed=42)

        batch = next(iter(sampler))
        assert len(batch) == batch_size
        b_labels = labels[batch]
        counts = np.bincount(b_labels, minlength=3)
        assert counts[2] == 4  # Drawn with replacement

    def test_weighted_random_sampler_builder(self):
        labels = np.array([0] * 80 + [1] * 20)
        sampler = build_weighted_random_sampler(labels)
        assert len(sampler) == 100


class TestCNNBiLSTMAttentionBackbone:
    def test_forward_pass_and_latent_dimension(self):
        input_dim = 141
        latent_dim = 128
        backbone = CNNBiLSTMAttentionBackbone(
            input_dim=input_dim,
            conv_channels=64,
            kernel_size=3,
            lstm_hidden=64,
            attn_dim=64,
            latent_dim=latent_dim,
        )

        x = torch.randn(4, input_dim)
        out = backbone(x)
        assert out.shape == (4, latent_dim)

    def test_single_sample_batchnorm_safety(self):
        backbone = CNNBiLSTMAttentionBackbone(input_dim=50, latent_dim=64)
        backbone.train()
        x_single = torch.randn(1, 50)
        # Should not raise ValueError on batch size 1
        out = backbone(x_single)
        assert out.shape == (1, 64)

    def test_build_backbone_factory(self):
        bb = build_backbone("cnn_bilstm_attention", input_dim=100, latent_dim=128)
        assert isinstance(bb, CNNBiLSTMAttentionBackbone)

        bb_falcon = build_backbone("falcon", input_dim=100, latent_dim=128)
        assert isinstance(bb_falcon, CNNBiLSTMAttentionBackbone)


class TestKLDriftDetector:
    def test_reference_window_initialization(self):
        detector = KLDriftDetector(window_size=20, threshold=0.2)
        # Feed 20 stable values (low error)
        for _ in range(20):
            d, w = detector.update(0.0)
            assert not d

        assert detector.ref_hist is not None

    def test_drift_trigger_on_distribution_shift(self):
        detector = KLDriftDetector(window_size=30, threshold=0.1)
        # Phase 1: Reference window with all 0s
        for _ in range(30):
            detector.update(0.0)

        # Phase 2: Shift distribution to all 1s
        drift_detected = False
        for _ in range(30):
            d, _ = detector.update(1.0)
            if d:
                drift_detected = True
                break

        assert drift_detected
        assert detector.raw_statistic > 0.1

    def test_build_drift_detector_factory(self):
        det = build_drift_detector("kl_drift", window_size=40)
        assert isinstance(det, KLDriftDetector)


class TestFalconMethodIntegration:
    def test_multi_task_distillation_and_replay(self):
        cfg = ModelConfig(
            backbone_type="falcon",
            input_dim=30,
            latent_dim=64,
            classes_per_task=3,
            num_total_classes=6,
        )
        model = FCILNet(cfg)
        il_method = FalconMethod(
            buffer_size_per_class=5,
            replay_ratio=0.5,
            focal_gamma=2.0,
            beta_old=1.0,
            gamma_new=0.0,  # test without independent new teacher first
        )

        # Task 0 training
        x_t0 = torch.randn(30, 30)
        y_t0 = torch.randint(0, 3, (30,))
        ds_t0 = TensorDataset(x_t0, y_t0)
        loader_t0 = DataLoader(ds_t0, batch_size=10)

        il_method.before_task(0, model, loader_t0)
        loss, loss_dict = il_method.compute_loss(model, x_t0[:10], y_t0[:10], nn.CrossEntropyLoss(), 0)
        assert loss.item() > 0.0
        assert "focal_loss" in loss_dict
        il_method.after_task(0, model, loader_t0)

        assert len(il_method.replay_buffer) > 0

        # Task 1 expansion
        model.expand_classes(3)
        assert model.current_classes == 6

        x_t1 = torch.randn(30, 30)
        y_t1 = torch.randint(3, 6, (30,))
        ds_t1 = TensorDataset(x_t1, y_t1)
        loader_t1 = DataLoader(ds_t1, batch_size=10)

        il_method.before_task(1, model, loader_t1)
        assert il_method.old_teacher is not None
        assert il_method.old_num_classes == 3

        loss_t1, loss_dict_t1 = il_method.compute_loss(model, x_t1[:10], y_t1[:10], nn.CrossEntropyLoss(), 1)
        assert loss_t1.item() > 0.0
        assert "l_old" in loss_dict_t1
        assert loss_dict_t1["l_old"] >= 0.0

    def test_adapter_with_falcon(self):
        cfg = ModelConfig(backbone_type="falcon", input_dim=20, latent_dim=32, classes_per_task=3, num_total_classes=6)
        model = FCILNet(cfg)
        il_cfg = ILConfig(method_name="falcon")
        il_method = build_il_method(il_cfg)

        adapter = ILMethodStrategyAdapter(model=model, il_method=il_method, lr=0.001, device="cpu", classes_per_task=3)
        adapter.before_task(0, 3)

        x = torch.randn(20, 20)
        y = torch.randint(0, 3, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        metrics = adapter.train_task(loader, task_id=0, n_epochs=1)
        assert "loss" in metrics
        assert metrics["loss"] > 0.0
