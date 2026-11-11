"""
Falcon Federated Incremental Learning Method.
Implements:
1. Multi-teacher knowledge distillation (Historical teacher + New-class teacher).
2. Memory replay mechanism with representative instance queue.
3. Focal Loss optimization for addressing severe class imbalance.

Reference:
    Yang et al., "Falcon: Federated Incremental Learning Framework for Intrusion
    Detection in Heterogeneous Consumer-Centric Internet of Things",
    IEEE Transactions on Consumer Electronics, Vol. 72, No. 3, August 2026.
"""

from typing import Dict, List, Optional, Tuple, Any
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet
from training.losses import FocalLoss


class FalconMethod(BaseILMethod):
    """
    Falcon Continual / Incremental Learning Strategy.
    
    Combines:
    1. Multi-Teacher Knowledge Distillation:
       - Teacher 1 (Historical Model M_old): preserves representations of previous classes.
       - Teacher 2 (New-Class Model M_new): guides learning of newly emerging attack semantics.
    2. Memory Replay:
       - Instance queue D_old of representative historical samples replayed at ratio delta.
    3. Focal Loss:
       - Enhances gradients of hard/minority samples to avoid benign-majority collapse.
    """

    def __init__(
        self,
        buffer_size_per_class: int = 20,
        replay_ratio: float = 0.2,
        alpha_cls: float = 1.0,
        beta_old: float = 1.0,
        gamma_new: float = 0.5,
        temperature: float = 2.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 1.0,
        grad_clip: float = 2.0,
        use_herding: bool = True,
        train_new_teacher_epochs: int = 5,
    ):
        super().__init__(name="falcon")
        self.buffer_size_per_class = int(buffer_size_per_class)
        self.replay_ratio = float(replay_ratio)
        self.alpha_cls = float(alpha_cls)
        self.beta_old = float(beta_old)
        self.gamma_new = float(gamma_new)
        self.temperature = float(temperature)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.grad_clip = float(grad_clip)
        self.use_herding = bool(use_herding)
        self.train_new_teacher_epochs = int(train_new_teacher_epochs)

        # 1. Memory replay queue: class_id -> (features_tensor, labels_tensor)
        self.replay_buffer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # 2. Historical Teacher Model snapshot
        self.old_teacher: Optional[FCILNet] = None
        self.old_num_classes: int = 0
        self.prev_seen_classes: int = 0

        # 3. New-Class Teacher Model
        self.new_teacher: Optional[nn.Module] = None
        self.new_num_classes: int = 0

        # 4. Focal Loss criterion
        self.focal_criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean")

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Lifecycle hook before starting training on task_id."""
        if task_id > 0:
            # Snapshot previous model as Historical Teacher
            self.old_teacher = model.clone_model().to(device)
            self.old_teacher.eval()
            for p in self.old_teacher.parameters():
                p.requires_grad = False

            if self.prev_seen_classes > 0:
                self.old_num_classes = self.prev_seen_classes
            else:
                cpt = getattr(model.config, "classes_per_task", 3) if hasattr(model, "config") and model.config else 3
                self.old_num_classes = min(model.current_classes - cpt, task_id * cpt) if model.current_classes > cpt else cpt

            # Train or initialize New-Class Teacher if data is available
            if train_loader is not None and len(train_loader) > 0 and self.gamma_new > 0:
                self._train_new_class_teacher(model, train_loader, device)
        else:
            self.old_teacher = None
            self.old_num_classes = 0
            self.new_teacher = None
            self.new_num_classes = 0

    def _train_new_class_teacher(
        self,
        base_model: FCILNet,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Train an independent new-class teacher model on incoming samples (Section III-D).
        """
        # Create an independent teacher copy
        self.new_teacher = copy.deepcopy(base_model).to(device)
        self.new_teacher.train()

        opt = torch.optim.AdamW(
            [p for p in self.new_teacher.parameters() if p.requires_grad],
            lr=0.002,
            weight_decay=1e-2,
        )

        for _ in range(self.train_new_teacher_epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits = self.new_teacher(bx, limit_to_current=True)
                loss = self.focal_criterion(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.new_teacher.parameters(), self.grad_clip)
                opt.step()

        self.new_teacher.eval()
        for p in self.new_teacher.parameters():
            p.requires_grad = False

    def _get_replay_batch(
        self,
        target_size: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a batch of historical exemplars from memory replay queue."""
        if not self.replay_buffer or target_size <= 0:
            return None

        samples_x, samples_y = [], []
        classes = list(self.replay_buffer.keys())
        quota_per_class = max(1, target_size // len(classes))

        for cls_id, (cls_x, cls_y) in self.replay_buffer.items():
            n_avail = len(cls_x)
            if n_avail == 0:
                continue
            idx = np.random.choice(n_avail, size=min(n_avail, quota_per_class), replace=False)
            samples_x.append(cls_x[idx])
            samples_y.append(cls_y[idx])

        if not samples_x:
            return None

        rx = torch.cat(samples_x, dim=0).to(device)
        ry = torch.cat(samples_y, dim=0).to(device)
        return rx, ry

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Falcon multi-teacher loss combining:
        L = alpha * L_Focal(z_s, y) + beta * L_old(z_old, z_s) + gamma * L_new(z_new, z_s).
        """
        batch_size = x.size(0)

        # 1. Memory Replay Injection
        if task_id > 0 and self.replay_buffer:
            target_replay_size = int(batch_size * self.replay_ratio)
            replay_data = self._get_replay_batch(target_replay_size, device)
            if replay_data is not None:
                rx, ry = replay_data
                combined_x = torch.cat([x, rx], dim=0)
                combined_y = torch.cat([y, ry], dim=0)
            else:
                combined_x, combined_y = x, y
        else:
            combined_x, combined_y = x, y

        # 2. Student model forward pass
        z_s = model(combined_x, limit_to_current=True)

        # 3. Supervised Classification Loss (Focal Loss)
        l_cls = self.focal_criterion(z_s, combined_y)

        # 4. Historical Teacher Distillation (Eq. 22)
        l_old = torch.tensor(0.0, device=device)
        if task_id > 0 and self.old_teacher is not None and self.old_num_classes > 0:
            with torch.no_grad():
                z_old = self.old_teacher(combined_x, limit_to_current=True)

            # Softmax distillation on historical classes slice
            p_old = F.softmax(z_old[:, :self.old_num_classes] / self.temperature, dim=1)
            log_q_s_old = F.log_softmax(z_s[:, :self.old_num_classes] / self.temperature, dim=1)

            # KL divergence: KL(p_old || q_s_old) = sum(p_old * (log p_old - log q_s_old))
            l_old = (self.temperature ** 2) * F.kl_div(log_q_s_old, p_old, reduction="batchmean").clamp(min=0.0)

        # 5. New-Class Teacher Distillation (Eq. 23)
        l_new = torch.tensor(0.0, device=device)
        if task_id > 0 and self.new_teacher is not None and self.old_num_classes < z_s.size(1):
            with torch.no_grad():
                z_new_full = self.new_teacher(combined_x, limit_to_current=True)

            # Slice for new classes
            p_new = F.softmax(z_new_full[:, self.old_num_classes:] / self.temperature, dim=1)
            log_q_s_new = F.log_softmax(z_s[:, self.old_num_classes:] / self.temperature, dim=1)

            l_new = (self.temperature ** 2) * F.kl_div(log_q_s_new, p_new, reduction="batchmean").clamp(min=0.0)

        # 6. Total Combined Loss (Eq. 24)
        total_loss = self.alpha_cls * l_cls + self.beta_old * l_old + self.gamma_new * l_new

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "focal_loss": float(l_cls.item()),
            "l_old": float(l_old.item()),
            "l_new": float(l_new.item()),
        }

        return total_loss, loss_dict

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Store representative exemplars for each class in task_id into the memory queue.
        Uses latent feature herding (if use_herding=True) or stratified selection.
        """
        if train_loader is None or len(train_loader) == 0:
            return

        model.eval()
        all_x, all_y, all_feat = [], [], []

        with torch.no_grad():
            for bx, by in train_loader:
                bx_dev = bx.to(device)
                feat = model.get_features(bx_dev)
                all_x.append(bx.cpu())
                all_y.append(by.cpu())
                all_feat.append(feat.cpu())

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_feat = torch.cat(all_feat, dim=0)

        unique_classes = torch.unique(all_y).tolist()

        for cls_id in unique_classes:
            mask = (all_y == cls_id)
            cls_samples = all_x[mask]
            cls_labels = all_y[mask]
            cls_feats = all_feat[mask]

            n_samples = len(cls_samples)
            if n_samples == 0:
                continue

            budget = min(self.buffer_size_per_class, n_samples)

            if self.use_herding and budget < n_samples:
                # Feature mean
                class_mean = cls_feats.mean(dim=0, keepdim=True)
                selected_indices = []
                current_sum = torch.zeros_like(class_mean)

                # Iterative herding selection
                for k in range(budget):
                    candidates = (current_sum + cls_feats) / (k + 1)
                    dist = torch.norm(class_mean - candidates, dim=1)
                    # Mask already selected
                    for s_idx in selected_indices:
                        dist[s_idx] = float("inf")
                    best_idx = torch.argmin(dist).item()
                    selected_indices.append(best_idx)
                    current_sum += cls_feats[best_idx : best_idx + 1]

                self.replay_buffer[int(cls_id)] = (
                    cls_samples[selected_indices].clone(),
                    cls_labels[selected_indices].clone(),
                )
            else:
                perm = torch.randperm(n_samples)[:budget]
                self.replay_buffer[int(cls_id)] = (
                    cls_samples[perm].clone(),
                    cls_labels[perm].clone(),
                )

        self.prev_seen_classes = model.current_classes

