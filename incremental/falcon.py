"""
Falcon Incremental Learning Strategy.
Implements IncrementalStrategy interface for Falcon framework.

Reference:
    Yang et al., "Falcon: Federated Incremental Learning Framework for Intrusion
    Detection in Heterogeneous Consumer-Centric Internet of Things",
    IEEE Transactions on Consumer Electronics, Vol. 72, No. 3, August 2026.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from incremental.base_strategy import IncrementalStrategy
from training.losses import FocalLoss
from data.sampler import compute_class_weights


class Falcon(IncrementalStrategy):
    """
    Falcon Incremental Strategy.
    Integrates Focal Loss, Multi-Teacher Knowledge Distillation, and Memory Replay.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        device: str = "cpu",
        buffer_size_per_class: int = 20,
        replay_ratio: float = 0.2,
        alpha_cls: float = 1.0,
        beta_old: float = 1.0,
        gamma_new: float = 0.5,
        temperature: float = 2.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 1.0,
        grad_clip: float = 2.0,
        **kwargs,
    ):
        super().__init__(model, optimizer_fn, device, **kwargs)
        self.buffer_size_per_class = int(buffer_size_per_class)
        self.replay_ratio = float(replay_ratio)
        self.alpha_cls = float(alpha_cls)
        self.beta_old = float(beta_old)
        self.gamma_new = float(gamma_new)
        self.temperature = float(temperature)
        self.grad_clip = float(grad_clip)

        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        self.replay_buffer: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)

        self.old_model: Optional[nn.Module] = None
        self.new_teacher: Optional[nn.Module] = None
        self.old_num_classes: int = 0

    def _train_new_class_teacher(self, train_loader: DataLoader) -> None:
        """Train specialized New-Class Teacher M_new on current task data D_new (Paper 2 Sec. III-D2)."""
        self.new_teacher = copy.deepcopy(self.model).to(self.device)
        self.new_teacher.train()
        for p in self.new_teacher.parameters():
            p.requires_grad = True

        teacher_opt = torch.optim.Adam(self.new_teacher.parameters(), lr=1e-3)
        max_batches = 15
        batches_seen = 0
        for data, target in train_loader:
            bx = data.to(self.device)
            by = target.to(self.device)
            new_mask = by >= self.old_num_classes
            if not torch.any(new_mask):
                continue
            bx_new = bx[new_mask]
            by_new = by[new_mask] - self.old_num_classes

            teacher_opt.zero_grad()
            logits = self.new_teacher(bx_new)
            new_logits = logits[:, self.old_num_classes:self.n_classes_so_far]
            if new_logits.size(1) == 0:
                break
            loss = F.cross_entropy(new_logits, by_new.clamp(0, new_logits.size(1) - 1))
            loss.backward()
            teacher_opt.step()

            batches_seen += 1
            if batches_seen >= max_batches:
                break

        self.new_teacher.eval()
        for p in self.new_teacher.parameters():
            p.requires_grad = False

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        self.current_task = task_id
        if task_id > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad = False
            self.old_num_classes = self.n_classes_so_far

        self.n_classes_so_far += n_new_classes

    def _sample_replay_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.replay_buffer or batch_size <= 0:
            return None

        samples_x, samples_y = [], []
        classes = list(self.replay_buffer.keys())
        quota = max(1, batch_size // len(classes))

        for cls_id, items in self.replay_buffer.items():
            if not items:
                continue
            idx = np.random.choice(len(items), size=min(len(items), quota), replace=False)
            for i in idx:
                samples_x.append(items[i][0])
                samples_y.append(items[i][1])

        if not samples_x:
            return None

        rx = torch.stack(samples_x).to(self.device)
        ry = torch.tensor(samples_y, dtype=torch.long, device=self.device)
        return rx, ry

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs,
    ) -> Dict[str, float]:
        self.model.to(self.device)

        # Calibrate inverse class-frequency weights (Eq. 1) for Focal Loss (Eq. 5)
        all_y = []
        for _, target in train_loader:
            all_y.append(target.detach().cpu())
        if all_y:
            y_cat = torch.cat(all_y, dim=0)
            c_weights = compute_class_weights(y_cat, num_classes=self.n_classes_so_far).to(self.device)
            self.focal_criterion = FocalLoss(alpha=c_weights, gamma=self.focal_gamma, reduction="mean")

        if task_id > 0 and self.gamma_new > 0:
            self._train_new_class_teacher(train_loader)
        else:
            self.new_teacher = None

        self.model.train()
        optimizer = self.optimizer_fn(self.model.parameters())

        total_loss = 0.0
        n_batches = 0

        for epoch in range(n_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Mix replay: D_train = D_new U replay(D_old, delta)
                if task_id > 0 and self.replay_buffer:
                    replay_size = int(data.size(0) * self.replay_ratio)
                    replay_data = self._sample_replay_batch(replay_size)
                    if replay_data is not None:
                        rx, ry = replay_data
                        comb_x = torch.cat([data, rx], dim=0)
                        comb_y = torch.cat([target, ry], dim=0)
                    else:
                        comb_x, comb_y = data, target
                else:
                    comb_x, comb_y = data, target

                optimizer.zero_grad()
                logits = self.model(comb_x)
                if logits.size(1) > self.n_classes_so_far:
                    logits = logits[:, :self.n_classes_so_far]

                l_cls = self.focal_criterion(logits, comb_y)

                # Historical Teacher Distillation L_old (Eq. 22)
                l_old = torch.tensor(0.0, device=self.device)
                if task_id > 0 and self.old_model is not None and self.old_num_classes > 0:
                    with torch.no_grad():
                        old_logits = self.old_model(comb_x)
                    p_old = F.softmax(old_logits[:, :self.old_num_classes] / self.temperature, dim=1)
                    log_q_s_old = F.log_softmax(logits[:, :self.old_num_classes] / self.temperature, dim=1)
                    l_old = (self.temperature ** 2) * F.kl_div(log_q_s_old, p_old, reduction="batchmean").clamp(min=0.0)

                # New-Class Teacher Distillation L_new (Eq. 23)
                l_new = torch.tensor(0.0, device=self.device)
                if (
                    task_id > 0
                    and self.new_teacher is not None
                    and self.gamma_new > 0
                    and logits.size(1) > self.old_num_classes
                ):
                    with torch.no_grad():
                        new_t_logits = self.new_teacher(comb_x)
                    t_slice = new_t_logits[:, self.old_num_classes:logits.size(1)]
                    s_slice = logits[:, self.old_num_classes:]
                    if t_slice.size(1) > 0 and t_slice.size(1) == s_slice.size(1):
                        p_new = F.softmax(t_slice / self.temperature, dim=1)
                        log_q_s_new = F.log_softmax(s_slice / self.temperature, dim=1)
                        l_new = (self.temperature ** 2) * F.kl_div(log_q_s_new, p_new, reduction="batchmean").clamp(min=0.0)

                # Total Dual-Teacher Distillation Loss (Eq. 24)
                loss = self.alpha_cls * l_cls + self.beta_old * l_old + self.gamma_new * l_new
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        return {"loss": avg_loss}

    def after_task(self, task_id: int, **kwargs) -> None:
        train_loader = kwargs.get("train_loader", None)
        if train_loader is None:
            return

        self.model.eval()
        class_samples: Dict[int, List[torch.Tensor]] = defaultdict(list)

        with torch.no_grad():
            for data, target in train_loader:
                for d, t in zip(data, target):
                    lbl = int(t.item())
                    if len(class_samples[lbl]) < self.buffer_size_per_class * 2:
                        class_samples[lbl].append(d.cpu())

        for lbl, samples in class_samples.items():
            selected = samples[:self.buffer_size_per_class]
            self.replay_buffer[lbl] = [(s, torch.tensor(lbl)) for s in selected]
