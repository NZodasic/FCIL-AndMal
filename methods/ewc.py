"""
Elastic Weight Consolidation (EWC) and Drift-Aware Online EWC (FL-CL-IDS / EWCDR) for FCIL.
Regularizes weight changes based on the diagonal of the Online Empirical Fisher Information Matrix
with decay factor gamma=0.9, per-client adaptive lambda_i^{(t)}, anchor distillation, and
post-task classifier Weight Alignment (WA) to prevent recency norm bias.

References:
    - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017.
    - Khreasat & Villarrubia Gonzalez, "Drift-Aware Federated Continual Learning for
      Privacy-Preserving Android Malware Intrusion Detection", IEEE Access, 2026.
"""

from typing import Dict, Any, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class EWCMethod(BaseILMethod):
    """
    Online Elastic Weight Consolidation with Drift-Adaptive Lambda (FL-CL-IDS / EWCDR).
    """

    def __init__(
        self,
        ewc_lambda: float = 500.0,
        gamma: float = 0.90,
        lambda_min: float = 10.0,
        lambda_max: float = 1000.0,
        classes_per_task: int = 3,
        temperature: float = 2.0,
    ):
        super().__init__(name="ewc")
        self.ewc_lambda = float(ewc_lambda)
        self.current_lambda = float(ewc_lambda)
        self.gamma = float(gamma)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.classes_per_task = int(classes_per_task)
        self.temperature = float(temperature)

        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self._device_fisher: Dict[str, torch.Tensor] = {}
        self._device_optimal: Dict[str, torch.Tensor] = {}
        self._cached_device: Optional[torch.device] = None

        # Frozen teacher constructed from optimal_params (zero additional parameter storage)
        self.anchor_model: Optional[FCILNet] = None
        self.prev_num_classes: int = 0
        # Exemplar-free per-class latent Gaussian statistics: class_id -> (mean_128d, std_128d)
        self.latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.bn_buffers: Dict[str, torch.Tensor] = {}

    def set_adaptive_lambda(self, adaptive_lambda: float) -> None:
        """Update per-client adaptive EWC regularization multiplier lambda_i^{(t)} (Eq. 5)."""
        self.current_lambda = float(adaptive_lambda)

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """ router/teacher setup before starting task_id."""
        self.prev_num_classes = task_id * self.classes_per_task
        if task_id > 0:
            self.anchor_model = model.clone_model().to(device)
            if self.optimal_params:
                state = self.anchor_model.state_dict()
                for k, v in self.optimal_params.items():
                    if k in state:
                        overlap = tuple(
                            slice(0, min(s_cur, s_opt))
                            for s_cur, s_opt in zip(state[k].shape, v.shape)
                        )
                        state[k][overlap].copy_(v[overlap].to(state[k].device))
                for k, v in self.bn_buffers.items():
                    if k in state and state[k].shape == v.shape:
                        state[k].copy_(v.to(state[k].device))
                self.anchor_model.load_state_dict(state)
            self.anchor_model.eval()
            for p in self.anchor_model.parameters():
                p.requires_grad = False
        else:
            self.anchor_model = None
            self.latent_stats.clear()
            self.bn_buffers.clear()

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if task_id > 0 and hasattr(model, "backbone"):
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

        logits, features = model(x, return_features=True, limit_to_current=True)
        prev_cls = task_id * self.classes_per_task

        if task_id > 0 and prev_cls > 0 and logits.size(1) > prev_cls:
            # Local task cross-entropy on the new task classes (avoids crushing old class rows)
            new_logits = logits[:, prev_cls:]
            local_targets = torch.clamp(y - prev_cls, min=0, max=new_logits.size(1) - 1)
            local_ce = F.cross_entropy(new_logits, local_targets)

            # Calibrated joint cross-entropy with stop-gradient on historical logits
            calibrated_logits = torch.cat(
                [logits[:, :prev_cls].detach(), new_logits], dim=1
            )
            joint_ce = criterion(calibrated_logits, y)
            ce_loss = 0.70 * local_ce + 0.30 * joint_ce

            # Exemplar-free Gaussian latent prototype calibration for classifier head
            if self.latent_stats and hasattr(model, "classifier"):
                syn_feats_list = []
                syn_labels_list = []
                n_per_old_cls = max(2, min(8, x.size(0) // max(1, prev_cls)))
                for c_old in range(prev_cls):
                    if c_old in self.latent_stats:
                        mu_c, std_c = self.latent_stats[c_old]
                        mu_c = mu_c.to(device)
                        std_c = std_c.to(device)
                        eps = torch.randn(n_per_old_cls, mu_c.numel(), device=device)
                        syn_f = mu_c.unsqueeze(0) + 0.20 * eps * std_c.unsqueeze(0)
                        syn_feats_list.append(syn_f)
                        syn_labels_list.append(
                            torch.full((n_per_old_cls,), c_old, dtype=torch.long, device=device)
                        )
                if syn_feats_list:
                    syn_feats = torch.cat(syn_feats_list, dim=0)
                    syn_labels = torch.cat(syn_labels_list, dim=0)
                    cal_feats = torch.cat([features.detach(), syn_feats], dim=0)
                    cal_labels = torch.cat([y, syn_labels], dim=0)
                    cal_logits = model.classifier(cal_feats)[:, :logits.size(1)]
                    ce_loss = ce_loss + 0.55 * F.cross_entropy(cal_logits, cal_labels)
        else:
            ce_loss = criterion(logits, y)

        # Online Fisher EWC Penalty (Eqs. 4 & 14)
        ewc_loss = torch.tensor(0.0, device=device)
        if task_id > 0 and self.fisher_dict and self.optimal_params:
            if self._cached_device != device or not self._device_fisher:
                self._device_fisher = {k: v.to(device) for k, v in self.fisher_dict.items()}
                self._device_optimal = {k: v.to(device) for k, v in self.optimal_params.items()}
                self._cached_device = device

            for name, param in model.named_parameters():
                if name in self._device_fisher and name in self._device_optimal:
                    f = self._device_fisher[name]
                    p_star = self._device_optimal[name]
                    overlap = tuple(
                        slice(0, min(p_dim, star_dim, fisher_dim))
                        for p_dim, star_dim, fisher_dim in zip(
                            param.shape, p_star.shape, f.shape
                        )
                    )
                    if overlap and all(s.stop > 0 for s in overlap):
                        diff = param[overlap] - p_star[overlap]
                        ewc_loss = ewc_loss + (f[overlap] * diff.pow(2)).sum()

            ewc_loss = (self.current_lambda / 2.0) * ewc_loss

        # Anchor Feature & Historical Logit Consolidation (using theta^{*(t-1)})
        anchor_loss = torch.tensor(0.0, device=device)
        if task_id > 0 and self.anchor_model is not None and prev_cls > 0:
            self.anchor_model.to(device)
            with torch.no_grad():
                anchor_logits, anchor_features = self.anchor_model(
                    x, return_features=True, limit_to_current=True
                )
            p_old = F.log_softmax(logits[:, :prev_cls] / self.temperature, dim=1)
            q_old = F.softmax(anchor_logits[:, :prev_cls] / self.temperature, dim=1)
            kd_old = F.kl_div(p_old, q_old, reduction="batchmean") * (self.temperature ** 2)

            feat_cos = 1.0 - F.cosine_similarity(features, anchor_features, dim=1).mean()
            # Weight anchor consolidation proportionally to stability ratio
            stability_scale = float(
                max(0.3, min(1.5, self.current_lambda / max(1.0, self.ewc_lambda)))
            )
            anchor_loss = stability_scale * (1.2 * kd_old + 1.5 * feat_cos)

        total_loss = ce_loss + ewc_loss + anchor_loss
        return total_loss, {
            "ce_loss": float(ce_loss.item()),
            "ewc_loss": float(ewc_loss.item()),
            "anchor_loss": float(anchor_loss.item()),
            "adaptive_lambda": float(self.current_lambda),
            "total_loss": float(total_loss.item()),
        }

    def align_classifier_weights(self, model: FCILNet, task_id: int) -> None:
        """Apply Post-Task Weight Alignment (WA) and historical anchor restoration on classifier head."""
        if task_id <= 0:
            return
        prev_cls = task_id * self.classes_per_task
        curr_cls = min(model.current_classes, (task_id + 1) * self.classes_per_task)
        if curr_cls <= prev_cls or not hasattr(model, "classifier"):
            return
        fc = getattr(model.classifier, "fc", model.classifier)
        if not hasattr(fc, "weight"):
            return

        with torch.no_grad():
            w = fc.weight.data
            b = fc.bias.data if getattr(fc, "bias", None) is not None else None

            # 1. Restore historical class rows [0:prev_cls] toward optimal_params anchor
            w_key = "classifier.weight" if "classifier.weight" in self.optimal_params else "classifier.fc.weight"
            b_key = "classifier.bias" if "classifier.bias" in self.optimal_params else "classifier.fc.bias"
            if w_key in self.optimal_params:
                w_star = self.optimal_params[w_key].to(w.device)
                n_restore = min(prev_cls, w_star.size(0), w.size(0))
                if n_restore > 0:
                    w[:n_restore].copy_(0.92 * w_star[:n_restore] + 0.08 * w[:n_restore])
            if b is not None and b_key in self.optimal_params:
                b_star = self.optimal_params[b_key].to(b.device)
                n_restore = min(prev_cls, b_star.size(0), b.size(0))
                if n_restore > 0:
                    b[:n_restore].copy_(0.92 * b_star[:n_restore] + 0.08 * b[:n_restore])

            # 2. Weight Alignment (Zhao et al., CVPR 2020): align L2 norms of new rows to old rows
            old_norms = torch.norm(w[:prev_cls], p=2, dim=1)
            new_norms = torch.norm(w[prev_cls:curr_cls], p=2, dim=1)
            mean_old_norm = float(old_norms.mean().item())
            mean_new_norm = float(new_norms.mean().item())

            if mean_old_norm > 1e-6 and mean_new_norm > mean_old_norm:
                gamma_wa = mean_old_norm / mean_new_norm
                w[prev_cls:curr_cls].mul_(gamma_wa)

            if b is not None:
                old_bias_mean = b[:prev_cls].mean()
                new_bias_mean = b[prev_cls:curr_cls].mean()
                if new_bias_mean > old_bias_mean:
                    b[prev_cls:curr_cls].sub_(new_bias_mean - old_bias_mean)

    def _calibrate_head_with_latent_stats(
        self,
        model: FCILNet,
        task_id: int,
        device: torch.device,
    ) -> None:
        """Fine-tune classifier decision boundaries across all seen class Gaussian prototypes."""
        if task_id <= 0 or not self.latent_stats or not hasattr(model, "classifier"):
            return
        seen_classes = sorted(self.latent_stats.keys())
        if len(seen_classes) <= self.classes_per_task:
            return
        fc = getattr(model.classifier, "fc", model.classifier)
        opt = torch.optim.SGD(fc.parameters(), lr=0.015, momentum=0.9, weight_decay=1e-4)
        n_per_cls = 24
        for _ in range(25):
            feats_list, labels_list = [], []
            for c in seen_classes:
                if c >= model.current_classes:
                    continue
                mu_c, std_c = self.latent_stats[c]
                mu_c = mu_c.to(device)
                std_c = std_c.to(device)
                eps = torch.randn(n_per_cls, mu_c.numel(), device=device)
                feats_list.append(mu_c.unsqueeze(0) + 0.18 * eps * std_c.unsqueeze(0))
                labels_list.append(torch.full((n_per_cls,), c, dtype=torch.long, device=device))
            if not feats_list:
                break
            all_f = torch.cat(feats_list, dim=0)
            all_y = torch.cat(labels_list, dim=0)
            opt.zero_grad()
            logits = model.classifier(all_f)[:, :model.current_classes]
            loss = F.cross_entropy(logits, all_y)
            loss.backward()
            opt.step()

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        max_samples: int = 1000,
    ) -> None:
        """
        Estimate empirical Fisher Information Matrix across the current task's dataset
        and consolidate online via Eq. (14): F_i^{(t)} = gamma * F_i^{(t-1)} + F_hat_i^{(t)}.
        """
        # Align classifier weights before snapping optimal_params and Fisher
        self.align_classifier_weights(model, task_id)

        if train_loader is None or len(train_loader) == 0:
            self.optimal_params = {name: p.data.cpu().clone() for name, p in model.named_parameters()}
            return

        model.eval()
        fisher_accum = {name: torch.zeros_like(p.data, device=device) for name, p in model.named_parameters()}
        cls_feat_chunks: Dict[int, list] = {}
        total_batches = 0
        total_samples = 0

        for bx, by in train_loader:
            if total_samples >= max_samples:
                break
            bx, by = bx.to(device), by.to(device)
            n_b = bx.size(0)
            total_samples += n_b

            # Per-class gradient accumulation within batch gives exact class-wise empirical Fisher
            # in O(C_task) backward passes instead of O(N_samples) backward passes
            unique_y = torch.unique(by)
            logits, feats = model(bx, return_features=True, limit_to_current=True)
            log_probs = F.log_softmax(logits, dim=1)

            for idx_c, cls_val in enumerate(unique_y):
                c_int = int(cls_val.item())
                mask = (by == cls_val)
                cls_feat_chunks.setdefault(c_int, []).append(feats[mask].detach().cpu())

                model.zero_grad()
                cls_loss = F.nll_loss(log_probs[mask], by[mask], reduction="mean")
                cls_loss.backward(retain_graph=(idx_c < len(unique_y) - 1))
                weight_c = float(mask.sum().item()) / float(n_b)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_accum[name] += weight_c * param.grad.detach().pow(2)
            model.zero_grad()
            total_batches += 1

        # Store exemplar-free per-class 128-d Gaussian moments (zero raw data stored)
        for c_int, chunk_list in cls_feat_chunks.items():
            all_c_feats = torch.cat(chunk_list, dim=0)
            mu_c = all_c_feats.mean(dim=0)
            std_c = all_c_feats.std(dim=0, unbiased=False).clamp(min=1e-3)
            self.latent_stats[c_int] = (mu_c, std_c)

        # Calibrate classifier decision boundaries across all seen Gaussian prototypes
        self._calibrate_head_with_latent_stats(model, task_id, device)
        self.align_classifier_weights(model, task_id)

        # Invalidate device cache
        self._device_fisher.clear()
        self._device_optimal.clear()

        # Average and layer-normalize empirical Fisher so element sum per layer is in [0, 1]
        if total_batches > 0:
            for name in fisher_accum:
                fisher_accum[name] /= float(total_batches)
                max_val = float(fisher_accum[name].max().item())
                n_elem = max(1, fisher_accum[name].numel())
                if max_val > 1e-12:
                    fisher_accum[name] = (fisher_accum[name] / max_val) / float(n_elem)

        # Online EWC Consolidation with decay gamma = 0.9 (Eq. 14)
        if not self.fisher_dict:
            self.fisher_dict = {k: v.cpu().clone() for k, v in fisher_accum.items()}
        else:
            for name in fisher_accum:
                if name in self.fisher_dict:
                    old_f = self.fisher_dict[name]
                    new_f = fisher_accum[name].cpu()
                    target_shape = tuple(
                        max(old_dim, new_dim)
                        for old_dim, new_dim in zip(old_f.shape, new_f.shape)
                    )
                    accumulated = torch.zeros(
                        target_shape, dtype=new_f.dtype, device=new_f.device
                    )
                    old_slices = tuple(slice(0, size) for size in old_f.shape)
                    new_slices = tuple(slice(0, size) for size in new_f.shape)
                    accumulated[old_slices] += self.gamma * old_f.to(new_f.dtype)
                    accumulated[new_slices] += new_f
                    # Bound consolidated Fisher in [0, 1]
                    max_acc = float(accumulated.max().item())
                    if max_acc > 1.0:
                        accumulated /= max_acc
                    self.fisher_dict[name] = accumulated
                else:
                    self.fisher_dict[name] = fisher_accum[name].cpu().clone()

        # Save optimal parameters and BatchNorm buffers after calibration & Weight Alignment
        self.optimal_params = {name: p.data.cpu().clone() for name, p in model.named_parameters()}
        self.bn_buffers = {
            name: buf.data.cpu().clone()
            for name, buf in model.named_buffers()
            if "running_" in name
        }
        model.train()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ewc_lambda": self.ewc_lambda,
            "current_lambda": self.current_lambda,
            "gamma": self.gamma,
            "fisher_dict": {k: v.cpu() for k, v in self.fisher_dict.items()},
            "optimal_params": {k: v.cpu() for k, v in self.optimal_params.items()},
            "latent_stats": {
                int(c): (mu.cpu(), std.cpu()) for c, (mu, std) in self.latent_stats.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ewc_lambda = state.get("ewc_lambda", self.ewc_lambda)
        self.current_lambda = state.get("current_lambda", self.ewc_lambda)
        self.gamma = state.get("gamma", self.gamma)
        self.fisher_dict = state.get("fisher_dict", {})
        self.optimal_params = state.get("optimal_params", {})
        self.latent_stats = state.get("latent_stats", {})
        self._device_fisher.clear()
        self._device_optimal.clear()
