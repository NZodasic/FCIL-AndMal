"""
Self-Paced Class-Incremental Learning (SPCIL).
Implements adaptive hardness-aware sample weighting curriculum for Android malware detection.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class SPCILMethod(BaseILMethod):
    """
    Self-Paced Class-Incremental Learning (SPCIL).
    Applies self-paced regularizer dynamically filtering or down-weighting ambiguous/noisy
    samples early in the incremental task to stabilize decision boundaries, combined with
    self-paced historical logit anchoring and post-task Weight Alignment (WA).
    """

    def __init__(self, lambda_init: float = 0.5, lambda_step: float = 0.1, temperature: float = 2.0):
        super().__init__(name="spcil")
        self.pacing_lambda = float(lambda_init)
        self.lambda_step = float(lambda_step)
        self.temperature = float(temperature)
        self.step_counter = 0
        self.prev_model: Optional[FCILNet] = None
        self.prev_num_classes: int = 0
        self.optimal_old_head: Optional[Dict[str, torch.Tensor]] = None
        self.latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.step_counter = 0
        cpt = (
            getattr(model.config, "classes_per_task", 3)
            if hasattr(model, "config") and model.config is not None
            else 3
        )
        self.prev_num_classes = task_id * cpt
        if task_id > 0:
            self.prev_model = model.clone_model().to(device)
            self.prev_model.eval()
            for p in self.prev_model.parameters():
                p.requires_grad = False
            if hasattr(model, "classifier"):
                fc = getattr(model.classifier, "fc", model.classifier)
                if hasattr(fc, "weight"):
                    self.optimal_old_head = {
                        "weight": fc.weight.data[:self.prev_num_classes].cpu().clone(),
                    }
                    if getattr(fc, "bias", None) is not None:
                        self.optimal_old_head["bias"] = fc.bias.data[:self.prev_num_classes].cpu().clone()
        else:
            self.prev_model = None
            self.optimal_old_head = None
            self.latent_stats.clear()

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if task_id > 0 and hasattr(model, "backbone"):
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

        logits, features = model(x, return_features=True, limit_to_current=True)
        prev_cls = self.prev_num_classes

        # Sample-wise cross entropy loss
        if task_id > 0 and prev_cls > 0 and logits.size(1) > prev_cls:
            new_logits = logits[:, prev_cls:]
            local_targets = torch.clamp(y - prev_cls, min=0, max=new_logits.size(1) - 1)
            local_losses = F.cross_entropy(new_logits, local_targets, reduction="none")
            calibrated_logits = torch.cat([logits[:, :prev_cls].detach(), new_logits], dim=1)
            joint_losses = F.cross_entropy(calibrated_logits, y, reduction="none")
            per_sample_loss = 0.65 * local_losses + 0.35 * joint_losses
        else:
            per_sample_loss = F.cross_entropy(logits, y, reduction="none")

        # Self-paced adaptive curriculum threshold (quantile-calibrated to prevent minority class starvation)
        effective_lambda = self.pacing_lambda + (self.step_counter * self.lambda_step * 0.02)
        with torch.no_grad():
            q_ratio = min(0.95, 0.65 + 0.005 * self.step_counter)
            adaptive_thresh = max(
                effective_lambda,
                float(torch.quantile(per_sample_loss.detach(), q_ratio).item()),
            )
            # Soft + hard mixture weights so hard classes still receive smooth curriculum updates
            hard_mask = (per_sample_loss <= adaptive_thresh).float()
            soft_weights = torch.exp(-0.25 * F.relu(per_sample_loss - adaptive_thresh))
            weights = 0.7 * hard_mask + 0.3 * soft_weights

        if weights.sum() == 0:
            weights = torch.ones_like(per_sample_loss)

        weighted_loss = (per_sample_loss * weights).sum() / weights.sum()

        # Self-paced historical knowledge preservation for task_id > 0
        distill_loss = torch.tensor(0.0, device=device)
        if task_id > 0 and self.prev_model is not None and prev_cls > 0:
            self.prev_model.to(device)
            with torch.no_grad():
                prev_logits, prev_features = self.prev_model(
                    x, return_features=True, limit_to_current=True
                )
            p_old = F.log_softmax(logits[:, :prev_cls] / self.temperature, dim=1)
            q_old = F.softmax(prev_logits[:, :prev_cls] / self.temperature, dim=1)
            kd_loss = F.kl_div(p_old, q_old, reduction="batchmean") * (self.temperature ** 2)
            feat_loss = 1.0 - F.cosine_similarity(features, prev_features, dim=1).mean()
            distill_loss = 0.7 * kd_loss + 0.7 * feat_loss

            if self.latent_stats and hasattr(model, "classifier"):
                syn_feats_list, syn_labels_list = [], []
                n_per_old = max(2, min(6, x.size(0) // max(1, prev_cls)))
                for c_old in range(prev_cls):
                    if c_old in self.latent_stats:
                        mu_c, std_c = self.latent_stats[c_old]
                        mu_c = mu_c.to(device)
                        std_c = std_c.to(device)
                        eps = torch.randn(n_per_old, mu_c.numel(), device=device)
                        syn_feats_list.append(mu_c.unsqueeze(0) + 0.25 * eps * std_c.unsqueeze(0))
                        syn_labels_list.append(
                            torch.full((n_per_old,), c_old, dtype=torch.long, device=device)
                        )
                if syn_feats_list:
                    syn_f = torch.cat(syn_feats_list, dim=0)
                    syn_y = torch.cat(syn_labels_list, dim=0)
                    cal_logits = model.classifier(torch.cat([features.detach(), syn_f], dim=0))[
                        :, :logits.size(1)
                    ]
                    distill_loss = distill_loss + 0.35 * F.cross_entropy(
                        cal_logits, torch.cat([y, syn_y], dim=0)
                    )

        total_loss = weighted_loss + distill_loss
        self.step_counter += 1

        return total_loss, {
            "ce_loss": float(weighted_loss.item()),
            "pacing_lambda": float(effective_lambda),
            "retained_ratio": float((per_sample_loss <= adaptive_thresh).float().mean().item()),
            "total_loss": float(total_loss.item()),
        }

    def align_classifier_weights(self, model: FCILNet, task_id: int) -> None:
        """Align new classifier rows with historical classifier rows after task/round."""
        if task_id <= 0 or not hasattr(model, "classifier"):
            return
        fc = getattr(model.classifier, "fc", model.classifier)
        if not hasattr(fc, "weight"):
            return
        prev_cls = self.prev_num_classes
        curr_cls = model.current_classes
        if prev_cls <= 0 or curr_cls <= prev_cls:
            return

        with torch.no_grad():
            w = fc.weight.data
            b = fc.bias.data if getattr(fc, "bias", None) is not None else None

            if self.optimal_old_head is not None and "weight" in self.optimal_old_head:
                w_old_star = self.optimal_old_head["weight"].to(w.device)
                n_rest = min(prev_cls, w_old_star.size(0), w.size(0))
                w[:n_rest].copy_(0.88 * w_old_star[:n_rest] + 0.12 * w[:n_rest])
                if b is not None and "bias" in self.optimal_old_head:
                    b_old_star = self.optimal_old_head["bias"].to(b.device)
                    b[:n_rest].copy_(0.88 * b_old_star[:n_rest] + 0.12 * b[:n_rest])

            old_norms = torch.norm(w[:prev_cls], p=2, dim=1)
            new_norms = torch.norm(w[prev_cls:curr_cls], p=2, dim=1)
            mean_old = float(old_norms.mean().item())
            mean_new = float(new_norms.mean().item())
            if mean_old > 1e-6 and mean_new > mean_old:
                gamma_wa = mean_old / mean_new
                w[prev_cls:curr_cls].mul_(gamma_wa)
            if b is not None:
                old_b = b[:prev_cls].mean()
                new_b = b[prev_cls:curr_cls].mean()
                if new_b > old_b:
                    b[prev_cls:curr_cls].sub_(new_b - old_b)

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        if train_loader is not None and len(train_loader) > 0:
            model.eval()
            cls_chunks: Dict[int, list] = {}
            samples = 0
            with torch.no_grad():
                for bx, by in train_loader:
                    if samples >= 800:
                        break
                    bx, by = bx.to(device), by.to(device)
                    samples += bx.size(0)
                    _, feats = model(bx, return_features=True, limit_to_current=True)
                    for c_val in torch.unique(by):
                        c_int = int(c_val.item())
                        cls_chunks.setdefault(c_int, []).append(feats[by == c_val].cpu())
            for c_int, ch_list in cls_chunks.items():
                all_f = torch.cat(ch_list, dim=0)
                self.latent_stats[c_int] = (
                    all_f.mean(dim=0),
                    all_f.std(dim=0, unbiased=False).clamp(min=1e-3),
                )
            model.train()
        self.align_classifier_weights(model, task_id)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pacing_lambda": self.pacing_lambda,
            "lambda_step": self.lambda_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.pacing_lambda = state.get("pacing_lambda", self.pacing_lambda)
        self.lambda_step = state.get("lambda_step", self.lambda_step)
