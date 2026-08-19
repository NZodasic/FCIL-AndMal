"""
Exemplar Memory Replay with Herding Selection (iCaRL-style) for FCIL.
Maintains a bounded replay buffer of past malware family features to prevent catastrophic forgetting.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class ReplayHerdingMethod(BaseILMethod):
    """
    Exemplar-based Continual Learning using Herding in latent feature space (Rebuffi et al., CVPR 2017).
    """

    def __init__(self, buffer_size_per_class: int = 20, use_herding: bool = True):
        super().__init__(name="replay")
        self.buffer_size_per_class = buffer_size_per_class
        self.use_herding = use_herding
        # Memory buffer: class_id -> (features_tensor, labels_tensor)
        self.buffer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Mix current batch with replay buffer samples
        if self.buffer and task_id > 0:
            replay_x_list, replay_y_list = [], []
            for cls_id, (cls_x, cls_y) in self.buffer.items():
                if len(cls_x) > 0:
                    # Randomly sample from buffer
                    idx = np.random.choice(len(cls_x), size=min(len(cls_x), max(1, x.size(0) // (len(self.buffer) * 2))), replace=False)
                    replay_x_list.append(cls_x[idx])
                    replay_y_list.append(cls_y[idx])

            if replay_x_list:
                rx = torch.cat(replay_x_list, dim=0).to(device)
                ry = torch.cat(replay_y_list, dim=0).to(device)
                combined_x = torch.cat([x, rx], dim=0)
                combined_y = torch.cat([y, ry], dim=0)
            else:
                combined_x, combined_y = x, y
        else:
            combined_x, combined_y = x, y

        logits = model(combined_x, limit_to_current=True)
        ce_loss = criterion(logits, combined_y)
        return ce_loss, {"ce_loss": float(ce_loss.item()), "total_loss": float(ce_loss.item())}

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """Construct/update exemplar memory using Herding on latent embeddings."""
        if train_loader is None or len(train_loader) == 0:
            return

        model.eval()
        # Collect all samples and their embeddings for this task
        all_x, all_y, all_feat = [], [], []
        with torch.no_grad():
            for bx, by in train_loader:
                bx_dev = bx.to(device)
                feat = model.get_features(bx_dev)
                all_x.append(bx.cpu())
                all_y.append(by.cpu())
                all_feat.append(feat.cpu())

        if not all_x:
            model.train()
            return

        all_x_t = torch.cat(all_x, dim=0)
        all_y_t = torch.cat(all_y, dim=0)
        all_feat_t = torch.cat(all_feat, dim=0)

        unique_classes = torch.unique(all_y_t).tolist()

        for cls_id in unique_classes:
            cls_mask = (all_y_t == cls_id)
            cls_x = all_x_t[cls_mask]
            cls_feat = all_feat_t[cls_mask]
            n_cls = cls_x.size(0)

            if n_cls == 0:
                continue

            m = min(self.buffer_size_per_class, n_cls)

            if self.use_herding and n_cls > m:
                # Herding selection: iteratively pick sample that brings running mean closest to true class mean
                cls_mean = cls_feat.mean(dim=0, keepdim=True)  # (1, latent_dim)
                selected_indices: List[int] = []
                running_sum = torch.zeros_like(cls_mean)

                for k in range(m):
                    # Candidate means
                    candidate_sums = running_sum + cls_feat
                    candidate_means = candidate_sums / (k + 1)
                    dists = torch.norm(candidate_means - cls_mean, dim=1)
                    
                    # Mask already selected
                    for s_idx in selected_indices:
                        dists[s_idx] = float("inf")

                    best_idx = int(torch.argmin(dists).item())
                    selected_indices.append(best_idx)
                    running_sum += cls_feat[best_idx:best_idx + 1]

                selected_x = cls_x[selected_indices]
                selected_y = torch.full((m,), cls_id, dtype=torch.long)
            else:
                # Random selection fallback
                perm = torch.randperm(n_cls)[:m]
                selected_x = cls_x[perm]
                selected_y = torch.full((len(perm),), cls_id, dtype=torch.long)

            self.buffer[cls_id] = (selected_x, selected_y)

        model.train()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "buffer_size_per_class": self.buffer_size_per_class,
            "buffer": {
                cls_id: (x.cpu(), y.cpu()) for cls_id, (x, y) in self.buffer.items()
            }
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.buffer_size_per_class = state.get("buffer_size_per_class", self.buffer_size_per_class)
        raw_buf = state.get("buffer", {})
        self.buffer = {int(k): (v[0], v[1]) for k, v in raw_buf.items()}
