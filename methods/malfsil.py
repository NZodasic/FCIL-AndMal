"""
MALFSIL: Multi-Facet Federated Self-Paced Incremental Learning (Proposed Method).
Combines local herding replay, old-task logit distillation, and server-side global
class prototype alignment to mitigate both local and global catastrophic forgetting under non-IID skew.
"""

from typing import Dict, List, Any, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class MALFSILMethod(BaseILMethod):
    """
    MALFSIL Framework (Proposed Novel FCIL Architecture).
    Three-Tier Defense Mechanism:
    1. Local Exemplar Herding: Compact memory replay buffer per class.
    2. Adaptive Logit Distillation: Knowledge distillation on previous class logits.
    3. Global Prototype Regularization: Distance alignment to global server-aggregated class centroids.
    """

    def __init__(
        self,
        buffer_size_per_class: int = 20,
        distill_weight: float = 1.0,
        proto_weight: float = 0.5,
        temperature: float = 2.0,
        margin: float = 0.2,
        use_herding: bool = True
    ):
        super().__init__(name="malfsil")
        self.buffer_size_per_class = buffer_size_per_class
        self.distill_weight = distill_weight
        self.proto_weight = proto_weight
        self.temperature = temperature
        self.margin = margin
        self.use_herding = use_herding

        # 1. Local Replay Buffer: class_id -> (X, y)
        self.buffer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # 2. Teacher model snapshot for distillation
        self.prev_model: Optional[FCILNet] = None
        self.prev_num_classes: int = 0

        # 3. Global prototypes: class_id -> prototype_vector (Tensor)
        self.global_prototypes: Dict[int, torch.Tensor] = {}

    def set_global_prototypes(self, prototypes: Dict[int, torch.Tensor]) -> None:
        """Inject server-aggregated global class prototypes."""
        self.global_prototypes = {k: v.clone() for k, v in prototypes.items()}

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        if task_id > 0:
            self.prev_model = model.clone_model().to(device)
            self.prev_model.eval()
            self.prev_num_classes = self.prev_model.current_classes

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Exemplar Replay Augmentation
        if self.buffer and task_id > 0:
            replay_x_list, replay_y_list = [], []
            for cls_id, (cls_x, cls_y) in self.buffer.items():
                if len(cls_x) > 0:
                    k_samples = min(len(cls_x), max(1, x.size(0) // (len(self.buffer) * 2)))
                    idx = np.random.choice(len(cls_x), size=k_samples, replace=False)
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

        # Forward pass with feature extraction
        logits, features = model(combined_x, return_features=True, limit_to_current=True)
        ce_loss = criterion(logits, combined_y)

        # 2. Knowledge Distillation Loss
        distill_loss = torch.tensor(0.0, device=device)
        if task_id > 0 and self.prev_model is not None and self.prev_num_classes > 0:
            with torch.no_grad():
                prev_logits = self.prev_model(combined_x, limit_to_current=True)

            p_soft = F.log_softmax(logits[:, :self.prev_num_classes] / self.temperature, dim=1)
            q_soft = F.softmax(prev_logits[:, :self.prev_num_classes] / self.temperature, dim=1)
            distill_loss = F.kl_div(p_soft, q_soft, reduction="batchmean") * (self.temperature ** 2)
            distill_loss = self.distill_weight * distill_loss

        # 3. Global Prototype Alignment Loss
        proto_loss = torch.tensor(0.0, device=device)
        if self.global_prototypes:
            proto_losses = []
            for i in range(combined_x.size(0)):
                label_i = int(combined_y[i].item())
                feat_i = features[i]
                if label_i in self.global_prototypes:
                    target_proto = self.global_prototypes[label_i].to(device)
                    # Pull towards ground-truth prototype
                    pos_dist = F.mse_loss(feat_i, target_proto)

                    # Push away from other prototypes
                    neg_dists = []
                    for other_cls, other_proto in self.global_prototypes.items():
                        if other_cls != label_i:
                            d = torch.norm(feat_i - other_proto.to(device), p=2)
                            neg_dists.append(d)

                    if neg_dists:
                        min_neg = torch.min(torch.stack(neg_dists))
                        margin_loss = F.relu(self.margin - min_neg)
                        proto_losses.append(pos_dist + margin_loss)
                    else:
                        proto_losses.append(pos_dist)

            if proto_losses:
                proto_loss = self.proto_weight * torch.mean(torch.stack(proto_losses))

        total_loss = ce_loss + distill_loss + proto_loss
        return total_loss, {
            "ce_loss": float(ce_loss.item()),
            "distill_loss": float(distill_loss.item()),
            "proto_loss": float(proto_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    def compute_local_prototypes(
        self,
        model: FCILNet,
        train_loader: DataLoader,
        device: torch.device = torch.device("cpu")
    ) -> Dict[int, Tuple[torch.Tensor, int]]:
        """
        Extract client's local class prototypes: class_id -> (mean_feature_tensor, sample_count).
        """
        model.eval()
        class_embeddings: Dict[int, List[torch.Tensor]] = {}

        with torch.no_grad():
            for bx, by in train_loader:
                bx_dev = bx.to(device)
                feat = model.get_features(bx_dev)
                for i in range(bx.size(0)):
                    cls_id = int(by[i].item())
                    if cls_id not in class_embeddings:
                        class_embeddings[cls_id] = []
                    class_embeddings[cls_id].append(feat[i].cpu())

        local_protos = {}
        for cls_id, emb_list in class_embeddings.items():
            stacked = torch.stack(emb_list, dim=0)
            mean_proto = stacked.mean(dim=0)
            local_protos[cls_id] = (mean_proto, len(emb_list))

        model.train()
        return local_protos

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """Herding exemplar buffer update."""
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

        if not all_x:
            model.train()
            return

        all_x_t = torch.cat(all_x, dim=0)
        all_y_t = torch.cat(all_y, dim=0)
        all_feat_t = torch.cat(all_feat, dim=0)

        for cls_id in torch.unique(all_y_t).tolist():
            cls_mask = (all_y_t == cls_id)
            cls_x = all_x_t[cls_mask]
            cls_feat = all_feat_t[cls_mask]
            n_cls = cls_x.size(0)

            if n_cls == 0:
                continue

            m = min(self.buffer_size_per_class, n_cls)

            if self.use_herding and n_cls > m:
                cls_mean = cls_feat.mean(dim=0, keepdim=True)
                selected_indices: List[int] = []
                running_sum = torch.zeros_like(cls_mean)

                for k in range(m):
                    candidate_sums = running_sum + cls_feat
                    candidate_means = candidate_sums / (k + 1)
                    dists = torch.norm(candidate_means - cls_mean, dim=1)
                    for s_idx in selected_indices:
                        dists[s_idx] = float("inf")
                    best_idx = int(torch.argmin(dists).item())
                    selected_indices.append(best_idx)
                    running_sum += cls_feat[best_idx:best_idx + 1]

                selected_x = cls_x[selected_indices]
                selected_y = torch.full((m,), cls_id, dtype=torch.long)
            else:
                perm = torch.randperm(n_cls)[:m]
                selected_x = cls_x[perm]
                selected_y = torch.full((len(perm),), cls_id, dtype=torch.long)

            self.buffer[cls_id] = (selected_x, selected_y)

        model.train()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "buffer_size_per_class": self.buffer_size_per_class,
            "distill_weight": self.distill_weight,
            "proto_weight": self.proto_weight,
            "temperature": self.temperature,
            "margin": self.margin,
            "buffer": {k: (v[0].cpu(), v[1].cpu()) for k, v in self.buffer.items()},
            "global_prototypes": {k: v.cpu() for k, v in self.global_prototypes.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.buffer_size_per_class = state.get("buffer_size_per_class", self.buffer_size_per_class)
        self.distill_weight = state.get("distill_weight", self.distill_weight)
        self.proto_weight = state.get("proto_weight", self.proto_weight)
        self.temperature = state.get("temperature", self.temperature)
        self.margin = state.get("margin", self.margin)
        raw_buf = state.get("buffer", {})
        self.buffer = {int(k): (v[0], v[1]) for k, v in raw_buf.items()}
        raw_protos = state.get("global_prototypes", {})
        self.global_prototypes = {int(k): v for k, v in raw_protos.items()}
