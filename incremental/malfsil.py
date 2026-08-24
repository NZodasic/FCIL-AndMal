"""Deprecated legacy MALFSIL experiment.

This module predates the paper-aligned MalFSCIL implementation in
``methods/malfscil.py``. It is retained only for compatibility with the legacy
``experiments/run_fl.py`` stack and must not be described as MalFSCIL.

The legacy experiment combines:
1. Local replay with herding
2. Knowledge distillation
3. Server-side prototype aggregation

"""

from typing import Dict, List
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from incremental.base_strategy import IncrementalStrategy


class MALFSIL(IncrementalStrategy):
    """Deprecated replay/distillation/prototype FCIL experiment.

    Combines:
    - Local experience replay (herding selection)
    - Knowledge distillation from old model
    - Server-side prototype aggregation (for global forgetting reduction)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        device: str = 'cuda',
        buffer_size_per_class: int = 20,
        lwf_alpha: float = 1.0,
        lwf_temperature: float = 2.0,
        prototype_weight: float = 0.1,
        **kwargs
    ):
        """Initialize the deprecated MALFSIL strategy.

        Args:
            model: Neural network model.
            optimizer_fn: Optimizer function.
            device: Device for training.
            buffer_size_per_class: Replay buffer size per class.
            lwf_alpha: Weight for distillation loss.
            lwf_temperature: Temperature for distillation.
            prototype_weight: Weight for prototype loss.
            **kwargs: Additional arguments.
        """
        super().__init__(model, optimizer_fn, device, **kwargs)
        self.buffer_size_per_class = buffer_size_per_class
        self.lwf_alpha = lwf_alpha
        self.lwf_temperature = lwf_temperature
        self.prototype_weight = prototype_weight

        # Local replay buffer
        self.replay_buffer: Dict[int, List[torch.Tensor]] = defaultdict(list)

        # Class prototypes (aggregated from server)
        self.class_prototypes: Dict[int, torch.Tensor] = {}

    def set_prototypes(self, prototypes: Dict[int, torch.Tensor]) -> None:
        """Set class prototypes from server aggregation.

        Args:
            prototypes: Dictionary mapping class_id to prototype vector.
        """
        self.class_prototypes = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in prototypes.items()
        }

    def get_prototypes(self) -> Dict[int, torch.Tensor]:
        """Get current class prototypes for server aggregation.

        Returns:
            Dictionary mapping class_id to prototype vector.
        """
        prototypes = {}
        self.model.eval()

        with torch.no_grad():
            for cls, data_list in self.replay_buffer.items():
                if not data_list:
                    continue

                # Get features for all samples of this class
                cls_data = torch.cat(data_list, dim=0).to(self.device)

                if hasattr(self.model, 'get_features'):
                    features = self.model.get_features(cls_data)
                else:
                    features = cls_data

                # Compute mean as prototype
                prototype = features.mean(dim=0)
                prototypes[cls] = prototype.cpu()

        return prototypes

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Prepare for new task."""
        super().before_task(task_id, n_new_classes)

        if self.old_model is not None:
            del self.old_model
            self.old_model = None

    def after_task(self, task_id: int, train_loader: DataLoader, **kwargs) -> None:
        """Update replay buffer after training."""
        all_data = []
        all_targets = []

        for data, targets in train_loader:
            all_data.append(data)
            all_targets.append(targets)

        all_data = torch.cat(all_data, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        unique_classes = torch.unique(all_targets).tolist()

        for cls in unique_classes:
            cls_mask = all_targets == cls
            cls_data = all_data[cls_mask]

            # Herding selection
            selected = self._herding_selection(cls_data)
            self.replay_buffer[cls] = [selected]

    def _herding_selection(self, data: torch.Tensor) -> torch.Tensor:
        """Select samples using herding."""
        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, 'get_features'):
                features = self.model.get_features(data.to(self.device))
            else:
                features = data.to(self.device)

            class_mean = features.mean(dim=0)
            distances = torch.cdist(features, class_mean.unsqueeze(0)).squeeze()
            _, indices = torch.topk(
                distances,
                min(self.buffer_size_per_class, len(distances)),
                largest=False
            )

        return data[indices].cpu()

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        data: torch.Tensor = None,
        **kwargs
    ):
        """Compute combined loss."""
        # Cross-entropy loss
        ce_loss = F.cross_entropy(outputs, targets)

        # Distillation loss
        distill_loss = 0.0
        if self.old_model is not None and self.current_task > 0:
            with torch.no_grad():
                old_outputs = self.old_model(data)

            n_old_classes = old_outputs.size(1)
            new_old_logits = outputs[:, :n_old_classes]

            T = self.lwf_temperature
            old_probs = F.softmax(old_outputs / T, dim=1)
            new_log_probs = F.log_softmax(new_old_logits / T, dim=1)
            distill_loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean') * (T ** 2)

        # Prototype loss
        prototype_loss = 0.0
        if self.class_prototypes and hasattr(self.model, 'get_features'):
            features = self.model.get_features(data)

            for cls, proto in self.class_prototypes.items():
                cls_mask = targets == cls
                if cls_mask.sum() > 0:
                    cls_features = features[cls_mask]
                    proto_loss = F.mse_loss(
                        cls_features,
                        proto.unsqueeze(0).expand(len(cls_features), -1)
                    )
                    prototype_loss += proto_loss

        total_loss = ce_loss + self.lwf_alpha * distill_loss + self.prototype_weight * prototype_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'distill_loss': distill_loss.item() if isinstance(distill_loss, torch.Tensor) else 0.0,
            'prototype_loss': prototype_loss.item() if isinstance(prototype_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }

        return total_loss, metrics

    def get_replay_loader(self, batch_size: int):
        """Create DataLoader from replay buffer."""
        if not self.replay_buffer:
            return None

        all_data = []
        all_targets = []

        for cls, data_list in self.replay_buffer.items():
            for data_tensor in data_list:
                data_tensor = data_tensor.view(-1, data_tensor.size(-1))
                all_data.append(data_tensor)
                all_targets.extend([cls] * len(data_tensor))

        if not all_data:
            return None

        replay_data = torch.cat(all_data, dim=0)
        replay_targets = torch.tensor(all_targets, dtype=torch.long)

        dataset = TensorDataset(replay_data, replay_targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(self, train_loader: DataLoader, optimizer, **kwargs) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_metrics = defaultdict(float)
        correct = 0
        total = 0

        replay_loader = self.get_replay_loader(train_loader.batch_size)
        replay_iter = iter(replay_loader) if replay_loader else None

        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            if replay_iter:
                try:
                    replay_data, replay_targets = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_data, replay_targets = next(replay_iter)

                data = torch.cat([data, replay_data.to(self.device)], dim=0)
                targets = torch.cat([targets, replay_targets.to(self.device)], dim=0)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss, metrics = self.compute_loss(outputs, targets, data=data)

            loss.backward()
            optimizer.step()

            for k, v in metrics.items():
                total_metrics[k] += v

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        n_batches = len(train_loader)
        return {
            **{k: v / n_batches for k, v in total_metrics.items()},
            'accuracy': 100. * correct / total
        }

    def train_task(self, train_loader, task_id, n_epochs, **kwargs) -> Dict[str, float]:
        """Train on a single task."""
        optimizer = self.optimizer_fn(self.model.parameters())

        metrics = {'epochs': []}
        for epoch in range(n_epochs):
            epoch_metrics = self.train_epoch(train_loader, optimizer)
            metrics['epochs'].append(epoch_metrics)

        self.after_task(task_id, train_loader)

        return {
            'ce_loss': sum(m['ce_loss'] for m in metrics['epochs']) / n_epochs,
            'accuracy': sum(m['accuracy'] for m in metrics['epochs']) / n_epochs,
            'task_id': task_id
        }
