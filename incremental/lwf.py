"""Learning without Forgetting (LwF) strategy.

Prevents catastrophic forgetting using knowledge distillation
from the old model.

Reference:
    Li & Hoiem. "Learning without Forgetting" (TPAMI 2017)

"""

from typing import Dict, Tuple
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from incremental.base_strategy import IncrementalStrategy


class LwF(IncrementalStrategy):
    """Learning without Forgetting strategy.

    Uses knowledge distillation to preserve old knowledge while
    learning new classes. The distillation loss encourages the
    new model to produce similar outputs to the old model on
    new task data.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        device: str = 'cuda',
        lwf_alpha: float = 1.0,
        lwf_temperature: float = 2.0,
        **kwargs
    ):
        """Initialize LwF.

        Args:
            model: Neural network model.
            optimizer_fn: Optimizer function.
            device: Device for training.
            lwf_alpha: Weight for distillation loss.
            lwf_temperature: Temperature for distillation.
            **kwargs: Additional arguments.
        """
        super().__init__(model, optimizer_fn, device, **kwargs)
        self.lwf_alpha = lwf_alpha
        self.lwf_temperature = lwf_temperature

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        data: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with distillation.

        Args:
            outputs: Model outputs (current model).
            targets: Target labels.
            data: Input data (needed for distillation).

        Returns:
            Tuple of (loss, metrics).
        """
        # Standard cross-entropy loss for all classes
        ce_loss = F.cross_entropy(outputs, targets)

        # Distillation loss from old model
        distill_loss = 0.0
        if self.old_model is not None and self.current_task > 0:
            with torch.no_grad():
                old_outputs = self.old_model(data)

            # Get number of old classes
            n_old_classes = old_outputs.size(1)

            # Distillation on old class logits
            new_old_logits = outputs[:, :n_old_classes]

            distill_loss = self._distillation_loss(
                new_old_logits, old_outputs
            )

        total_loss = ce_loss + self.lwf_alpha * distill_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'distill_loss': distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
            'total_loss': total_loss.item()
        }

        return total_loss, metrics

    def _distillation_loss(
        self,
        new_logits: torch.Tensor,
        old_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
            new_logits: Logits from current model for old classes.
            old_logits: Logits from old model.

        Returns:
            Distillation loss.
        """
        # Temperature scaling
        T = self.lwf_temperature

        # Softmax with temperature
        old_probs = F.softmax(old_logits / T, dim=1)
        new_log_probs = F.log_softmax(new_logits / T, dim=1)

        # KL divergence loss
        loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')

        # Scale by T^2 (as in the original paper)
        return loss * (T ** 2)

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> Dict[str, float]:
        """Train for one epoch with LwF.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer.

        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        total_metrics = defaultdict(float)
        correct = 0
        total = 0

        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss, metrics = self.compute_loss(outputs, targets, data=data)

            loss.backward()
            optimizer.step()

            # Accumulate metrics
            for key, value in metrics.items():
                total_metrics[key] += value

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Average metrics
        n_batches = len(train_loader)
        avg_metrics = {
            key: value / n_batches
            for key, value in total_metrics.items()
        }
        avg_metrics['accuracy'] = 100. * correct / total

        return avg_metrics

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train on a single task with LwF.

        Args:
            train_loader: DataLoader for training data.
            task_id: Current task ID.
            n_epochs: Number of training epochs.

        Returns:
            Dictionary of training metrics.
        """
        optimizer = self.optimizer_fn(self.model.parameters())

        metrics = {'epochs': []}
        for epoch in range(n_epochs):
            epoch_metrics = self.train_epoch(train_loader, optimizer)
            metrics['epochs'].append(epoch_metrics)

        # Average metrics
        avg_metrics = {
            'ce_loss': sum(m['ce_loss'] for m in metrics['epochs']) / n_epochs,
            'distill_loss': sum(m.get('distill_loss', 0)
                               for m in metrics['epochs']) / n_epochs,
            'accuracy': sum(m['accuracy'] for m in metrics['epochs']) / n_epochs,
            'task_id': task_id
        }

        return avg_metrics


from collections import defaultdict
