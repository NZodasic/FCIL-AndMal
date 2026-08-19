"""Self-Paced Class-Incremental Learning (SPCIL) strategy.

Curriculum learning for class-incremental learning.

Reference:
    Tao et al. "Few-Shot Class-Incremental Learning" (CVPR 2020)
    Adapted for malware domain.

"""

from typing import Dict
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from incremental.base_strategy import IncrementalStrategy


class SPCIL(IncrementalStrategy):
    """Self-Paced Class-Incremental Learning.

    Uses curriculum learning to gradually introduce harder samples,
    combined with knowledge distillation from old model.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        device: str = 'cuda',
        spcil_mu: float = 0.5,
        lwf_temperature: float = 2.0,
        **kwargs
    ):
        """Initialize SPCIL.

        Args:
            model: Neural network model.
            optimizer_fn: Optimizer function.
            device: Device for training.
            spcil_mu: Pacing parameter for curriculum learning.
            lwf_temperature: Temperature for distillation.
            **kwargs: Additional arguments.
        """
        super().__init__(model, optimizer_fn, device, **kwargs)
        self.spcil_mu = spcil_mu
        self.lwf_temperature = lwf_temperature

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        data: torch.Tensor = None,
        sample_weights: torch.Tensor = None,
        **kwargs
    ):
        """Compute loss with self-paced weighting.

        Args:
            outputs: Model outputs.
            targets: Target labels.
            data: Input data.
            sample_weights: Weights for curriculum learning.

        Returns:
            Loss and metrics.
        """
        # Standard cross-entropy
        if sample_weights is not None:
            ce_loss = F.cross_entropy(outputs, targets, reduction='none')
            ce_loss = (ce_loss * sample_weights).mean()
        else:
            ce_loss = F.cross_entropy(outputs, targets)

        # Distillation from old model
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

        total_loss = ce_loss + self.spcil_mu * distill_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'distill_loss': distill_loss.item() if isinstance(distill_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }

        return total_loss, metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0,
        n_epochs: int = 1,
        **kwargs
    ) -> Dict[str, float]:
        """Train for one epoch with self-paced weighting.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer.
            epoch: Current epoch number.
            n_epochs: Total number of epochs.

        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Compute pacing: start with easy samples, gradually include harder ones
        pacing = min(1.0, epoch / (n_epochs * 0.5))  # Linear pacing

        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            # Compute sample difficulty (prediction confidence)
            with torch.no_grad():
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                confidence = probs.max(dim=1)[0]

            # Sample weights based on pacing
            # Easy samples: high confidence
            # Hard samples: low confidence
            easy_mask = confidence >= (1.0 - pacing)
            sample_weights = easy_mask.float()

            # Ensure at least some samples are used
            if sample_weights.sum() < 2:
                sample_weights = torch.ones_like(sample_weights)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss, metrics = self.compute_loss(
                outputs, targets, data=data, sample_weights=sample_weights
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total,
            'pacing': pacing
        }

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train on a single task with SPCIL.

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
            epoch_metrics = self.train_epoch(
                train_loader, optimizer, epoch=epoch, n_epochs=n_epochs
            )
            metrics['epochs'].append(epoch_metrics)

        # Average metrics
        avg_metrics = {
            'loss': sum(m['loss'] for m in metrics['epochs']) / n_epochs,
            'accuracy': sum(m['accuracy'] for m in metrics['epochs']) / n_epochs,
            'task_id': task_id
        }

        return avg_metrics
