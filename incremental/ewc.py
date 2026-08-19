"""Elastic Weight Consolidation (EWC) strategy.

Prevents catastrophic forgetting by penalizing changes to important weights.

Reference:
    Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks"
    (PNAS 2017)

"""

from typing import Dict, Optional
from collections import defaultdict
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from incremental.base_strategy import IncrementalStrategy


class EWC(IncrementalStrategy):
    """Elastic Weight Consolidation strategy.

    Uses Fisher Information to estimate importance of weights
    and penalizes changes to important weights when learning new tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        device: str = 'cuda',
        ewc_lambda: float = 1000.0,
        **kwargs
    ):
        """Initialize EWC.

        Args:
            model: Neural network model.
            optimizer_fn: Optimizer function.
            device: Device for training.
            ewc_lambda: Regularization strength for EWC penalty.
            **kwargs: Additional arguments.
        """
        super().__init__(model, optimizer_fn, device, **kwargs)
        self.ewc_lambda = ewc_lambda

        # Store Fisher Information and optimal params for each task
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Prepare for new task.

        Args:
            task_id: Current task ID.
            n_new_classes: Number of new classes.
        """
        # Save optimal parameters from previous task
        if task_id > 0:
            self.optimal_params[task_id - 1] = {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

        super().before_task(task_id, n_new_classes)

    def after_task(self, task_id: int, train_loader: DataLoader, **kwargs) -> None:
        """Compute Fisher Information after training.

        Args:
            task_id: Current task ID.
            train_loader: DataLoader for computing Fisher.
        """
        self.fisher_matrices[task_id] = self._compute_fisher(train_loader)

    def _compute_fisher(
        self,
        train_loader: DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information matrix (diagonal approximation).

        Args:
            train_loader: DataLoader for computing Fisher.
            num_samples: Number of samples to use (None = all).

        Returns:
            Dictionary mapping parameter name to Fisher values.
        """
        self.model.eval()
        fisher = defaultdict(lambda: 0.0)

        n_samples = 0
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(data)
            log_probs = F.log_softmax(outputs, dim=1)

            # Use predicted class (unsupervised) or true class (supervised)
            # Here we use a sampled class from the output distribution
            probs = F.softmax(outputs, dim=1)
            sampled_targets = torch.multinomial(probs, 1).squeeze()

            # Compute negative log-likelihood
            nll = F.nll_loss(log_probs, sampled_targets)
            nll.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            n_samples += data.size(0)
            if num_samples is not None and n_samples >= num_samples:
                break

        # Average over samples
        for name in fisher:
            fisher[name] = fisher[name] / n_samples

        return dict(fisher)

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ):
        """Compute loss with EWC penalty.

        Args:
            outputs: Model outputs.
            targets: Target labels.

        Returns:
            Tuple of (loss, metrics).
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(outputs, targets)

        # EWC penalty
        ewc_loss = 0.0
        if self.current_task > 0:
            for task_id in range(self.current_task):
                if task_id in self.fisher_matrices and task_id in self.optimal_params:
                    for name, param in self.model.named_parameters():
                        if name in self.fisher_matrices[task_id]:
                            fisher = self.fisher_matrices[task_id][name]
                            optimal_param = self.optimal_params[task_id][name]
                            ewc_loss += (fisher * (param - optimal_param) ** 2).sum()

        total_loss = ce_loss + self.ewc_lambda * ewc_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'ewc_loss': ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss,
            'total_loss': total_loss.item()
        }

        return total_loss, metrics

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train on a single task with EWC.

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

        # Compute Fisher after training
        self.after_task(task_id, train_loader)

        # Average metrics
        avg_metrics = {
            'ce_loss': sum(m.get('ce_loss', m.get('loss', 0))
                          for m in metrics['epochs']) / n_epochs,
            'accuracy': sum(m['accuracy'] for m in metrics['epochs']) / n_epochs,
            'task_id': task_id
        }

        return avg_metrics
