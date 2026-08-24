"""Base class for incremental learning strategies.

"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class IncrementalStrategy(ABC):
    """Abstract base class for incremental learning strategies.

    All incremental learning methods should inherit from this class
    and implement the required methods.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        device: str = 'cuda',
        **kwargs
    ):
        """Initialize strategy.

        Args:
            model: Neural network model.
            optimizer_fn: Function to create optimizer (e.g., lambda params: Adam(params, lr=0.001)).
            device: Device for training.
            **kwargs: Additional strategy-specific arguments.
        """
        self.model = model
        self.optimizer_fn = optimizer_fn
        self.device = device
        self.current_task = 0
        self.n_classes_so_far = 0

        # Store old model for distillation
        self.old_model: Optional[nn.Module] = None

        # Move model to device
        self.model.to(device)

    @abstractmethod
    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train on a single task.

        Args:
            train_loader: DataLoader for training data.
            task_id: Current task ID.
            n_epochs: Number of training epochs.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of training metrics.
        """
        pass

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Prepare for training on a new task.

        Called before train_task for each task.

        Args:
            task_id: Current task ID.
            n_new_classes: Number of new classes in this task.
        """
        self.current_task = task_id

        # Expand classifier for new classes
        if task_id > 0 and hasattr(self.model, 'expand_classifier'):
            self.model.expand_classifier(n_new_classes)
            self.model.to(self.device)

        # Save old model for distillation
        if task_id > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
            self.old_model.to(self.device)

        self.n_classes_so_far += n_new_classes

    def after_task(self, task_id: int, **kwargs) -> None:
        """Cleanup after training on a task.

        Called after train_task for each task.

        Args:
            task_id: Current task ID.
            **kwargs: Additional arguments.
        """
        pass

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch.

        Args:
            outputs: Model outputs.
            targets: Target labels.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        # Standard cross-entropy loss
        loss = nn.CrossEntropyLoss()(outputs, targets)
        metrics = {'ce_loss': loss.item()}
        return loss, metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss, metrics = self.compute_loss(outputs, targets, **kwargs)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

    def get_state(self) -> Dict[str, Any]:
        """Get strategy state for checkpointing.

        Returns:
            Dictionary containing strategy state.
        """
        return {
            'current_task': self.current_task,
            'n_classes_so_far': self.n_classes_so_far,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy state from checkpoint.

        Args:
            state: State dictionary.
        """
        self.current_task = state.get('current_task', 0)
        self.n_classes_so_far = state.get('n_classes_so_far', 0)
