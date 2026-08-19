"""Federated Learning Client.

Local training for federated learning with incremental learning support.

"""

from typing import Dict, Optional, Any
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from incremental.base_strategy import IncrementalStrategy


class FLClient:
    """Federated Learning Client.

    Handles local training and communication with the server.
    Supports incremental learning strategies.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        strategy: IncrementalStrategy,
        device: str = 'cuda'
    ):
        """Initialize FL Client.

        Args:
            client_id: Unique client identifier.
            model: Local model.
            strategy: Incremental learning strategy.
            device: Device for training.
        """
        self.client_id = client_id
        self.model = model
        self.strategy = strategy
        self.device = device

        # Training statistics
        self.n_samples = 0
        self.n_steps = 0

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get model state for server aggregation.

        Returns:
            Model state dictionary.
        """
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Set model state from server.

        Args:
            state_dict: Model state dictionary.
        """
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def local_train(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform local training.

        Args:
            train_loader: DataLoader for local data.
            n_epochs: Number of local epochs.
            **kwargs: Additional arguments for strategy.

        Returns:
            Dictionary of training metrics.
        """
        # Update sample count
        if hasattr(train_loader, 'dataset'):
            self.n_samples = len(train_loader.dataset)

        # Train using strategy
        metrics = self.strategy.train_task(
            train_loader,
            task_id=self.strategy.current_task,
            n_epochs=n_epochs,
            **kwargs
        )

        # Update step count
        self.n_steps = n_epochs * len(train_loader)

        metrics['client_id'] = self.client_id
        metrics['n_samples'] = self.n_samples
        metrics['n_steps'] = self.n_steps

        return metrics

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Prepare for new task.

        Args:
            task_id: Current task ID.
            n_new_classes: Number of new classes.
        """
        self.strategy.before_task(task_id, n_new_classes)

    def after_task(self, task_id: int, **kwargs) -> None:
        """Cleanup after task.

        Args:
            task_id: Current task ID.
            **kwargs: Additional arguments.
        """
        self.strategy.after_task(task_id, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics.

        Returns:
            Dictionary of client statistics.
        """
        return {
            'client_id': self.client_id,
            'n_samples': self.n_samples,
            'n_steps': self.n_steps,
            'current_task': self.strategy.current_task,
            'n_classes_so_far': self.strategy.n_classes_so_far
        }
