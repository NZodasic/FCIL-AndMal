"""Fine-tune strategy (baseline).

Simple sequential fine-tuning without any forgetting prevention.
Serves as lower bound baseline.

"""

from typing import Dict
from torch.utils.data import DataLoader

from incremental.base_strategy import IncrementalStrategy


class FineTune(IncrementalStrategy):
    """Fine-tune baseline strategy.

    Simply trains sequentially on each task without any mechanism
    to prevent catastrophic forgetting. Serves as the lower bound
    for comparison.
    """

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train on a single task with simple fine-tuning.

        Args:
            train_loader: DataLoader for training data.
            task_id: Current task ID.
            n_epochs: Number of training epochs.

        Returns:
            Dictionary of training metrics.
        """
        # Create optimizer
        optimizer = self.optimizer_fn(self.model.parameters())

        # Training loop
        metrics = {'epochs': []}
        for epoch in range(n_epochs):
            epoch_metrics = self.train_epoch(train_loader, optimizer)
            metrics['epochs'].append(epoch_metrics)

        # Average metrics
        avg_loss = sum(m['loss'] for m in metrics['epochs']) / n_epochs
        avg_acc = sum(m['accuracy'] for m in metrics['epochs']) / n_epochs

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'task_id': task_id
        }
