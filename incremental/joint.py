"""Joint training strategy (upper bound).

Trains on all data from all tasks simultaneously.
Serves as upper bound for comparison (not truly incremental).

"""

from typing import Dict, List
from torch.utils.data import DataLoader, ConcatDataset

from incremental.base_strategy import IncrementalStrategy


class JointTraining(IncrementalStrategy):
    """Joint training strategy (upper bound).

    Accumulates all training data and trains jointly.
    This is not truly incremental but serves as an upper bound
    for comparing forgetting performance.
    """

    def __init__(self, *args, **kwargs):
        """Initialize JointTraining."""
        super().__init__(*args, **kwargs)
        self.all_train_loaders: List[DataLoader] = []

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Prepare for training on a new task.

        Args:
            task_id: Current task ID.
            n_new_classes: Number of new classes.
        """
        super().before_task(task_id, n_new_classes)

        # For joint training, we don't need old model
        self.old_model = None

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train on accumulated data from all tasks.

        Args:
            train_loader: DataLoader for current task data.
            task_id: Current task ID.
            n_epochs: Number of training epochs.

        Returns:
            Dictionary of training metrics.
        """
        # Accumulate data loaders
        self.all_train_loaders.append(train_loader)

        # Create combined dataset
        # Note: In practice, we should combine datasets, not loaders
        # This is a simplified version
        combined_loader = train_loader
        if len(self.all_train_loaders) > 1:
            # Concatenate datasets from all loaders
            all_datasets = []
            for loader in self.all_train_loaders:
                if hasattr(loader, 'dataset'):
                    all_datasets.append(loader.dataset)

            if all_datasets:
                combined_dataset = ConcatDataset(all_datasets)
                combined_loader = DataLoader(
                    combined_dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    pin_memory=train_loader.pin_memory
                )

        # Create optimizer
        optimizer = self.optimizer_fn(self.model.parameters())

        # Training loop
        metrics = {'epochs': []}
        for epoch in range(n_epochs):
            epoch_metrics = self.train_epoch(combined_loader, optimizer)
            metrics['epochs'].append(epoch_metrics)

        # Average metrics
        avg_loss = sum(m['loss'] for m in metrics['epochs']) / n_epochs
        avg_acc = sum(m['accuracy'] for m in metrics['epochs']) / n_epochs

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'task_id': task_id,
            'n_tasks_trained': len(self.all_train_loaders)
        }
