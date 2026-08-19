"""Experience Replay strategy.

Prevents forgetting by storing and replaying samples from previous tasks.

Reference:
    Rebuffi et al. "iCaRL: Incremental Classifier and Representation Learning"
    (CVPR 2017)

"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from incremental.base_strategy import IncrementalStrategy


class Replay(IncrementalStrategy):
    """Experience Replay strategy.

    Stores a subset of samples from previous tasks and replays them
    during training on new tasks. Uses herding selection to choose
    representative samples.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_fn,
        device: str = 'cuda',
        buffer_size_per_class: int = 20,
        selection_method: str = 'herding',  # 'random', 'herding'
        **kwargs
    ):
        """Initialize Replay.

        Args:
            model: Neural network model.
            optimizer_fn: Optimizer function.
            device: Device for training.
            buffer_size_per_class: Number of samples to store per class.
            selection_method: Method for selecting replay samples.
            **kwargs: Additional arguments.
        """
        super().__init__(model, optimizer_fn, device, **kwargs)
        self.buffer_size_per_class = buffer_size_per_class
        self.selection_method = selection_method

        # Replay buffer: class_id -> list of (data, target) tuples
        self.replay_buffer: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Prepare for new task.

        Args:
            task_id: Current task ID.
            n_new_classes: Number of new classes.
        """
        super().before_task(task_id, n_new_classes)

        # Clear old model reference to save memory
        if self.old_model is not None:
            del self.old_model
            self.old_model = None

    def after_task(self, task_id: int, train_loader: DataLoader, **kwargs) -> None:
        """Update replay buffer after training.

        Args:
            task_id: Current task ID.
            train_loader: DataLoader with current task data.
        """
        # Collect all data from current task
        all_data = []
        all_targets = []

        for data, targets in train_loader:
            all_data.append(data)
            all_targets.append(targets)

        all_data = torch.cat(all_data, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Update buffer for each class in current task
        unique_classes = torch.unique(all_targets).tolist()

        for cls in unique_classes:
            cls_mask = all_targets == cls
            cls_data = all_data[cls_mask]

            if self.selection_method == 'random':
                # Random selection
                indices = torch.randperm(len(cls_data))[:self.buffer_size_per_class]
                selected = cls_data[indices]

            elif self.selection_method == 'herding':
                # Herding selection (class mean closest)
                selected = self._herding_selection(cls_data)

            else:
                raise ValueError(f"Unknown selection method: {self.selection_method}")

            self.replay_buffer[cls] = [selected]

    def _herding_selection(self, data: torch.Tensor) -> torch.Tensor:
        """Select samples using herding (closest to class mean).

        Args:
            data: Tensor of samples for a class.

        Returns:
            Selected samples.
        """
        self.model.eval()

        with torch.no_grad():
            # Extract features
            if hasattr(self.model, 'get_features'):
                features = self.model.get_features(data.to(self.device))
            else:
                # Fallback: use data directly
                features = data.to(self.device)

            # Compute class mean
            class_mean = features.mean(dim=0)

            # Select samples closest to mean
            distances = torch.cdist(features, class_mean.unsqueeze(0)).squeeze()
            _, indices = torch.topk(distances, min(self.buffer_size_per_class, len(distances)), largest=False)

        return data[indices].cpu()

    def get_replay_loader(
        self,
        batch_size: int,
        n_batches: int = 1
    ) -> Optional[DataLoader]:
        """Create DataLoader from replay buffer.

        Args:
            batch_size: Batch size.
            n_batches: Number of batches to return.

        Returns:
            DataLoader or None if buffer is empty.
        """
        if not self.replay_buffer:
            return None

        # Concatenate all buffer samples
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

        # Create dataset
        dataset = TensorDataset(replay_data, replay_targets)

        # Return DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> Dict[str, float]:
        """Train for one epoch with replay.

        Args:
            train_loader: DataLoader for current task data.
            optimizer: Optimizer.

        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        replay_count = 0

        # Get replay loader
        replay_loader = self.get_replay_loader(train_loader.batch_size)
        replay_iter = iter(replay_loader) if replay_loader is not None else None

        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            # Mix with replay samples if available
            if replay_iter is not None:
                try:
                    replay_data, replay_targets = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_data, replay_targets = next(replay_iter)

                replay_data = replay_data.to(self.device)
                replay_targets = replay_targets.to(self.device)

                # Concatenate
                data = torch.cat([data, replay_data], dim=0)
                targets = torch.cat([targets, replay_targets], dim=0)
                replay_count += len(replay_data)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total,
            'replay_samples': replay_count
        }

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Train on a single task with replay.

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

        # Update replay buffer
        self.after_task(task_id, train_loader)

        # Average metrics
        avg_metrics = {
            'loss': sum(m['loss'] for m in metrics['epochs']) / n_epochs,
            'accuracy': sum(m['accuracy'] for m in metrics['epochs']) / n_epochs,
            'task_id': task_id,
            'buffer_size': sum(len(v) for v in self.replay_buffer.values())
        }

        return avg_metrics
