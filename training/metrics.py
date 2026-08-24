"""Evaluation metrics for FCIL experiments.

Implements standard metrics for incremental learning evaluation.

"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class MetricsTracker:
    """Track and compute evaluation metrics."""

    def __init__(self, n_classes: int):
        """Initialize metrics tracker.

        Args:
            n_classes: Total number of classes.
        """
        self.n_classes = n_classes
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.predictions: List[int] = []
        self.targets: List[int] = []
        self.task_accuracies: Dict[int, List[float]] = {}

    def update(self, predictions: np.ndarray, targets: np.ndarray, task_id: int) -> None:
        """Update metrics with batch predictions.

        Args:
            predictions: Predicted labels.
            targets: True labels.
            task_id: Current task ID.
        """
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())

        # Track per-task accuracy
        if task_id not in self.task_accuracies:
            self.task_accuracies[task_id] = []

        acc = accuracy_score(targets, predictions)
        self.task_accuracies[task_id].append(acc)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of computed metrics.
        """
        if not self.predictions:
            return {}

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        accuracy = accuracy_score(targets, preds)

        aggregate_metrics = {}
        for average in ('macro', 'micro', 'weighted'):
            precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
                targets, preds, average=average, zero_division=0
            )
            aggregate_metrics[f'precision_{average}'] = precision_avg * 100
            aggregate_metrics[f'recall_{average}'] = recall_avg * 100
            aggregate_metrics[f'f1_{average}'] = f1_avg * 100

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )

        metrics = {
            'accuracy': accuracy * 100,
            **aggregate_metrics,
            'macro_f1': aggregate_metrics['f1_macro'],
            'mean_precision': np.mean(precision) * 100,
            'mean_recall': np.mean(recall) * 100,
            'confusion_matrix': confusion_matrix(targets, preds).tolist(),
        }

        # Per-class F1
        for i in range(min(self.n_classes, len(f1))):
            metrics[f'f1_class_{i}'] = f1[i] * 100

        return metrics

    def compute_forgetting(
        self,
        task_performances: Dict[int, float]
    ) -> Dict[str, float]:
        """Compute forgetting metrics.

        Args:
            task_performances: Dict mapping task_id to performance (accuracy/F1).

        Returns:
            Forgetting metrics.
        """
        if len(task_performances) < 2:
            return {'forgetting': 0.0, 'avg_forgetting': 0.0}

        # Forgetting for each task (except current)
        forgetting_per_task = {}

        for task_id in sorted(task_performances.keys())[:-1]:  # Exclude current task
            # Find best performance for this task
            best_perf = task_performances[task_id]

            # Find current performance (last measurement)
            current_perf = task_performances[task_id]

            # Forgetting = best - current
            forgetting = max(0, best_perf - current_perf)
            forgetting_per_task[task_id] = forgetting

        avg_forgetting = np.mean(list(forgetting_per_task.values()))

        return {
            'forgetting_per_task': forgetting_per_task,
            'avg_forgetting': avg_forgetting
        }


def compute_backward_transfer(
    accuracies: List[float]
) -> float:
    """Compute Backward Transfer (BWT) metric.

    BWT measures the influence of learning new tasks on previous tasks.
    Negative BWT indicates forgetting.

    Args:
        accuracies: List of final accuracies after each task.

    Returns:
        Backward transfer score.
    """
    if len(accuracies) < 2:
        return 0.0

    # BWT = average_{i < j} (acc_j,i - acc_i,i)
    # Simplified: difference between last and first accuracy
    bwt = accuracies[-1] - accuracies[0]

    return bwt


def compute_forward_transfer(
    accuracies: List[float],
    baseline_accuracies: List[float]
) -> float:
    """Compute Forward Transfer (FWT) metric.

    FWT measures the benefit of learning previous tasks on new tasks.

    Args:
        accuracies: List of accuracies on new tasks.
        baseline_accuracies: List of baseline (random init) accuracies.

    Returns:
        Forward transfer score.
    """
    if len(accuracies) != len(baseline_accuracies):
        return 0.0

    fwt = np.mean([
        acc - baseline
        for acc, baseline in zip(accuracies, baseline_accuracies)
    ])

    return fwt


def compute_average_incremental_accuracy(
    task_accuracies: List[float]
) -> float:
    """Compute Average Incremental Accuracy (AIA).

    Average accuracy across all task transitions.

    Args:
        task_accuracies: List of accuracies after each task.

    Returns:
        Average incremental accuracy.
    """
    return np.mean(task_accuracies)
