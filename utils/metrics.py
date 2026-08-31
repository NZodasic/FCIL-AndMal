"""
Academic Evaluation Metrics for Federated Class-Incremental Learning.
Implements Macro-F1, Catastrophic Forgetting Matrix, Backward Transfer (BWT),
Forward Transfer (FWT), Per-Class Metrics, and Benign-vs-Malware Disaggregation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import torch


CLASSIFICATION_METRIC_KEYS: Tuple[str, ...] = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)

CLASSIFICATION_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Macro Precision",
    "recall_macro": "Macro Recall",
    "f1_macro": "Macro F1",
    "precision_micro": "Micro Precision",
    "recall_micro": "Micro Recall",
    "f1_micro": "Micro F1",
    "precision_weighted": "Weighted Precision",
    "recall_weighted": "Weighted Recall",
    "f1_weighted": "Weighted F1",
}


def primary_classification_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Return the ten required test metrics in their reporting order."""
    missing = [key for key in CLASSIFICATION_METRIC_KEYS if key not in metrics]
    if missing:
        raise KeyError(f"Evaluation result is missing metrics: {', '.join(missing)}")
    return {key: float(metrics[key]) for key in CLASSIFICATION_METRIC_KEYS}


def format_classification_metrics(metrics: Dict[str, Any]) -> str:
    """Format the required metrics as percentages for console/file logs."""
    primary_metrics = primary_classification_metrics(metrics)
    return " | ".join(
        f"{CLASSIFICATION_METRIC_LABELS[key]}: {value * 100:.2f}%"
        for key, value in primary_metrics.items()
    )


def format_confusion_matrix(
    metrics: Dict[str, Any],
    label_names: Optional[Dict[int, str]] = None,
) -> str:
    """Format a labeled confusion matrix for task-final test logs."""
    matrix = metrics.get("confusion_matrix", [])
    labels = list(metrics.get("confusion_matrix_labels", []))
    if not matrix:
        return "Confusion Matrix: unavailable (no evaluated samples)"

    matrix_array = np.asarray(matrix, dtype=np.int64)
    if matrix_array.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix dimensions must match its labels")

    displayed_labels = [
        label_names.get(int(label), str(label)) if label_names else str(label)
        for label in labels
    ]
    rows = "\n".join(
        f"  {label}: {row.tolist()}"
        for label, row in zip(displayed_labels, matrix_array)
    )
    return (
        "Confusion Matrix (rows=true, columns=predicted)\n"
        f"  labels: {displayed_labels}\n{rows}"
    )


def compute_fscil_session_metrics(
    task_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the session-level metrics reported by the MalFSCIL paper."""
    accuracies = [float(result["accuracy"]) for result in task_results]
    if not accuracies:
        return {
            "session_accuracies": [],
            "performance_degradation": 0.0,
            "average_accuracy": 0.0,
            "area_under_time": 0.0,
        }

    performance_degradation = accuracies[0] - accuracies[-1]
    average_accuracy = float(np.mean(accuracies))
    if len(accuracies) == 1:
        area_under_time = accuracies[0]
    else:
        trapezoids = [
            (left + right) / 2.0
            for left, right in zip(accuracies[:-1], accuracies[1:])
        ]
        area_under_time = float(np.mean(trapezoids))
    return {
        "session_accuracies": accuracies,
        "performance_degradation": performance_degradation,
        "average_accuracy": average_accuracy,
        "area_under_time": area_under_time,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seen_classes: List[int],
    label_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Compute rigorous multi-class metrics focused on Macro-F1 and per-family performance.
    
    Args:
        y_true: Ground truth integer class labels.
        y_pred: Predicted integer class labels.
        seen_classes: List of class IDs seen up to the current task.
        label_names: Mapping from class ID to string name.
        
    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    empty_metrics = {
        "accuracy": 0.0,
        "precision_macro": 0.0,
        "precision_micro": 0.0,
        "precision_weighted": 0.0,
        "recall_macro": 0.0,
        "recall_micro": 0.0,
        "recall_weighted": 0.0,
        "f1_macro": 0.0,
        "f1_micro": 0.0,
        "f1_weighted": 0.0,
    }
    if len(y_true) == 0:
        return {
            **empty_metrics,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "confusion_matrix": [],
            "confusion_matrix_labels": list(seen_classes),
        }

    # Restrict evaluation to seen classes
    mask = np.isin(y_true, seen_classes)
    y_true_seen = y_true[mask]
    y_pred_seen = y_pred[mask]

    if len(y_true_seen) == 0:
        return {
            **empty_metrics,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "confusion_matrix": [],
            "confusion_matrix_labels": list(seen_classes),
        }

    acc = float(accuracy_score(y_true_seen, y_pred_seen))

    # Calculate precision, recall, f1
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_seen, y_pred_seen, labels=seen_classes, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_seen, y_pred_seen, labels=seen_classes, average="micro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_seen, y_pred_seen, labels=seen_classes, average="weighted", zero_division=0
    )

    # Per-class breakdown
    prec_per_cls, rec_per_cls, f1_per_cls, support = precision_recall_fscore_support(
        y_true_seen, y_pred_seen, labels=seen_classes, average=None, zero_division=0
    )

    per_class_f1: Dict[str, float] = {}
    per_class_acc: Dict[str, float] = {}
    malware_f1_list: List[float] = []
    benign_f1: Optional[float] = None

    for idx, cls_id in enumerate(seen_classes):
        cls_name = label_names.get(cls_id, f"Class_{cls_id}") if label_names else f"Class_{cls_id}"
        f1_val = float(f1_per_cls[idx])
        per_class_f1[cls_name] = f1_val

        # Per-class accuracy
        cls_mask = y_true_seen == cls_id
        if np.sum(cls_mask) > 0:
            per_class_acc[cls_name] = float(np.mean(y_pred_seen[cls_mask] == cls_id))
        else:
            per_class_acc[cls_name] = 0.0

        if cls_name == "Benign" or cls_id == 0:
            benign_f1 = f1_val
        else:
            malware_f1_list.append(f1_val)

    f1_malware_avg = float(np.mean(malware_f1_list)) if len(malware_f1_list) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision_macro": float(prec_macro),
        "precision_micro": float(prec_micro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_micro": float(rec_micro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        # Compatibility aliases used by existing continual-learning reports.
        "macro_f1": float(f1_macro),
        "micro_f1": float(f1_micro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "f1_benign": benign_f1 if benign_f1 is not None else 0.0,
        "f1_malware_avg": f1_malware_avg,
        "per_class_f1": per_class_f1,
        "per_class_accuracy": per_class_acc,
        "total_evaluated_samples": int(len(y_true_seen)),
        "confusion_matrix": confusion_matrix(
            y_true_seen, y_pred_seen, labels=seen_classes
        ).tolist(),
        "confusion_matrix_labels": list(seen_classes),
    }


class ContinualEvaluationMatrix:
    """
    Maintains the incremental task evaluation matrix R where R[i, j] is the performance
    on Task j test set after completing Task i training.
    
    Computes:
    - Average Accuracy at step t: A_t = 1/(t+1) * sum_{j=0}^t R[t, j]
    - Average Forgetting at step t: F_t = 1/t * sum_{j=0}^{t-1} (R[j, j] - R[t, j])
    - Backward Transfer (BWT): BWT = 1/(T-1) * sum_{j=0}^{T-2} (R[T-1, j] - R[j, j])
    """

    def __init__(self, n_tasks: int = 5):
        self.n_tasks = n_tasks
        # R_acc[i, j]: Accuracy on task j after task i
        self.R_acc = np.full((n_tasks, n_tasks), np.nan)
        # R_f1[i, j]: Macro-F1 on task j after task i
        self.R_f1 = np.full((n_tasks, n_tasks), np.nan)

    def update(self, current_task: int, evaluated_task: int, accuracy: float, macro_f1: float) -> None:
        """Record performance on a specific task's subset after current_task."""
        self.R_acc[current_task, evaluated_task] = accuracy
        self.R_f1[current_task, evaluated_task] = macro_f1

    def get_average_accuracy(self, current_task: int) -> float:
        """Calculate mean accuracy over all tasks seen up to current_task."""
        vals = self.R_acc[current_task, :current_task + 1]
        valid_vals = vals[~np.isnan(vals)]
        return float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

    def get_average_f1(self, current_task: int) -> float:
        """Calculate mean Macro-F1 over all tasks seen up to current_task."""
        vals = self.R_f1[current_task, :current_task + 1]
        valid_vals = vals[~np.isnan(vals)]
        return float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

    def get_forgetting(self, current_task: int) -> float:
        """
        Calculate Average Forgetting at current_task:
        F_t = (1 / current_task) * sum_{j=0}^{current_task-1} (R[j, j] - R[current_task, j])
        """
        if current_task == 0:
            return 0.0
        forgetting_drops = []
        for j in range(current_task):
            peak = self.R_acc[j, j]
            current = self.R_acc[current_task, j]
            if not np.isnan(peak) and not np.isnan(current):
                forgetting_drops.append(max(0.0, peak - current))
        return float(np.mean(forgetting_drops)) if forgetting_drops else 0.0

    def get_backward_transfer(self) -> float:
        """
        Compute final Backward Transfer (BWT) across all tasks.
        Negative values indicate catastrophic forgetting.
        """
        last_task = self.n_tasks - 1
        bwt_list = []
        for j in range(last_task):
            initial = self.R_acc[j, j]
            final = self.R_acc[last_task, j]
            if not np.isnan(initial) and not np.isnan(final):
                bwt_list.append(final - initial)
        return float(np.mean(bwt_list)) if bwt_list else 0.0

    def get_summary_dict(self) -> Dict[str, Any]:
        """Export matrix and summary metrics as clean dictionary."""
        return {
            "R_acc": np.where(np.isnan(self.R_acc), None, self.R_acc).tolist(),
            "R_f1": np.where(np.isnan(self.R_f1), None, self.R_f1).tolist(),
            "final_average_accuracy": self.get_average_accuracy(self.n_tasks - 1),
            "final_average_f1": self.get_average_f1(self.n_tasks - 1),
            "final_forgetting": self.get_forgetting(self.n_tasks - 1),
            "backward_transfer": self.get_backward_transfer(),
        }
