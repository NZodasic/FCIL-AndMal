"""
Comprehensive Multi-Task Continual Evaluator.
Executes rigorous performance assessments across all historical and current malware families.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TASK_LABEL_MAP, LABEL2ID, ID2LABEL
from models.fcil_model import FCILNet
from data.dataset import TabularMalwareDataset
from utils.metrics import compute_classification_metrics, ContinualEvaluationMatrix


class ContinualEvaluator:
    """
    Evaluates global and local FCIL models against pure held-out test distributions,
    computing Macro-F1, Benign vs Malware metrics, and task-specific forgetting drops.
    """

    def __init__(
        self,
        test_X: np.ndarray,
        test_y: np.ndarray,
        batch_size: int = 512,
        device: torch.device = torch.device("cpu")
    ):
        self.test_X = test_X
        self.test_y = test_y
        self.batch_size = batch_size
        self.device = device
        self.continual_matrix = ContinualEvaluationMatrix(n_tasks=5)

    def evaluate_all_seen_tasks(
        self,
        model: FCILNet,
        current_task_id: int
    ) -> Dict[str, Any]:
        """
        Evaluate model on all classes seen from Task 0 up to current_task_id.
        Updates continual evaluation matrix for forgetting tracking.
        """
        model.eval()
        model.to(self.device)

        # Determine all seen class IDs
        seen_classes = []
        for t in range(current_task_id + 1):
            for lbl in TASK_LABEL_MAP[t]:
                seen_classes.append(LABEL2ID[lbl])

        # Filter test dataset for seen classes
        mask = np.isin(self.test_y, seen_classes)
        sub_X = self.test_X[mask]
        sub_y = self.test_y[mask]

        if len(sub_y) == 0:
            empty_metrics = compute_classification_metrics(
                y_true=np.asarray([], dtype=np.int64),
                y_pred=np.asarray([], dtype=np.int64),
                seen_classes=seen_classes,
                label_names=ID2LABEL,
            )
            return {
                **empty_metrics,
                "per_task_evaluation": {},
                "average_forgetting": 0.0,
                "continual_avg_accuracy": 0.0,
                "continual_avg_macro_f1": 0.0,
                "continual_matrix": self.continual_matrix.get_summary_dict(),
            }

        loader = DataLoader(
            TabularMalwareDataset(sub_X, sub_y),
            batch_size=self.batch_size,
            shuffle=False
        )

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(self.device)
                logits = model(bx, limit_to_current=True)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(by.numpy())

        all_preds_arr = np.array(all_preds)
        all_targets_arr = np.array(all_targets)

        # 1. Global Metrics across all seen classes
        global_metrics = compute_classification_metrics(
            y_true=all_targets_arr,
            y_pred=all_preds_arr,
            seen_classes=seen_classes,
            label_names=ID2LABEL
        )

        # 2. Per-Task Slices to update Continual Evaluation Matrix R[current_task_id, evaluated_task]
        per_task_results = {}
        for t in range(current_task_id + 1):
            t_classes = [LABEL2ID[lbl] for lbl in TASK_LABEL_MAP[t]]
            t_mask = np.isin(all_targets_arr, t_classes)
            if np.sum(t_mask) > 0:
                t_targets = all_targets_arr[t_mask]
                t_preds = all_preds_arr[t_mask]
                t_metrics = compute_classification_metrics(
                    y_true=t_targets,
                    y_pred=t_preds,
                    seen_classes=t_classes,
                    label_names=ID2LABEL
                )
                self.continual_matrix.update(
                    current_task=current_task_id,
                    evaluated_task=t,
                    accuracy=t_metrics["accuracy"],
                    macro_f1=t_metrics["macro_f1"]
                )
                per_task_results[f"task_{t}"] = t_metrics

        # Compute running continual metrics
        avg_forgetting = self.continual_matrix.get_forgetting(current_task_id)
        avg_acc = self.continual_matrix.get_average_accuracy(current_task_id)
        avg_f1 = self.continual_matrix.get_average_f1(current_task_id)

        return {
            **global_metrics,
            "per_task_evaluation": per_task_results,
            "average_forgetting": avg_forgetting,
            "continual_avg_accuracy": avg_acc,
            "continual_avg_macro_f1": avg_f1,
            "continual_matrix": self.continual_matrix.get_summary_dict(),
        }


class Evaluator:
    """Model evaluator for incremental learning."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        n_classes: int = 15
    ):
        self.model = model
        self.device = device
        self.n_classes = n_classes

    def evaluate(
        self,
        test_loader: DataLoader,
        task_id: Optional[int] = None
    ) -> Dict[str, Any]:
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                targets = targets.cpu().numpy()
                all_predictions.extend(predictions.tolist())
                all_targets.extend(targets.tolist())

        current_task = task_id if task_id is not None else 4
        seen_classes = []
        for seen_task in range(current_task + 1):
            seen_classes.extend(LABEL2ID[label] for label in TASK_LABEL_MAP[seen_task])
        return compute_classification_metrics(
            np.asarray(all_targets),
            np.asarray(all_predictions),
            seen_classes,
            ID2LABEL,
        )
