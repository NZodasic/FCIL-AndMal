"""
Sequential Fine-Tuning (Lower Bound Baseline) and Joint Cumulative (Upper Bound).
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class FineTuneMethod(BaseILMethod):
    """
    Standard sequential fine-tuning baseline without anti-forgetting mechanisms.
    Serves as the empirical lower bound for continual learning performance.
    """

    def __init__(self):
        super().__init__(name="finetune")

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = model(x, limit_to_current=True)
        ce_loss = criterion(logits, y)
        return ce_loss, {"ce_loss": float(ce_loss.item()), "total_loss": float(ce_loss.item())}


class JointCumulativeMethod(BaseILMethod):
    """
    Joint / Cumulative training on all data seen up to the current task.
    Serves as the empirical upper bound for continual learning performance.
    """

    def __init__(self):
        super().__init__(name="joint")

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = model(x, limit_to_current=True)
        ce_loss = criterion(logits, y)
        return ce_loss, {"ce_loss": float(ce_loss.item()), "total_loss": float(ce_loss.item())}
