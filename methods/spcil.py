"""
Self-Paced Class-Incremental Learning (SPCIL).
Implements adaptive hardness-aware sample weighting curriculum for Android malware detection.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class SPCILMethod(BaseILMethod):
    """
    Self-Paced Class-Incremental Learning (SPCIL).
    Applies self-paced regularizer dynamically filtering or down-weighting ambiguous/noisy
    samples early in the incremental task to stabilize decision boundaries.
    """

    def __init__(self, lambda_init: float = 0.5, lambda_step: float = 0.1):
        super().__init__(name="spcil")
        self.pacing_lambda = lambda_init
        self.lambda_step = lambda_step
        self.step_counter = 0

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.step_counter = 0

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
        # Sample-wise cross entropy loss
        per_sample_loss = F.cross_entropy(logits, y, reduction="none")

        # Self-paced weights: v_i = 1 if loss_i < lambda else 0 (or soft sigmoid weighting)
        effective_lambda = self.pacing_lambda + (self.step_counter * self.lambda_step * 0.01)
        weights = (per_sample_loss < effective_lambda).float()

        # Prevent zero gradient collapse
        if weights.sum() == 0:
            weights = torch.ones_like(per_sample_loss)

        weighted_loss = (per_sample_loss * weights).sum() / weights.sum()
        self.step_counter += 1

        return weighted_loss, {
            "ce_loss": float(weighted_loss.item()),
            "pacing_lambda": float(effective_lambda),
            "retained_ratio": float(weights.mean().item()),
            "total_loss": float(weighted_loss.item()),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pacing_lambda": self.pacing_lambda,
            "lambda_step": self.lambda_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.pacing_lambda = state.get("pacing_lambda", self.pacing_lambda)
        self.lambda_step = state.get("lambda_step", self.lambda_step)
