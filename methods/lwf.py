"""
Learning without Forgetting (LwF) for FCIL.
Distills outputs from previous task models to preserve historical classification boundaries.
"""

from typing import Dict, Any, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class LwFMethod(BaseILMethod):
    """
    Learning without Forgetting (Li & Hoiem, TPAMI 2017).
    Applies temperature-scaled Kullback-Leibler divergence on previous class logits.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 1.0):
        super().__init__(name="lwf")
        self.temperature = temperature
        self.alpha = alpha
        self.prev_model: Optional[FCILNet] = None
        self.prev_num_classes: int = 0

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        if task_id > 0:
            self.prev_model = model.clone_model().to(device)
            self.prev_model.eval()
            self.prev_num_classes = self.prev_model.current_classes

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

        distill_loss = torch.tensor(0.0, device=device)
        if task_id > 0 and self.prev_model is not None and self.prev_num_classes > 0:
            with torch.no_grad():
                prev_logits = self.prev_model(x, limit_to_current=True)

            # Distillation on previous classes slice
            p_soft = F.log_softmax(logits[:, :self.prev_num_classes] / self.temperature, dim=1)
            q_soft = F.softmax(prev_logits[:, :self.prev_num_classes] / self.temperature, dim=1)
            distill_loss = F.kl_div(p_soft, q_soft, reduction="batchmean") * (self.temperature ** 2)
            distill_loss = self.alpha * distill_loss

        total_loss = ce_loss + distill_loss
        return total_loss, {
            "ce_loss": float(ce_loss.item()),
            "distill_loss": float(distill_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "prev_num_classes": self.prev_num_classes,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.temperature = state.get("temperature", self.temperature)
        self.alpha = state.get("alpha", self.alpha)
        self.prev_num_classes = state.get("prev_num_classes", self.prev_num_classes)
