"""
Elastic Weight Consolidation (EWC) for FCIL.
Regularizes weight changes based on the diagonal of the empirical Fisher Information Matrix.
"""

from typing import Dict, Any, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class EWCMethod(BaseILMethod):
    """
    Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017).
    Calculates Fisher Information Matrix diagonal to penalize changes to parameters
    critical for previous tasks.
    """

    def __init__(self, ewc_lambda: float = 5000.0):
        super().__init__(name="ewc")
        self.ewc_lambda = ewc_lambda
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

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

        ewc_loss = torch.tensor(0.0, device=device)
        if task_id > 0 and self.fisher_dict:
            for name, param in model.named_parameters():
                if name in self.fisher_dict and name in self.optimal_params:
                    f = self.fisher_dict[name].to(device)
                    p_star = self.optimal_params[name].to(device)
                    # Handle size changes in classifier weights
                    min_rows = min(param.shape[0], p_star.shape[0])
                    if param.dim() > 1:
                        diff = param[:min_rows] - p_star[:min_rows]
                        ewc_loss += (f[:min_rows] * (diff ** 2)).sum()
                    else:
                        diff = param[:min_rows] - p_star[:min_rows]
                        ewc_loss += (f[:min_rows] * (diff ** 2)).sum()

            ewc_loss = (self.ewc_lambda / 2.0) * ewc_loss

        total_loss = ce_loss + ewc_loss
        return total_loss, {
            "ce_loss": float(ce_loss.item()),
            "ewc_loss": float(ewc_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """
        Estimate empirical Fisher Information Matrix across the current task's dataset.
        """
        if train_loader is None or len(train_loader) == 0:
            return

        model.eval()
        fisher_accum = {name: torch.zeros_like(p.data) for name, p in model.named_parameters()}
        total_samples = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            model.zero_grad()
            logits = model(bx, limit_to_current=True)
            log_probs = F.log_softmax(logits, dim=1)
            # Sample from model's predictive distribution
            n_samples = bx.size(0)
            total_samples += n_samples

            for i in range(n_samples):
                label = torch.multinomial(log_probs[i].exp(), 1).squeeze()
                loss = -log_probs[i, label]
                loss.backward(retain_graph=(i < n_samples - 1))

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_accum[name] += param.grad.data ** 2
            model.zero_grad()

        # Normalize Fisher
        if total_samples > 0:
            for name in fisher_accum:
                fisher_accum[name] /= total_samples

        # Accumulate with previous Fisher matrices
        if not self.fisher_dict:
            self.fisher_dict = {k: v.cpu().clone() for k, v in fisher_accum.items()}
        else:
            for name in fisher_accum:
                if name in self.fisher_dict:
                    old_f = self.fisher_dict[name]
                    new_f = fisher_accum[name].cpu()
                    min_r = min(old_f.shape[0], new_f.shape[0])
                    old_f[:min_r] += new_f[:min_r]
                    self.fisher_dict[name] = old_f

        # Save optimal parameters
        self.optimal_params = {name: p.data.cpu().clone() for name, p in model.named_parameters()}
        model.train()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ewc_lambda": self.ewc_lambda,
            "fisher_dict": {k: v.cpu() for k, v in self.fisher_dict.items()},
            "optimal_params": {k: v.cpu() for k, v in self.optimal_params.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ewc_lambda = state.get("ewc_lambda", self.ewc_lambda)
        self.fisher_dict = state.get("fisher_dict", {})
        self.optimal_params = state.get("optimal_params", {})
