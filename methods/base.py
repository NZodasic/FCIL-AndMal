"""
Abstract Base Class for Class-Incremental Learning (CIL) Algorithms.
Defines lifecycle hooks for pre-task setup, loss computation, and post-task consolidation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fcil_model import FCILNet


class BaseILMethod(ABC):
    """
    Abstract interface for Continual / Incremental Learning strategies.
    """

    def __init__(self, name: str):
        self.name = name

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """Hook executed before starting training on task_id."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss combining classification objective with IL regularization terms.
        
        Returns:
            (total_loss_tensor, loss_breakdown_dict)
        """
        raise NotImplementedError

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """Hook executed after completing training on task_id (e.g. Fisher computation, herding)."""
        pass

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        """Return trainable method-owned parameters not stored on the model."""
        return ()

    def state_dict(self) -> Dict[str, Any]:
        """Export internal state for checkpointing."""
        return {"name": self.name}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        pass
