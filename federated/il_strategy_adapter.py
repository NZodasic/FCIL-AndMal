"""
Adapter: bridges methods.BaseILMethod → FLClient-compatible strategy interface.

FLClient expects a `strategy` object with:
  - strategy.current_task (int)
  - strategy.n_classes_so_far (int)
  - strategy.before_task(task_id, n_new_classes)
  - strategy.after_task(task_id, **kwargs)
  - strategy.train_task(train_loader, task_id, n_epochs, **kwargs) → Dict

This adapter wraps any BaseILMethod so it can be used in FLClient without
modifying the existing client code.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet


class ILMethodStrategyAdapter:
    """Adapter wrapping BaseILMethod as an FLClient-compatible strategy."""

    def __init__(
        self,
        model: FCILNet,
        il_method: BaseILMethod,
        lr: float = 0.001,
        device: str = "cpu",
        classes_per_task: int = 3,
    ):
        self.model = model
        self.il_method = il_method
        self.lr = lr
        self.device_str = device
        self.device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        self.classes_per_task = classes_per_task

        # FLClient-required attributes
        self.current_task: int = 0
        self.n_classes_so_far: int = classes_per_task  # Task 0 starts with first 3

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Expand model head and call IL pre-task hook."""
        self.current_task = task_id
        target_classes = (task_id + 1) * self.classes_per_task
        if self.model.current_classes < target_classes:
            self.model.expand_classes(num_new_classes=target_classes - self.model.current_classes)
        self.model.to(self.device)
        self.il_method.before_task(
            task_id=task_id,
            model=self.model,
            device=self.device,
        )
        self.n_classes_so_far = target_classes

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run local training for one FL round (n_epochs epochs)."""
        self.model.to(self.device)
        self.model.train()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        trainable.extend(self.il_method.auxiliary_parameters())
        optimizer = optim.Adam(trainable, lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_batches = 0
        for _ in range(n_epochs):
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss, _ = self.il_method.compute_loss(
                    model=self.model,
                    x=bx,
                    y=by,
                    criterion=criterion,
                    task_id=task_id,
                    device=self.device,
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        return {"loss": avg_loss, "task_id": task_id}

    def after_task(self, task_id: int, **kwargs) -> None:
        """Call IL post-task hook."""
        train_loader = kwargs.get("train_loader", None)
        self.il_method.after_task(
            task_id=task_id,
            model=self.model,
            train_loader=train_loader,
            device=self.device,
        )

    # Prototype support for MALFSIL
    def get_prototypes(self) -> Dict[int, torch.Tensor]:
        if hasattr(self.il_method, "global_prototypes"):
            return self.il_method.global_prototypes
        return {}

    def set_prototypes(self, prototypes: Dict[int, torch.Tensor]) -> None:
        if hasattr(self.il_method, "set_global_prototypes"):
            self.il_method.set_global_prototypes(prototypes)
