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
from utils.device import resolve_device


class ILMethodStrategyAdapter:
    """Adapter wrapping BaseILMethod as an FLClient-compatible strategy."""

    def __init__(
        self,
        model: FCILNet,
        il_method: BaseILMethod,
        lr: float = 0.001,
        device: Any = "auto",
        classes_per_task: int = 3,
    ):
        self.model = model
        self.il_method = il_method
        self.lr = lr
        self.device = resolve_device(device)
        self.classes_per_task = classes_per_task

        # FLClient-required attributes
        self.current_task: int = 0
        self.n_classes_so_far: int = classes_per_task  # Task 0 starts with first 3
        self._last_train_loader: Optional[DataLoader] = None

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

    def set_adaptive_lambda(self, adaptive_lambda: float) -> None:
        """Forward per-client adaptive lambda_i^{(t)} to underlying IL method (FL-CL-IDS)."""
        if hasattr(self.il_method, "set_adaptive_lambda"):
            self.il_method.set_adaptive_lambda(adaptive_lambda)

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        n_epochs: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run local training for one FL round (n_epochs epochs)."""
        self._last_train_loader = train_loader
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
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        # Apply post-round classifier weight alignment if supported by IL method
        if task_id > 0 and hasattr(self.il_method, "align_classifier_weights"):
            self.il_method.align_classifier_weights(self.model, task_id)

        avg_loss = total_loss / max(1, total_batches)
        return {"loss": avg_loss, "task_id": task_id}

    def after_task(self, task_id: int, **kwargs) -> None:
        """Call IL post-task hook."""
        train_loader = kwargs.get("train_loader", self._last_train_loader)
        self.il_method.after_task(
            task_id=task_id,
            model=self.model,
            train_loader=train_loader,
            device=self.device,
        )

    # Prototype support for MALFSIL
    def get_prototypes(self) -> Dict[int, Any]:
        if hasattr(self.il_method, "compute_local_prototypes") and self._last_train_loader is not None:
            return self.il_method.compute_local_prototypes(
                self.model, self._last_train_loader, self.device
            )
        if hasattr(self.il_method, "global_prototypes"):
            return self.il_method.global_prototypes
        return {}

    def set_prototypes(self, prototypes: Dict[int, torch.Tensor]) -> None:
        if hasattr(self.il_method, "set_global_prototypes"):
            self.il_method.set_global_prototypes(prototypes)
