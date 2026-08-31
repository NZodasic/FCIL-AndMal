"""Weight-only checkpoint persistence for centralized and federated runs."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.fcil_model import FCILNet


class CheckpointManager:
    """Persist model state dictionaries without serializing model objects."""

    def __init__(self, checkpoint_dir: str, logger: Optional[Any] = None):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_macro_f1 = -1.0
        self.best_task_id = -1

    @staticmethod
    def _model_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def _atomic_save(state: Dict[str, Any], checkpoint_path: Path) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            torch.save(state, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def save_weights_checkpoint(
        self,
        model: nn.Module,
        *,
        task_id: int,
        step_type: str,
        step_id: int,
        global_step: Optional[int] = None,
    ) -> str:
        """Save CPU model weights for one epoch or communication round."""
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
        if step_type not in {"epoch", "round"}:
            raise ValueError("step_type must be 'epoch' or 'round'")
        if step_id <= 0:
            raise ValueError("step_id must be positive")

        task_dir = Path(self.checkpoint_dir) / f"task_{task_id + 1:02d}"
        checkpoint_path = task_dir / f"{step_type}_{step_id:04d}_weights.pt"
        state = {
            "checkpoint_type": "weights_only",
            "model_state_dict": self._model_state_dict_cpu(model),
            "task_id": task_id,
            "step_type": step_type,
            "step_id": step_id,
            "global_step": global_step if global_step is not None else step_id,
            "current_classes": getattr(model, "current_classes", 0),
        }
        self._atomic_save(state, checkpoint_path)

        if self.logger:
            self.logger.info(
                f"Weight checkpoint saved: Task {task_id + 1}, "
                f"{step_type} {step_id} -> {checkpoint_path}"
            )
        return str(checkpoint_path)

    def save_task_checkpoint(
        self,
        task_id: int,
        round_id: int,
        global_model: FCILNet,
        continual_matrix_dict: Dict[str, Any],
        client_states: Dict[int, Dict[str, Any]],
        server_prototype_state: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compatibility wrapper that now writes only model weights."""
        return self.save_weights_checkpoint(
            global_model,
            task_id=task_id,
            step_type="round",
            step_id=round_id,
            global_step=round_id,
        )

    def save_best_model(
        self,
        task_id: int,
        macro_f1: float,
        global_model: FCILNet,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Track the best score and save a weight-only model state."""
        if macro_f1 <= self.best_macro_f1:
            return None

        self.best_macro_f1 = macro_f1
        self.best_task_id = task_id
        best_path = Path(self.checkpoint_dir) / "model_best_weights.pt"
        state = {
            "checkpoint_type": "weights_only",
            "model_state_dict": self._model_state_dict_cpu(global_model),
            "task_id": task_id,
            "best_macro_f1": macro_f1,
            "current_classes": getattr(global_model, "current_classes", 0),
        }
        self._atomic_save(state, best_path)
        if self.logger:
            self.logger.info(
                f"Best weights updated: Task {task_id + 1}, "
                f"Macro-F1 {macro_f1 * 100:.2f}%"
            )
        return str(best_path)

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[FCILNet] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Load a weight checkpoint and optionally restore a model."""
        if checkpoint_path is None:
            candidates = list(Path(self.checkpoint_dir).glob("task_*/*_weights.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No weight checkpoints found in: {self.checkpoint_dir}"
                )
            checkpoint_path = str(
                max(candidates, key=lambda path: path.stat().st_mtime_ns)
            )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            state = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
        except TypeError:
            state = torch.load(checkpoint_path, map_location=device)

        if model is not None:
            ckpt_classes = state.get(
                "current_classes", getattr(model, "current_classes", 0)
            )
            current_classes = getattr(model, "current_classes", 0)
            if current_classes < ckpt_classes and hasattr(model, "expand_classes"):
                model.expand_classes(ckpt_classes - current_classes)
            model_state = state.get("model_state_dict", state.get("model_state"))
            if model_state is None:
                raise KeyError("Checkpoint does not contain model weights")
            model.load_state_dict(model_state)

        if self.logger:
            self.logger.info(
                f"Loaded weight checkpoint: {checkpoint_path} "
                f"(Task {state.get('task_id', -1) + 1})"
            )
        return state
