"""
Academic Checkpointing and Model State Serialization Suite.
Handles full experiment checkpointing, task snapshots, best-metric tracking,
and seamless training resumption across server and client states.
"""

import os
import shutil
from typing import Dict, Any, Optional
import torch

from models.fcil_model import FCILNet
from utils.logger import ExperimentLogger


class CheckpointManager:
    """
    Manages robust serialization and recovery of global model states, client continual states,
    evaluation matrices, and experiment configuration.
    """

    def __init__(self, checkpoint_dir: str, logger: Optional[Any] = None):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_macro_f1 = -1.0
        self.best_task_id = -1

    def save_task_checkpoint(
        self,
        task_id: int,
        round_id: int,
        global_model: FCILNet,
        continual_matrix_dict: Dict[str, Any],
        client_states: Dict[int, Dict[str, Any]],
        server_prototype_state: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete snapshot at the boundary of a completed task.
        """
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_task_{task_id}.pt")
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")

        state = {
            "task_id": task_id,
            "round_id": round_id,
            "current_classes": getattr(global_model, "current_classes", 0),
            "model_state": global_model.get_state_dict_cpu() if hasattr(global_model, "get_state_dict_cpu") else global_model.state_dict(),
            "model_config": getattr(global_model.config, "__dict__", {}),
            "continual_matrix": continual_matrix_dict,
            "client_states": client_states,
            "server_prototypes": server_prototype_state,
            "extra_meta": extra_meta or {},
        }

        torch.save(state, ckpt_path)
        torch.save(state, latest_path)

        if self.logger:
            self.logger.info(f"💾 Checkpoint saved for Task {task_id + 1} at: {ckpt_path}")
        return ckpt_path

    def save_best_model(
        self,
        task_id: int,
        macro_f1: float,
        global_model: FCILNet,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track and save the best model based on Macro-F1 score."""
        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.best_task_id = task_id
            best_path = os.path.join(self.checkpoint_dir, "model_best.pt")

            state = {
                "task_id": task_id,
                "best_macro_f1": macro_f1,
                "current_classes": getattr(global_model, "current_classes", 0),
                "model_state": global_model.get_state_dict_cpu() if hasattr(global_model, "get_state_dict_cpu") else global_model.state_dict(),
                "model_config": getattr(global_model.config, "__dict__", {}),
                "extra_meta": extra_meta or {},
            }
            torch.save(state, best_path)
            if self.logger:
                self.logger.info(f"🏆 New best model saved! (Task {task_id + 1}, Macro-F1: {macro_f1 * 100:.2f}%)")

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[FCILNet] = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore model parameters if provided.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device)

        if model is not None:
            ckpt_classes = state.get("current_classes", getattr(model, "current_classes", 0))
            current_classes = getattr(model, "current_classes", 0)
            if current_classes < ckpt_classes and hasattr(model, "expand_classes"):
                model.expand_classes(ckpt_classes - current_classes)
            model.load_state_dict(state["model_state"])

        if self.logger:
            self.logger.info(f"📂 Loaded checkpoint from: {checkpoint_path} (Task: {state.get('task_id', -1) + 1})")

        return state
