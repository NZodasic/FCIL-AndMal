"""Checkpoint management for experiments.

Provides comprehensive checkpoint saving and loading.

"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages model checkpoints and experiment state.

    Supports hierarchical checkpoint structure:
    - Task-level checkpoints
    - Round-level checkpoints (for FL)
    - Best model tracking
    - Resume capability
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        keep_last_n: int = 3
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints.
            experiment_name: Name of the experiment.
            keep_last_n: Number of recent checkpoints to keep.
        """
        self.base_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.keep_last_n = keep_last_n

        # Create directory structure
        self.exp_dir = self.base_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint history
        self.checkpoint_history: list = []

    def get_task_dir(self, task_id: int) -> Path:
        """Get checkpoint directory for a task.

        Args:
            task_id: Task ID.

        Returns:
            Path to task checkpoint directory.
        """
        task_dir = self.exp_dir / f"task_{task_id}"
        task_dir.mkdir(exist_ok=True)
        return task_dir

    def get_round_dir(self, task_id: int, round_id: int) -> Path:
        """Get checkpoint directory for a specific round.

        Args:
            task_id: Task ID.
            round_id: Round ID.

        Returns:
            Path to round checkpoint directory.
        """
        task_dir = self.get_task_dir(task_id)
        round_dir = task_dir / f"round_{round_id}"
        round_dir.mkdir(exist_ok=True)
        return round_dir

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        task_id: int,
        round_id: Optional[int] = None,
        is_best: bool = False,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a checkpoint.

        Args:
            state_dict: State dictionary to save.
            task_id: Current task ID.
            round_id: Current round ID (None for task-level checkpoint).
            is_best: Whether this is the best model so far.
            metadata: Additional metadata to save.

        Returns:
            Path to saved checkpoint.
        """
        if round_id is not None:
            save_dir = self.get_round_dir(task_id, round_id)
            filename = "checkpoint.pt"
        else:
            save_dir = self.get_task_dir(task_id)
            filename = "task_checkpoint.pt"

        # Save main checkpoint
        checkpoint_path = save_dir / filename
        torch.save(state_dict, checkpoint_path)

        # Save metadata
        if metadata:
            meta_path = save_dir / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Save as best if specified
        if is_best:
            best_path = self.exp_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)

            best_meta_path = self.exp_dir / "best_metadata.json"
            if metadata:
                with open(best_meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

        # Update history
        self.checkpoint_history.append({
            'task_id': task_id,
            'round_id': round_id,
            'path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat()
        })

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def save_model(
        self,
        model: nn.Module,
        task_id: int,
        round_id: Optional[int] = None,
        is_best: bool = False,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save model checkpoint.

        Args:
            model: PyTorch model.
            task_id: Current task ID.
            round_id: Current round ID.
            is_best: Whether this is the best model.
            metadata: Additional metadata.

        Returns:
            Path to saved checkpoint.
        """
        state_dict = {
            'model_state_dict': model.state_dict(),
            'task_id': task_id,
            'round_id': round_id
        }
        return self.save_checkpoint(state_dict, task_id, round_id, is_best, metadata)

    def load_checkpoint(
        self,
        task_id: Optional[int] = None,
        round_id: Optional[int] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            task_id: Task ID to load (None for latest).
            round_id: Round ID to load (None for task-level).
            load_best: Whether to load the best model.

        Returns:
            Checkpoint state dictionary.
        """
        if load_best:
            checkpoint_path = self.exp_dir / "best_model.pt"
        elif task_id is not None and round_id is not None:
            checkpoint_path = self.get_round_dir(task_id, round_id) / "checkpoint.pt"
        elif task_id is not None:
            checkpoint_path = self.get_task_dir(task_id) / "task_checkpoint.pt"
        else:
            # Load latest checkpoint
            if not self.checkpoint_history:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = Path(self.checkpoint_history[-1]['path'])

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path)

    def load_model(
        self,
        model: nn.Module,
        task_id: Optional[int] = None,
        round_id: Optional[int] = None,
        load_best: bool = False
    ) -> nn.Module:
        """Load model from checkpoint.

        Args:
            model: PyTorch model to load into.
            task_id: Task ID to load.
            round_id: Round ID to load.
            load_best: Whether to load the best model.

        Returns:
            Loaded model.
        """
        checkpoint = self.load_checkpoint(task_id, round_id, load_best)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return

        # Keep only the last N checkpoints
        to_remove = self.checkpoint_history[:-self.keep_last_n]

        for ckpt_info in to_remove:
            ckpt_path = Path(ckpt_info['path'])
            if ckpt_path.exists():
                ckpt_path.unlink()

        self.checkpoint_history = self.checkpoint_history[-self.keep_last_n:]

    def save_experiment_state(
        self,
        state: Dict[str, Any],
        filename: str = "experiment_state.json"
    ) -> None:
        """Save experiment state for resume.

        Args:
            state: Experiment state dictionary.
            filename: State file name.
        """
        state_path = self.exp_dir / filename
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_experiment_state(
        self,
        filename: str = "experiment_state.json"
    ) -> Optional[Dict[str, Any]]:
        """Load experiment state for resume.

        Args:
            filename: State file name.

        Returns:
            Experiment state dictionary or None.
        """
        state_path = self.exp_dir / filename
        if not state_path.exists():
            return None

        with open(state_path, 'r') as f:
            return json.load(f)

    def list_checkpoints(self) -> list:
        """List all available checkpoints.

        Returns:
            List of checkpoint information.
        """
        return self.checkpoint_history.copy()
