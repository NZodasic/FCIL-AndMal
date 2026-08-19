"""
Centralized Continual Learning Trainer.
Supports centralized sequential training for lower-bound (Fine-tune), upper-bound (Joint),
and centralized continual baselines (EWC, LwF, Replay, SPCIL, MALFSIL).
"""

from typing import Dict, List, Any, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import ExperimentConfig, TASK_LABEL_MAP, LABEL2ID
from models.fcil_model import FCILNet
from methods import build_il_method
from data.dataset import TabularMalwareDataset
from training.evaluator import ContinualEvaluator
from training.checkpoint import CheckpointManager
from utils.logger import AcademicLogger


class CentralizedTrainer:
    """
    Orchestrates centralized continual learning experiments across the 5 tasks.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        full_train_X: Dict[int, Any],  # task_id -> X array
        full_train_y: Dict[int, Any],  # task_id -> y array
        evaluator: ContinualEvaluator,
        logger: AcademicLogger
    ):
        self.config = config
        self.full_train_X = full_train_X
        self.full_train_y = full_train_y
        self.evaluator = evaluator
        self.logger = logger
        self.device = torch.device(config.fl.device if torch.cuda.is_available() and config.fl.device != "cpu" else "cpu")

        # Initialize model
        self.model = FCILNet(config.model).to(self.device)
        self.il_method = build_il_method(config.il)
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=os.path.join(config.get_exp_dir(), "checkpoints"),
            logger=logger
        )

    def train_all_tasks(self, epochs_per_task: int = 10) -> Dict[str, Any]:
        """Run sequential training over all 5 tasks."""
        self.logger.section(f"Starting Centralized Continual Training: Method = {self.config.il.method_name.upper()}")
        all_task_results = []
        cumulative_X, cumulative_y = [], []

        for task_id in range(5):
            task_labels = TASK_LABEL_MAP[task_id]
            self.logger.subsection(f"Task {task_id + 1}/5: {task_labels} (Classes {self.model.current_classes})")

            # 1. Expand model capacity for new task classes if beyond Task 0
            if task_id > 0:
                self.model.expand_classes(num_new_classes=3)
                self.model.to(self.device)

            # 2. Prepare task data
            if self.config.il.method_name == "joint":
                # Cumulative training: combine current task data with all past data
                cumulative_X.append(self.full_train_X[task_id])
                cumulative_y.append(self.full_train_y[task_id])
                curr_X = np.concatenate(cumulative_X, axis=0)
                curr_y = np.concatenate(cumulative_y, axis=0)
            else:
                curr_X = self.full_train_X[task_id]
                curr_y = self.full_train_y[task_id]

            ds = TabularMalwareDataset(curr_X, curr_y)
            train_loader = DataLoader(ds, batch_size=self.config.fl.batch_size, shuffle=True)

            # 3. IL pre-task hook
            self.il_method.before_task(
                task_id=task_id,
                model=self.model,
                train_loader=train_loader,
                device=self.device
            )

            # 4. Train loop
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.fl.lr, weight_decay=self.config.fl.weight_decay)
            criterion = nn.CrossEntropyLoss()
            self.model.train()

            for ep in range(epochs_per_task):
                epoch_loss = 0.0
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    loss, loss_dict = self.il_method.compute_loss(
                        model=self.model,
                        x=bx,
                        y=by,
                        criterion=criterion,
                        task_id=task_id,
                        device=self.device
                    )
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (ep + 1) % max(1, epochs_per_task // 2) == 0 or ep == epochs_per_task - 1:
                    avg_loss = epoch_loss / max(1, len(train_loader))
                    self.logger.info(f"  [Task {task_id + 1} | Epoch {ep + 1}/{epochs_per_task}] Loss: {avg_loss:.4f}")

            # 5. IL post-task hook
            self.il_method.after_task(
                task_id=task_id,
                model=self.model,
                train_loader=train_loader,
                device=self.device
            )

            # 6. Comprehensive multi-task evaluation
            eval_metrics = self.evaluator.evaluate_all_seen_tasks(self.model, task_id)
            eval_metrics["task_id"] = task_id
            eval_metrics["task_name"] = f"Task_{task_id + 1}"
            all_task_results.append(eval_metrics)

            self.logger.info(
                f"  📊 Post-Task {task_id + 1} Evaluation -> Macro-F1: {eval_metrics['macro_f1'] * 100:.2f}% | "
                f"Accuracy: {eval_metrics['accuracy'] * 100:.2f}% | "
                f"Avg Forgetting: {eval_metrics['average_forgetting'] * 100:.2f}% | "
                f"Malware F1: {eval_metrics['f1_malware_avg'] * 100:.2f}%"
            )

            # 7. Save task checkpoint & log
            self.checkpoint_mgr.save_task_checkpoint(
                task_id=task_id,
                round_id=epochs_per_task,
                global_model=self.model,
                continual_matrix_dict=self.evaluator.continual_matrix.get_summary_dict(),
                client_states={}
            )
            self.checkpoint_mgr.save_best_model(
                task_id=task_id,
                macro_f1=eval_metrics["macro_f1"],
                global_model=self.model
            )
            self.logger.log_task_evaluation(task_id, f"Task_{task_id + 1}", eval_metrics)

        final_summary = {
            "all_task_results": all_task_results,
            "continual_matrix": self.evaluator.continual_matrix.get_summary_dict(),
        }
        self.logger.log_final_results(final_summary)
        return final_summary
