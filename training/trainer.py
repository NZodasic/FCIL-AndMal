"""
Centralized Continual Learning Trainer.
Supports centralized sequential training for lower-bound (Fine-tune), upper-bound (Joint),
and centralized continual baselines (EWC, LwF, Replay, SPCIL, MALFSIL).
"""

from typing import Dict, List, Any, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    CENTRALIZED_BATCH_SIZE,
    ExperimentConfig,
    TASK_LABEL_MAP,
    LABEL2ID,
    ID2LABEL,
)
from models.fcil_model import FCILNet
from methods import build_il_method
from data.dataset import TabularMalwareDataset
from data.few_shot import build_few_shot_session
from training.evaluator import ContinualEvaluator
from training.checkpoint import CheckpointManager
from utils.logger import AcademicLogger
from utils.results import export_experiment_results, resolve_test_location
from utils.metrics import compute_fscil_session_metrics


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
        if config.fl.batch_size != CENTRALIZED_BATCH_SIZE:
            self.logger.warning(
                f"Centralized batch size is fixed at {CENTRALIZED_BATCH_SIZE}; "
                f"overriding {config.fl.batch_size}."
            )
            config.fl.batch_size = CENTRALIZED_BATCH_SIZE
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
        checkpoint_paths = []
        fscil_manifests = []
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
            elif self.config.il.method_name == "malfscil" and task_id > 0:
                expected_classes = [LABEL2ID[label] for label in task_labels]
                if len(expected_classes) != self.config.il.fscil_n_way:
                    raise ValueError(
                        f"Task {task_id + 1} has {len(expected_classes)} classes; "
                        f"configured n_way is {self.config.il.fscil_n_way}"
                    )
                session = build_few_shot_session(
                    self.full_train_X[task_id],
                    self.full_train_y[task_id],
                    k_shot=self.config.il.fscil_k_shot,
                    query_per_class=self.config.il.fscil_query_per_class,
                    mask_probability=self.config.il.fscil_mask_probability,
                    expected_classes=expected_classes,
                    seed=self.config.seed + task_id,
                )
                curr_X = session.support_X
                curr_y = session.support_y
                self.logger.info(
                    f"  FSCIL session: {len(expected_classes)}-way "
                    f"{self.config.il.fscil_k_shot}-shot; "
                    f"support={len(session.support_y)}, "
                    f"augmented_query={len(session.query_y)}"
                )
                fscil_manifests.append({
                    "task_id": task_id,
                    "n_way": len(expected_classes),
                    "k_shot": self.config.il.fscil_k_shot,
                    "query_per_class": self.config.il.fscil_query_per_class,
                    "support_indices": session.support_indices.tolist(),
                })
            else:
                curr_X = self.full_train_X[task_id]
                curr_y = self.full_train_y[task_id]
                if self.config.il.method_name == "malfscil":
                    fscil_manifests.append({
                        "task_id": task_id,
                        "base_session_samples": int(len(curr_y)),
                    })

            ds = TabularMalwareDataset(curr_X, curr_y)
            if len(ds) == 0:
                raise ValueError(
                    f"Task {task_id + 1} has 0 training samples! This occurs if partition files were generated "
                    f"before applying the Stage 2 fused join fix. Please re-run Stage 2 partitioning:\n"
                    f"  python3 -m data.partition --dataset {os.path.join(self.config.scenario.prepared_data_dir, self.config.scenario.feature_type, 'train.parquet')} "
                    f"--feature_type {self.config.scenario.feature_type} --n_clients {self.config.scenario.n_clients} --output_dir {self.config.scenario.partition_output_dir}"
                )

            train_loader = DataLoader(ds, batch_size=self.config.fl.batch_size, shuffle=True)
            optimization_loaders = [train_loader]
            if (
                self.config.il.method_name == "malfscil"
                and task_id > 0
                and len(session.query_y) > 0
            ):
                query_dataset = TabularMalwareDataset(
                    session.query_X, session.query_y
                )
                optimization_loaders.append(
                    DataLoader(
                        query_dataset,
                        batch_size=self.config.fl.batch_size,
                        shuffle=True,
                    )
                )

            # 3. IL pre-task hook
            self.il_method.before_task(
                task_id=task_id,
                model=self.model,
                train_loader=train_loader,
                device=self.device
            )

            # 4. Train loop
            trainable_parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            trainable_parameters.extend(self.il_method.auxiliary_parameters())
            optimizer = optim.Adam(
                trainable_parameters,
                lr=self.config.fl.lr,
                weight_decay=self.config.fl.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()
            self.model.train()

            for ep in range(epochs_per_task):
                epoch_loss = 0.0
                epoch_batches = 0
                # MalFSCIL pre-trains on support, then fine-tunes on query.
                for optimization_loader in optimization_loaders:
                    for bx, by in optimization_loader:
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
                        epoch_batches += 1

                avg_loss = epoch_loss / max(1, epoch_batches)
                is_final_epoch = ep == epochs_per_task - 1
                if (ep + 1) % 5 == 0 and not is_final_epoch:
                    # Interim test every 5 epochs (excluding the final epoch,
                    # which is handled with confusion matrix after the loop).
                    interim_metrics = self.evaluator.evaluate_all_seen_tasks(self.model, task_id)
                    context = (
                        f"Centralized Task {task_id + 1} | "
                        f"Epoch {ep + 1}/{epochs_per_task} | Test"
                    )
                    self.logger.info(f"{context} | Loss: {avg_loss:.4f}")
                    self.logger.log_evaluation(
                        interim_metrics,
                        context=context,
                        task_id=task_id,
                        step=ep + 1,
                    )
                    self.model.train()
                elif is_final_epoch:
                    # Log final epoch loss; full evaluation with confusion matrix
                    # follows after the loop.
                    self.logger.info(
                        f"Centralized Task {task_id + 1} | "
                        f"Epoch {ep + 1}/{epochs_per_task} | Loss: {avg_loss:.4f}"
                    )

            # 5. IL post-task hook
            self.il_method.after_task(
                task_id=task_id,
                model=self.model,
                train_loader=train_loader,
                device=self.device
            )

            checkpoint_path = self.checkpoint_mgr.save_weights_checkpoint(
                self.model,
                task_id=task_id,
                step_type="epoch",
                step_id=epochs_per_task,
                global_step=(task_id + 1) * epochs_per_task,
            )

            # 6. Comprehensive multi-task evaluation
            eval_metrics = self.evaluator.evaluate_all_seen_tasks(self.model, task_id)
            eval_metrics["task_id"] = task_id
            eval_metrics["task_name"] = f"Task_{task_id + 1}"
            all_task_results.append(eval_metrics)

            self.logger.log_evaluation(
                eval_metrics,
                context=(
                    f"Centralized Task {task_id + 1} | "
                    f"Epoch {epochs_per_task}/{epochs_per_task} | Final Test"
                ),
                task_id=task_id,
                step=epochs_per_task,
                include_confusion_matrix=True,
                label_names=ID2LABEL,
            )
            self.logger.info(
                f"Centralized Task {task_id + 1} | Continual Avg Accuracy: "
                f"{eval_metrics['continual_avg_accuracy'] * 100:.2f}% | "
                f"Avg Forgetting: {eval_metrics['average_forgetting'] * 100:.2f}%"
            )

            # 7. Track final-epoch and best-performing weight files.
            checkpoint_paths.append(checkpoint_path)
            self.checkpoint_mgr.save_best_model(
                task_id=task_id,
                macro_f1=eval_metrics["macro_f1"],
                global_model=self.model
            )
            self.logger.log_task_evaluation(task_id, f"Task_{task_id + 1}", eval_metrics)

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            workbook_path = export_experiment_results(
                workbook_path=os.path.join(self.config.output_root, "evaluation_results.xlsx"),
                experiment_name=self.config.exp_name,
                method=self.config.il.method_name,
                setting="centralized",
                rounds_per_task=epochs_per_task,
                client_num="N/A",
                patch_size=self.config.fl.batch_size,
                test_location=resolve_test_location(
                    self.config.scenario.prepared_data_dir,
                    self.config.scenario.feature_type,
                ),
                task_results=all_task_results,
                checkpoint_paths=checkpoint_paths,
                artifact_dir=os.path.join(
                    self.config.get_exp_dir(), "confusion_matrices"
                ),
            )

        final_summary = {
            "all_task_results": all_task_results,
            "continual_matrix": self.evaluator.continual_matrix.get_summary_dict(),
            "evaluation_workbook": workbook_path,
        }
        if self.config.il.method_name == "malfscil":
            final_summary["fscil_summary"] = compute_fscil_session_metrics(
                all_task_results
            )
            final_summary["fscil_manifests"] = fscil_manifests
        self.logger.log_final_results(final_summary)
        return final_summary
