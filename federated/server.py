"""Federated Learning Server.

Orchestrates federated learning with incremental tasks.

"""

from typing import Dict, List, Optional, Any
import copy
import gc

import torch
import torch.nn as nn

from federated.client import FLClient
from federated.aggregators.base import BaseAggregator, FedAvg
from federated.drift.controller import ServerDriftController
from config import ID2LABEL
from utils.device import resolve_device, clear_device_cache
from utils.metrics import format_classification_metrics, format_confusion_matrix


class FLServer:
    """Federated Learning Server.

    Manages global model, coordinates clients, and performs aggregation.
    Supports class-incremental learning scenarios.
    """

    def __init__(
        self,
        global_model: Optional[nn.Module] = None,
        aggregator: Optional[BaseAggregator] = None,
        device: Any = "auto",
        config: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        logger: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        drift_controller: Optional[ServerDriftController] = None,
        enable_drift_filtering: bool = False,
    ):
        if global_model is None and config is not None:
            from models.fcil_model import FCILNet
            global_model = FCILNet(config.model)
        self.global_model = global_model
        self.config = config
        self.evaluator = evaluator
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.aggregator = aggregator or FedAvg()
        self.device = resolve_device(device)

        # FL-MalDrift Server-Side Controller & Gating
        self.enable_drift_filtering = enable_drift_filtering
        self.drift_controller = drift_controller

        if not self.enable_drift_filtering and config is not None:
            if getattr(config, 'enable_drift', False):
                self.enable_drift_filtering = True
            elif hasattr(config, 'fl') and getattr(config.fl, 'enable_drift', False):
                self.enable_drift_filtering = True
            elif hasattr(config, 'drift') and getattr(config.drift, 'enabled', False):
                self.enable_drift_filtering = True

        if self.enable_drift_filtering and self.drift_controller is None:
            warmup = 3
            window_size = 10
            alpha = 0.8
            k = 1.5
            p_star = 0.7
            eta = 0.05
            drift_cfg = getattr(config, 'drift', None) if config else None
            if drift_cfg is not None:
                warmup = getattr(drift_cfg, 'warmup_rounds', warmup)
                window_size = getattr(drift_cfg, 'window_size', window_size)
                alpha = getattr(drift_cfg, 'alpha', alpha)
                k = getattr(drift_cfg, 'k', k)
                p_star = getattr(drift_cfg, 'p_star', p_star)
                eta = getattr(drift_cfg, 'eta', eta)
            self.drift_controller = ServerDriftController(
                warmup_rounds=warmup,
                window_size=window_size,
                alpha=alpha,
                k=k,
                target_participation_budget=p_star,
                gain_eta=eta,
            )

        # Clients
        self.clients: Dict[int, FLClient] = {}

        # Current task
        self.current_task = 0

        # Global prototypes for MALFSIL
        self.global_prototypes: Dict[int, torch.Tensor] = {}

        # Metrics history
        self.history: List[Dict[str, Any]] = []

    def register_client(self, client: FLClient) -> None:
        """Register a client with the server.

        Args:
            client: FLClient instance.
        """
        self.clients[client.client_id] = client

    def register_clients(self, clients: List[FLClient]) -> None:
        """Register multiple clients.

        Args:
            clients: List of FLClient instances.
        """
        for client in clients:
            self.register_client(client)

    def distribute_model(self, client_ids: Optional[List[int]] = None) -> None:
        """Distribute global model to clients.

        Args:
            client_ids: List of client IDs to distribute to (None = all).
        """
        global_state = self.global_model.state_dict()

        target_clients = client_ids if client_ids else self.clients.keys()

        for cid in target_clients:
            if cid in self.clients:
                self.clients[cid].set_model_state(global_state)

    def aggregate_models(
        self,
        client_ids: List[int],
        client_weights: Optional[List[float]] = None,
        client_steps: Optional[List[int]] = None
    ) -> nn.Module:
        """Aggregate models from selected clients.

        Args:
            client_ids: List of participating client IDs.
            client_weights: Optional weights for each client.
            client_steps: Optional step counts for each client.

        Returns:
            Updated global model.
        """
        # Collect client model state dictionaries (avoids allocating 50 model duplicates every round)
        client_states = [
            self.clients[cid].model.state_dict()
            for cid in client_ids
        ]

        # Aggregate
        if isinstance(self.aggregator, FedAvg):
            self.global_model = self.aggregator.aggregate(
                self.global_model,
                client_states,
                client_weights
            )
        else:
            # For FedNova or other aggregators that need steps
            self.global_model = self.aggregator.aggregate(
                self.global_model,
                client_states,
                client_weights,
                client_steps
            )

        del client_states
        clear_device_cache()

        return self.global_model

    def aggregate_prototypes(self, client_ids: List[int]) -> None:
        """Aggregate class prototypes from clients (for MALFSIL).

        Args:
            client_ids: List of participating client IDs.
        """
        # Collect prototypes from clients
        all_prototypes: Dict[int, List[torch.Tensor]] = {}

        for cid in client_ids:
            client = self.clients[cid]
            if hasattr(client.strategy, 'get_prototypes'):
                prototypes = client.strategy.get_prototypes()
                for cls, proto in prototypes.items():
                    if cls not in all_prototypes:
                        all_prototypes[cls] = []
                    all_prototypes[cls].append(proto)

        # Average prototypes
        for cls, proto_list in all_prototypes.items():
            if proto_list:
                stacked = torch.stack(proto_list)
                self.global_prototypes[cls] = stacked.mean(dim=0)

        # Distribute global prototypes to clients
        for cid in client_ids:
            client = self.clients[cid]
            if hasattr(client.strategy, 'set_prototypes'):
                client.strategy.set_prototypes(self.global_prototypes)

    def run_round(
        self,
        client_ids: List[int],
        train_loaders: Dict[int, Any],
        n_epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Run one federated learning round with optional FL-MalDrift gating.

        Args:
            client_ids: List of participating client IDs.
            train_loaders: Dict mapping client_id to DataLoader.
            n_epochs: Number of local epochs.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of round metrics.
        """
        # Distribute global model
        self.distribute_model(client_ids)

        # Local training
        client_metrics = []
        client_weights_map: Dict[int, float] = {}
        client_steps_map: Dict[int, int] = {}
        client_scores: Dict[int, float] = {}

        for cid in client_ids:
            if cid in train_loaders:
                metrics = self.clients[cid].local_train(
                    train_loaders[cid],
                    n_epochs=n_epochs,
                    **kwargs
                )
                client_metrics.append(metrics)
                client_weights_map[cid] = metrics.get('n_samples', 1)
                client_steps_map[cid] = metrics.get('n_steps', 1)
                client_scores[cid] = float(metrics.get('drift_score', 0.0))

        # FL-MalDrift Admission Filtering (Algorithm 2)
        admitted_client_ids = [cid for cid in client_ids if cid in train_loaders]
        drift_round_info = {}
        if self.enable_drift_filtering and self.drift_controller is not None:
            current_round_idx = len(self.history) + 1
            admitted_client_ids, tau_t, drift_round_info = self.drift_controller.step(
                round_idx=current_round_idx,
                client_scores=client_scores,
            )
            # Ensure admitted_client_ids only includes clients that actually trained
            admitted_client_ids = [cid for cid in admitted_client_ids if cid in client_weights_map]

            # Paper (Patel et al. 2026, Sec. 4.4): During warm-up, record each client's
            # post-training divergence so theta_base can be calibrated to the per-client
            # median divergence.  Once warm-up ends, finalize theta_base for each client.
            is_warmup = drift_round_info.get('warmup_active', False)
            for cid in client_ids:
                if cid in self.clients and cid in train_loaders:
                    client = self.clients[cid]
                    if is_warmup:
                        client.record_warmup_divergence()
                    else:
                        client.calibrate_theta_base()

        # Extract weights and step counts for admitted clients
        admitted_weights = [client_weights_map[cid] for cid in admitted_client_ids]
        admitted_steps = [client_steps_map[cid] for cid in admitted_client_ids]

        # Aggregate models from admitted stable clients
        if admitted_client_ids:
            self.aggregate_models(admitted_client_ids, admitted_weights, admitted_steps)
            # Aggregate prototypes (for MALFSIL) from admitted clients
            self.aggregate_prototypes(admitted_client_ids)

        # Client Rollback: Quarantined/excluded clients have their local parameters
        # rolled back to the newly aggregated global model to prevent drift contamination
        if self.enable_drift_filtering and self.drift_controller is not None:
            excluded_cids = [cid for cid in client_ids if cid in train_loaders and cid not in admitted_client_ids]
            if excluded_cids:
                global_state = self.global_model.state_dict()
                for cid in excluded_cids:
                    if cid in self.clients:
                        self.clients[cid].set_model_state(global_state)

        # Return metrics
        round_metrics = {
            'round': len(self.history),
            'n_clients': len(client_ids),
            'n_admitted': len(admitted_client_ids),
            'admitted_clients': admitted_client_ids,
            'client_metrics': client_metrics,
        }
        if drift_round_info:
            round_metrics['drift_control'] = drift_round_info

        return round_metrics

    def run_task(
        self,
        task_id: int,
        n_new_classes: int,
        client_ids: List[int],
        train_loaders: Dict[int, Any],
        n_rounds: int,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Run federated learning for one task.

        Args:
            task_id: Current task ID.
            n_new_classes: Number of new classes in this task.
            client_ids: List of participating client IDs.
            train_loaders: Dict mapping client_id to DataLoader.
            n_rounds: Number of communication rounds.
            n_epochs: Number of local epochs per round.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of task metrics.
        """
        print(f"\n{'='*60}")
        print(f"Task {task_id}: Starting FL with {len(client_ids)} clients")
        print(f"{'='*60}")

        self.current_task = task_id

        # Prepare clients for new task
        for cid in client_ids:
            self.clients[cid].before_task(task_id, n_new_classes)

        # Expand global model if needed
        if task_id > 0:
            classes_per_task = (
                self.config.model.classes_per_task
                if self.config is not None and self.config.model is not None
                else n_new_classes
            )
            target_classes = (task_id + 1) * classes_per_task
            if self.global_model.current_classes < target_classes:
                needed = target_classes - self.global_model.current_classes
                if hasattr(self.global_model, 'expand_classes'):
                    self.global_model.expand_classes(needed)
                elif hasattr(self.global_model, 'expand_classifier'):
                    self.global_model.expand_classifier(needed)
                else:
                    raise AttributeError(
                        "Global model does not support incremental class expansion"
                    )
            self.global_model.to(self.device)

        # Run FL rounds
        round_metrics = []
        round_checkpoint_paths = []
        for round_idx in range(n_rounds):
            metrics = self.run_round(
                client_ids,
                train_loaders,
                n_epochs,
                **kwargs
            )
            local_round = round_idx + 1
            global_round = task_id * n_rounds + local_round
            metrics["round"] = local_round
            metrics["global_round"] = global_round
            if self.checkpoint_manager is not None and (round_idx == n_rounds - 1 or (round_idx + 1) % 10 == 0):
                checkpoint_path = self.checkpoint_manager.save_weights_checkpoint(
                    self.global_model,
                    task_id=task_id,
                    step_type="round",
                    step_id=local_round,
                    global_step=global_round,
                )
                metrics["checkpoint_path"] = checkpoint_path
                round_checkpoint_paths.append(checkpoint_path)

            if (round_idx + 1) % 10 == 0:
                is_final_round = (round_idx == n_rounds - 1)
                if not is_final_round and self.evaluator is not None:
                    interim_eval = self.evaluator.evaluate_all_seen_tasks(
                        self.global_model, task_id
                    )
                    metrics.update(interim_eval)
                    context = (
                        f"FL Task {task_id + 1} | Global Round {global_round} "
                        f"(task round {local_round}/{n_rounds}) | Interim Test"
                    )
                    if self.logger is not None and hasattr(self.logger, "log_evaluation"):
                        self.logger.log_evaluation(
                            interim_eval,
                            context=context,
                            task_id=task_id,
                            step=global_round,
                            round_id=global_round,
                            include_confusion_matrix=False,
                            label_names=ID2LABEL,
                        )
                    else:
                        print(f"{context} | {format_classification_metrics(interim_eval)}")

            round_metrics.append(metrics)

            if (round_idx + 1) % 10 == 0 or round_idx == 0:
                print(f"  Round {round_idx + 1}/{n_rounds} completed")

        # Cleanup after task
        for cid in client_ids:
            if cid in train_loaders:
                self.clients[cid].after_task(
                    task_id,
                    train_loader=train_loaders[cid]
                )
        gc.collect()
        clear_device_cache()

        # Evaluation happens once, after the task's final communication round.
        final_evaluation = {}
        if self.evaluator is not None:
            final_evaluation = self.evaluator.evaluate_all_seen_tasks(
                self.global_model, task_id
            )
            global_round = (task_id + 1) * n_rounds
            context = (
                f"FL Task {task_id + 1} | Global Round {global_round} "
                f"(task round {n_rounds}/{n_rounds}) | Final Test"
            )
            if self.logger is not None and hasattr(self.logger, "log_evaluation"):
                self.logger.log_evaluation(
                    final_evaluation,
                    context=context,
                    task_id=task_id,
                    step=global_round,
                    round_id=global_round,
                    include_confusion_matrix=True,
                    label_names=ID2LABEL,
                )
            else:
                print(f"{context} | {format_classification_metrics(final_evaluation)}")
                print(format_confusion_matrix(final_evaluation, label_names=ID2LABEL))

        task_metrics = {
            'task_id': task_id,
            'n_rounds': n_rounds,
            'n_clients': len(client_ids),
            'round_metrics': round_metrics,
            'round_checkpoint_paths': round_checkpoint_paths,
            'final_checkpoint_path': (
                round_checkpoint_paths[-1] if round_checkpoint_paths else None
            ),
            **final_evaluation,
        }

        self.history.append(task_metrics)

        return task_metrics
