"""Federated Learning Server.

Orchestrates federated learning with incremental tasks.

"""

from typing import Dict, List, Optional, Any
import copy

import torch
import torch.nn as nn

from federated.client import FLClient
from federated.aggregators.base import BaseAggregator, FedAvg
from config import ID2LABEL
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
        device: str = 'cuda',
        config: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        logger: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
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
        self.device = device

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
        # Collect client models
        client_models = [
            copy.deepcopy(self.clients[cid].model)
            for cid in client_ids
        ]

        # Aggregate
        if isinstance(self.aggregator, FedAvg):
            self.global_model = self.aggregator.aggregate(
                self.global_model,
                client_models,
                client_weights
            )
        else:
            # For FedNova or other aggregators that need steps
            self.global_model = self.aggregator.aggregate(
                self.global_model,
                client_models,
                client_weights,
                client_steps
            )

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
        """Run one federated learning round.

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
        client_weights = []
        client_steps = []

        for cid in client_ids:
            if cid in train_loaders:
                metrics = self.clients[cid].local_train(
                    train_loaders[cid],
                    n_epochs=n_epochs,
                    **kwargs
                )
                client_metrics.append(metrics)
                client_weights.append(metrics.get('n_samples', 1))
                client_steps.append(metrics.get('n_steps', 1))

        # Aggregate models
        self.aggregate_models(client_ids, client_weights, client_steps)

        # Aggregate prototypes (for MALFSIL)
        self.aggregate_prototypes(client_ids)

        # Return metrics
        return {
            'round': len(self.history),
            'n_clients': len(client_ids),
            'client_metrics': client_metrics
        }

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
            if hasattr(self.global_model, 'expand_classes'):
                self.global_model.expand_classes(n_new_classes)
            elif hasattr(self.global_model, 'expand_classifier'):
                self.global_model.expand_classifier(n_new_classes)
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
            if self.checkpoint_manager is not None:
                checkpoint_path = self.checkpoint_manager.save_weights_checkpoint(
                    self.global_model,
                    task_id=task_id,
                    step_type="round",
                    step_id=local_round,
                    global_step=global_round,
                )
                metrics["checkpoint_path"] = checkpoint_path
                round_checkpoint_paths.append(checkpoint_path)
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
