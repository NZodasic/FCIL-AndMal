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
from federated.drift.controller import ServerDriftController, AdaptiveLambdaController
from federated.drift.detectors import PageHinkleyDetector
from config import ID2LABEL
from utils.device import resolve_device, clear_device_cache
from utils.metrics import format_classification_metrics, format_confusion_matrix


class FLServer:
    """Federated Learning Server.

    Manages global model, coordinates clients, and performs aggregation.
    Supports class-incremental learning scenarios, FL-MalDrift, and FL-CL-IDS.
    """

    def __init__(
        self,
        global_model: Optional[nn.Module] = None,
        aggregator: Optional[BaseAggregator] = None,
        device: Any = "auto",
        config: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        validation_evaluator: Optional[Any] = None,
        logger: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        drift_controller: Optional[ServerDriftController] = None,
        adaptive_lambda_controller: Optional[AdaptiveLambdaController] = None,
        enable_drift_filtering: bool = False,
    ):
        if global_model is None and config is not None:
            from models.fcil_model import FCILNet
            global_model = FCILNet(config.model)
        self.global_model = global_model
        self.config = config
        self.evaluator = evaluator
        self.validation_evaluator = validation_evaluator
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.aggregator = aggregator or FedAvg()
        self.device = resolve_device(device)

        # FL-MalDrift / FL-CL-IDS Server-Side Controllers & Gating
        self.enable_drift_filtering = enable_drift_filtering
        self.drift_controller = drift_controller
        self.adaptive_lambda_controller = adaptive_lambda_controller

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

        if self.adaptive_lambda_controller is None:
            drift_cfg = getattr(config, 'drift', None) if config else None
            il_cfg = getattr(config, 'il', None) if config else None
            lam_min = getattr(drift_cfg, 'lambda_min', 10.0) if drift_cfg else 10.0
            lam_max = getattr(drift_cfg, 'lambda_max', 1000.0) if drift_cfg else 1000.0
            tau_low = getattr(drift_cfg, 'tau_low', 0.05) if drift_cfg else 0.05
            tau_high = getattr(drift_cfg, 'tau_high', 0.30) if drift_cfg else 0.30
            rho_min = getattr(drift_cfg, 'rho_min', 0.50) if drift_cfg else 0.50
            if il_cfg is not None and getattr(il_cfg, 'method_name', '') in ('ewc', 'ewcdr'):
                self.adaptive_lambda_controller = AdaptiveLambdaController(
                    lambda_min=lam_min,
                    lambda_max=lam_max,
                    tau_low=tau_low,
                    tau_high=tau_high,
                    rho_min=rho_min,
                )

        # Clients
        self.clients: Dict[int, FLClient] = {}

        # Current task and local round counter
        self.current_task = 0
        self.current_round_in_task = 0

        # Global prototypes for MALFSIL
        self.global_prototypes: Dict[int, torch.Tensor] = {}

        # Global consolidated Fisher & optimal parameters for EWC / FL-CL-IDS
        self.global_fisher_dict: Dict[str, torch.Tensor] = {}
        self.global_optimal_params: Dict[str, torch.Tensor] = {}

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

        Supports both (mean_proto, count) sample-weighted tuples and plain Tensor prototypes.

        Args:
            client_ids: List of participating client IDs.
        """
        weighted_sums: Dict[int, torch.Tensor] = {}
        weight_totals: Dict[int, float] = {}

        for cid in client_ids:
            client = self.clients[cid]
            if hasattr(client.strategy, 'get_prototypes'):
                prototypes = client.strategy.get_prototypes()
                for cls, proto_val in prototypes.items():
                    if isinstance(proto_val, (tuple, list)) and len(proto_val) == 2:
                        proto_tensor, count = proto_val[0].detach().cpu(), float(max(1, proto_val[1]))
                    elif isinstance(proto_val, torch.Tensor):
                        proto_tensor, count = proto_val.detach().cpu(), 1.0
                    else:
                        continue
                    cls_id = int(cls)
                    if cls_id not in weighted_sums:
                        weighted_sums[cls_id] = proto_tensor * count
                        weight_totals[cls_id] = count
                    else:
                        weighted_sums[cls_id] = weighted_sums[cls_id] + proto_tensor * count
                        weight_totals[cls_id] += count

        # Update global prototypes using sample-weighted average (Eq. 11)
        for cls_id, proto_sum in weighted_sums.items():
            if weight_totals.get(cls_id, 0.0) > 0:
                self.global_prototypes[cls_id] = proto_sum / weight_totals[cls_id]

        # Distribute global prototypes to all registered clients
        for cid, client in self.clients.items():
            if hasattr(client.strategy, 'set_prototypes'):
                client.strategy.set_prototypes(self.global_prototypes)

    def consolidate_global_ewc_state(self, client_ids: List[int]) -> None:
        """Aggregate consolidated Fisher, optimal_params, and latent_stats across clients."""
        fisher_lists: Dict[str, List[torch.Tensor]] = {}
        opt_lists: Dict[str, List[torch.Tensor]] = {}
        latent_mu_lists: Dict[int, List[torch.Tensor]] = {}
        latent_std_lists: Dict[int, List[torch.Tensor]] = {}

        for cid in client_ids:
            if cid not in self.clients:
                continue
            strat = self.clients[cid].strategy
            il_mod = getattr(strat, "il_method", strat)
            f_dict = getattr(il_mod, "fisher_dict", None)
            if f_dict:
                for k, v in f_dict.items():
                    fisher_lists.setdefault(k, []).append(v.detach().cpu())
            o_dict = getattr(il_mod, "optimal_params", None)
            if o_dict:
                for k, v in o_dict.items():
                    opt_lists.setdefault(k, []).append(v.detach().cpu())
            l_stats = getattr(il_mod, "latent_stats", None)
            if l_stats:
                for c_int, (mu_c, std_c) in l_stats.items():
                    latent_mu_lists.setdefault(int(c_int), []).append(mu_c.detach().cpu())
                    latent_std_lists.setdefault(int(c_int), []).append(std_c.detach().cpu())

        global_latent_stats: Dict[int, Any] = {}
        for c_int, mu_list in latent_mu_lists.items():
            if mu_list:
                avg_mu = torch.stack(mu_list, dim=0).mean(dim=0)
                avg_std = torch.stack(latent_std_lists[c_int], dim=0).mean(dim=0).clamp(min=1e-3)
                global_latent_stats[c_int] = (avg_mu, avg_std)

        if global_latent_stats:
            for cid, client in self.clients.items():
                il_mod = getattr(client.strategy, "il_method", client.strategy)
                if hasattr(il_mod, "latent_stats"):
                    for c_int, (mu_c, std_c) in global_latent_stats.items():
                        il_mod.latent_stats[c_int] = (mu_c.clone(), std_c.clone())

        if not fisher_lists:
            # Even for non-EWC methods (e.g., SPCIL/MALFSIL/Replay), sync aligned classifier head
            aligned_w, aligned_b = [], []
            for cid in client_ids:
                if cid in self.clients and hasattr(self.clients[cid].model, "classifier"):
                    fc = getattr(self.clients[cid].model.classifier, "fc", self.clients[cid].model.classifier)
                    if hasattr(fc, "weight"):
                        aligned_w.append(fc.weight.data.detach().cpu())
                    if getattr(fc, "bias", None) is not None:
                        aligned_b.append(fc.bias.data.detach().cpu())
            if aligned_w and hasattr(self.global_model, "classifier"):
                g_fc = getattr(self.global_model.classifier, "fc", self.global_model.classifier)
                if hasattr(g_fc, "weight"):
                    g_fc.weight.data.copy_(torch.stack(aligned_w, dim=0).mean(dim=0).to(g_fc.weight.device))
                if aligned_b and getattr(g_fc, "bias", None) is not None:
                    g_fc.bias.data.copy_(torch.stack(aligned_b, dim=0).mean(dim=0).to(g_fc.bias.device))
            return

        for k, t_list in fisher_lists.items():
            max_shape = tuple(max(t.shape[d] for t in t_list) for d in range(t_list[0].ndim))
            acc = torch.zeros(max_shape, dtype=t_list[0].dtype)
            for t in t_list:
                slices = tuple(slice(0, s) for s in t.shape)
                acc[slices] += t
            acc /= float(len(t_list))
            max_v = float(acc.max().item())
            if max_v > 1.0:
                acc /= max_v
            self.global_fisher_dict[k] = acc

        if opt_lists:
            for k, t_list in opt_lists.items():
                max_shape = tuple(max(t.shape[d] for t in t_list) for d in range(t_list[0].ndim))
                acc = torch.zeros(max_shape, dtype=t_list[0].dtype)
                for t in t_list:
                    slices = tuple(slice(0, s) for s in t.shape)
                    acc[slices] += t
                acc /= float(len(t_list))
                self.global_optimal_params[k] = acc
        else:
            self.global_optimal_params = {
                name: p.data.detach().cpu().clone()
                for name, p in self.global_model.named_parameters()
            }

        # Sync consolidated global Fisher and optimal_params to all clients (including newly joining clients)
        for cid, client in self.clients.items():
            strat = client.strategy
            il_mod = getattr(strat, "il_method", strat)
            if hasattr(il_mod, "fisher_dict") and hasattr(il_mod, "optimal_params"):
                il_mod.fisher_dict = {k: v.clone() for k, v in self.global_fisher_dict.items()}
                il_mod.optimal_params = {k: v.clone() for k, v in self.global_optimal_params.items()}
                if hasattr(il_mod, "_device_fisher"):
                    il_mod._device_fisher.clear()
                if hasattr(il_mod, "_device_optimal"):
                    il_mod._device_optimal.clear()

    def _is_fl_cl_ids_active(self, client_ids: List[int]) -> bool:
        """Return True if Page-Hinkley + AdaptiveLambdaController (FL-CL-IDS) is active."""
        if not self.enable_drift_filtering or self.adaptive_lambda_controller is None:
            return False
        for cid in client_ids:
            if cid in self.clients and isinstance(self.clients[cid].drift_detector, PageHinkleyDetector):
                return True
        return False

    def run_round(
        self,
        client_ids: List[int],
        train_loaders: Dict[int, Any],
        n_epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Run one federated learning round with FL-CL-IDS or FL-MalDrift gating.

        Args:
            client_ids: List of participating client IDs.
            train_loaders: Dict mapping client_id to DataLoader.
            n_epochs: Number of local epochs.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of round metrics.
        """
        self.current_round_in_task += 1

        # Distribute global model
        self.distribute_model(client_ids)

        active_scheduled = [cid for cid in client_ids if cid in train_loaders]
        client_metrics = []
        client_weights_map: Dict[int, float] = {}
        client_steps_map: Dict[int, int] = {}
        client_scores: Dict[int, float] = {}
        drift_round_info: Dict[str, Any] = {}

        # Branch 1: FL-CL-IDS (Page-Hinkley Covariate Drift + Adaptive Lambda + Round-Skip Protocol)
        if self._is_fl_cl_ids_active(active_scheduled):
            for cid in active_scheduled:
                score_i = self.clients[cid].evaluate_covariate_drift(train_loaders[cid])
                client_scores[cid] = float(score_i)

            admitted_client_ids, skipped_client_ids, lambdas, drift_round_info = (
                self.adaptive_lambda_controller.plan_round(
                    client_drift_scores=client_scores,
                    round_in_task=self.current_round_in_task,
                    task_id=self.current_task,
                )
            )

            for cid in active_scheduled:
                if cid in lambdas:
                    self.clients[cid].set_adaptive_lambda(lambdas[cid])

            for cid in admitted_client_ids:
                metrics = self.clients[cid].local_train(
                    train_loaders[cid],
                    n_epochs=n_epochs,
                    **kwargs
                )
                metrics['adaptive_lambda'] = float(lambdas.get(cid, 0.0))
                metrics['round_skipped'] = False
                client_metrics.append(metrics)
                client_weights_map[cid] = metrics.get('n_samples', 1)
                client_steps_map[cid] = metrics.get('n_steps', 1)

            for cid in skipped_client_ids:
                client_metrics.append({
                    'client_id': cid,
                    'n_samples': len(train_loaders[cid].dataset) if hasattr(train_loaders[cid], 'dataset') else 0,
                    'n_steps': 0,
                    'drift_score': float(client_scores.get(cid, 0.0)),
                    'adaptive_lambda': float(lambdas.get(cid, 0.0)),
                    'round_skipped': True,
                })
        else:
            # Branch 2: Standard FL or FL-MalDrift (Error-Stream Detector + Server Quarantine)
            for cid in active_scheduled:
                metrics = self.clients[cid].local_train(
                    train_loaders[cid],
                    n_epochs=n_epochs,
                    **kwargs
                )
                client_metrics.append(metrics)
                client_weights_map[cid] = metrics.get('n_samples', 1)
                client_steps_map[cid] = metrics.get('n_steps', 1)
                client_scores[cid] = float(metrics.get('drift_score', 0.0))

            admitted_client_ids = list(active_scheduled)
            if self.enable_drift_filtering and self.drift_controller is not None:
                current_round_idx = len(self.history) + 1
                admitted_client_ids, tau_t, drift_round_info = self.drift_controller.step(
                    round_idx=current_round_idx,
                    client_scores=client_scores,
                )
                admitted_client_ids = [cid for cid in admitted_client_ids if cid in client_weights_map]

                is_warmup = drift_round_info.get('warmup_active', False)
                for cid in active_scheduled:
                    if cid in self.clients:
                        client = self.clients[cid]
                        if is_warmup:
                            client.record_warmup_divergence()
                        else:
                            client.calibrate_theta_base()

        # Extract weights and step counts for admitted clients
        admitted_weights = [client_weights_map[cid] for cid in admitted_client_ids]
        admitted_steps = [client_steps_map[cid] for cid in admitted_client_ids]

        # Aggregate models from admitted clients
        if admitted_client_ids:
            self.aggregate_models(admitted_client_ids, admitted_weights, admitted_steps)
            # Aggregate prototypes (for MALFSIL) from admitted clients
            self.aggregate_prototypes(admitted_client_ids)

        # Client Sync: Non-admitted / skipped clients receive the updated global model
        if self.enable_drift_filtering and (self.drift_controller is not None or self.adaptive_lambda_controller is not None):
            excluded_cids = [cid for cid in active_scheduled if cid not in admitted_client_ids]
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
        self.current_round_in_task = 0

        # Ensure all participating clients (including newly joining clients in Tasks 2..5)
        # start from the converged previous-task global model before snapshotting teacher/anchor
        if task_id > 0:
            self.distribute_model(client_ids)

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
                interim_evaluator = self.validation_evaluator or self.evaluator
                if not is_final_round and interim_evaluator is not None:
                    interim_eval = interim_evaluator.evaluate_all_seen_tasks(
                        self.global_model, task_id
                    )
                    metrics.update(interim_eval)
                    eval_split_label = "Interim Val" if self.validation_evaluator is not None else "Interim Test"
                    context = (
                        f"FL Task {task_id + 1} | Global Round {global_round} "
                        f"(task round {local_round}/{n_rounds}) | {eval_split_label}"
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

        # Distribute converged global model to clients before after_task so Fisher/Herding
        # are computed on the final aggregated global model parameters
        self.distribute_model(client_ids)
        for cid in client_ids:
            if cid in train_loaders:
                self.clients[cid].after_task(
                    task_id,
                    train_loader=train_loaders[cid]
                )
        # Consolidate global Fisher and optimal_params across clients (for EWC / FL-CL-IDS)
        self.consolidate_global_ewc_state(client_ids)
        # Sync any classifier weight alignment from global optimal_params back to global_model
        if self.global_optimal_params:
            state = self.global_model.state_dict()
            for k, v in self.global_optimal_params.items():
                if k in state and state[k].shape == v.shape:
                    state[k].copy_(v.to(state[k].device))
            self.global_model.load_state_dict(state)

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
                if hasattr(self.logger, "log_task_evaluation"):
                    self.logger.log_task_evaluation(
                        task_id=task_id,
                        eval_metrics=final_evaluation,
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
