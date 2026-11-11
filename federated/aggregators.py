"""
Federated Model Aggregators: FedAvg and FedNova.
Implements standard weighted averaging and normalized gradient aggregation to address client quantity skew.
"""

from typing import List, Dict, Any, Tuple, Optional
import copy
import torch
import torch.nn as nn


class BaseAggregator:
    """Interface for Federated Aggregation algorithms."""

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int],
        client_steps: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class FedAvgAggregator(BaseAggregator):
    """
    Standard Federated Averaging (McMahan et al., AISTATS 2017).
    Weights client parameter updates strictly proportional to their sample counts.
    """

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int],
        client_steps: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            return global_state

        total_samples = sum(client_sample_counts)
        if total_samples == 0:
            return global_state

        new_global_state = {}
        for key in global_state.keys():
            if key not in client_states[0]:
                new_global_state[key] = global_state[key].clone()
                continue

            # Weight average
            accum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for state, n_k in zip(client_states, client_sample_counts):
                weight = n_k / total_samples
                accum += weight * state[key].to(torch.float32)

            # Preserve integer dtypes if applicable
            if global_state[key].dtype in (torch.int64, torch.int32, torch.long):
                new_global_state[key] = accum.round().to(global_state[key].dtype)
            else:
                new_global_state[key] = accum.to(global_state[key].dtype)

        return new_global_state


class FedNovaAggregator(BaseAggregator):
    """
    Federated Normalized Averaging (Wang et al., NeurIPS 2020).
    Normalizes client updates by effective local optimization steps to prevent objective inconsistency
    when clients train for differing numbers of steps or suffer extreme sample quantity skew.
    """

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int],
        client_steps: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            return global_state

        total_samples = sum(client_sample_counts)
        if total_samples == 0:
            return global_state

        # If step counts are not provided, estimate from sample counts
        if client_steps is None or len(client_steps) != len(client_states):
            client_steps = [max(1, count) for count in client_sample_counts]

        # Normalized gradient per client: delta_k = (w_0 - w_k) / a_k where a_k = steps
        # Effective global step: tau_eff = sum(p_k * a_k)
        p_weights = [count / total_samples for count in client_sample_counts]
        tau_eff = sum(p * a for p, a in zip(p_weights, client_steps))

        new_global_state = {}
        for key in global_state.keys():
            if key not in client_states[0] or not global_state[key].is_floating_point():
                new_global_state[key] = global_state[key].clone()
                continue

            w_0 = global_state[key].to(torch.float32)
            normalized_accum = torch.zeros_like(w_0)

            for state, p_k, a_k in zip(client_states, p_weights, client_steps):
                w_k = state[key].to(torch.float32)
                # Client update
                delta_k = (w_0 - w_k) / max(1.0, float(a_k))
                normalized_accum += p_k * delta_k

            # FedNova updated parameter: w_new = w_0 - tau_eff * sum(p_k * delta_k)
            w_new = w_0 - (tau_eff * normalized_accum)
            new_global_state[key] = w_new.to(global_state[key].dtype)

        return new_global_state


class FedSGDAggregator(BaseAggregator):
    """Federated Stochastic Gradient Descent (FedSGD) Aggregator."""

    def __init__(self, server_lr: float = 1.0):
        super().__init__(name="fedsgd")
        self.server_lr = float(server_lr)

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int],
        client_steps: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        total_samples = sum(client_sample_counts)
        if total_samples == 0 or len(client_states) == 0:
            return copy.deepcopy(global_state)

        p_weights = [count / total_samples for count in client_sample_counts]
        new_global_state: Dict[str, torch.Tensor] = {}

        for key in global_state.keys():
            if key not in client_states[0] or not global_state[key].is_floating_point():
                new_global_state[key] = global_state[key].clone()
                continue

            w_0 = global_state[key].to(torch.float32)
            pseudo_grad = torch.zeros_like(w_0)
            for state, p_k in zip(client_states, p_weights):
                w_k = state[key].to(w_0.device, dtype=torch.float32)
                if w_k.shape == w_0.shape:
                    pseudo_grad += p_k * (w_k - w_0)

            w_new = w_0 + self.server_lr * pseudo_grad
            new_global_state[key] = w_new.to(global_state[key].dtype)

        return new_global_state


# Class Aliases for compatibility
FedAvg = FedAvgAggregator
FedNova = FedNovaAggregator
FedSGD = FedSGDAggregator


def build_aggregator(aggregator_name: str, **kwargs) -> BaseAggregator:
    """Factory builder for Federated Aggregators."""
    name = aggregator_name.lower()
    if name == "fedavg":
        return FedAvgAggregator()
    elif name == "fednova":
        return FedNovaAggregator()
    elif name == "fedsgd":
        return FedSGDAggregator(**kwargs)
    else:
        raise ValueError(
            f"Unknown aggregator: {aggregator_name}. Choose 'fedavg', 'fednova', or 'fedsgd'."
        )


__all__ = [
    'BaseAggregator',
    'FedAvgAggregator',
    'FedNovaAggregator',
    'FedSGDAggregator',
    'FedAvg',
    'FedNova',
    'FedSGD',
    'build_aggregator',
]
