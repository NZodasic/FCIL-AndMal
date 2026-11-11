"""Federated averaging aggregators.

Implements FedAvg and FedNova for federated learning.

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import copy

import torch
import torch.nn as nn


class BaseAggregator(ABC):
    """Abstract base class for federated aggregators."""

    @abstractmethod
    def aggregate(
        self,
        global_model: Union[nn.Module, Dict[str, torch.Tensor]],
        client_models: List[Union[nn.Module, Dict[str, torch.Tensor]]],
        client_weights: Optional[List[float]] = None
    ) -> Union[nn.Module, Dict[str, torch.Tensor]]:
        """Aggregate client models into global model."""
        pass


class FedAvg(BaseAggregator):
    """Federated Averaging (FedAvg) algorithm."""

    def aggregate(
        self,
        global_model: Union[nn.Module, Dict[str, torch.Tensor]],
        client_models: List[Union[nn.Module, Dict[str, torch.Tensor]]],
        client_weights: Optional[List[float]] = None,
        client_steps: Optional[List[int]] = None
    ) -> Union[nn.Module, Dict[str, torch.Tensor]]:
        if not client_models:
            return global_model

        is_dict = isinstance(global_model, dict)
        global_state = global_model if is_dict else global_model.state_dict()
        client_states = [c if isinstance(c, dict) else c.state_dict() for c in client_models]

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)

        total_weight = sum(client_weights)
        if total_weight == 0:
            total_weight = 1.0
        client_weights = [w / total_weight for w in client_weights]

        aggregated_state = {}
        for key in global_state.keys():
            aggregated = sum(
                client_states[i][key] * client_weights[i]
                for i in range(len(client_states))
            )
            aggregated_state[key] = aggregated

        if is_dict:
            return aggregated_state
        else:
            global_model.load_state_dict(aggregated_state)
            return global_model


class FedNova(BaseAggregator):
    """FedNova: Normalized averaging with heterogeneous local step correction (Wang et al., NeurIPS 2020)."""

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def aggregate(
        self,
        global_model: Union[nn.Module, Dict[str, torch.Tensor]],
        client_models: List[Union[nn.Module, Dict[str, torch.Tensor]]],
        client_weights: Optional[List[float]] = None,
        client_steps: Optional[List[int]] = None
    ) -> Union[nn.Module, Dict[str, torch.Tensor]]:
        if not client_models:
            return global_model

        is_dict = isinstance(global_model, dict)
        global_state = global_model if is_dict else global_model.state_dict()
        client_states = [c if isinstance(c, dict) else c.state_dict() for c in client_models]

        n_clients = len(client_states)
        if client_weights is None:
            client_weights = [1.0 / n_clients] * n_clients

        total_w = sum(client_weights)
        if total_w <= 0:
            p_weights = [1.0 / n_clients] * n_clients
        else:
            p_weights = [float(w) / float(total_w) for w in client_weights]

        if client_steps is None:
            steps_list = [1.0] * n_clients
        else:
            steps_list = [ float(max(1, s)) for s in client_steps ]

        # Effective step count tau_eff = sum_k p_k * tau_k (Eq. 8)
        tau_eff = sum(p * s for p, s in zip(p_weights, steps_list))
        # Normalized update coefficient a_k = p_k * (tau_eff / tau_k)
        raw_coeffs = [p * (tau_eff / max(1.0, s)) for p, s in zip(p_weights, steps_list)]
        coeff_sum = sum(raw_coeffs)
        if coeff_sum <= 0:
            norm_coeffs = p_weights
        else:
            norm_coeffs = [c / coeff_sum for c in raw_coeffs]

        aggregated_state = {}
        for key, g_val in global_state.items():
            if g_val.dtype.is_floating_point:
                delta_sum = sum(
                    (client_states[i][key].to(g_val.device) - g_val) * norm_coeffs[i]
                    for i in range(n_clients)
                )
                aggregated_state[key] = g_val + delta_sum
            else:
                aggregated_state[key] = client_states[0][key].to(g_val.device).clone()

        if is_dict:
            return aggregated_state
        else:
            global_model.load_state_dict(aggregated_state)
            return global_model


class FedSGD(BaseAggregator):
    """Federated Stochastic Gradient Descent (FedSGD) aggregator.

    Aggregates client pseudo-gradients (model updates relative to the broadcasted
    global model) weighted by sample counts and applies a server-side update step:
        theta_{t+1} = theta_t + server_lr * sum_k (n_k / N) * (theta_{k,t} - theta_t)

    References:
        - McMahan et al., "Communication-Efficient Learning of Deep Networks from
          Decentralized Data", AISTATS 2017.
        - Patel et al., "FL-MalDrift: a federated learning framework for malware
          detection under local concept drift", Scientific Reports, 2026.
    """

    def __init__(self, server_lr: float = 1.0):
        self.server_lr = float(server_lr)

    def aggregate(
        self,
        global_model: Any,
        client_models: List[Any],
        client_weights: Optional[List[float]] = None,
        client_steps: Optional[List[int]] = None,
    ) -> Any:
        n_clients = len(client_models)
        if n_clients == 0:
            return global_model

        is_dict = isinstance(global_model, dict)
        global_state = global_model if is_dict else global_model.state_dict()
        client_states = [
            m if isinstance(m, dict) else m.state_dict() for m in client_models
        ]

        if client_weights is None:
            p_weights = [1.0 / n_clients] * n_clients
        else:
            total_w = sum(client_weights)
            if total_w <= 0:
                p_weights = [1.0 / n_clients] * n_clients
            else:
                p_weights = [float(w) / float(total_w) for w in client_weights]

        aggregated_state = {}
        for key, g_val in global_state.items():
            if g_val.dtype.is_floating_point:
                pseudo_grad = torch.zeros_like(g_val, dtype=torch.float32)
                for i in range(n_clients):
                    c_val = client_states[i][key].to(g_val.device)
                    if c_val.shape == g_val.shape:
                        pseudo_grad += p_weights[i] * (c_val.float() - g_val.float())
                updated = g_val.float() + self.server_lr * pseudo_grad
                aggregated_state[key] = updated.to(dtype=g_val.dtype)
            else:
                aggregated_state[key] = client_states[0][key].to(g_val.device).clone()

        if is_dict:
            return aggregated_state
        else:
            global_model.load_state_dict(aggregated_state)
            return global_model

