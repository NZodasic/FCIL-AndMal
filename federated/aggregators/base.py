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
    """FedNova: Normalized averaging with momentum correction."""

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

        if client_steps is None:
            client_steps = [1] * n_clients

        total_steps = sum(client_steps)
        if total_steps == 0:
            total_steps = 1
        normalized_steps = [steps / total_steps * n_clients for steps in client_steps]

        effective_weights = [w * norm_step for w, norm_step in zip(client_weights, normalized_steps)]
        total_weight = sum(effective_weights)
        if total_weight == 0:
            total_weight = 1.0
        effective_weights = [w / total_weight for w in effective_weights]

        aggregated_state = {}
        for key in global_state.keys():
            aggregated = sum(
                client_states[i][key] * effective_weights[i]
                for i in range(n_clients)
            )
            aggregated_state[key] = aggregated

        if is_dict:
            return aggregated_state
        else:
            global_model.load_state_dict(aggregated_state)
            return global_model
