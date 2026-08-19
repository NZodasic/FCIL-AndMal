"""Aggregators package."""

from federated.aggregators.base import BaseAggregator, FedAvg, FedNova

FedAvgAggregator = FedAvg
FedNovaAggregator = FedNova

__all__ = [
    'BaseAggregator',
    'FedAvg',
    'FedNova',
    'FedAvgAggregator',
    'FedNovaAggregator',
]
