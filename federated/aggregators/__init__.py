"""Aggregators package."""

from federated.aggregators.base import BaseAggregator, FedAvg, FedNova

FedAvgAggregator = FedAvg
FedNovaAggregator = FedNova


def build_aggregator(aggregator_name: str) -> BaseAggregator:
    """Factory builder for Federated Aggregators."""
    name = aggregator_name.lower()
    if name == "fedavg":
        return FedAvg()
    elif name == "fednova":
        return FedNova()
    else:
        raise ValueError(f"Unknown aggregator: {aggregator_name}. Choose 'fedavg' or 'fednova'.")


__all__ = [
    'BaseAggregator',
    'FedAvg',
    'FedNova',
    'FedAvgAggregator',
    'FedNovaAggregator',
    'build_aggregator',
]
