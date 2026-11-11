"""Aggregators package."""

from federated.aggregators.base import BaseAggregator, FedAvg, FedNova, FedSGD

FedAvgAggregator = FedAvg
FedNovaAggregator = FedNova
FedSGDAggregator = FedSGD


def build_aggregator(aggregator_name: str, **kwargs) -> BaseAggregator:
    """Factory builder for Federated Aggregators."""
    name = aggregator_name.lower()
    if name == "fedavg":
        return FedAvg()
    elif name == "fednova":
        return FedNova()
    elif name == "fedsgd":
        return FedSGD(**kwargs)
    else:
        raise ValueError(
            f"Unknown aggregator: {aggregator_name}. Choose 'fedavg', 'fednova', or 'fedsgd'."
        )


__all__ = [
    'BaseAggregator',
    'FedAvg',
    'FedNova',
    'FedSGD',
    'FedAvgAggregator',
    'FedNovaAggregator',
    'FedSGDAggregator',
    'build_aggregator',
]
