"""Federated Concept Drift Detection Module (FL-MalDrift).

Provides client-side streaming drift detectors, unified normalization,
and server-side dynamic EWMA participation threshold controller.
"""

from federated.drift.detectors import (
    BaseDriftDetector,
    ADWINDetector,
    DDMDetector,
    EDDMDetector,
    HDDMDetector,
    KLDriftDetector,
    PageHinkleyDetector,
    build_drift_detector,
)
from federated.drift.normalizer import DriftScoreNormalizer
from federated.drift.controller import ServerDriftController, AdaptiveLambdaController

__all__ = [
    "BaseDriftDetector",
    "ADWINDetector",
    "DDMDetector",
    "EDDMDetector",
    "HDDMDetector",
    "KLDriftDetector",
    "PageHinkleyDetector",
    "build_drift_detector",
    "DriftScoreNormalizer",
    "ServerDriftController",
    "AdaptiveLambdaController",
]
