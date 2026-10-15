"""Utilities package."""

from utils.checkpoint import CheckpointManager
from utils.device import resolve_device, get_device_info, is_mps_available, is_cuda_available, clear_device_cache
from utils.logging import ExperimentLogger, get_logger
from utils.metrics import (
    ContinualEvaluationMatrix,
    compute_classification_metrics,
    compute_fscil_session_metrics,
)

__all__ = [
    'CheckpointManager',
    'ExperimentLogger',
    'get_logger',
    'compute_classification_metrics',
    'compute_fscil_session_metrics',
    'ContinualEvaluationMatrix',
    'resolve_device',
    'get_device_info',
    'is_mps_available',
    'is_cuda_available',
    'clear_device_cache',
]

