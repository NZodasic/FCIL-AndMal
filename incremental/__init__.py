"""Incremental learning strategies package."""

from incremental.base_strategy import IncrementalStrategy
from incremental.fine_tune import FineTune
from incremental.joint import JointTraining
from incremental.ewc import EWC
from incremental.lwf import LwF
from incremental.replay import Replay
from incremental.spcil import SPCIL
from incremental.malfsil import MALFSIL

__all__ = [
    'IncrementalStrategy',
    'FineTune',
    'JointTraining',
    'EWC',
    'LwF',
    'Replay',
    'SPCIL',
    'MALFSIL',
]
