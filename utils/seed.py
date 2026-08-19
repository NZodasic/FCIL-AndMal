"""
Reproducibility and Determinism Utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set seeds across all random number generators for strict academic reproducibility.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_slot_seed(base_seed: int, task_id: int, client_id: int) -> int:
    """
    Compute deterministic slot seed for data partitioning per client and task.
    Formula: base_seed ^ (task_id * 1000 + client_id)
    """
    return base_seed ^ (task_id * 1000 + client_id)
