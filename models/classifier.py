"""
Dynamically Expanding Incremental Classification Head.
Implements weight-preserving expansion for sequential class additions across FCIL tasks.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicIncrementalClassifier(nn.Module):
    """
    Expanding Linear Classifier Head that grows its output dimension dynamically
    as new malware families are introduced in each task, preserving previously learned weights.
    """

    def __init__(
        self,
        in_features: int,
        initial_classes: int = 3,
        max_classes: int = 15,
        use_cosine_norm: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.max_classes = max_classes
        self.current_classes = initial_classes
        self.use_cosine_norm = use_cosine_norm

        self.weight = nn.Parameter(torch.empty((initial_classes, in_features)))
        self.bias = nn.Parameter(torch.empty(initial_classes)) if not use_cosine_norm else None
        self._init_weights(0, initial_classes)

    def _init_weights(self, start_idx: int, end_idx: int) -> None:
        """Kaiming uniform weight initialization for class slice [start_idx:end_idx]."""
        nn.init.kaiming_uniform_(self.weight[start_idx:end_idx], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[start_idx:end_idx])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias[start_idx:end_idx], -bound, bound)

    def expand_classes(self, num_new_classes: int) -> None:
        """
        Dynamically allocate output rows for newly introduced classes while preserving
        weights of already trained classes.
        """
        old_classes = self.current_classes
        new_classes = old_classes + num_new_classes

        if new_classes > self.max_classes:
            new_classes = self.max_classes

        if new_classes <= old_classes:
            return

        old_weight = self.weight.data
        old_bias = self.bias.data if self.bias is not None else None

        # Re-allocate parameter tensors
        self.weight = nn.Parameter(torch.empty((new_classes, self.in_features), device=old_weight.device))
        self.weight.data[:old_classes] = old_weight

        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty(new_classes, device=old_bias.device))
            self.bias.data[:old_classes] = old_bias

        # Initialize only the newly added rows
        self._init_weights(old_classes, new_classes)
        self.current_classes = new_classes

    def forward(self, x: torch.Tensor, limit_to_current: bool = True) -> torch.Tensor:
        """
        Compute logits over active classes.
        """
        if self.use_cosine_norm:
            w_norm = F.normalize(self.weight, p=2, dim=1)
            x_norm = F.normalize(x, p=2, dim=1)
            logits = F.linear(x_norm, w_norm)
        else:
            logits = F.linear(x, self.weight, self.bias)

        if limit_to_current:
            return logits[:, :self.current_classes]
        return logits
