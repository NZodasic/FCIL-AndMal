"""Base model class with incremental learning support.

"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class IncrementalModel(nn.Module):
    """Base class for incremental learning models.

    Supports expanding classifier heads as new tasks arrive.
    """

    def __init__(self, initial_classes: int = 3):
        """Initialize model.

        Args:
            initial_classes: Number of classes in first task.
        """
        super().__init__()
        self.current_classes = initial_classes
        self.seen_classes = initial_classes
        self.classifier = None

    def expand_classifier(self, new_classes: int) -> None:
        """Expand classifier to accommodate new classes.

        This method implements the "expanding head" approach for
        class-incremental learning. New neurons are added to the
        output layer for new classes.

        Args:
            new_classes: Number of new classes to add.
        """
        if self.classifier is None:
            raise ValueError("Classifier not initialized")

        old_classes = self.current_classes
        self.current_classes += new_classes
        self.seen_classes = self.current_classes

        # Get old classifier weights
        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data if self.classifier.bias is not None else None

        # Create new classifier
        in_features = self.classifier.in_features
        new_classifier = nn.Linear(in_features, self.current_classes)

        # Copy old weights
        with torch.no_grad():
            new_classifier.weight[:old_classes] = old_weight
            if old_bias is not None:
                new_classifier.bias[:old_classes] = old_bias

        self.classifier = new_classifier

    def freeze_old_params(self) -> None:
        """Freeze parameters related to old classes.

        Used in some incremental learning strategies to prevent
        catastrophic forgetting.
        """
        # To be implemented by subclasses if needed
        pass

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Logits tensor.
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits

    def get_old_logits(self, x: torch.Tensor, n_old_classes: int) -> torch.Tensor:
        """Get logits for old classes only.

        Used for knowledge distillation in LwF and related methods.

        Args:
            x: Input tensor.
            n_old_classes: Number of old classes.

        Returns:
            Logits for old classes.
        """
        logits = self.forward(x)
        return logits[:, :n_old_classes]


def initialize_weights(module: nn.Module) -> None:
    """Initialize network weights.

    Uses Kaiming initialization for conv layers and
    Xavier initialization for linear layers.

    Args:
        module: PyTorch module.
    """
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
