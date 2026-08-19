"""
Full Federated Class-Incremental Neural Network (FCILNet).
Combines configurable feature backbones with dynamically expanding incremental classifier heads.
"""

from typing import Tuple, Union, Optional, Dict, Any
import copy
import torch
import torch.nn as nn

from config import ModelConfig
from models.backbones import build_backbone
from models.classifier import DynamicIncrementalClassifier


class FCILNet(nn.Module):
    """
    Unified Architecture for Federated Class-Incremental Learning on CIC-AndMal-2020.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(
            backbone_type=config.backbone_type,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            static_dim=config.static_input_dim,
            dynamic_dim=config.dynamic_input_dim,
            dropout=config.dropout
        )
        self.classifier = DynamicIncrementalClassifier(
            in_features=config.latent_dim,
            initial_classes=config.classes_per_task,  # Starts with 3 classes for Task 0
            max_classes=config.num_total_classes      # 15 classes
        )
        self.current_classes = config.classes_per_task

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        limit_to_current: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.backbone(x)
        logits = self.classifier(features, limit_to_current=limit_to_current)
        if return_features:
            return logits, features
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation before classification layer."""
        return self.backbone(x)

    def expand_classes(self, num_new_classes: int = 3) -> None:
        """Expand classifier head to accommodate newly introduced malware families."""
        self.classifier.expand_classes(num_new_classes)
        self.current_classes = self.classifier.current_classes

    def clone_model(self) -> "FCILNet":
        """Create deep copy of model for snapshotting / teacher distillation."""
        return copy.deepcopy(self)

    def get_state_dict_cpu(self) -> Dict[str, Any]:
        """Export state dictionary with all tensors on CPU for serialization."""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}
