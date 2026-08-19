"""Static feature CNN/MLP model.

For processing static Android features (binary/one-hot encoded).

"""

from typing import List
import torch
import torch.nn as nn

from models.base_model import IncrementalModel, initialize_weights


class StaticCNN(IncrementalModel):
    """CNN for static features.

    Input: (batch_size, static_input_dim) binary/one-hot features
    Output: (batch_size, n_classes) logits
    """

    def __init__(
        self,
        input_dim: int = 500,  # After dimensionality reduction
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.5,
        initial_classes: int = 3
    ):
        """Initialize StaticCNN.

        Args:
            input_dim: Input feature dimension (after reduction).
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout probability.
            initial_classes: Number of classes in first task.
        """
        super().__init__(initial_classes)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build feature extractor
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # 1D Conv layer for local feature extraction
            if i == 0:
                # First layer: project to hidden dim
                layers.append(nn.Linear(prev_dim, hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))

            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Classifier (will be expanded during incremental learning)
        self.classifier = nn.Linear(hidden_dims[-1], initial_classes)

        # Initialize weights
        self.apply(initialize_weights)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier.

        Args:
            x: Input tensor (batch_size, input_dim).

        Returns:
            Feature tensor (batch_size, hidden_dims[-1]).
        """
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim).

        Returns:
            Logits tensor (batch_size, n_classes).
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits


class StaticMLP(IncrementalModel):
    """Simple MLP for static features.

    Alternative to CNN for baseline comparison.
    """

    def __init__(
        self,
        input_dim: int = 500,
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.5,
        initial_classes: int = 3
    ):
        """Initialize StaticMLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout probability.
            initial_classes: Number of classes in first task.
        """
        super().__init__(initial_classes)

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], initial_classes)

        self.apply(initialize_weights)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.get_features(x)
        return self.classifier(features)
