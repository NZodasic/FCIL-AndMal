"""Dynamic feature CNN/TCN model.

For processing dynamic Android features (numerical counts/measurements).

"""

from typing import List
import torch
import torch.nn as nn

from models.base_model import IncrementalModel, initialize_weights
from models.layers.tcn_layer import TemporalConvNet


class DynamicCNN(IncrementalModel):
    """CNN for dynamic features.

    Input: (batch_size, input_dim) numerical features
           or (batch_size, 2, input_dim//2) for before+after
    Output: (batch_size, n_classes) logits
    """

    def __init__(
        self,
        input_dim: int = 141,
        use_before_after: bool = False,
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.5,
        initial_classes: int = 3
    ):
        """Initialize DynamicCNN.

        Args:
            input_dim: Input feature dimension (141 or 282).
            use_before_after: Whether to treat as 2 channels (before/after).
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout probability.
            initial_classes: Number of classes in first task.
        """
        super().__init__(initial_classes)

        self.input_dim = input_dim
        self.use_before_after = use_before_after

        if use_before_after:
            # Treat as 2 channels: (batch, 2, input_dim//2)
            feature_dim = input_dim // 2
            self.input_channels = 2
        else:
            feature_dim = input_dim
            self.input_channels = 1

        # 1D Convolution layers
        conv_layers = []
        in_channels = self.input_channels
        out_channels = 64

        for _ in range(2):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate / 2)
            ])
            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened size
        self.flat_size = in_channels * feature_dim

        # Fully connected layers
        fc_layers = []
        prev_dim = self.flat_size

        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.fc_layers = nn.Sequential(*fc_layers)

        # Classifier
        self.classifier = nn.Linear(hidden_dims[-1], initial_classes)

        self.apply(initialize_weights)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        if self.use_before_after:
            # x: (batch, input_dim) -> (batch, 2, input_dim//2)
            batch_size = x.size(0)
            x = x.view(batch_size, 2, -1)
        else:
            # x: (batch, input_dim) -> (batch, 1, input_dim)
            x = x.unsqueeze(1)

        # Conv layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc_layers(x)

        return x

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


class DynamicTCN(IncrementalModel):
    """TCN for dynamic features.

    Uses Temporal Convolutional Network to capture temporal patterns
    in dynamic features.

    Reference: Bai et al. "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling" (2018)
    """

    def __init__(
        self,
        input_dim: int = 141,
        use_before_after: bool = False,
        num_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
        initial_classes: int = 3
    ):
        """Initialize DynamicTCN.

        Args:
            input_dim: Input feature dimension.
            use_before_after: Whether to treat as 2 channels.
            num_channels: List of TCN channel sizes.
            kernel_size: Convolution kernel size.
            dropout_rate: Dropout probability.
            initial_classes: Number of classes in first task.
        """
        super().__init__(initial_classes)

        self.input_dim = input_dim
        self.use_before_after = use_before_after

        if use_before_after:
            feature_dim = input_dim // 2
            input_channels = 2
        else:
            feature_dim = input_dim
            input_channels = 1

        # TCN layers
        self.tcn = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout_rate
        )

        # Global average pooling + classifier
        self.classifier = nn.Linear(num_channels[-1], initial_classes)

        self.apply(initialize_weights)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using TCN.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        if self.use_before_after:
            batch_size = x.size(0)
            x = x.view(batch_size, 2, -1)
        else:
            x = x.unsqueeze(1)

        # TCN: (batch, channels, seq_len) -> (batch, channels, seq_len)
        x = self.tcn(x)

        # Global average pooling over sequence
        x = x.mean(dim=2)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits
