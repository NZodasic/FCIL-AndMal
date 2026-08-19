"""Temporal Convolutional Network (TCN) layer.

Based on Bai et al. "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" (2018)

"""

from typing import List
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Chomp layer to remove extra padding from causal convolution."""

    def __init__(self, chomp_size: int):
        """Initialize Chomp1d.

        Args:
            chomp_size: Number of elements to remove from end.
        """
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, channels, seq_len).

        Returns:
            Chomped tensor (batch, channels, seq_len - chomp_size).
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single TCN temporal block with residual connection."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        """Initialize TemporalBlock.

        Args:
            n_inputs: Number of input channels.
            n_outputs: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            dilation: Convolution dilation.
            padding: Convolution padding.
            dropout: Dropout probability.
        """
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize network weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (batch, channels, seq_len).

        Returns:
            Output tensor (batch, channels, seq_len).
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network.

    Stack of temporal blocks with exponentially increasing dilation.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """Initialize TemporalConvNet.

        Args:
            num_inputs: Number of input channels.
            num_channels: List of output channels for each layer.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, channels, seq_len).

        Returns:
            Output tensor (batch, channels, seq_len).
        """
        return self.network(x)
