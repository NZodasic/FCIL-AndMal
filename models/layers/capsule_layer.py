"""Capsule Network layer.

Based on Sabour et al. "Dynamic Routing Between Capsules" (2017)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLayer(nn.Module):
    """Capsule layer with dynamic routing.

    Reference: Sabour et al. "Dynamic Routing Between Capsules" (NIPS 2017)
    """

    def __init__(
        self,
        num_capsules: int,
        num_route_nodes: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = None,
        stride: int = None,
        num_iterations: int = 3
    ):
        """Initialize CapsuleLayer.

        Args:
            num_capsules: Number of output capsules.
            num_route_nodes: Number of input capsules (for routing).
            in_channels: Input channels per capsule.
            out_channels: Output channels per capsule (capsule dimension).
            kernel_size: Kernel size for convolutional capsules.
            stride: Stride for convolutional capsules.
            num_iterations: Number of routing iterations.
        """
        super().__init__()

        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        if num_route_nodes != -1:
            # Fully connected capsules with routing
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)
            )
        else:
            # Convolutional capsules without routing
            self.capsules = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride)
                for _ in range(num_capsules)
            ])

    def squash(self, s: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Squash activation function.

        Args:
            s: Input tensor.
            dim: Dimension to squash along.

        Returns:
            Squashed tensor.
        """
        squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * s / (squared_norm.sqrt() + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic routing.

        Args:
            x: Input tensor.

        Returns:
            Output capsules.
        """
        if self.num_route_nodes != -1:
            # Dynamic routing
            # x: (batch, num_route_nodes, in_channels)
            batch_size = x.size(0)

            # Compute predictions
            # u: (batch, num_route_nodes, in_channels, 1)
            u = x[:, None, :, :, None]

            # w: (num_capsules, num_route_nodes, in_channels, out_channels)
            w = self.route_weights[None, :, :, :, :]

            # u_hat: (batch, num_capsules, num_route_nodes, out_channels)
            u_hat = (w * u).sum(dim=3)

            # Routing coefficients
            b = torch.zeros(batch_size, self.num_capsules, self.num_route_nodes, 1,
                          device=x.device)

            # Dynamic routing
            for i in range(self.num_iterations):
                c = F.softmax(b, dim=1)  # (batch, num_capsules, num_route_nodes, 1)
                s = (c * u_hat).sum(dim=2)  # (batch, num_capsules, out_channels)
                v = self.squash(s)  # (batch, num_capsules, out_channels)

                if i < self.num_iterations - 1:
                    # Update routing coefficients
                    b = b + (u_hat * v[:, :, None, :]).sum(dim=-1, keepdim=True)

            return v
        else:
            # Convolutional capsules
            outputs = [capsule(x) for capsule in self.capsules]
            outputs = torch.stack(outputs, dim=1)
            return self.squash(outputs)


class PrimaryCapsule(nn.Module):
    """Primary capsule layer (convolutional capsules)."""

    def __init__(
        self,
        num_capsules: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1
    ):
        """Initialize PrimaryCapsule.

        Args:
            num_capsules: Number of capsule types.
            in_channels: Input channels.
            out_channels: Output channels per capsule.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
        """
        super().__init__()

        self.capsules = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, stride)
            for _ in range(num_capsules)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, in_channels, seq_len).

        Returns:
            Output capsules (batch, num_capsules, out_channels, out_len).
        """
        outputs = [capsule(x) for capsule in self.capsules]
        outputs = torch.stack(outputs, dim=1)  # (batch, num_caps, out_ch, out_len)

        # Flatten spatial dimensions
        batch_size = outputs.size(0)
        num_caps = outputs.size(1)
        out_ch = outputs.size(2)

        # Reshape to (batch, num_caps * out_len, out_ch)
        outputs = outputs.view(batch_size, num_caps * outputs.size(3), out_ch)

        # Apply squash
        squared_norm = (outputs ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        outputs = scale * outputs / (squared_norm.sqrt() + 1e-8)

        return outputs
