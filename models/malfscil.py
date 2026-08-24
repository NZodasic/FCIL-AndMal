"""Core neural modules for the MalFSCIL method."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalFeatureAdapter(nn.Module):
    """VAE heads around the project's feature backbone for tabular malware data."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(latent_dim, min(256, input_dim * 2))
        self.log_variance = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self, mean: torch.Tensor, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_variance = self.log_variance(mean)
        if sample:
            standard_deviation = torch.exp(0.5 * log_variance)
            latent = mean + standard_deviation * torch.randn_like(standard_deviation)
        else:
            latent = mean
        reconstruction = self.decoder(latent)
        return latent, reconstruction, log_variance

    @staticmethod
    def loss(
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gaussian reconstruction NLL adapted to unbounded tabular counts. The
        # denominator prevents high-magnitude columns from overwhelming CE/KL.
        reconstruction_scale = target.detach().pow(2).mean().clamp_min(1e-8)
        reconstruction_loss = F.mse_loss(reconstruction, target) / reconstruction_scale
        kl_loss = -0.5 * torch.mean(
            1.0 + log_variance - mean.pow(2) - log_variance.exp()
        )
        return reconstruction_loss, kl_loss


class PrototypeGraphAttention(nn.Module):
    """Evolve all classifier prototypes as a fully connected attention graph."""

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.query = nn.Linear(feature_dim, attention_dim, bias=False)
        self.key = nn.Linear(feature_dim, attention_dim, bias=False)

    def forward(
        self, prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self.query(prototypes)
        keys = self.key(prototypes)
        scores = queries @ keys.transpose(0, 1) / math.sqrt(keys.size(1))
        attention = F.softmax(scores, dim=1)
        evolved = F.normalize(prototypes + attention @ prototypes, p=2, dim=1)
        return evolved, attention


def cosine_logits(
    features: torch.Tensor, prototypes: torch.Tensor, scale: float
) -> torch.Tensor:
    """Compute scaled cosine classifier logits."""
    return scale * F.linear(
        F.normalize(features, p=2, dim=1),
        F.normalize(prototypes, p=2, dim=1),
    )


def additive_angular_margin_logits(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    *,
    scale: float,
    margin: float,
) -> torch.Tensor:
    """Apply the paper's additive angular margin to target-class logits."""
    cosine = F.linear(
        F.normalize(features, p=2, dim=1),
        F.normalize(prototypes, p=2, dim=1),
    ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    target_cosine = torch.cos(
        torch.acos(cosine.gather(1, labels.unsqueeze(1))) + margin
    )
    margin_cosine = cosine.scatter(1, labels.unsqueeze(1), target_cosine)
    return scale * margin_cosine
