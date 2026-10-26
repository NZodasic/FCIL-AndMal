"""
Loss Functions for Falcon and FCIL Framework.
Implements:
1. FocalLoss: Multi-class focal loss with focusing parameter gamma and class weighting alpha
   (Lin et al., ICCV 2017; Yang et al., IEEE TCE 2026, Eqs. 3-5).
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for mitigating extreme class imbalance in network intrusion
    and Android malware detection.
    
    Formula:
        p_{t, i} = exp(-\\ell_{CE}(z_i, y_i))
        \\ell_{Focal}(x_i, y_i) = -\\alpha_{y_i} * (1 - p_{t, i})^\\gamma * \\log(p_{t, i})
                               = \\alpha_{y_i} * (1 - p_{t, i})^\\gamma * \\ell_{CE}(z_i, y_i)
    
    Args:
        alpha: Class weighting factor. Can be:
               - None: uniform weighting (alpha = 1.0)
               - float: scalar weight applied uniformly
               - torch.Tensor: per-class weight vector of shape (C,)
        gamma: Focusing parameter controlling suppression of easy negative examples (default: 2.0).
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            raise TypeError(f"Unsupported alpha type: {type(alpha)}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss over mini-batch.
        
        Args:
            logits: Predicted class logits (batch_size, num_classes).
            targets: Ground-truth class labels (batch_size,).
            
        Returns:
            Computed scalar loss (if reduction in ['mean', 'sum']) or per-sample loss.
        """
        if logits.dim() < 2:
            raise ValueError(f"Expected 2D logits (batch_size, num_classes), got shape {logits.shape}")

        num_classes = logits.size(1)

        # Standard cross entropy per sample (unreduced)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # p_t: probability of true class
        p_t = torch.exp(-ce_loss)

        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t).clamp(min=0.0, max=1.0) ** self.gamma

        focal_loss = modulating_factor * ce_loss

        # Apply class balancing alpha if provided
        if self.alpha is not None:
            if self.alpha.dim() == 0:
                focal_loss = self.alpha.to(logits.device) * focal_loss
            elif self.alpha.dim() == 1:
                alpha_tensor = self.alpha.to(logits.device)
                if alpha_tensor.size(0) >= num_classes:
                    alpha_t = alpha_tensor[targets]
                else:
                    # Pad alpha if model expanded classes dynamically
                    pad = torch.ones(num_classes - alpha_tensor.size(0), device=logits.device)
                    alpha_tensor = torch.cat([alpha_tensor, pad])
                    alpha_t = alpha_tensor[targets]
                focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
