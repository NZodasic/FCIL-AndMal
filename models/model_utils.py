"""
Model Diagnostics and Inspection Utilities.
Provides parameter counting, model summaries, and parameter freezing/copy helpers.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a PyTorch module.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Produce structured summary dictionary of model parameters and layer structure.
    """
    total, trainable = count_parameters(model)
    layer_info = []
    for name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        layer_info.append({"name": name, "type": module.__class__.__name__, "params": mod_total})

    return {
        "model_class": model.__class__.__name__,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "layers": layer_info,
    }


def freeze_parameters(model: nn.Module) -> None:
    """Freeze all model parameters (set requires_grad = False)."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_parameters(model: nn.Module) -> None:
    """Unfreeze all model parameters (set requires_grad = True)."""
    for param in model.parameters():
        param.requires_grad = True
