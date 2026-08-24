"""MalFSCIL: few-shot class-incremental learning for malware detection.

This tabular adaptation follows Chai et al. (TIFS 2024): base-session
classification is co-trained with a VAE objective, then the feature extractor is
frozen and classifier prototypes evolve through graph attention under softmax
and additive angular-margin objectives.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseILMethod
from models.fcil_model import FCILNet
from models.malfscil import (
    PrototypeGraphAttention,
    VariationalFeatureAdapter,
    additive_angular_margin_logits,
    cosine_logits,
)


class MalFSCILMethod(BaseILMethod):
    """Paper-aligned MalFSCIL training strategy."""

    def __init__(
        self,
        *,
        vae_weight: float = 1.0,
        kl_weight: float = 1.0,
        arc_weight: float = 0.5,
        arc_scale: float = 30.0,
        arc_margin: float = 0.5,
        graph_attention_dim: int = 64,
    ):
        super().__init__(name="malfscil")
        if not 0.0 <= arc_weight <= 1.0:
            raise ValueError("arc_weight must be in [0, 1]")
        if vae_weight < 0.0 or kl_weight < 0.0:
            raise ValueError("VAE loss weights cannot be negative")
        if arc_scale <= 0.0 or arc_margin < 0.0:
            raise ValueError("ArcFace scale must be positive and margin non-negative")

        self.vae_weight = vae_weight
        self.kl_weight = kl_weight
        self.arc_weight = arc_weight
        self.arc_scale = arc_scale
        self.arc_margin = arc_margin
        self.graph_attention_dim = graph_attention_dim
        self.current_task = 0
        self.vae: Optional[VariationalFeatureAdapter] = None
        self.prototype_graph: Optional[PrototypeGraphAttention] = None

    def _initialize_modules(self, model: FCILNet, device: torch.device) -> None:
        if self.vae is None:
            self.vae = VariationalFeatureAdapter(
                input_dim=model.config.input_dim,
                latent_dim=model.config.latent_dim,
            ).to(device)
        if self.prototype_graph is None:
            self.prototype_graph = PrototypeGraphAttention(
                feature_dim=model.config.latent_dim,
                attention_dim=self.graph_attention_dim,
            ).to(device)

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        modules = (self.vae, self.prototype_graph)
        for module in modules:
            if module is not None:
                yield from (p for p in module.parameters() if p.requires_grad)

    @staticmethod
    def _set_feature_extractor_trainable(model: FCILNet, trainable: bool) -> None:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = trainable

    def before_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.current_task = task_id
        self._initialize_modules(model, device)
        if task_id == 0:
            self._set_feature_extractor_trainable(model, True)
            for parameter in self.vae.parameters():
                parameter.requires_grad = True
            return

        # The paper freezes F after the base session and trains only C_new.
        self._set_feature_extractor_trainable(model, False)
        for parameter in self.vae.parameters():
            parameter.requires_grad = False
        model.classifier.use_cosine_norm = True
        if train_loader is not None:
            self._write_support_prototypes(model, train_loader, device)

    def compute_loss(
        self,
        model: FCILNet,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        task_id: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._initialize_modules(model, device)
        if task_id == 0:
            mean = model.get_features(x)
            latent, reconstruction, log_variance = self.vae(mean, sample=True)
            logits = model.classifier(latent, limit_to_current=True)
            classification_loss = criterion(logits, y)
            reconstruction_loss, kl_loss = self.vae.loss(
                reconstruction, x, mean, log_variance
            )
            vae_loss = reconstruction_loss + self.kl_weight * kl_loss
            total_loss = classification_loss + self.vae_weight * vae_loss
            return total_loss, {
                "ce_loss": float(classification_loss.item()),
                "reconstruction_loss": float(reconstruction_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "vae_loss": float(vae_loss.item()),
                "total_loss": float(total_loss.item()),
            }

        # Keep BatchNorm statistics fixed along with the frozen parameters.
        model.backbone.eval()
        with torch.no_grad():
            features = model.get_features(x)
        active_prototypes = model.classifier.weight[:model.current_classes]
        evolved_prototypes, attention = self.prototype_graph(active_prototypes)
        softmax_loss = F.cross_entropy(
            cosine_logits(features, evolved_prototypes, self.arc_scale), y
        )
        arc_loss = F.cross_entropy(
            additive_angular_margin_logits(
                features,
                evolved_prototypes,
                y,
                scale=self.arc_scale,
                margin=self.arc_margin,
            ),
            y,
        )
        total_loss = (
            (1.0 - self.arc_weight) * softmax_loss
            + self.arc_weight * arc_loss
        )
        return total_loss, {
            "softmax_loss": float(softmax_loss.item()),
            "arc_loss": float(arc_loss.item()),
            "attention_entropy": float(
                (-(attention * attention.clamp_min(1e-12).log()).sum(dim=1).mean()).item()
            ),
            "total_loss": float(total_loss.item()),
        }

    def _write_support_prototypes(
        self,
        model: FCILNet,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        was_training = model.training
        model.eval()
        features_by_class: Dict[int, List[torch.Tensor]] = {}
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                features = model.get_features(batch_x.to(device))
                for class_id in torch.unique(batch_y).tolist():
                    class_mask = batch_y == class_id
                    features_by_class.setdefault(int(class_id), []).append(
                        features[class_mask.to(device)].detach()
                    )

            for class_id, feature_parts in features_by_class.items():
                if class_id >= model.current_classes:
                    raise ValueError(
                        f"Class {class_id} exceeds the active classifier size"
                    )
                prototype = torch.cat(feature_parts, dim=0).mean(dim=0)
                model.classifier.weight[class_id].copy_(
                    F.normalize(prototype, p=2, dim=0)
                )
                if model.classifier.bias is not None:
                    model.classifier.bias[class_id].zero_()
        if was_training:
            model.train()

    def after_task(
        self,
        task_id: int,
        model: FCILNet,
        train_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if task_id == 0 and train_loader is not None:
            self._write_support_prototypes(model, train_loader, device)
        model.classifier.use_cosine_norm = True
        if task_id > 0:
            with torch.no_grad():
                active = model.classifier.weight[:model.current_classes]
                evolved, _ = self.prototype_graph(active)
                model.classifier.weight[:model.current_classes].copy_(evolved)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current_task": self.current_task,
            "vae_weight": self.vae_weight,
            "kl_weight": self.kl_weight,
            "arc_weight": self.arc_weight,
            "arc_scale": self.arc_scale,
            "arc_margin": self.arc_margin,
            "graph_attention_dim": self.graph_attention_dim,
            "vae": (
                {key: value.cpu() for key, value in self.vae.state_dict().items()}
                if self.vae is not None
                else None
            ),
            "prototype_graph": (
                {
                    key: value.cpu()
                    for key, value in self.prototype_graph.state_dict().items()
                }
                if self.prototype_graph is not None
                else None
            ),
        }
