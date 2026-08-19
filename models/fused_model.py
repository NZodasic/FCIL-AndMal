"""Fused model combining static and dynamic features.

"""

from typing import List
import torch
import torch.nn as nn

from models.base_model import IncrementalModel, initialize_weights


class FusedModel(IncrementalModel):
    """Fused model combining static and dynamic features.

    Architecture:
    1. Static features -> Static encoder
    2. Dynamic features -> Dynamic encoder
    3. Concatenate embeddings
    4. Fusion layer
    5. Classifier
    """

    def __init__(
        self,
        static_input_dim: int = 500,
        dynamic_input_dim: int = 141,
        use_before_after: bool = False,
        static_hidden_dims: List[int] = [256, 128],
        dynamic_hidden_dims: List[int] = [128, 64],
        fusion_dim: int = 128,
        dropout_rate: float = 0.5,
        fusion_method: str = 'concat',  # 'concat', 'attention', 'gated'
        initial_classes: int = 3
    ):
        """Initialize FusedModel.

        Args:
            static_input_dim: Dimension of static features.
            dynamic_input_dim: Dimension of dynamic features.
            use_before_after: Whether dynamic has before/after channels.
            static_hidden_dims: Hidden dims for static encoder.
            dynamic_hidden_dims: Hidden dims for dynamic encoder.
            fusion_dim: Dimension of fused representation.
            dropout_rate: Dropout probability.
            fusion_method: How to fuse features.
            initial_classes: Number of classes in first task.
        """
        super().__init__(initial_classes)

        self.fusion_method = fusion_method

        # Static encoder
        static_layers = []
        prev_dim = static_input_dim
        for hidden_dim in static_hidden_dims:
            static_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.static_encoder = nn.Sequential(*static_layers)
        self.static_dim = static_hidden_dims[-1]

        # Dynamic encoder
        if use_before_after:
            dynamic_feature_dim = dynamic_input_dim // 2
            dynamic_in_channels = 2
        else:
            dynamic_feature_dim = dynamic_input_dim
            dynamic_in_channels = 1

        dynamic_conv_layers = []
        in_ch = dynamic_in_channels
        out_ch = 32
        for _ in range(2):
            dynamic_conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
            out_ch *= 2

        self.dynamic_conv = nn.Sequential(*dynamic_conv_layers)
        dynamic_flat_size = in_ch * dynamic_feature_dim

        dynamic_fc_layers = []
        prev_dim = dynamic_flat_size
        for hidden_dim in dynamic_hidden_dims:
            dynamic_fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.dynamic_fc = nn.Sequential(*dynamic_fc_layers)
        self.dynamic_dim = dynamic_hidden_dims[-1]

        # Fusion
        combined_dim = self.static_dim + self.dynamic_dim

        if fusion_method == 'concat':
            fusion_input_dim = combined_dim
        elif fusion_method == 'attention':
            fusion_input_dim = combined_dim
            self.attention = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.Tanh(),
                nn.Linear(combined_dim // 2, 2),
                nn.Softmax(dim=1)
            )
        elif fusion_method == 'gated':
            fusion_input_dim = combined_dim
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Classifier
        self.classifier = nn.Linear(fusion_dim, initial_classes)

        self.apply(initialize_weights)

    def encode_static(self, x: torch.Tensor) -> torch.Tensor:
        """Encode static features.

        Args:
            x: Static features (batch, static_input_dim).

        Returns:
            Static embedding (batch, static_dim).
        """
        return self.static_encoder(x)

    def encode_dynamic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode dynamic features.

        Args:
            x: Dynamic features (batch, dynamic_input_dim).

        Returns:
            Dynamic embedding (batch, dynamic_dim).
        """
        # Reshape for conv
        if x.dim() == 2:
            # Assume single channel
            x = x.unsqueeze(1)

        x = self.dynamic_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dynamic_fc(x)
        return x

    def fuse_features(
        self,
        static_emb: torch.Tensor,
        dynamic_emb: torch.Tensor
    ) -> torch.Tensor:
        """Fuse static and dynamic embeddings.

        Args:
            static_emb: Static embedding (batch, static_dim).
            dynamic_emb: Dynamic embedding (batch, dynamic_dim).

        Returns:
            Fused embedding (batch, fusion_dim).
        """
        combined = torch.cat([static_emb, dynamic_emb], dim=1)

        if self.fusion_method == 'concat':
            fused = combined
        elif self.fusion_method == 'attention':
            weights = self.attention(combined)  # (batch, 2)
            static_weight = weights[:, 0:1]
            dynamic_weight = weights[:, 1:2]
            fused = torch.cat([
                static_emb * static_weight,
                dynamic_emb * dynamic_weight
            ], dim=1)
        elif self.fusion_method == 'gated':
            gate = self.gate(combined)
            fused = combined * gate
        else:
            fused = combined

        return self.fusion(fused)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (not used for fused model)."""
        raise NotImplementedError("Use forward_with_components for fused model")

    def forward_with_components(
        self,
        static_x: torch.Tensor,
        dynamic_x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with separate inputs.

        Args:
            static_x: Static features (batch, static_input_dim).
            dynamic_x: Dynamic features (batch, dynamic_input_dim).

        Returns:
            Logits (batch, n_classes).
        """
        static_emb = self.encode_static(static_x)
        dynamic_emb = self.encode_dynamic(dynamic_x)
        fused = self.fuse_features(static_emb, dynamic_emb)
        logits = self.classifier(fused)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with concatenated input.

        Args:
            x: Concatenated static+dynamic features.

        Returns:
            Logits.
        """
        # Split input (assumes static first, then dynamic)
        static_x = x[:, :self.static_encoder[0].in_features]
        dynamic_x = x[:, self.static_encoder[0].in_features:]
        return self.forward_with_components(static_x, dynamic_x)
