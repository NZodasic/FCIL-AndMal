"""
Neural Network Backbones for Android Malware Feature Representation.
Implements:
1. Hybrid TCN + CNN Backbone (Primary Proposed Architecture for Dynamic Telemetry)
2. 1D-CNN Backbone (Ablation)
3. TCN Backbone (Ablation)
4. MLP Backbone (Static Features Baseline)
5. Fused Multi-Modal Backbone (Static MLP + Dynamic Hybrid TCN-CNN)
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNormContext:
    """Context manager preventing BatchNorm crashes when a training batch has size 1.

    When batch_size is 1, PyTorch's BatchNorm raises ValueError because sample variance
    cannot be calculated with N=1. Temporarily setting training BatchNorms to eval mode
    uses running statistics for that single-sample forward pass without affecting gradients.
    """
    def __init__(self, module: nn.Module, batch_size: int):
        self.module = module
        self.single_sample = (batch_size == 1 and module.training)
        self.bns = []

    def __enter__(self):
        if self.single_sample:
            for m in self.module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.training:
                    m.eval()
                    self.bns.append(m)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.bns:
            m.train()


class MLPBackbone(nn.Module):
    """
    Multi-Layer Perceptron encoder with Batch Normalization, LeakyReLU, and Dropout.
    Suitable for static sparse features and general tabular inputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.2,
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(latent_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with SafeBatchNormContext(self, x.size(0)):
            return self.net(x)


class CNN1DBackbone(nn.Module):
    """
    1D-Convolutional Neural Network for local pattern extraction from dynamic runtime metrics.
    Treats input features as 1D sequence (batch_size, 1, seq_len).
    """

    def __init__(
        self,
        input_dim: int,
        channels: List[int] = [32, 64, 128],
        latent_dim: int = 64,
        dropout: float = 0.2,
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        conv_layers = []
        in_c = 1
        for out_c in channels:
            conv_layers.append(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1, bias=False))
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm1d(out_c))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            if dropout > 0:
                conv_layers.append(nn.Dropout(p=dropout))
            in_c = out_c

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with SafeBatchNormContext(self, x.size(0)):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            feat_maps = self.conv(x)
            pooled = self.pool(feat_maps).squeeze(2)
            out = self.fc(pooled)
            return out


class TemporalBlock(nn.Module):
    """Dilated residual causal block for Temporal Convolutional Networks (TCN)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with SafeBatchNormContext(self, x.size(0)):
            res = x if self.downsample is None else self.downsample(x)
            out = self.conv1(x)
            # Chomp causal padding
            out = out[:, :, :-self.padding] if self.padding > 0 else out
            out = self.dropout1(self.relu1(self.bn1(out)))

            out = self.conv2(out)
            out = out[:, :, :-self.padding] if self.padding > 0 else out
            out = self.dropout2(self.relu2(self.bn2(out)))

            return F.relu(out + res)


class TCNBackbone(nn.Module):
    """
    Temporal Convolutional Network backbone for sequential/temporal dynamic features.
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: List[int] = [32, 64, 128],
        latent_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = []
        in_c = 1
        for i, out_c in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_c, out_c, kernel_size, stride=1, dilation=dilation, dropout=dropout))
            in_c = out_c

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with SafeBatchNormContext(self, x.size(0)):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out = self.tcn(x)
            pooled = self.pool(out).squeeze(2)
            return self.fc(pooled)



# ==============================================================================
# PRIMARY PROPOSED BACKBONE: HYBRID TCN + CNN ARCHITECTURE
# ==============================================================================
class HybridTCNCNNBackbone(nn.Module):
    """
    Hybrid TCN + CNN Neural Architecture for Dynamic Android Malware Telemetry.
    
    Architecture Design:
    1. Multi-Scale 1D-CNN Stage:
       - Multi-scale convolutional filter banks (kernels 3 & 5) extract fine-grained
         local spatial correlation, n-gram syscall sequences, and runtime metric patterns.
    2. Dilated Residual TCN Stage:
       - Cascaded causal dilated convolutional residual blocks (dilations 1, 2, 4, 8)
         model long-range temporal evolutions and multi-phase reboot behaviors.
    3. Squeeze-and-Excitation / Adaptive Temporal Pooling:
       - Channel-wise attention recalibration + adaptive pooling into compact latent embedding.
    """

    def __init__(
        self,
        input_dim: int = 141,
        cnn_channels: List[int] = [32, 64],
        tcn_channels: List[int] = [64, 128],
        latent_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 1. CNN Feature Extraction Stage
        # Multi-scale convolutional stage (Kernel 3 + Kernel 5)
        self.cnn_conv1_k3 = nn.Conv1d(1, cnn_channels[0], kernel_size=3, padding=1, bias=False)
        self.cnn_conv1_k5 = nn.Conv1d(1, cnn_channels[0], kernel_size=5, padding=2, bias=False)
        self.cnn_bn1 = nn.BatchNorm1d(cnn_channels[0] * 2)
        self.cnn_relu = nn.GELU()

        self.cnn_conv2 = nn.Conv1d(cnn_channels[0] * 2, cnn_channels[1], kernel_size=3, padding=1, bias=False)
        self.cnn_bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.cnn_pool = nn.MaxPool1d(kernel_size=2)
        self.cnn_dropout = nn.Dropout(dropout)

        # 2. TCN Temporal Modeling Stage
        tcn_layers = []
        in_c = cnn_channels[1]
        for i, out_c in enumerate(tcn_channels):
            dilation = 2 ** i
            tcn_layers.append(TemporalBlock(in_c, out_c, kernel_size=kernel_size, stride=1, dilation=dilation, dropout=dropout))
            in_c = out_c
        self.tcn = nn.Sequential(*tcn_layers)

        # 3. Channel Attention (Squeeze-and-Excitation)
        self.se_fc = nn.Sequential(
            nn.Linear(tcn_channels[-1], tcn_channels[-1] // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(tcn_channels[-1] // 4, tcn_channels[-1], bias=False),
            nn.Sigmoid()
        )

        # 4. Latent Projection Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_head = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with SafeBatchNormContext(self, x.size(0)):
            # Input shape: (Batch, Input_Dim) -> (Batch, 1, Input_Dim)
            if x.dim() == 2:
                x = x.unsqueeze(1)

            # 1. Multi-scale CNN Stage
            c3 = self.cnn_conv1_k3(x)
            c5 = self.cnn_conv1_k5(x)
            cnn_out = torch.cat([c3, c5], dim=1)
            cnn_out = self.cnn_relu(self.cnn_bn1(cnn_out))
            cnn_out = self.cnn_dropout(self.cnn_pool(self.cnn_relu(self.cnn_bn2(self.cnn_conv2(cnn_out)))))

            # 2. TCN Temporal Dilated Residual Stage
            tcn_out = self.tcn(cnn_out)

            # 3. Channel Attention Recalibration
            pooled_temp = self.global_pool(tcn_out).squeeze(2)  # (Batch, tcn_channels[-1])
            se_weights = self.se_fc(pooled_temp)                # (Batch, tcn_channels[-1])
            calibrated = pooled_temp * se_weights

            # 4. Latent Embedding Projection
            embedding = self.latent_head(calibrated)
            return embedding


class FusedMultiModalBackbone(nn.Module):
    """
    Dual-branch encoder that processes static features via MLP and dynamic features via Hybrid TCN+CNN,
    fusing the resulting latent embeddings through a joint multi-layer projection head.
    """

    def __init__(
        self,
        static_dim: int = 300,
        dynamic_dim: int = 141,
        latent_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.latent_dim = latent_dim

        # Static branch (MLP)
        self.static_branch = MLPBackbone(
            input_dim=static_dim,
            hidden_dims=[256, 128],
            latent_dim=64,
            dropout=dropout
        )

        # Dynamic branch (Hybrid TCN + CNN)
        self.dynamic_branch = HybridTCNCNNBackbone(
            input_dim=dynamic_dim,
            cnn_channels=[32, 64],
            tcn_channels=[64, 128],
            latent_dim=64,
            dropout=dropout
        )

        # Fusion projection head
        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with SafeBatchNormContext(self, x.size(0)):
            # Split concatenated input into static and dynamic slices
            x_static = x[:, :self.static_dim]
            x_dynamic = x[:, self.static_dim:]

            emb_static = self.static_branch(x_static)
            emb_dynamic = self.dynamic_branch(x_dynamic)

            fused = torch.cat([emb_static, emb_dynamic], dim=1)
            return self.fusion_fc(fused)


# ==============================================================================
# FALCON PROPOSED BACKBONE: CNN-BiLSTM-ATTENTION ARCHITECTURE
# ==============================================================================
class CNNBiLSTMAttentionBackbone(nn.Module):
    """
    Falcon CNN-BiLSTM-Attention Backbone for Traffic and Malware Representation.
    
    Architecture (Yang et al., IEEE TCE 2026, Section III-B & IV-A-2):
    1. 1D-CNN Stage:
       - 1D convolutional layer with 64 channels, kernel size 3, padding 1, ReLU.
       - Extracts fine-grained local patterns and short-term correlations.
    2. Bidirectional LSTM Stage:
       - 2-layer BiLSTM with hidden size 64 per direction (output dimension 128 = 2 * 64),
         dropout 0.2, batch_first=True.
       - Captures bidirectional long-range temporal dependencies.
    3. Self-Attention Stage:
       - Trainable projection matrices W_q, W_k, W_v with projection dimension d_k = 64.
       - Scaled dot-product attention: s_{t,j} = (q_t^T k_j) / sqrt(d_k).
       - Softmax normalization across sequence length.
       - Aggregated representation: z_t = sum_j alpha_{t,j} v_j.
       - Global sequence pooling: z = mean_t(z_t).
    4. Projection Head:
       - Projects to target latent_dim (default: 128) with BatchNorm1d and ReLU.
    """

    def __init__(
        self,
        input_dim: int = 141,
        conv_channels: int = 64,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        attn_dim: int = 64,
        latent_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        import math
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attn_dim = attn_dim
        self.scale = 1.0 / math.sqrt(attn_dim)

        # 1. 1D-CNN Stage
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_drop = nn.Dropout(p=dropout)

        # 2. 2-layer BiLSTM Stage
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        bilstm_out_dim = lstm_hidden * 2  # 128

        # 3. Self-Attention projections (Eqs. 13-16)
        self.w_q = nn.Linear(bilstm_out_dim, attn_dim)
        self.w_k = nn.Linear(bilstm_out_dim, attn_dim)
        self.w_v = nn.Linear(bilstm_out_dim, attn_dim)

        # 4. Latent projection
        self.proj = nn.Sequential(
            nn.Linear(attn_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with SafeBatchNormContext(self, x.size(0)):
            if x.dim() == 2:
                x_seq = x.unsqueeze(1)
            elif x.dim() == 3:
                x_seq = x
            else:
                raise ValueError(f"Expected 2D or 3D input, got {x.shape}")

            # 1. 1D-CNN along sequence dimension
            c = self.conv_drop(self.relu(self.bn(self.conv(x_seq))))

            # 2. BiLSTM
            c_transposed = c.permute(0, 2, 1)  # (B, L, C)
            h, _ = self.lstm(c_transposed)     # (B, L, 2*H)

            # 3. Self-Attention
            q = self.w_q(h)  # (B, L, d_k)
            k = self.w_k(h)  # (B, L, d_k)
            v = self.w_v(h)  # (B, L, d_k)

            scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            z_t = torch.bmm(attn_weights, v)

            # Global temporal pooling
            z = z_t.mean(dim=1)  # (B, d_k)

            # 4. Latent projection
            return self.proj(z)


def build_backbone(
    backbone_type: str,
    input_dim: int,
    latent_dim: int = 64,
    static_dim: int = 300,
    dynamic_dim: int = 141,
    dropout: float = 0.2
) -> nn.Module:
    """
    Factory builder for neural backbones supporting Hybrid TCN+CNN, Falcon CNN-BiLSTM-Attention,
    and ablation models.
    """
    b_type = backbone_type.lower()
    if b_type in ["hybrid_tcn_cnn", "cnn_tcn", "tcn_cnn", "hybrid"]:
        return HybridTCNCNNBackbone(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
    elif b_type in ["cnn_bilstm_attention", "falcon", "falcon_backbone", "cnn_bilstm_attn"]:
        return CNNBiLSTMAttentionBackbone(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
    elif b_type in ["cnn1d", "cnn"]:
        return CNN1DBackbone(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
    elif b_type == "tcn":
        return TCNBackbone(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
    elif b_type == "mlp":
        return MLPBackbone(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
    elif b_type == "fused":
        return FusedMultiModalBackbone(static_dim=static_dim, dynamic_dim=dynamic_dim, latent_dim=latent_dim, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown backbone type: {backbone_type}. "
            "Choose from ['hybrid_tcn_cnn', 'cnn_bilstm_attention', 'falcon', 'cnn1d', 'tcn', 'mlp', 'fused']"
        )

