"""
Hybrid CNN-LSTM-Transformer Model for Cognitive Workload & Fatigue Monitoring.

This module implements a multi-modal fusion architecture combining:
1. Spatial CNN Extractor - Extracts visual features from facial frames
2. Temporal Encoder (LSTM/TCN) - Models temporal dynamics of visual + physiological signals
3. Transformer Fusion Layer - Cross-modal attention for feature integration
4. Task-Specific Heads - Fatigue classification, workload regression, blink detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import timm
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Container for model outputs."""
    fatigue_logits: torch.Tensor  # [B, num_classes]
    workload_score: torch.Tensor  # [B, 1]
    blink_logits: torch.Tensor    # [B, 2]
    fused_features: torch.Tensor  # [B, d_model]
    attention_weights: Optional[torch.Tensor] = None


class SpatialCNNExtractor(nn.Module):
    """
    Spatial CNN feature extractor for facial frames.
    Uses EfficientNet/MobileNet/ResNet backbone with custom head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.output_dim = output_dim

        # Load backbone from timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(3,),  # Last feature map
        )

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            feat = self.backbone(dummy)[0]
            self.feature_dim = feat.shape[1]

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Froze {backbone} backbone parameters")

        # Adaptive pooling + projection head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frames [B, C, H, W] or [B, T, C, H, W]
        Returns:
            Spatial features [B, output_dim] or [B, T, output_dim]
        """
        # Handle temporal input [B, T, C, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self._extract_features(x)
            return features.view(B, T, -1)
        else:
            return self._extract_features(x)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from single frames."""
        feats = self.backbone(x)[0]  # [B, C', H', W']
        pooled = self.global_pool(feats).flatten(1)  # [B, C']
        projected = self.projection(pooled)  # [B, output_dim]
        return projected


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for sequence modeling.
    Supports LSTM, GRU, or TCN (Temporal Convolutional Network).
    """

    def __init__(
        self,
        encoder_type: str = "lstm",
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if self.encoder_type in ("lstm", "gru"):
            rnn_cls = nn.LSTM if self.encoder_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
            self.output_dim = hidden_dim * self.num_directions

        elif self.encoder_type == "tcn":
            # Temporal Convolutional Network
            self.tcn = nn.Sequential(
                self._tcn_block(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
                self._tcn_block(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
                self._tcn_block(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
            )
            self.output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.layer_norm = nn.LayerNorm(self.output_dim)

    def _tcn_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> nn.Module:
        """Create a TCN residual block."""
        padding = (kernel_size - 1) * dilation // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, T, input_dim]
            lengths: Optional sequence lengths [B] for packing
        Returns:
            Temporal features [B, T, output_dim]
        """
        if self.encoder_type in ("lstm", "gru"):
            if lengths is not None:
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
            output, _ = self.rnn(x)
            if lengths is not None:
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:  # TCN expects [B, C, T]
            x = x.transpose(1, 2)
            output = self.tcn(x)
            output = output.transpose(1, 2)

        return self.layer_norm(output)


class TransformerFusion(nn.Module):
    """
    Transformer-based cross-modal fusion layer.
    Fuses visual-temporal features with physiological features using self-attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # Encoder for visual-temporal stream
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Encoder for physiological stream
        self.physio_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Cross-modal fusion via decoder (visual attends to physio)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.cross_fusion = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Modality embeddings
        self.modality_embedding = nn.Embedding(2, d_model)  # 0: visual, 1: physio

        # Output projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,  # [B, T_v, d_model]
        physio_features: torch.Tensor,  # [B, T_p, d_model]
        visual_mask: Optional[torch.Tensor] = None,
        physio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse visual and physiological features via cross-attention.
        Returns:
            fused_features: [B, T_v, d_model]
            attention_weights: [B, T_v, T_p] from last decoder layer
        """
        B, T_v, _ = visual_features.shape
        _, T_p, _ = physio_features.shape

        # Add modality embeddings
        visual_features = visual_features + self.modality_embedding(
            torch.zeros(B, T_v, dtype=torch.long, device=visual_features.device)
        )
        physio_features = physio_features + self.modality_embedding(
            torch.ones(B, T_p, dtype=torch.long, device=physio_features.device)
        )

        # Positional encoding
        visual_features = self.pos_encoder(visual_features)
        physio_features = self.pos_encoder(physio_features)

        # Encode each modality
        visual_encoded = self.visual_encoder(visual_features, src_key_padding_mask=visual_mask)
        physio_encoded = self.physio_encoder(physio_features, src_key_padding_mask=physio_mask)

        # Cross-modal fusion: visual attends to physiological
        fused = self.cross_fusion(
            tgt=visual_encoded,
            memory=physio_encoded,
            tgt_key_padding_mask=visual_mask,
            memory_key_padding_mask=physio_mask,
        )

        # Concatenate and project
        combined = torch.cat([visual_encoded, fused], dim=-1)
        fused_features = self.fusion_proj(combined)

        # Return attention weights from last decoder layer (approximate)
        # Note: PyTorch doesn't expose attention weights directly from TransformerDecoder
        # This would require a custom implementation or hooks
        attention_weights = None

        return fused_features, attention_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T, d_model] or [T, B, d_model]
        """
        if x.dim() == 3 and x.shape[0] != self.pe.shape[0]:
            # [B, T, d_model] format
            x = x + self.pe[:x.size(1)].transpose(0, 1)
        else:
            # [T, B, d_model] format
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TaskHeads(nn.Module):
    """
    Task-specific output heads for multi-task learning.
    """

    def __init__(
        self,
        input_dim: int,
        fatigue_classes: int = 3,
        workload_dim: int = 1,
        blink_classes: int = 2,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        # Fatigue Classification Head
        self.fatigue_head = self._build_head(input_dim, fatigue_classes, hidden_dims, dropout)

        # Workload Regression Head
        self.workload_head = self._build_head(input_dim, workload_dim, hidden_dims, dropout)

        # Blink Detection Head
        self.blink_head = self._build_head(input_dim, blink_classes, hidden_dims[:1], dropout)

    def _build_head(self, in_dim: int, out_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Fused features [B, d_model] or [B, T, d_model]
        Returns:
            fatigue_logits: [B, num_classes] or [B, T, num_classes]
            workload_score: [B, 1] or [B, T, 1]
            blink_logits: [B, 2] or [B, T, 2]
        """
        # Handle temporal input
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
            fatigue = self.fatigue_head(x).view(B, T, -1)
            workload = self.workload_head(x).view(B, T, -1)
            blink = self.blink_head(x).view(B, T, -1)
        else:
            fatigue = self.fatigue_head(x)
            workload = self.workload_head(x)
            blink = self.blink_head(x)

        return fatigue, workload, blink


class HybridFatigueModel(nn.Module):
    """
    Main hybrid model combining CNN + LSTM + Transformer for fatigue monitoring.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Spatial CNN Extractor
        self.cnn = SpatialCNNExtractor(
            backbone=config["cnn"]["backbone"],
            pretrained=config["cnn"]["pretrained"],
            freeze_backbone=config["cnn"]["freeze_backbone"],
            output_dim=config["cnn"]["output_dim"],
            dropout=config["cnn"]["dropout"],
        )

        # Physiological feature encoder
        physio_input_dim = config.get("physiological", {}).get("feature_dim", 32)
        self.physio_encoder = nn.Sequential(
            nn.Linear(physio_input_dim, config["cnn"]["output_dim"]),
            nn.LayerNorm(config["cnn"]["output_dim"]),
            nn.GELU(),
            nn.Dropout(config["cnn"]["dropout"]),
        )

        # Temporal Encoder
        temporal_config = config["temporal"]
        self.temporal_encoder = TemporalEncoder(
            encoder_type=temporal_config["type"],
            input_dim=config["cnn"]["output_dim"] * 2,  # Visual + Physio
            hidden_dim=temporal_config["hidden_dim"],
            num_layers=temporal_config["num_layers"],
            bidirectional=temporal_config["bidirectional"],
            dropout=temporal_config["dropout"],
        )

        # Transformer Fusion
        fusion_config = config["transformer"]
        self.fusion = TransformerFusion(
            d_model=fusion_config["d_model"],
            nhead=fusion_config["nhead"],
            num_encoder_layers=fusion_config["num_encoder_layers"],
            num_decoder_layers=fusion_config["num_decoder_layers"],
            dim_feedforward=fusion_config["dim_feedforward"],
            dropout=fusion_config["dropout"],
            activation=fusion_config["activation"],
        )

        # Project temporal output to fusion d_model
        temporal_output_dim = self.temporal_encoder.output_dim
        self.temporal_proj = nn.Linear(temporal_output_dim, fusion_config["d_model"])

        # Task Heads
        heads_config = config["heads"]
        self.task_heads = TaskHeads(
            input_dim=fusion_config["d_model"],
            fatigue_classes=heads_config["fatigue_classification"]["num_classes"],
            workload_dim=heads_config["workload_regression"]["output_dim"],
            blink_classes=heads_config["blink_detection"]["num_classes"],
            hidden_dims=heads_config["fatigue_classification"]["hidden_dims"],
            dropout=heads_config["fatigue_classification"]["dropout"],
        )

        logger.info(f"Initialized HybridFatigueModel with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(
        self,
        frames: torch.Tensor,  # [B, T, C, H, W] or [B, C, H, W]
        physio_features: Optional[torch.Tensor] = None,  # [B, T, physio_dim] or [B, physio_dim]
        lengths: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass through the hybrid model.

        Args:
            frames: Video frames [B, T, C, H, W] or single frame [B, C, H, W]
            physio_features: Physiological features [B, T, D] or [B, D]
            lengths: Sequence lengths for packing [B]

        Returns:
            ModelOutput with predictions
        """
        # Handle single frame input
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)  # [B, 1, C, H, W]
        B, T, C, H, W = frames.shape

        # Extract spatial features from frames
        visual_features = self.cnn(frames)  # [B, T, cnn_dim]

        # Process physiological features
        if physio_features is not None:
            if physio_features.dim() == 2:
                physio_features = physio_features.unsqueeze(1).expand(-1, T, -1)
            physio_encoded = self.physio_encoder(physio_features)  # [B, T, cnn_dim]
        else:
            # Zero physiological features if not provided
            physio_encoded = torch.zeros_like(visual_features)

        # Concatenate visual and physiological features
        combined_features = torch.cat([visual_features, physio_encoded], dim=-1)  # [B, T, 2*cnn_dim]

        # Temporal encoding
        temporal_features = self.temporal_encoder(combined_features, lengths)  # [B, T, temporal_dim]

        # Project to fusion dimension
        temporal_projected = self.temporal_proj(temporal_features)  # [B, T, d_model]

        # Split for fusion (use visual as query, physio as key/value)
        # For simplicity, we use the same features for both streams
        # In practice, you might separate them
        visual_stream = temporal_projected
        physio_stream = temporal_projected

        # Transformer fusion
        fused_features, attention_weights = self.fusion(visual_stream, physio_stream)

        # Global temporal pooling for final prediction
        # Use attention-weighted pooling or simple mean
        if lengths is not None:
            # Masked mean pooling
            mask = torch.arange(T, device=lengths.device).expand(B, T) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (fused_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = fused_features.mean(dim=1)  # [B, d_model]

        # Task heads
        fatigue_logits, workload_score, blink_logits = self.task_heads(pooled)

        return ModelOutput(
            fatigue_logits=fatigue_logits,
            workload_score=workload_score,
            blink_logits=blink_logits,
            fused_features=pooled,
            attention_weights=attention_weights,
        )

    def get_model_size(self) -> Dict[str, int]:
        """Return parameter counts for each component."""
        return {
            "cnn": sum(p.numel() for p in self.cnn.parameters()),
            "physio_encoder": sum(p.numel() for p in self.physio_encoder.parameters()),
            "temporal": sum(p.numel() for p in self.temporal_encoder.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "heads": sum(p.numel() for p in self.task_heads.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


def create_model(config: Dict) -> HybridFatigueModel:
    """Factory function to create model from config."""
    return HybridFatigueModel(config)


if __name__ == "__main__":
    # Test model creation
    import yaml

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Add physiological feature dim
    config["physiological"] = config.get("physiological", {})
    config["physiological"]["feature_dim"] = 32

    model = create_model(config)
    print(f"Model created with {model.get_model_size()['total']:,} parameters")

    # Test forward pass
    B, T = 2, 30
    frames = torch.randn(B, T, 3, 224, 224)
    physio = torch.randn(B, T, 32)

    with torch.no_grad():
        output = model(frames, physio)
        print(f"Fatigue logits: {output.fatigue_logits.shape}")
        print(f"Workload score: {output.workload_score.shape}")
        print(f"Blink logits: {output.blink_logits.shape}")
        print(f"Fused features: {output.fused_features.shape}")