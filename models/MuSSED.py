"""
MuSSED: Multi-Station Seismic Event Detection
DETR-inspired event detection model with temporal encoder backbone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from models.MuSSeg import MuSSeg


class TemporalEncoder(nn.Module):
    """
    Standalone temporal encoder for multi-station seismic data.
    Produces station-aware temporal feature maps [B, C, T'].

    Uses late station attention at the final encoder level (bottleneck).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        depth: int = 5,
        kernel_size: int = 7,
        stride: int = 4,
        dilation: int = 1,
        filters_root: int = 8,
        norm: str = "std",
        feature_dropout: float = 0.0,
        bottleneck_attention: bool = False,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_dropout: float = 0.0,
        bottleneck_attn_ff_mult: int = 2,
        station_attn_heads: int = 4,
        station_attn_dropout: float = 0.0,
        station_attn_ff_mult: int = 2,
        volcano_name: str | None = None,
        use_distance_attn_bias: bool = False,
        use_distance_bottleneck_emb: bool = False,
        use_station_weighted_skips: bool = False,
    ):
        """
        Args:
            in_channels: Number of input channels (typically 3)
            num_classes: Number of event classes
            depth: Encoder depth (number of down-sampling levels)
            kernel_size: Convolution kernel size
            stride: Down-sampling stride per level
            dilation: Dilation rate(s)
            filters_root: Base number of filters (doubled each level)
            norm: Normalization type ('std' for batch norm)
            feature_dropout: Dropout rate
            bottleneck_attention: Use temporal attention at bottleneck
            volcano_name: Required if use_distance_attn_bias=True
            use_distance_attn_bias: Incorporate station distance into attention
            use_distance_bottleneck_emb: Incorporate station distance into embeddings
            use_station_weighted_skips: Learn per-station skip-merge weights
        """
        super().__init__()

        # Late attention always at final encoder level (depth-1)
        station_attention_levels = [depth - 1]

        # Build underlying encoder
        encoder_kwargs = {
            "in_channels": in_channels,
            "classes": num_classes,
            "depth": depth,
            "kernel_size": kernel_size,
            "stride": stride,
            "dilation": dilation,
            "filters_root": filters_root,
            "norm": norm,
            "feature_dropout": feature_dropout,
            "bottleneck_attention": bottleneck_attention,
            "station_interaction": "late_attention",
            "station_message_levels": [],
            "station_message_aggregation": "sum",
            "station_message_ratio": 1.0,
            "station_attention_levels": station_attention_levels,
            "pre_bottleneck_station_attn_merge": False,
            "bottleneck_attn_heads": bottleneck_attn_heads,
            "bottleneck_attn_dropout": bottleneck_attn_dropout,
            "bottleneck_attn_ff_mult": bottleneck_attn_ff_mult,
            "station_attn_heads": station_attn_heads,
            "station_attn_dropout": station_attn_dropout,
            "station_attn_ff_mult": station_attn_ff_mult,
            "volcano_name": volcano_name,
            "use_distance_attn_bias": use_distance_attn_bias,
            "use_distance_bottleneck_emb": use_distance_bottleneck_emb,
            "use_station_weighted_skips": use_station_weighted_skips,
        }

        self._encoder = MuSSeg(**encoder_kwargs)
        self.output_channels = self._encoder.bottleneck_channels
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input waveforms to temporal feature map.

        Args:
            x: [B, S, T] input waveforms

        Returns:
            encoder_features: [B, C, T'] temporal feature map
        """
        if x.ndim == 4 and x.shape[2] == 1:
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(f"TemporalEncoder expects [B, S, T], got {tuple(x.shape)}")

        # Reshape to [B, S, 1, T] for processing
        x = x[:, :, None, :]
        x = self._encoder._apply_station_conv(
            x, self._encoder.inc_shared, self._encoder.in_bn_shared
        )

        skips = []
        store_station_skips = (
            self._encoder.use_station_weighted_skips
            or self._encoder.station_dist is not None
        )

        # Downsampling path
        for level, (conv_same, bn1, conv_down, bn2) in enumerate(
            self._encoder.down_branch
        ):
            x = self._encoder._apply_station_conv(x, conv_same, bn1)

            if level in self._encoder.station_attention_levels:
                if self._encoder.use_distance_attn_bias:
                    station_dist = self._encoder._station_dist_for_count(
                        int(x.shape[1]),
                        device=x.device,
                        dtype=x.dtype,
                    ).squeeze(-1)
                    dist_diff = (
                        station_dist.unsqueeze(0) - station_dist.unsqueeze(1)
                    ).abs()
                    dist_bias = self._encoder.dist_attn_bias_proj(
                        dist_diff.unsqueeze(-1)
                    ).squeeze(-1)
                    x = self._encoder.station_attention_blocks[str(level)](
                        x, dist_bias=dist_bias
                    )
                else:
                    x = self._encoder.station_attention_blocks[str(level)](x)

            if conv_down is not None:
                skips.append(
                    x if store_station_skips else self._encoder._station_max(x)
                )
                x = self._encoder._pad_shared_downsample(
                    x,
                    kernel_size=self._encoder.encoder_kernel_sizes[level],
                    stride=self._encoder.encoder_strides[level],
                    dilation=self._encoder.encoder_dilations[level],
                )
                x = self._encoder._apply_station_conv(x, conv_down, bn2)

        # Bottleneck (attention)
        if self._encoder.bottleneck_attention:
            if self._encoder.station_interaction == "none":
                # Merge stations before bottleneck attention
                x = self._encoder._merge_stations(x)
                x = self._encoder._apply_bottleneck_attention(x)
            else:
                # Keep station-wise, apply per-station attention
                bsz, n_stations, channels, t_len = x.shape
                x_flat = x.reshape(bsz * n_stations, channels, t_len)
                x_flat = self._encoder._apply_bottleneck_attention(x_flat)
                x = x_flat.reshape(bsz, n_stations, channels, t_len)
                if self._encoder.use_distance_bottleneck_emb:
                    station_dist = self._encoder._station_dist_for_count(
                        n_stations,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    dist_emb = self._encoder.dist_bottleneck_proj(station_dist)
                    x = x + dist_emb[None, :, :, None]

        # Merge stations to [B, C, T']
        if x.ndim == 4:
            x = self._encoder._merge_stations(x)

        return x


class DETRTransformerDecoder(nn.Module):
    """
    Minimal DETR transformer decoder with self-attention on queries
    and cross-attention to temporal features.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(
        self,
        queries: torch.Tensor,  # [B, Nq, d_model]
        memory: torch.Tensor,  # [B, T', d_model]
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Nq, d_model] learnable queries
            memory: [B, T', d_model] encoder features (temporal)

        Returns:
            decoder_output: [B, Nq, d_model] decoded queries
        """
        return self.decoder(queries, memory)


class DetectionHead(nn.Module):
    """
    Prediction heads for event properties.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 6,
    ):
        super().__init__()
        self.class_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.center_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.start_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.end_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, Nq, input_dim]

        Returns:
            Dict with keys:
                - class_logits: [B, Nq, num_classes]
                - center: [B, Nq, 1]
                - start: [B, Nq, 1]
                - end: [B, Nq, 1]
                - confidence: [B, Nq, 1]
        """
        return {
            "class_logits": self.class_head(x),
            "center": self.center_head(x),
            "start": self.start_head(x),
            "end": self.end_head(x),
            "confidence": self.confidence_head(x),
        }


class MuSSED(nn.Module):
    """
    MuSSED: Multi-Station Seismic Event Detection

    DETR-inspired event detection architecture.
    Detects a fixed set of events (not per-timestep segmentation).

    Architecture:
    - Temporal Encoder: multi-station aware, produces [B, C, T']
    - Event Queries: learnable set of Nq queries
    - Transformer Decoder: self-attention on queries + cross-attention to features
    - Detection Head: per-query predictions (class, location, confidence)
    """

    def __init__(
        self,
        # Encoder parameters
        in_channels: int = 3,
        num_classes: int = 6,
        depth: int = 5,
        kernel_size: int = 7,
        stride: int = 4,
        dilation: int = 1,
        filters_root: int = 8,
        norm: str = "std",
        feature_dropout: float = 0.0,
        bottleneck_attention: bool = False,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_dropout: float = 0.0,
        bottleneck_attn_ff_mult: int = 2,
        station_attn_heads: int = 4,
        station_attn_dropout: float = 0.0,
        station_attn_ff_mult: int = 2,
        volcano_name: str | None = None,
        use_distance_attn_bias: bool = False,
        use_distance_bottleneck_emb: bool = False,
        use_station_weighted_skips: bool = False,
        # Detection head parameters
        num_queries: int = 3,
        query_dim: int = 256,
        hidden_dim: int = 512,
        num_decoder_heads: int = 4,
        num_decoder_layers: int = 3,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()

        # Build temporal encoder
        self.encoder = TemporalEncoder(
            in_channels=in_channels,
            num_classes=num_classes,
            depth=depth,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            filters_root=filters_root,
            norm=norm,
            feature_dropout=feature_dropout,
            bottleneck_attention=bottleneck_attention,
            bottleneck_attn_heads=bottleneck_attn_heads,
            bottleneck_attn_dropout=bottleneck_attn_dropout,
            bottleneck_attn_ff_mult=bottleneck_attn_ff_mult,
            station_attn_heads=station_attn_heads,
            station_attn_dropout=station_attn_dropout,
            station_attn_ff_mult=station_attn_ff_mult,
            volcano_name=volcano_name,
            use_distance_attn_bias=use_distance_attn_bias,
            use_distance_bottleneck_emb=use_distance_bottleneck_emb,
            use_station_weighted_skips=use_station_weighted_skips,
        )

        encoder_channels = self.encoder.output_channels

        # Temporal feature projection: encoder_channels -> query_dim
        self.temporal_proj = nn.Linear(encoder_channels, query_dim)

        # Learnable event queries
        self.num_queries = int(num_queries)
        self.event_queries = nn.Parameter(
            torch.randn(1, self.num_queries, query_dim) / (query_dim**0.5)
        )

        # Positional encoding for temporal features
        self.positional_encoding = PositionalEncoding(query_dim, max_len=8192)

        # DETR transformer decoder
        self.decoder = DETRTransformerDecoder(
            d_model=query_dim,
            nhead=num_decoder_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            dropout=decoder_dropout,
        )

        # Detection heads
        self.detection_head = DetectionHead(
            input_dim=query_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        self.query_dim = query_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for event detection.

        Args:
            x: [B, S, T] input waveforms
               B: batch size
               S: number of stations (variable)
               T: time samples (variable)

        Returns:
            Dict with keys:
                - class_logits: [B, Nq, num_classes]
                - center: [B, Nq, 1] normalized event center time
                - start: [B, Nq, 1] normalized event start time
                - end: [B, Nq, 1] normalized event end time
                - confidence: [B, Nq, 1] objectness score
                - encoder_features: [B, C, T'] for interpretability
                - attention_weights: Optional, if computing attention
        """
        batch_size = x.shape[0]

        # 1. Encode temporal features
        encoder_features = self.encoder(x)  # [B, C, T']
        _, _, t_prime = encoder_features.shape

        # 2. Project encoder features
        encoder_proj = encoder_features.transpose(1, 2)  # [B, T', C]
        encoder_proj = self.temporal_proj(encoder_proj)  # [B, T', query_dim]

        # 3. Add positional encoding
        pos_enc = self.positional_encoding(encoder_proj)  # [B, T', query_dim]
        memory = encoder_proj + pos_enc

        # 4. Expand queries to batch size
        queries = self.event_queries.expand(batch_size, -1, -1)  # [B, Nq, query_dim]

        # 5. Decode (transformer decoder)
        decoder_out = self.decoder(queries, memory)  # [B, Nq, query_dim]

        # 6. Predict event properties
        predictions = self.detection_head(decoder_out)

        # Add encoder features for interpretability
        predictions["encoder_features"] = encoder_features

        return predictions

    def freeze_encoder(self):
        """Freeze encoder parameters (useful for transfer learning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal dimension.
    Borrowed from Transformer literature.
    """

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[..., :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            positional encoding: [B, T, d_model]
        """
        return self.dropout(self.pe[:, : x.size(1), :])
