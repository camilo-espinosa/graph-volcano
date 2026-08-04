"""
MuSSED: Multi-Station Seismic Event Detection

Standalone DETR-style event detector for multi-station seismic waveforms. The
model emits a fixed set of events (class + temporal interval + confidence)
directly, avoiding per-timestep segmentation and catalog post-processing.

This module is fully self-contained and does not depend on the MuSSeg
segmentation model.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _broadcast_level_param(
    name: str, value: int | Sequence[int], num_levels: int
) -> list[int]:
    """Expand a scalar into one value per level, or validate a per-level list."""
    if num_levels == 0:
        return []
    if isinstance(value, int):
        values = [value] * num_levels
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [int(v) for v in value]
        if len(values) != num_levels:
            raise ValueError(
                f"{name} must have length {num_levels}, got {len(values)}: {value!r}."
            )
    else:
        raise TypeError(
            f"{name} must be an int or a list/tuple of ints, "
            f"got {type(value).__name__}."
        )
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} values must be positive integers, got {values}.")
    return values


class StationAttentionBlock(nn.Module):
    """Self-attention across stations, applied to time-pooled descriptors.

    Input/Output: [B, S, C, T]. Stations are pooled over time, attend to one
    another, and the per-station update is broadcast back over the time axis.
    Permutation-equivariant over the station dimension.
    """

    def __init__(self, channels: int, heads: int = 4, ff_mult: int = 2):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(
                f"station-attention channels ({channels}) must be divisible "
                f"by heads ({heads})."
            )
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Linear(channels * ff_mult, channels),
        )
        self.last_attn_weights: torch.Tensor | None = None
        self.capture_attention_weights = False

    def forward(
        self,
        x: torch.Tensor,
        station_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = x.mean(dim=-1)  # [B, S, C]
        normed = self.norm1(pooled)
        if self.capture_attention_weights:
            attn_out, attn_weights = self.attn(
                normed,
                normed,
                normed,
                need_weights=True,
                average_attn_weights=False,
                key_padding_mask=station_key_padding_mask,
            )
            self.last_attn_weights = attn_weights.detach()
        else:
            attn_out, _ = self.attn(
                normed,
                normed,
                normed,
                need_weights=False,
                key_padding_mask=station_key_padding_mask,
            )
            self.last_attn_weights = None
        pooled = pooled + attn_out
        pooled = pooled + self.ff(self.norm2(pooled))
        return x + pooled[:, :, :, None]


class TemporalBottleneckAttention(nn.Module):
    """Pre-norm transformer block applying self-attention over time.

    Input/Output: [N, C, T]. Used at the encoder bottleneck to inject global
    temporal context.
    """

    def __init__(self, channels: int, heads: int = 4, ff_mult: int = 2):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(
                f"bottleneck channels ({channels}) must be divisible by "
                f"heads ({heads})."
            )
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Linear(channels * ff_mult, channels),
        )
        self.last_attn_weights: torch.Tensor | None = None
        self.capture_attention_weights = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [N, T, C]
        normed = self.norm1(x)
        if self.capture_attention_weights:
            attn_out, attn_weights = self.attn(
                normed,
                normed,
                normed,
                need_weights=True,
                average_attn_weights=True,
            )
            self.last_attn_weights = attn_weights.detach()
        else:
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            self.last_attn_weights = None
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x.transpose(1, 2)  # [N, C, T]


class MultiStationTemporalEncoder(nn.Module):
    """U-Net-style temporal encoder with weight-shared per-station convolutions.

    Each station is processed independently by shared convolutions. Stations
    interact once via attention at the deepest level, a temporal bottleneck
    attention injects global context, and stations are finally merged by a
    permutation-invariant max over the station axis.

    Input:  [B, S, T]  (or [B, S, 1, T]); each station is a single waveform.
    Output: [B, C, T'] with C = filters_root * 2**(depth - 1) and
            T' = T // prod(strides).
    """

    def __init__(
        self,
        depth: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
        dilation: int | Sequence[int],
        filters_root: int,
        bottleneck_attention: bool,
        bottleneck_attn_heads: int,
        bottleneck_attn_ff_mult: int,
        station_attn_heads: int,
        station_attn_ff_mult: int,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}.")

        self.depth = int(depth)
        self.kernel_sizes = _broadcast_level_param(
            "kernel_size", kernel_size, self.depth
        )
        self.dilations = _broadcast_level_param("dilation", dilation, self.depth)
        self.strides = _broadcast_level_param("stride", stride, self.depth - 1)
        self.use_bottleneck_attention = bool(bottleneck_attention)

        # Shared per-station stem: 1 channel -> filters_root.
        self.stem_conv = nn.Conv1d(
            1,
            filters_root,
            self.kernel_sizes[0],
            padding="same",
            dilation=self.dilations[0],
        )
        self.stem_bn = nn.BatchNorm1d(filters_root, eps=1e-3)

        # Downsampling levels: (conv_same, bn_same, conv_down, bn_down).
        self.levels = nn.ModuleList()
        prev_filters = filters_root
        for level in range(self.depth):
            filters = filters_root * (2**level)
            kernel = self.kernel_sizes[level]
            dilation_l = self.dilations[level]
            conv_same = nn.Conv1d(
                prev_filters,
                filters,
                kernel,
                padding="same",
                dilation=dilation_l,
                bias=False,
            )
            bn_same = nn.BatchNorm1d(filters, eps=1e-3)
            prev_filters = filters
            if level == self.depth - 1:
                conv_down = None
                bn_down = None
            else:
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    kernel,
                    self.strides[level],
                    padding=0,
                    dilation=dilation_l,
                    bias=False,
                )
                bn_down = nn.BatchNorm1d(filters, eps=1e-3)
            self.levels.append(nn.ModuleList([conv_same, bn_same, conv_down, bn_down]))

        self.output_channels = filters_root * (2 ** (self.depth - 1))
        self.station_attention = StationAttentionBlock(
            self.output_channels, station_attn_heads, station_attn_ff_mult
        )
        if self.use_bottleneck_attention:
            self.bottleneck_attention = TemporalBottleneckAttention(
                self.output_channels, bottleneck_attn_heads, bottleneck_attn_ff_mult
            )
        else:
            self.bottleneck_attention = None

    @staticmethod
    def _station_conv(
        x: torch.Tensor, conv: nn.Conv1d, bn: nn.BatchNorm1d
    ) -> torch.Tensor:
        # Apply a shared conv + BN + ReLU independently per station.
        bsz, n_stations, channels, t_len = x.shape
        y = x.reshape(bsz * n_stations, channels, t_len)
        y = F.relu(bn(conv(y)))
        return y.reshape(bsz, n_stations, y.shape[1], y.shape[2])

    @staticmethod
    def _pad_for_downsample(
        x: torch.Tensor, *, kernel_size: int, stride: int, dilation: int
    ) -> torch.Tensor:
        # "same"-style symmetric padding for the strided downsampling conv.
        bsz, n_stations, channels, t_len = x.shape
        out_len = math.ceil(t_len / stride)
        effective_kernel = dilation * (kernel_size - 1) + 1
        total_pad = max(0, (out_len - 1) * stride + effective_kernel - t_len)
        if total_pad == 0:
            return x
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        y = x.reshape(bsz * n_stations, channels, t_len)
        y = F.pad(y, (pad_left, pad_right))
        return y.reshape(bsz, n_stations, channels, y.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[2] == 1:
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(
                f"encoder expects [B, S, T] or [B, S, 1, T], got {tuple(x.shape)}."
            )

        # Zero-valued stations encode missingness and must be masked out.
        station_missing_mask = x.eq(0).all(dim=-1)  # [B, S], True means missing
        if station_missing_mask.all(dim=1).any():
            raise RuntimeError(
                "Encountered sample(s) with all stations missing (all-zero waveforms)."
            )
        station_valid_scale = (~station_missing_mask).to(x.dtype)[:, :, None, None]

        x = x[:, :, None, :]  # [B, S, 1, T]
        x = self._station_conv(x, self.stem_conv, self.stem_bn)
        x = x * station_valid_scale

        for level, (conv_same, bn_same, conv_down, bn_down) in enumerate(self.levels):
            x = self._station_conv(x, conv_same, bn_same)
            x = x * station_valid_scale
            if level == self.depth - 1:
                x = self.station_attention(
                    x, station_key_padding_mask=station_missing_mask
                )
                x = x * station_valid_scale
            if conv_down is not None:
                x = self._pad_for_downsample(
                    x,
                    kernel_size=self.kernel_sizes[level],
                    stride=self.strides[level],
                    dilation=self.dilations[level],
                )
                x = self._station_conv(x, conv_down, bn_down)
                x = x * station_valid_scale

        if self.bottleneck_attention is not None:
            bsz, n_stations, channels, t_len = x.shape
            x = x.reshape(bsz * n_stations, channels, t_len)
            x = self.bottleneck_attention(x)
            x = x.reshape(bsz, n_stations, channels, t_len)
            x = x * station_valid_scale

        # Permutation-invariant station merge.
        return x.max(dim=1).values  # [B, C, T']


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
    """Per-query prediction heads: class, temporal interval, and confidence.

    Interval parameterization is controlled by ``constrain_intervals``:

    - ``True`` (default): predict a ``center`` and a ``width`` (both squashed
      to ``[0, 1]`` via sigmoid), then derive ``start = center - width / 2`` and
      ``end = center + width / 2`` (clamped to ``[0, 1]``). This guarantees
      ``0 <= start <= center <= end <= 1`` by construction.
    - ``False``: predict ``start``, ``center`` and ``end`` independently (each
      squashed to ``[0, 1]``). Outputs stay bounded but carry no ordering
      guarantee, which is useful as an ablation of the ordering constraint.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 6,
        constrain_intervals: bool = True,
    ):
        super().__init__()
        self.constrain_intervals = bool(constrain_intervals)

        def mlp(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.class_head = mlp(num_classes)
        self.confidence_head = mlp(1)
        self.center_head = mlp(1)
        if self.constrain_intervals:
            self.width_head = mlp(1)
        else:
            self.start_head = mlp(1)
            self.end_head = mlp(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, Nq, input_dim]

        Returns:
            Dict with keys:
                - class_logits: [B, Nq, num_classes] (raw logits)
                - center: [B, Nq, 1] in [0, 1]
                - start: [B, Nq, 1] in [0, 1]
                - end: [B, Nq, 1] in [0, 1]
                - confidence: [B, Nq, 1] (raw logit; apply sigmoid in the loss)
        """
        center = torch.sigmoid(self.center_head(x))
        if self.constrain_intervals:
            half_width = 0.5 * torch.sigmoid(self.width_head(x))
            start = (center - half_width).clamp(0.0, 1.0)
            end = (center + half_width).clamp(0.0, 1.0)
        else:
            start = torch.sigmoid(self.start_head(x))
            end = torch.sigmoid(self.end_head(x))

        return {
            "class_logits": self.class_head(x),
            "center": center,
            "start": start,
            "end": end,
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
    - Detection Head: per-query predictions (class, interval, confidence)
    """

    def __init__(
        self,
        num_classes: int = 6,
        # --- Encoder (fixed to the best MuSSeg NVCHVC config by default) ---
        depth: int = 4,
        kernel_size: int | Sequence[int] = 127,
        stride: int | Sequence[int] = 2,
        dilation: int | Sequence[int] = 1,
        filters_root: int = 16,
        bottleneck_attention: bool = True,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_ff_mult: int = 2,
        station_attn_heads: int = 4,
        station_attn_ff_mult: int = 2,
        # --- Detection decoder / heads ---
        num_queries: int = 10,
        query_dim: int = 128,
        hidden_dim: int = 256,
        num_decoder_heads: int = 4,
        num_decoder_layers: int = 2,
        decoder_dropout: float = 0.1,
        use_temporal_projection: bool = False,
        constrain_intervals: bool = True,
    ):
        super().__init__()

        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}.")

        # Build temporal encoder.
        self.encoder = MultiStationTemporalEncoder(
            depth=depth,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            filters_root=filters_root,
            bottleneck_attention=bottleneck_attention,
            bottleneck_attn_heads=bottleneck_attn_heads,
            bottleneck_attn_ff_mult=bottleneck_attn_ff_mult,
            station_attn_heads=station_attn_heads,
            station_attn_ff_mult=station_attn_ff_mult,
        )

        encoder_channels = self.encoder.output_channels

        # Temporal feature projection (optional).
        self.use_temporal_projection = bool(use_temporal_projection)
        if self.use_temporal_projection:
            self.temporal_proj = nn.Linear(encoder_channels, query_dim)
            memory_dim = query_dim
        else:
            # Without projection, the decoder consumes encoder features directly,
            # so query_dim must match the encoder output channels.
            if query_dim != encoder_channels:
                raise ValueError(
                    f"When use_temporal_projection=False, query_dim ({query_dim}) "
                    f"must equal encoder_channels ({encoder_channels}). Either set "
                    f"query_dim={encoder_channels} or use_temporal_projection=True."
                )
            memory_dim = encoder_channels

        # Transformer attention requires the model dim to split evenly across heads.
        if query_dim % num_decoder_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by "
                f"num_decoder_heads ({num_decoder_heads})."
            )

        # Learnable event queries.
        self.num_queries = int(num_queries)
        self.event_queries = nn.Parameter(
            torch.randn(1, self.num_queries, query_dim) / (query_dim**0.5)
        )

        # Sinusoidal positional encoding, computed on the fly for any length.
        self.positional_encoding = SinusoidalPositionalEncoding(memory_dim)

        # DETR transformer decoder.
        self.decoder = DETRTransformerDecoder(
            d_model=query_dim,
            nhead=num_decoder_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            dropout=decoder_dropout,
        )

        # Detection heads.
        self.detection_head = DetectionHead(
            input_dim=query_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            constrain_intervals=constrain_intervals,
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
                - center: [B, Nq, 1] normalized event center time in [0, 1]
                - start: [B, Nq, 1] normalized event start time in [0, 1]
                - end: [B, Nq, 1] normalized event end time in [0, 1]
                - confidence: [B, Nq, 1] objectness logit
                - encoder_features: [B, C, T'] for interpretability
        """
        batch_size = x.shape[0]

        # 1. Encode temporal features
        encoder_features = self.encoder(x)  # [B, C, T']

        # 2. Project encoder features (optional)
        encoder_proj = encoder_features.transpose(1, 2)  # [B, T', C]
        if self.use_temporal_projection:
            encoder_proj = self.temporal_proj(encoder_proj)  # [B, T', query_dim]

        # 3. Add positional encoding (sized to the actual sequence length)
        pos_enc = self.positional_encoding(
            encoder_proj.shape[1],
            device=encoder_proj.device,
            dtype=encoder_proj.dtype,
        )
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


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding computed on the fly for any length.

    Unlike a fixed pre-allocated table, encodings are generated per forward
    call from the requested sequence length, so the model is robust to inputs
    of arbitrary size without a hard-coded maximum.
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"positional encoding requires an even d_model, got {d_model}."
            )
        self.d_model = int(d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(
        self, length: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Args:
            length: sequence length T to encode
            device: device for the returned tensor
            dtype: dtype for the returned tensor

        Returns:
            positional encoding: [1, T, d_model]
        """
        if length < 1:
            raise ValueError(f"sequence length must be >= 1, got {length}.")
        position = torch.arange(length, device=device, dtype=torch.float).unsqueeze(1)
        angles = position * self.div_term.to(device=device)  # [T, d_model / 2]
        pe = torch.zeros(length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe.unsqueeze(0).to(dtype)  # [1, T, d_model]
