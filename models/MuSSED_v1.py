"""
MuSSED: Multi-Station Seismic Event Detection

Standalone DETR-style event detector for multi-station seismic waveforms. The
model emits a fixed set of events (class + temporal interval)
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


def _validate_interval_output_format(value: str) -> str:
    allowed_values = {"start_end", "center_duration"}
    if value not in allowed_values:
        raise ValueError(
            "interval_output_format must be one of "
            f"{sorted(allowed_values)}, got {value!r}."
        )
    return value


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


def _validate_memory_levels(
    memory_levels: Sequence[int], depth: int
) -> tuple[int, ...]:
    if not isinstance(memory_levels, Sequence) or isinstance(
        memory_levels, (str, bytes)
    ):
        raise TypeError(
            "memory_levels must be a list/tuple of ints, "
            f"got {type(memory_levels).__name__}."
        )
    levels = [int(level) for level in memory_levels]
    valid_levels = list(range(depth - 1))
    invalid_levels = [level for level in levels if level not in valid_levels]
    if invalid_levels:
        raise ValueError(
            "memory_levels values must be in "
            f"{valid_levels}, got {invalid_levels} from {levels}."
        )
    if len(set(levels)) != len(levels):
        raise ValueError(f"memory_levels contains duplicates: {levels}.")
    return tuple(levels)


def _validate_detection_head_mode(value: str) -> str:
    allowed_values = {"independent", "shared_trunk"}
    if value not in allowed_values:
        raise ValueError(
            "detection_head_mode must be one of "
            f"{sorted(allowed_values)}, got {value!r}."
        )
    return value


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
        self.last_station_weights: torch.Tensor | None = None
        self.capture_attention_weights = False

    def forward(
        self,
        x: torch.Tensor,
        station_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = x.mean(dim=-1)  # [B, S, C]
        normed = self.norm1(pooled)
        attn_out, attn_weights = self.attn(
            normed,
            normed,
            normed,
            need_weights=True,
            average_attn_weights=False,
            key_padding_mask=station_key_padding_mask,
        )
        if self.capture_attention_weights:
            self.last_attn_weights = attn_weights.detach()
        else:
            self.last_attn_weights = None
        station_weights = attn_weights.mean(dim=1).mean(dim=1)  # [B, S]
        station_weights = station_weights * station_weights.shape[-1]
        if station_key_padding_mask is not None:
            station_weights = station_weights.masked_fill(station_key_padding_mask, 0.0)
        self.last_station_weights = station_weights.detach()
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
    attention injects global context, and fusion is fixed by design:
    bottleneck uses station-attention-derived weights while non-bottleneck
    memory levels use plain max over stations.

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
        station_mask_abs_sum_threshold: float = 0.0,
        memory_levels: Sequence[int] = (),
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}.")
        if station_mask_abs_sum_threshold < 0:
            raise ValueError(
                "station_mask_abs_sum_threshold must be >= 0, got "
                f"{station_mask_abs_sum_threshold}."
            )

        self.depth = int(depth)
        self.kernel_sizes = _broadcast_level_param(
            "kernel_size", kernel_size, self.depth
        )
        self.dilations = _broadcast_level_param("dilation", dilation, self.depth)
        self.strides = _broadcast_level_param("stride", stride, self.depth - 1)
        self.use_bottleneck_attention = bool(bottleneck_attention)
        self.station_mask_abs_sum_threshold = float(station_mask_abs_sum_threshold)
        self.memory_levels = _validate_memory_levels(memory_levels, self.depth)

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
        self.level_channels = [filters_root * (2**level) for level in range(self.depth)]
        self.level_base_strides = [
            int(math.prod(self.strides[:level])) for level in range(self.depth)
        ]
        self.bottleneck_stride = float(self.level_base_strides[self.depth - 1])
        self.last_extra_level_effective_strides: dict[int, float] = {}
        self.last_bottleneck_effective_stride = self.bottleneck_stride
        self.capture_attention_weights = False
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

    @staticmethod
    def _merge_stations_max(
        x: torch.Tensor,
        station_missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = (~station_missing_mask)[:, :, None, None]
        x_masked = torch.where(valid_mask, x, torch.full_like(x, -torch.inf))
        merged = x_masked.max(dim=1).values
        if not torch.isfinite(merged).all():
            raise RuntimeError(
                "Non-finite values encountered during max station fusion."
            )
        return merged

    @staticmethod
    def _merge_bottleneck_with_attention(
        x: torch.Tensor,
        station_missing_mask: torch.Tensor,
        station_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if station_weights is None:
            raise RuntimeError(
                "Missing bottleneck station-attention weights for station fusion."
            )
        if station_weights.shape != station_missing_mask.shape:
            raise RuntimeError(
                "Station-attention weights and station mask shape mismatch: "
                f"{tuple(station_weights.shape)} vs {tuple(station_missing_mask.shape)}."
            )
        weights = station_weights.to(dtype=x.dtype)
        weights = weights.masked_fill(station_missing_mask, 0.0)
        weight_sums = weights.sum(dim=1, keepdim=True)
        if (weight_sums <= 0).any():
            raise RuntimeError(
                "Encountered zero bottleneck station-attention weight sum "
                "after masking missing stations."
            )
        normalized_weights = weights / weight_sums
        merged = (x * normalized_weights[:, :, None, None]).sum(dim=1)
        if not torch.isfinite(merged).all():
            raise RuntimeError(
                "Non-finite values encountered during bottleneck station fusion."
            )
        return merged

    def forward(
        self,
        x: torch.Tensor,
        return_extra_level_features: bool = False,
        return_decoder_skip_features: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict[int, torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, dict[int, torch.Tensor], list[torch.Tensor]]
    ):
        if x.ndim == 4 and x.shape[2] == 1:
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(
                f"encoder expects [B, S, T] or [B, S, 1, T], got {tuple(x.shape)}."
            )

        # Low-energy stations (abs-sum <= threshold) are treated as missing.
        station_abs_sum = x.abs().sum(dim=-1)  # [B, S]
        station_missing_mask = (
            station_abs_sum <= self.station_mask_abs_sum_threshold
        )  # [B, S], True means missing
        if station_missing_mask.all(dim=1).any():
            raise RuntimeError(
                "Encountered sample(s) with all stations masked as missing. "
                f"station_mask_abs_sum_threshold={self.station_mask_abs_sum_threshold}."
            )
        station_valid_scale = (~station_missing_mask).to(x.dtype)[:, :, None, None]

        x = x[:, :, None, :]  # [B, S, 1, T]
        x = self._station_conv(x, self.stem_conv, self.stem_bn)
        x = x * station_valid_scale

        raw_level_station_features: dict[int, torch.Tensor] = {}
        decoder_skip_features: list[torch.Tensor] = []
        bottleneck_station_weights: torch.Tensor | None = None
        self.station_attention.capture_attention_weights = (
            self.capture_attention_weights
        )

        for level, (conv_same, bn_same, conv_down, bn_down) in enumerate(self.levels):
            x = self._station_conv(x, conv_same, bn_same)
            x = x * station_valid_scale

            if level in self.memory_levels:
                raw_level_station_features[level] = x
            if level == self.depth - 1:
                x = self.station_attention(
                    x, station_key_padding_mask=station_missing_mask
                )
                bottleneck_station_weights = self.station_attention.last_station_weights
                x = x * station_valid_scale
            if conv_down is not None:
                if return_decoder_skip_features:
                    decoder_skip_features.append(
                        self._merge_stations_max(x, station_missing_mask)
                    )
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

        # Bottleneck uses station-attention-derived weighted fusion.
        bottleneck_features = self._merge_bottleneck_with_attention(
            x,
            station_missing_mask,
            bottleneck_station_weights,
        )  # [B, C, T']
        self.last_bottleneck_effective_stride = self.bottleneck_stride

        extra_level_features: dict[int, torch.Tensor] = {}
        self.last_extra_level_effective_strides = {}
        for level in self.memory_levels:
            level_features = self._merge_stations_max(
                raw_level_station_features[level],
                station_missing_mask,
            )
            extra_level_features[level] = level_features
            self.last_extra_level_effective_strides[level] = float(
                self.level_base_strides[level]
            )

        if return_extra_level_features and return_decoder_skip_features:
            return bottleneck_features, extra_level_features, decoder_skip_features
        if return_extra_level_features:
            return bottleneck_features, extra_level_features
        if return_decoder_skip_features:
            return bottleneck_features, decoder_skip_features
        return bottleneck_features


class _DETRDecoderLayer(nn.Module):
    """Transformer decoder layer exposing cross-attention weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        capture_cross_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))

        cross_attn_out, cross_attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            need_weights=capture_cross_attention,
            average_attn_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))

        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff_out))

        if not capture_cross_attention:
            return tgt, None
        return tgt, cross_attn_weights


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
        capture_attention_weights: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [
                _DETRDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.capture_attention_weights = bool(capture_attention_weights)
        self.last_cross_attn_weights: torch.Tensor | None = None

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
        x = queries
        self.last_cross_attn_weights = None
        for layer in self.layers:
            x, cross_attn_weights = layer(
                x,
                memory,
                capture_cross_attention=self.capture_attention_weights,
            )
            if self.capture_attention_weights and cross_attn_weights is not None:
                self.last_cross_attn_weights = cross_attn_weights.detach()
        return x

    def attention_weights_by_level(
        self, level_boundaries: dict[str, tuple[int, int]]
    ) -> dict[str, torch.Tensor]:
        if self.last_cross_attn_weights is None:
            raise RuntimeError(
                "No captured decoder cross-attention weights. "
                "Set capture_attention_weights=True and run a forward pass first."
            )
        normalized_weights = (
            self.last_cross_attn_weights
            / self.last_cross_attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        )
        by_level: dict[str, torch.Tensor] = {}
        for level_name, (start, end) in level_boundaries.items():
            by_level[level_name] = normalized_weights[..., start:end].sum(dim=-1)
        return by_level


class DetectionHead(nn.Module):
    """Single-event conv head over fused multiscale temporal features.

    Input is a single fused feature map [B, C, T] (no query decoder). The head
    emits one event per sample while preserving MuSSED output keys with an
    explicit singleton query dimension [B, 1, ...].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 6,
        interval_output_format: str = "start_end",
        detection_head_mode: str = "independent",
        trunk_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.interval_output_format = _validate_interval_output_format(
            interval_output_format
        )
        # Kept for constructor compatibility with prior configs.
        self.detection_head_mode = _validate_detection_head_mode(detection_head_mode)
        self.trunk_dims = tuple() if trunk_dims is None else tuple(int(v) for v in trunk_dims)

        if input_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f"input_dim and hidden_dim must be >= 1, got {input_dim}, {hidden_dim}."
            )

        self.trunk = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-3),
            nn.ReLU(),
        )

        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.center_logits_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        if self.interval_output_format == "start_end":
            self.start_head = nn.Linear(hidden_dim, 1)
            self.end_head = nn.Linear(hidden_dim, 1)
            self.duration_head = None
        else:
            self.duration_head = nn.Linear(hidden_dim, 1)
            self.start_head = None
            self.end_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T]

        Returns:
            Dict with keys:
                - class_logits: [B, 1, num_classes]
                - if interval_output_format == "start_end":
                    - start: [B, 1, 1] in [0, 1]
                    - end: [B, 1, 1] in [0, 1]
                    - center: [B, 1, 1] in [0, 1]
                - if interval_output_format == "center_duration":
                    - center: [B, 1, 1] in [0, 1]
                    - duration: [B, 1, 1] in [0, 1]
        """
        if x.ndim != 3:
            raise ValueError(f"DetectionHead expects [B, C, T], got {tuple(x.shape)}.")

        feat = self.trunk(x)
        pooled = feat.mean(dim=-1)

        class_logits = self.class_head(pooled)[:, None, :]

        center_logits = self.center_logits_head(feat).squeeze(1)
        center_weights = torch.softmax(center_logits, dim=-1)
        time_grid = torch.linspace(
            0.0,
            1.0,
            steps=center_weights.shape[-1],
            device=center_weights.device,
            dtype=center_weights.dtype,
        )
        center = (center_weights * time_grid[None, :]).sum(dim=-1, keepdim=True)
        center = center[:, None, :]

        predictions: Dict[str, torch.Tensor] = {
            "class_logits": class_logits,
            "center": center,
        }

        if self.interval_output_format == "start_end":
            raw_start = torch.sigmoid(self.start_head(pooled))[:, None, :]
            raw_end = torch.sigmoid(self.end_head(pooled))[:, None, :]
            start = torch.minimum(raw_start, raw_end)
            end = torch.maximum(raw_start, raw_end)
            predictions["start"] = start
            predictions["end"] = end
            predictions["center"] = 0.5 * (start + end)
        else:
            predictions["duration"] = torch.sigmoid(self.duration_head(pooled))[:, None, :]

        return predictions


class MuSSED(nn.Module):
    """
    MuSSED: Multi-Station Seismic Event Detection

    Single-event detection architecture with a MuSSeg-like encoder/decoder flow.

    Architecture:
    - Temporal Encoder: multi-station aware, produces [B, C, T']
    - Skip decoder: progressive upsampling with skip fusion
    - Detection Head: one event (class, interval) from final decoder features
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
        station_mask_abs_sum_threshold: float = 0.0,
        memory_levels: Sequence[int] = (),
        # --- Detection head ---
        num_queries: int = 1,
        query_dim: int = 128,
        hidden_dim: int = 256,
        decoder_ffn_dim: int | None = None,
        head_hidden_dim: int | None = None,
        num_decoder_heads: int = 4,
        num_decoder_layers: int = 2,
        decoder_dropout: float = 0.1,
        use_temporal_projection: bool = False,
        interval_output_format: str = "start_end",
        detection_head_mode: str = "independent",
        head_trunk_dims: Sequence[int] | None = None,
    ):
        super().__init__()

        head_hidden_dim_resolved = (
            int(hidden_dim) if head_hidden_dim is None else int(head_hidden_dim)
        )
        if head_hidden_dim_resolved < 1:
            raise ValueError(
                "head_hidden_dim must be >= 1 after resolution, got "
                f"{head_hidden_dim_resolved}."
            )

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
            station_mask_abs_sum_threshold=station_mask_abs_sum_threshold,
            memory_levels=memory_levels,
        )

        encoder_channels = self.encoder.output_channels
        self.memory_levels = self.encoder.memory_levels
        self.use_temporal_projection = bool(use_temporal_projection)

        # MuSSeg-like temporal decoder: upsample bottleneck and fuse skip features.
        self.decoder_blocks = nn.ModuleList()
        in_channels = encoder_channels
        # Skip features are stored for levels [0..depth-2] and consumed deepest->shallowest.
        for level in range(depth - 2, -1, -1):
            out_channels = self.encoder.level_channels[level]
            up_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3),
                nn.ReLU(),
            )
            fuse_conv = nn.Sequential(
                nn.Conv1d(
                    out_channels + self.encoder.level_channels[level],
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels, eps=1e-3),
                nn.ReLU(),
            )
            self.decoder_blocks.append(nn.ModuleList([up_conv, fuse_conv]))
            in_channels = out_channels

        # Detection heads.
        self.detection_head = DetectionHead(
            input_dim=self.encoder.level_channels[0],
            hidden_dim=head_hidden_dim_resolved,
            num_classes=num_classes,
            interval_output_format=interval_output_format,
            detection_head_mode=detection_head_mode,
            trunk_dims=head_trunk_dims,
        )

        self.query_dim = query_dim
        self.num_queries = 1
        self.num_classes = num_classes
        self.interval_output_format = self.detection_head.interval_output_format
        self.decoder_ffn_dim = (
            int(hidden_dim) if decoder_ffn_dim is None else int(decoder_ffn_dim)
        )
        self.head_hidden_dim = head_hidden_dim_resolved
        self.num_decoder_heads = int(num_decoder_heads)
        self.num_decoder_layers = int(num_decoder_layers)
        self.decoder_dropout = float(decoder_dropout)

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
                - if interval_output_format == "start_end":
                    - center: [B, Nq, 1] normalized event center time in [0, 1]
                    - start: [B, Nq, 1] normalized event start time in [0, 1]
                    - end: [B, Nq, 1] normalized event end time in [0, 1]
                - if interval_output_format == "center_duration":
                    - center: [B, Nq, 1] normalized event center time in [0, 1]
                    - duration: [B, Nq, 1] normalized event duration in [0, 1]
                - encoder_features: [B, C, T'] for interpretability
        """
        # 1. Encode temporal features and collect skip maps.
        encoder_out = self.encoder(
            x,
            return_extra_level_features=False,
            return_decoder_skip_features=True,
        )
        if isinstance(encoder_out, tuple):
            encoder_features, decoder_skips = encoder_out
        else:
            raise RuntimeError("Encoder did not return decoder skip features.")

        # 2. MuSSeg-like skip decoder from bottleneck to high temporal resolution.
        x_dec = encoder_features
        for block, skip in zip(self.decoder_blocks, reversed(decoder_skips)):
            up_conv, fuse_conv = block
            x_dec = F.interpolate(
                x_dec,
                size=skip.shape[-1],
                mode="linear",
                align_corners=False,
            )
            x_dec = up_conv(x_dec)
            x_dec = torch.cat([skip, x_dec], dim=1)
            x_dec = fuse_conv(x_dec)

        # 3. Predict event properties from final decoded feature map.
        predictions = self.detection_head(x_dec)

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
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        stride: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            length: sequence length T to encode
            device: device for the returned tensor
            dtype: dtype for the returned tensor
            stride: temporal stride used to map tokens onto a shared real-time axis

        Returns:
            positional encoding: [1, T, d_model]
        """
        if length < 1:
            raise ValueError(f"sequence length must be >= 1, got {length}.")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}.")
        position = (
            torch.arange(length, device=device, dtype=torch.float) * float(stride)
        ).unsqueeze(1)
        angles = position * self.div_term.to(device=device)  # [T, d_model / 2]
        pe = torch.zeros(length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe.unsqueeze(0).to(dtype)  # [1, T, d_model]
