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


def _validate_station_attention_mode(value: str) -> str:
    allowed_values = {
        "shared_memory_levels",
        "bottleneck_only",
        "per_level_memory_levels",
    }
    if value not in allowed_values:
        raise ValueError(
            "station_attention_mode must be one of "
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


class StationAttentionWeightsBlock(nn.Module):
    """Independent station-attention block that outputs one gate per station.

    Input: [B, S, C, T]
    Output: [B, S, 1, 1] (station gates derived from attention weights)
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

        pooled = pooled + attn_out
        pooled = pooled + self.ff(self.norm2(pooled))

        # One scalar importance weight per station, mean-preserving over stations.
        station_weights = attn_weights.mean(dim=1).mean(dim=1)  # [B, S]
        station_weights = station_weights * station_weights.shape[-1]
        if station_key_padding_mask is not None:
            station_weights = station_weights.masked_fill(station_key_padding_mask, 0.0)
        return station_weights[:, :, None, None]


class SharedMemoryLevelStationAttention(nn.Module):
    """Single station-attention mechanism over multiple encoder levels.

    Each selected level contributes a time-pooled station descriptor.
    Descriptors are projected to a shared channel space, averaged, and attended
    across stations once. The resulting attention map is reduced to one scalar
    weight per station, which is then used as a multiplicative gate.
    """

    def __init__(
        self,
        level_channels: dict[int, int],
        output_channels: int,
        heads: int = 4,
        ff_mult: int = 2,
    ):
        super().__init__()
        if output_channels % heads != 0:
            raise ValueError(
                f"shared station-attention channels ({output_channels}) must be "
                f"divisible by heads ({heads})."
            )
        self.level_projections = nn.ModuleDict(
            {
                str(level): nn.Linear(channels, output_channels)
                for level, channels in level_channels.items()
            }
        )
        self.norm1 = nn.LayerNorm(output_channels)
        self.attn = nn.MultiheadAttention(
            output_channels, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(output_channels)
        self.ff = nn.Sequential(
            nn.Linear(output_channels, output_channels * ff_mult),
            nn.GELU(),
            nn.Linear(output_channels * ff_mult, output_channels),
        )
        self.last_attn_weights: torch.Tensor | None = None
        self.capture_attention_weights = False

    def forward(
        self,
        level_features: dict[int, torch.Tensor],
        station_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(level_features) < 1:
            raise ValueError("level_features must contain at least one level.")

        descriptors: list[torch.Tensor] = []
        for level in sorted(level_features):
            if str(level) not in self.level_projections:
                raise KeyError(
                    f"Missing projection for level {level}. "
                    "Check station_attention_mode and memory_levels configuration."
                )
            pooled = level_features[level].mean(dim=-1)  # [B, S, C_level]
            descriptors.append(self.level_projections[str(level)](pooled))

        fused = torch.stack(descriptors, dim=0).mean(dim=0)  # [B, S, C_out]
        normed = self.norm1(fused)
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

        fused = fused + attn_out
        fused = fused + self.ff(self.norm2(fused))

        # One scalar importance weight per station, mean-preserving over stations.
        station_weights = attn_weights.mean(dim=1).mean(dim=1)  # [B, S]
        station_weights = station_weights * station_weights.shape[-1]
        if station_key_padding_mask is not None:
            station_weights = station_weights.masked_fill(station_key_padding_mask, 0.0)
        return station_weights[:, :, None, None]


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
        station_mask_abs_sum_threshold: float = 0.0,
        station_attention_mode: str = "shared_memory_levels",
        memory_levels: Sequence[int] = (),
        memory_level_pool_to: int | None = None,
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
        self.station_attention_mode = _validate_station_attention_mode(
            station_attention_mode
        )
        self.memory_levels = _validate_memory_levels(memory_levels, self.depth)
        if memory_level_pool_to is not None and memory_level_pool_to < 1:
            raise ValueError(
                "memory_level_pool_to must be >= 1 when provided, got "
                f"{memory_level_pool_to}."
            )
        self.memory_level_pool_to = memory_level_pool_to

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
        self.last_station_attn_weights_by_level: dict[int, torch.Tensor] = {}
        self.last_station_weights_by_level: dict[int, torch.Tensor] = {}
        self.last_station_weights_aggregate: torch.Tensor | None = None
        if self.station_attention_mode == "bottleneck_only":
            self.station_attention = StationAttentionBlock(
                self.output_channels, station_attn_heads, station_attn_ff_mult
            )
            self.station_attention_by_level = None
        elif self.station_attention_mode == "per_level_memory_levels":
            self.station_attention = None
            per_level_indices = sorted(set(self.memory_levels) | {self.depth - 1})
            self.station_attention_by_level = nn.ModuleDict(
                {
                    str(level): StationAttentionWeightsBlock(
                        self.level_channels[level],
                        station_attn_heads,
                        station_attn_ff_mult,
                    )
                    for level in per_level_indices
                }
            )
        else:
            shared_attention_levels = sorted(set(self.memory_levels) | {self.depth - 1})
            shared_level_channels = {
                level: self.level_channels[level] for level in shared_attention_levels
            }
            self.station_attention = SharedMemoryLevelStationAttention(
                level_channels=shared_level_channels,
                output_channels=self.output_channels,
                heads=station_attn_heads,
                ff_mult=station_attn_ff_mult,
            )
            self.station_attention_by_level = None
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
    def _station_weights_from_attention_map(
        attn_weights: torch.Tensor,
        station_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_weights.ndim != 4:
            raise ValueError(
                "station attention map must be [B, H, S, S], "
                f"got shape {tuple(attn_weights.shape)}."
            )
        station_weights = attn_weights.mean(dim=1).mean(dim=1)  # [B, S]
        station_weights = station_weights * station_weights.shape[-1]
        if station_key_padding_mask is not None:
            station_weights = station_weights.masked_fill(station_key_padding_mask, 0.0)
        return station_weights

    @staticmethod
    def _normalized_station_weights(station_weights: torch.Tensor) -> torch.Tensor:
        if station_weights.ndim != 2:
            raise ValueError(
                "station weights for merge must be [B, S], "
                f"got shape {tuple(station_weights.shape)}."
            )
        weight_sums = station_weights.sum(dim=1, keepdim=True)
        if (weight_sums <= 0).any():
            raise RuntimeError(
                "Station weights must have positive mass per sample for weighted merge."
            )
        return station_weights / weight_sums

    @staticmethod
    def _weighted_station_merge(
        x: torch.Tensor,
        station_weights: torch.Tensor,
    ) -> torch.Tensor:
        normalized = MultiStationTemporalEncoder._normalized_station_weights(
            station_weights
        )
        return (x * normalized[:, :, None, None]).sum(dim=1)

    def forward(
        self, x: torch.Tensor, return_extra_level_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
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
        level_station_weights: dict[int, torch.Tensor] = {}
        self.last_station_attn_weights_by_level = {}
        self.last_station_weights_by_level = {}
        self.last_station_weights_aggregate = None

        if self.station_attention_mode == "per_level_memory_levels":
            for block in self.station_attention_by_level.values():
                block.capture_attention_weights = self.capture_attention_weights

        for level, (conv_same, bn_same, conv_down, bn_down) in enumerate(self.levels):
            x = self._station_conv(x, conv_same, bn_same)
            x = x * station_valid_scale

            if (
                self.station_attention_mode == "per_level_memory_levels"
                and level in self.memory_levels
            ):
                level_block = self.station_attention_by_level[str(level)]
                level_weights = level_block(
                    x,
                    station_key_padding_mask=station_missing_mask,
                )
                level_station_weights[level] = level_weights[:, :, 0, 0].detach()
                if (
                    self.capture_attention_weights
                    and level_block.last_attn_weights is not None
                ):
                    self.last_station_attn_weights_by_level[level] = (
                        level_block.last_attn_weights.detach()
                    )

            if level in self.memory_levels:
                raw_level_station_features[level] = x
            if (
                self.station_attention_mode == "bottleneck_only"
                and level == self.depth - 1
            ):
                x = self.station_attention(
                    x, station_key_padding_mask=station_missing_mask
                )
                x = x * station_valid_scale
                bottleneck_weights = self.station_attention.last_station_weights
                if bottleneck_weights is None:
                    raise RuntimeError(
                        "bottleneck_only station attention did not produce station weights."
                    )
                level_station_weights[self.depth - 1] = bottleneck_weights.detach()
                if self.capture_attention_weights:
                    attn_map = self.station_attention.last_attn_weights
                    if attn_map is not None:
                        self.last_station_attn_weights_by_level[self.depth - 1] = (
                            attn_map.detach()
                        )
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

        if self.station_attention_mode == "per_level_memory_levels":
            bottleneck_block = self.station_attention_by_level[str(self.depth - 1)]
            bottleneck_weights = bottleneck_block(
                x,
                station_key_padding_mask=station_missing_mask,
            )
            level_station_weights[self.depth - 1] = bottleneck_weights[
                :, :, 0, 0
            ].detach()
            if (
                self.capture_attention_weights
                and bottleneck_block.last_attn_weights is not None
            ):
                self.last_station_attn_weights_by_level[self.depth - 1] = (
                    bottleneck_block.last_attn_weights.detach()
                )

        if self.station_attention_mode == "shared_memory_levels":
            attention_inputs = {
                level: raw_level_station_features[level] for level in self.memory_levels
            }
            attention_inputs[self.depth - 1] = x
            station_weights = self.station_attention(
                attention_inputs,
                station_key_padding_mask=station_missing_mask,
            )
            level_station_weights[self.depth - 1] = station_weights[:, :, 0, 0].detach()
            if (
                self.capture_attention_weights
                and self.station_attention.last_attn_weights is not None
            ):
                self.last_station_attn_weights_by_level[self.depth - 1] = (
                    self.station_attention.last_attn_weights.detach()
                )

        if level_station_weights:
            self.last_station_weights_by_level = {
                level: weights.detach()
                for level, weights in level_station_weights.items()
            }
            self.last_station_weights_aggregate = (
                torch.stack(list(self.last_station_weights_by_level.values()), dim=0)
                .mean(dim=0)
                .detach()
            )
        else:
            raise RuntimeError(
                "No station attention weights were collected; weighted station merge cannot proceed."
            )

        merge_weights = self.last_station_weights_aggregate
        if merge_weights is None:
            raise RuntimeError(
                "Station merge requires aggregate station attention weights, got None."
            )

        # Permutation-invariant weighted station merge.
        bottleneck_features = self._weighted_station_merge(
            x, merge_weights
        )  # [B, C, T']
        self.last_bottleneck_effective_stride = self.bottleneck_stride

        target_tokens = self.memory_level_pool_to
        extra_level_features: dict[int, torch.Tensor] = {}
        self.last_extra_level_effective_strides = {}
        for level in self.memory_levels:
            if level in self.last_station_weights_by_level:
                level_merge_weights = self.last_station_weights_by_level[level]
            else:
                level_merge_weights = merge_weights
            level_features = self._weighted_station_merge(
                raw_level_station_features[level],
                level_merge_weights,
            )
            raw_length = level_features.shape[-1]
            if target_tokens is not None and raw_length > target_tokens:
                level_features = F.adaptive_avg_pool1d(level_features, target_tokens)
            pooled_length = level_features.shape[-1]
            extra_level_features[level] = level_features
            self.last_extra_level_effective_strides[level] = float(
                self.level_base_strides[level]
            ) * (raw_length / pooled_length)

        if return_extra_level_features:
            return bottleneck_features, extra_level_features
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
    """Per-query prediction heads: class and temporal interval.

    Supports two interval parameterizations controlled by
    ``interval_output_format``:

    - ``center_duration``: predicts ``center`` and ``duration`` (both in [0, 1]).
        ``start/end`` are materialized later by normalization utilities.
    - ``start_end``: predicts ``start`` and ``end`` directly (both in [0, 1]),
        then enforces temporal ordering via element-wise min/max.

    In both modes, a ``center`` key is returned so downstream loss/matching code
    can use a shared interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 6,
        interval_output_format: str = "start_end",
    ):
        super().__init__()
        self.interval_output_format = _validate_interval_output_format(
            interval_output_format
        )

        def mlp(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.class_head = mlp(num_classes)
        if self.interval_output_format == "start_end":
            self.start_head = mlp(1)
            self.end_head = mlp(1)
            self.center_head = None
            self.duration_head = None
        else:
            self.center_head = mlp(1)
            self.duration_head = mlp(1)
            self.start_head = None
            self.end_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, Nq, input_dim]

        Returns:
            Dict with keys:
                - class_logits: [B, Nq, num_classes] (raw logits)
                - if interval_output_format == "start_end":
                    - start: [B, Nq, 1] in [0, 1]
                    - end: [B, Nq, 1] in [0, 1]
                    - center: [B, Nq, 1] in [0, 1]
                - if interval_output_format == "center_duration":
                    - center: [B, Nq, 1] in [0, 1]
                    - duration: [B, Nq, 1] in [0, 1]
        """
        predictions = {"class_logits": self.class_head(x)}
        if self.interval_output_format == "start_end":
            raw_start = torch.sigmoid(self.start_head(x))
            raw_end = torch.sigmoid(self.end_head(x))
            start = torch.minimum(raw_start, raw_end)
            end = torch.maximum(raw_start, raw_end)
            center = 0.5 * (start + end)
            predictions["start"] = start
            predictions["end"] = end
            predictions["center"] = center
        else:
            center = torch.sigmoid(self.center_head(x))
            duration = torch.sigmoid(self.duration_head(x))
            predictions["center"] = center
            predictions["duration"] = duration
        return predictions


class MuSSED(nn.Module):
    """
    MuSSED: Multi-Station Seismic Event Detection

    DETR-inspired event detection architecture.
    Detects a fixed set of events (not per-timestep segmentation).

    Architecture:
    - Temporal Encoder: multi-station aware, produces [B, C, T']
    - Event Queries: learnable set of Nq queries
    - Transformer Decoder: self-attention on queries + cross-attention to features
    - Detection Head: per-query predictions (class, interval)
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
        station_attention_mode: str = "shared_memory_levels",
        memory_levels: Sequence[int] = (),
        memory_level_pool_to: int | None = None,
        eval_memory_level_pool_to: int | None = None,
        # --- Detection decoder / heads ---
        num_queries: int = 10,
        query_dim: int = 128,
        hidden_dim: int = 256,
        num_decoder_heads: int = 4,
        num_decoder_layers: int = 2,
        decoder_dropout: float = 0.1,
        use_temporal_projection: bool = False,
        interval_output_format: str = "start_end",
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
            station_mask_abs_sum_threshold=station_mask_abs_sum_threshold,
            station_attention_mode=station_attention_mode,
            memory_levels=memory_levels,
            memory_level_pool_to=memory_level_pool_to,
        )
        if eval_memory_level_pool_to is not None and eval_memory_level_pool_to < 1:
            raise ValueError(
                "eval_memory_level_pool_to must be >= 1 when provided, got "
                f"{eval_memory_level_pool_to}."
            )
        self.eval_memory_level_pool_to = eval_memory_level_pool_to

        encoder_channels = self.encoder.output_channels
        self.memory_levels = self.encoder.memory_levels

        # Temporal feature projection (optional).
        self.use_temporal_projection = bool(use_temporal_projection)
        if self.use_temporal_projection:
            self.temporal_proj = nn.Linear(encoder_channels, query_dim)
        else:
            # Without projection, the decoder consumes encoder features directly,
            # so query_dim must match the encoder output channels.
            if query_dim != encoder_channels:
                raise ValueError(
                    f"When use_temporal_projection=False, query_dim ({query_dim}) "
                    f"must equal encoder_channels ({encoder_channels}). Either set "
                    f"query_dim={encoder_channels} or use_temporal_projection=True."
                )
        self.memory_level_projections = nn.ModuleDict(
            {
                str(level): nn.Linear(self.encoder.level_channels[level], query_dim)
                for level in self.memory_levels
            }
        )

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
        self.memory_level_embeddings = nn.ParameterDict(
            {
                "bottleneck": nn.Parameter(
                    torch.randn(1, 1, query_dim) / (query_dim**0.5)
                ),
                **{
                    str(level): nn.Parameter(
                        torch.randn(1, 1, query_dim) / (query_dim**0.5)
                    )
                    for level in self.memory_levels
                },
            }
        )

        # Sinusoidal positional encoding, computed on the fly for any length.
        self.positional_encoding = SinusoidalPositionalEncoding(query_dim)

        # DETR transformer decoder.
        self.decoder = DETRTransformerDecoder(
            d_model=query_dim,
            nhead=num_decoder_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            dropout=decoder_dropout,
            capture_attention_weights=False,
        )

        # Detection heads.
        self.detection_head = DetectionHead(
            input_dim=query_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            interval_output_format=interval_output_format,
        )

        self.query_dim = query_dim
        self.num_classes = num_classes
        self.interval_output_format = self.detection_head.interval_output_format

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
        batch_size = x.shape[0]

        # 1. Encode temporal features
        if not self.training and self.eval_memory_level_pool_to is not None:
            original_pool_to = self.encoder.memory_level_pool_to
            self.encoder.memory_level_pool_to = self.eval_memory_level_pool_to
            encoder_out = self.encoder(x, return_extra_level_features=True)
            self.encoder.memory_level_pool_to = original_pool_to
        else:
            encoder_out = self.encoder(x, return_extra_level_features=True)
        if isinstance(encoder_out, tuple):
            encoder_features, extra_level_features = encoder_out
        else:
            encoder_features = encoder_out
            extra_level_features = {}

        # 2. Project encoder features (optional)
        encoder_proj = encoder_features.transpose(1, 2)  # [B, T', C]
        if self.use_temporal_projection:
            encoder_proj = self.temporal_proj(encoder_proj)  # [B, T', query_dim]
        encoder_proj = encoder_proj + self.memory_level_embeddings["bottleneck"]

        # 3. Add positional encoding in shared real-time coordinates.
        pos_enc = self.positional_encoding(
            encoder_proj.shape[1],
            device=encoder_proj.device,
            dtype=encoder_proj.dtype,
            stride=self.encoder.last_bottleneck_effective_stride,
        )
        memory_parts = [encoder_proj + pos_enc]

        for level in self.memory_levels:
            level_features = extra_level_features[level]
            level_proj = self.memory_level_projections[str(level)](
                level_features.transpose(1, 2)
            )
            level_proj = level_proj + self.memory_level_embeddings[str(level)]
            level_pos_enc = self.positional_encoding(
                level_proj.shape[1],
                device=level_proj.device,
                dtype=level_proj.dtype,
                stride=self.encoder.last_extra_level_effective_strides[level],
            )
            memory_parts.append(level_proj + level_pos_enc)

        memory = torch.cat(memory_parts, dim=1)

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
