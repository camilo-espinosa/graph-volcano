"""
MuSSED: Multi-Station Seismic Event Detection

Lean single-path event detector for multi-station seismic waveforms.

Architecture:
- MultiStationTemporalEncoder: U-Net-like encoder with weight-shared per-station
    convolutions, station interaction via attention at the bottleneck, and optional
    multiscale memory levels.
- Level-Aligned Memory Bank: project each selected memory level to a shared
    query_dim and learned-upsample to a common temporal grid T0, keeping levels
    separated as [B, D, L, T0].
- Lightweight Fused Memory: level-temporal weighting merges [B, D, L, T0] into
    [B, D, T0] before query decoding to reduce VRAM.
- Segmentation-Aware Query Head: predicts class/confidence plus dense per-query
    mask/start/end heatmaps; start/end are decoded by differentiable expectation.

This module intentionally has no DETR path and no optional legacy branches.
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


class LightweightTemporalUpsample(nn.Module):
    """Lightweight learned temporal upsampling with depthwise ConvTranspose1d."""

    def __init__(self, channels: int, up_factor: int):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}.")
        if up_factor < 1:
            raise ValueError(f"up_factor must be >= 1, got {up_factor}.")

        self.up_factor = int(up_factor)
        if self.up_factor == 1:
            self.upsample = nn.Identity()
            return

        stages: list[nn.Module] = []
        remaining = self.up_factor
        while remaining % 2 == 0:
            stages.extend(
                [
                    nn.ConvTranspose1d(
                        channels,
                        channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels, eps=1e-3),
                    nn.ReLU(),
                ]
            )
            remaining //= 2

        if remaining != 1:
            raise ValueError(
                "LightweightTemporalUpsample currently supports power-of-two "
                f"factors, got up_factor={self.up_factor}."
            )

        stages.extend(
            [
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels, eps=1e-3),
                nn.ReLU(),
            ]
        )
        self.upsample = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.upsample(x)
        if y.shape[-1] == target_len:
            return y
        if y.shape[-1] > target_len:
            return y[..., :target_len]
        return F.pad(y, (0, target_len - y.shape[-1]))


class MultiStationTemporalEncoder(nn.Module):
    """U-Net-style temporal encoder with weight-shared per-station convolutions."""

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

        self.stem_conv = nn.Conv1d(
            1,
            filters_root,
            self.kernel_sizes[0],
            padding="same",
            dilation=self.dilations[0],
        )
        self.stem_bn = nn.BatchNorm1d(filters_root, eps=1e-3)

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
        bsz, n_stations, channels, t_len = x.shape
        y = x.reshape(bsz * n_stations, channels, t_len)
        y = F.relu(bn(conv(y)))
        return y.reshape(bsz, n_stations, y.shape[1], y.shape[2])

    @staticmethod
    def _pad_for_downsample(
        x: torch.Tensor, *, kernel_size: int, stride: int, dilation: int
    ) -> torch.Tensor:
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
        self, x: torch.Tensor, return_extra_level_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        if x.ndim == 4 and x.shape[2] == 1:
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(
                f"encoder expects [B, S, T] or [B, S, 1, T], got {tuple(x.shape)}."
            )

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

        if return_extra_level_features:
            return bottleneck_features, extra_level_features
        return bottleneck_features


class SegmentationQueryHead(nn.Module):
    """Mask-conditioned multi-query head with learned start/end decoding.

    Input: fused memory [B, D, T]. Outputs class/confidence, dense query mask
    logits, and normalized start/end times predicted through a learned
    boundary decoder. Query cross-attention is optional.
    """

    def __init__(
        self,
        input_dim: int,
        num_queries: int,
        query_head_channels: int,
        num_classes: int = 6,
        query_head_kernel_size: int = 31,
        use_query_cross_attention: bool = False,
        boundary_attn_heads: int = 4,
    ):
        super().__init__()

        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}.")
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}.")
        if query_head_channels < 1:
            raise ValueError(
                f"query_head_channels must be >= 1, got {query_head_channels}."
            )
        if query_head_kernel_size < 1 or query_head_kernel_size % 2 == 0:
            raise ValueError(
                "query_head_kernel_size must be a positive odd integer, got "
                f"{query_head_kernel_size}."
            )
        if use_query_cross_attention:
            if boundary_attn_heads < 1:
                raise ValueError(
                    f"boundary_attn_heads must be >= 1, got {boundary_attn_heads}."
                )
            if query_head_channels % boundary_attn_heads != 0:
                raise ValueError(
                    "query_head_channels must be divisible by boundary_attn_heads, got "
                    f"query_head_channels={query_head_channels}, "
                    f"boundary_attn_heads={boundary_attn_heads}."
                )
        self.num_queries = int(num_queries)
        self.query_head_channels = int(query_head_channels)
        self.use_query_cross_attention = bool(use_query_cross_attention)
        self.query_head_kernel_size = int(query_head_kernel_size)
        self.eps = 1e-6

        extractor_kernel_size = 3
        self.query_feature_extractor = nn.Sequential(
            nn.Conv1d(
                input_dim,
                self.num_queries * self.query_head_channels,
                kernel_size=extractor_kernel_size,
                padding=extractor_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.num_queries * self.query_head_channels, eps=1e-3),
            nn.ReLU(),
        )

        self.class_head = nn.Linear(self.query_head_channels, num_classes)
        self.confidence_head = nn.Linear(self.query_head_channels, 1)
        self.mask_logits_head = nn.Conv1d(
            self.query_head_channels,
            1,
            kernel_size=self.query_head_kernel_size,
            padding=self.query_head_kernel_size // 2,
        )
        if self.use_query_cross_attention:
            self.mask_context_proj = nn.Sequential(
                nn.Conv1d(
                    self.query_head_channels + 1,
                    self.query_head_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm1d(self.query_head_channels, eps=1e-3),
                nn.ReLU(),
            )
            self.query_cross_attn_norm = nn.LayerNorm(self.query_head_channels)
            self.query_cross_attn = nn.MultiheadAttention(
                embed_dim=self.query_head_channels,
                num_heads=int(boundary_attn_heads),
                dropout=0.0,
                batch_first=True,
            )
            self.query_cross_ff_norm = nn.LayerNorm(self.query_head_channels)
            self.query_cross_ff = nn.Sequential(
                nn.Linear(self.query_head_channels, self.query_head_channels * 2),
                nn.GELU(),
                nn.Linear(self.query_head_channels * 2, self.query_head_channels),
            )
        else:
            self.mask_context_proj = None
            self.query_cross_attn_norm = None
            self.query_cross_attn = None
            self.query_cross_ff_norm = None
            self.query_cross_ff = None
        self.start_end_logits_head = nn.Sequential(
            nn.Conv1d(
                self.query_head_channels * 2 + 1,
                self.query_head_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.query_head_channels, eps=1e-3),
            nn.ReLU(),
            nn.Conv1d(
                self.query_head_channels,
                2,
                kernel_size=self.query_head_kernel_size,
                padding=self.query_head_kernel_size // 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "SegmentationQueryHead expects [B, D, T], " f"got {tuple(x.shape)}."
            )

        batch_size, _, t_len = x.shape
        query_features = self.query_feature_extractor(x)
        query_features = query_features.view(
            batch_size,
            self.num_queries,
            self.query_head_channels,
            t_len,
        )

        flat_q = query_features.reshape(
            batch_size * self.num_queries,
            self.query_head_channels,
            t_len,
        )

        time_grid = torch.linspace(
            0.0,
            1.0,
            steps=t_len,
            device=x.device,
            dtype=x.dtype,
        )[None, :]

        mask_logits = self.mask_logits_head(flat_q).squeeze(1)
        mask_probs = torch.sigmoid(mask_logits).clamp(min=self.eps, max=1.0 - self.eps)
        mask_weights = mask_probs / mask_probs.sum(dim=-1, keepdim=True).clamp_min(
            self.eps
        )

        # Mask-conditioned token can be optionally refined by query-to-time cross attention.
        pooled = (flat_q * mask_weights[:, None, :]).sum(dim=-1)
        if self.use_query_cross_attention:
            query_token = pooled[:, None, :]
            mask_context = self.mask_context_proj(
                torch.cat([flat_q, mask_probs[:, None, :]], dim=1)
            ).transpose(1, 2)
            query_normed = self.query_cross_attn_norm(query_token)
            attn_out, _ = self.query_cross_attn(
                query=query_normed,
                key=mask_context,
                value=mask_context,
                need_weights=False,
            )
            query_token = query_token + attn_out
            query_token = query_token + self.query_cross_ff(
                self.query_cross_ff_norm(query_token)
            )
            pooled = query_token[:, 0, :]

        class_logits = self.class_head(pooled).view(batch_size, self.num_queries, -1)
        confidence_logits = self.confidence_head(pooled).view(
            batch_size, self.num_queries, 1
        )

        boundary_logits = self.start_end_logits_head(
            torch.cat(
                [
                    flat_q,
                    pooled[:, :, None].expand(-1, -1, t_len),
                    mask_probs[:, None, :],
                ],
                dim=1,
            )
        )
        start_prob = torch.softmax(boundary_logits[:, 0, :], dim=-1)
        end_prob = torch.softmax(boundary_logits[:, 1, :], dim=-1)
        start = (start_prob * time_grid).sum(dim=-1).unsqueeze(-1)
        end = (end_prob * time_grid).sum(dim=-1).unsqueeze(-1)
        start = start.view(batch_size, self.num_queries, 1)
        end = end.view(batch_size, self.num_queries, 1)
        start_ordered = torch.minimum(start, end)
        end_ordered = torch.maximum(start, end)

        predictions: Dict[str, torch.Tensor] = {
            "class_logits": class_logits,
            "confidence_logits": confidence_logits,
            "mask_logits": mask_logits.view(batch_size, self.num_queries, t_len),
            "start": start_ordered,
            "end": end_ordered,
        }

        return predictions


class MuSSED(nn.Module):
    """MuSSED: lean multi-station detector with fused-memory segmentation head."""

    def __init__(
        self,
        num_classes: int = 6,
        depth: int = 4,
        kernel_size: int | Sequence[int] = 127,
        stride: int | Sequence[int] = [2, 2, 2],
        dilation: int | Sequence[int] = [1, 1, 1, 1],
        filters_root: int = 16,
        bottleneck_attention: bool = True,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_ff_mult: int = 2,
        station_attn_heads: int = 4,
        station_attn_ff_mult: int = 2,
        station_mask_abs_sum_threshold: float = 1e1,
        memory_levels: Sequence[int] = (0, 1, 2),
        query_dim: int = 128,
        num_queries: int = 1,
        query_head_channels: int = 256,
        level_merge_kernel_size: int = 1,
        query_head_kernel_size: int = 31,
        use_query_cross_attention: bool = False,
        boundary_attn_heads: int = 4,
    ):
        super().__init__()

        if query_dim < 1:
            raise ValueError(f"query_dim must be >= 1, got {query_dim}.")
        if level_merge_kernel_size < 1 or level_merge_kernel_size % 2 == 0:
            raise ValueError(
                "level_merge_kernel_size must be a positive odd integer, got "
                f"{level_merge_kernel_size}."
            )

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

        self.memory_levels = self.encoder.memory_levels

        self.bottleneck_projection = nn.Linear(self.encoder.output_channels, query_dim)
        self.memory_level_projections = nn.ModuleDict(
            {
                str(level): nn.Linear(self.encoder.level_channels[level], query_dim)
                for level in self.memory_levels
            }
        )

        stride_by_level: dict[str, float] = {
            "bottleneck": float(self.encoder.bottleneck_stride),
            **{
                str(level): float(self.encoder.level_base_strides[level])
                for level in self.memory_levels
            },
        }
        if "0" in stride_by_level:
            self.reference_memory_level = "0"
        else:
            self.reference_memory_level = min(
                stride_by_level.keys(), key=lambda k: stride_by_level[k]
            )
        self.multiscale_memory_order = [self.reference_memory_level] + [
            key
            for key in sorted(stride_by_level.keys(), key=lambda k: stride_by_level[k])
            if key != self.reference_memory_level
        ]

        reference_stride = stride_by_level[self.reference_memory_level]
        self.memory_level_upsamplers = nn.ModuleDict()
        for level_key in self.multiscale_memory_order:
            level_stride = stride_by_level[level_key]
            ratio = level_stride / reference_stride
            up_factor = int(round(ratio))
            if not math.isclose(ratio, float(up_factor), rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    "Expected integer stride ratios for multiscale upsampling, "
                    f"got ratio={ratio} from level {level_key} with stride "
                    f"{level_stride} and reference stride {reference_stride}."
                )
            self.memory_level_upsamplers[level_key] = LightweightTemporalUpsample(
                channels=query_dim,
                up_factor=up_factor,
            )

        self.num_memory_levels = len(self.multiscale_memory_order)
        self.level_merge_kernel_size = int(level_merge_kernel_size)
        self.level_merger = nn.Sequential(
            nn.Conv2d(
                in_channels=query_dim,
                out_channels=int(query_head_channels),
                kernel_size=(self.num_memory_levels, self.level_merge_kernel_size),
                padding=(0, self.level_merge_kernel_size // 2),
                bias=False,
            ),
            nn.BatchNorm2d(int(query_head_channels), eps=1e-3),
            nn.ReLU(),
        )
        self.last_level_fusion_weights: torch.Tensor | None = None

        self.detection_head = SegmentationQueryHead(
            input_dim=int(query_head_channels),
            num_queries=int(num_queries),
            query_head_channels=int(query_head_channels),
            num_classes=int(num_classes),
            query_head_kernel_size=int(query_head_kernel_size),
            use_query_cross_attention=bool(use_query_cross_attention),
            boundary_attn_heads=int(boundary_attn_heads),
        )

        self.query_dim = int(query_dim)
        self.num_classes = int(num_classes)
        self.interval_output_format = "start_end"

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_features, extra_level_features = self.encoder(
            x, return_extra_level_features=True
        )

        bottleneck_proj = self.bottleneck_projection(encoder_features.transpose(1, 2))
        memory_parts: dict[str, torch.Tensor] = {
            "bottleneck": bottleneck_proj.transpose(1, 2)
        }

        for level in self.memory_levels:
            level_features = extra_level_features[level]
            level_proj = self.memory_level_projections[str(level)](
                level_features.transpose(1, 2)
            )
            memory_parts[str(level)] = level_proj.transpose(1, 2)

        target_len = memory_parts[self.reference_memory_level].shape[-1]
        aligned_parts: list[torch.Tensor] = []
        for level_key in self.multiscale_memory_order:
            part = memory_parts[level_key]
            part = self.memory_level_upsamplers[level_key](part, target_len)
            aligned_parts.append(part)

        multiscale_level_memory = torch.stack(aligned_parts, dim=2)  # [B, D, L, T0]
        merged_memory = self.level_merger(multiscale_level_memory)
        fused_memory = merged_memory.squeeze(2)
        if fused_memory.ndim != 3:
            raise RuntimeError(
                "Expected merged memory shape [B, H, T] after level merge, got "
                f"{tuple(fused_memory.shape)}."
            )
        self.last_level_fusion_weights = None

        predictions = self.detection_head(fused_memory)
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
