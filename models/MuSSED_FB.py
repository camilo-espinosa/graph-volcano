"""MuSSED-FB: event detection with a learnable spectral-envelope front end."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MuSSED import (
    SegmentationQueryHead,
    StationAttentionBlock,
    TemporalBottleneckAttention,
)


def _logit(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


class LearnableSincFilterbank(nn.Module):
    """Learnable Hann-windowed band-pass filters with bounded 1–15 Hz cutoffs."""

    def __init__(
        self,
        num_bands: int,
        sample_rate: float = 102.4,
        kernel_size: int = 251,
        low_limit: float = 1.0,
        high_limit: float = 15.0,
        min_bandwidth: float = 0.1,
    ):
        super().__init__()
        if num_bands < 1:
            raise ValueError(f"num_bands must be positive, got {num_bands}.")
        if sample_rate <= 2.0 * high_limit:
            raise ValueError(
                f"sample_rate must exceed twice high_limit ({high_limit} Hz)."
            )
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be an odd integer >= 3, got {kernel_size}."
            )
        if not 0.0 < low_limit < high_limit < sample_rate / 2.0:
            raise ValueError(
                "Require 0 < low_limit < high_limit < Nyquist; got "
                f"{low_limit}, {high_limit}, {sample_rate / 2.0}."
            )
        if min_bandwidth <= 0 or min_bandwidth >= high_limit - low_limit:
            raise ValueError(
                "min_bandwidth must be positive and narrower than the frequency range."
            )

        self.num_bands = int(num_bands)
        self.sample_rate = float(sample_rate)
        self.kernel_size = int(kernel_size)
        self.low_limit = float(low_limit)
        self.high_limit = float(high_limit)
        self.min_bandwidth = float(min_bandwidth)

        edges = torch.logspace(
            math.log10(self.low_limit),
            math.log10(self.high_limit),
            steps=self.num_bands + 1,
        )
        initial_lows = edges[:-1]
        initial_widths = edges[1:] - edges[:-1]

        low_max = self.high_limit - self.min_bandwidth
        low_fraction = (initial_lows - self.low_limit) / (
            low_max - self.low_limit
        )
        width_room = self.high_limit - initial_lows - self.min_bandwidth
        width_fraction = (initial_widths - self.min_bandwidth) / width_room
        self.low_cutoff_logits = nn.Parameter(_logit(low_fraction))
        self.bandwidth_logits = nn.Parameter(_logit(width_fraction))

        time = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        self.register_buffer("time", time)
        self.register_buffer("window", torch.hann_window(kernel_size, periodic=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_max = self.high_limit - self.min_bandwidth
        low_hz = self.low_limit + (low_max - self.low_limit) * torch.sigmoid(
            self.low_cutoff_logits
        )
        bandwidth_room = self.high_limit - low_hz - self.min_bandwidth
        bandwidth_hz = self.min_bandwidth + bandwidth_room * torch.sigmoid(
            self.bandwidth_logits
        )
        high_hz = low_hz + bandwidth_hz

        time = self.time.to(dtype=x.dtype)
        window = self.window.to(dtype=x.dtype)
        high = 2.0 * high_hz[:, None] / self.sample_rate
        low = 2.0 * low_hz[:, None] / self.sample_rate
        kernels = (high * torch.sinc(high * time) - low * torch.sinc(low * time))
        kernels = kernels * window[None, :]
        kernels = kernels / kernels.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(
            1e-8
        )
        return F.conv1d(
            x,
            kernels[:, None, :],
            padding=self.kernel_size // 2,
        )


class FilterbankMultiStationEncoder(nn.Module):
    """Reduced-rate spectral-envelope encoder with multiscale station fusion."""

    def __init__(
        self,
        num_bands: int,
        filters_root: int,
        sample_rate: float = 102.4,
        station_attn_heads: int = 4,
        station_attn_ff_mult: int = 2,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_ff_mult: int = 2,
        station_mask_abs_sum_threshold: float = 1e1,
    ):
        super().__init__()
        if filters_root < 1:
            raise ValueError(f"filters_root must be positive, got {filters_root}.")
        if station_mask_abs_sum_threshold < 0:
            raise ValueError(
                "station_mask_abs_sum_threshold must be non-negative, got "
                f"{station_mask_abs_sum_threshold}."
            )
        self.station_mask_abs_sum_threshold = float(station_mask_abs_sum_threshold)
        self.filterbank = LearnableSincFilterbank(
            num_bands=num_bands, sample_rate=sample_rate
        )
        self.raw_branch = nn.Sequential(
            nn.Conv1d(1, num_bands, kernel_size=15, stride=8, padding=7, bias=False),
            nn.BatchNorm1d(num_bands, eps=1e-3),
            nn.GELU(),
        )

        self.level0 = nn.Sequential(
            nn.Conv1d(2 * num_bands, filters_root, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(filters_root, eps=1e-3),
            nn.GELU(),
        )
        self.down0 = nn.Sequential(
            nn.Conv1d(
                filters_root, 2 * filters_root, kernel_size=5, stride=2, padding=2, bias=False
            ),
            nn.BatchNorm1d(2 * filters_root, eps=1e-3),
            nn.GELU(),
        )
        self.level1 = nn.Sequential(
            nn.Conv1d(2 * filters_root, 2 * filters_root, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(2 * filters_root, eps=1e-3),
            nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(
                2 * filters_root,
                4 * filters_root,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(4 * filters_root, eps=1e-3),
            nn.GELU(),
        )
        self.level2 = nn.Sequential(
            nn.Conv1d(4 * filters_root, 4 * filters_root, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(4 * filters_root, eps=1e-3),
            nn.GELU(),
        )
        self.station_attention = StationAttentionBlock(
            4 * filters_root, station_attn_heads, station_attn_ff_mult
        )
        self.bottleneck_attention = TemporalBottleneckAttention(
            4 * filters_root,
            bottleneck_attn_heads,
            bottleneck_attn_ff_mult,
        )
        self.output_channels = 4 * filters_root
        self.level_channels = {
            0: filters_root,
            1: 2 * filters_root,
        }
        self.capture_attention_weights = False

    @staticmethod
    def _max_fuse_stations(
        features: torch.Tensor, missing_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = (~missing_mask)[:, :, None, None]
        fused = torch.where(valid, features, torch.full_like(features, -torch.inf))
        result = fused.max(dim=1).values
        if not torch.isfinite(result).all():
            raise RuntimeError("Non-finite values encountered during station fusion.")
        return result

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, stations, time_steps = x.shape
        flat = x.reshape(batch_size * stations, 1, time_steps)
        band_features = self.filterbank(flat)
        band_envelopes = F.avg_pool1d(
            band_features.abs(), kernel_size=8, stride=8, ceil_mode=True
        )
        band_envelopes = torch.log1p(band_envelopes)
        raw_features = self.raw_branch(flat)
        if band_envelopes.shape[-1] != raw_features.shape[-1]:
            raise RuntimeError(
                "Filterbank and raw branches produced different time dimensions: "
                f"{band_envelopes.shape[-1]} and {raw_features.shape[-1]}."
            )
        features = torch.cat([band_envelopes, raw_features], dim=1)
        return features.reshape(
            batch_size, stations, features.shape[1], features.shape[-1]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        if x.ndim == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        if x.ndim != 3:
            raise ValueError(
                f"Expected input [B, S, T] or [B, S, 1, T], got {tuple(x.shape)}."
            )
        missing_mask = (
            x.abs().sum(dim=-1) <= self.station_mask_abs_sum_threshold
        )
        if missing_mask.all(dim=1).any():
            raise RuntimeError("At least one station must be present in every sample.")
        batch_size, stations, _ = x.shape
        valid_scale = (~missing_mask).to(x.dtype)[:, :, None, None]

        features = self._extract_features(x)
        y = self.level0(features.reshape(batch_size * stations, features.shape[2], -1))
        level0 = y.reshape(batch_size, stations, y.shape[1], y.shape[-1])
        level0 = level0 * valid_scale

        y = self.down0(level0.reshape(batch_size * stations, level0.shape[2], -1))
        y = self.level1(y)
        level1 = y.reshape(batch_size, stations, y.shape[1], y.shape[-1])
        level1 = level1 * valid_scale

        y = self.down1(level1.reshape(batch_size * stations, level1.shape[2], -1))
        y = self.level2(y)
        y = y.reshape(batch_size, stations, y.shape[1], y.shape[-1])
        y = self.station_attention(y, station_key_padding_mask=missing_mask)
        station_weights = self.station_attention.last_station_weights
        y = y * valid_scale

        _, _, channels, time_len = y.shape
        y = self.bottleneck_attention(
            y.reshape(batch_size * stations, channels, time_len)
        )
        y = y.reshape(batch_size, stations, channels, time_len) * valid_scale

        if station_weights is None or station_weights.shape != missing_mask.shape:
            raise RuntimeError("Station attention did not produce valid fusion weights.")
        weights = station_weights.to(y.dtype).masked_fill(missing_mask, 0.0)
        denominator = weights.sum(dim=1, keepdim=True)
        if (denominator <= 0).any():
            raise RuntimeError("No valid station-attention weights remain after masking.")
        bottleneck = (y * (weights / denominator)[:, :, None, None]).sum(dim=1)
        extras = {
            0: self._max_fuse_stations(level0, missing_mask),
            1: self._max_fuse_stations(level1, missing_mask),
        }
        return bottleneck, extras


class MuSSEDFB(nn.Module):
    """Single-query detector with learnable band-envelope features."""

    def __init__(
        self,
        num_classes: int = 6,
        num_bands: int = 16,
        filters_root: int = 24,
        sample_rate: float = 102.4,
        num_queries: int = 1,
        query_dim: int = 128,
        query_head_channels: int = 256,
        query_head_kernel_size: int = 3,
        station_attn_heads: int = 4,
        bottleneck_attn_heads: int = 4,
        station_mask_abs_sum_threshold: float = 1e1,
    ):
        super().__init__()
        self.encoder = FilterbankMultiStationEncoder(
            num_bands=num_bands,
            filters_root=filters_root,
            sample_rate=sample_rate,
            station_attn_heads=station_attn_heads,
            bottleneck_attn_heads=bottleneck_attn_heads,
            station_mask_abs_sum_threshold=station_mask_abs_sum_threshold,
        )
        self.memory_levels = (0, 1)
        self.bottleneck_projection = nn.Conv1d(
            self.encoder.output_channels, query_dim, kernel_size=1
        )
        self.memory_level_projections = nn.ModuleDict(
            {
                str(level): nn.Conv1d(
                    self.encoder.level_channels[level], query_dim, kernel_size=1
                )
                for level in self.memory_levels
            }
        )
        self.level_embeddings = nn.ParameterDict(
            {
                key: nn.Parameter(torch.randn(1, query_dim, 1) / (query_dim**0.5))
                for key in ("bottleneck", "0", "1")
            }
        )
        self.level_temporal_scorer = nn.Conv1d(
            3, 3, kernel_size=3, padding=1, groups=3
        )
        self.fused_refine = nn.Sequential(
            nn.Conv1d(query_dim, query_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(query_dim, eps=1e-3),
            nn.ReLU(),
        )
        self.detection_head = SegmentationQueryHead(
            input_dim=query_dim,
            num_queries=num_queries,
            query_head_channels=query_head_channels,
            num_classes=num_classes,
            query_head_kernel_size=query_head_kernel_size,
        )
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.interval_output_format = "start_end"

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bottleneck, extra_levels = self.encoder(x)
        target_len = extra_levels[0].shape[-1]
        memory_parts = [
            self.memory_level_projections["0"](extra_levels[0])
            + self.level_embeddings["0"],
            self.memory_level_projections["1"](extra_levels[1])
            + self.level_embeddings["1"],
            self.bottleneck_projection(bottleneck)
            + self.level_embeddings["bottleneck"],
        ]
        aligned = [
            F.interpolate(part, size=target_len, mode="linear", align_corners=False)
            for part in memory_parts
        ]
        multiscale = torch.stack(aligned, dim=2)
        level_logits = self.level_temporal_scorer(multiscale.mean(dim=1))
        level_weights = torch.softmax(level_logits, dim=1)
        fused = (multiscale * level_weights[:, None, :, :]).sum(dim=2)
        fused = self.fused_refine(fused)
        predictions = self.detection_head(fused)
        predictions["encoder_features"] = bottleneck
        return predictions

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
