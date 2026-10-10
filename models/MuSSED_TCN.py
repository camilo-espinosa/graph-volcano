"""MuSSED-TCN: a resolution-preserving dilated temporal detector."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.MuSSED import SegmentationQueryHead, StationAttentionBlock


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 17):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm1d(channels, eps=1e-3)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm1d(channels, eps=1e-3)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.depthwise_norm(self.depthwise(x)))
        y = self.pointwise_norm(self.pointwise(y))
        return self.activation(x + y)


class MultiStationDilatedEncoder(nn.Module):
    """Shared dilated temporal processing with permutation-equivariant station fusion."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        station_attn_heads: int = 4,
        station_attn_ff_mult: int = 2,
        station_mask_abs_sum_threshold: float = 1e1,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")
        if station_mask_abs_sum_threshold < 0:
            raise ValueError(
                "station_mask_abs_sum_threshold must be non-negative, got "
                f"{station_mask_abs_sum_threshold}."
            )

        self.station_mask_abs_sum_threshold = float(station_mask_abs_sum_threshold)
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=17, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(channels, eps=1e-3),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(channels, dilation=2 ** (i % 9))
                for i in range(num_blocks)
            ]
        )
        self.mid_station_attention = StationAttentionBlock(
            channels, station_attn_heads, station_attn_ff_mult
        )
        self.station_attention = StationAttentionBlock(
            channels, station_attn_heads, station_attn_ff_mult
        )
        self.mid_attention_index = max(0, num_blocks // 2 - 1)
        self.output_channels = channels
        self.capture_attention_weights = False

    @staticmethod
    def _valid_scale(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (~mask).to(dtype)[:, :, None, None]

    @staticmethod
    def _fuse_stations(
        features: torch.Tensor,
        missing_mask: torch.Tensor,
        station_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if station_weights is None or station_weights.shape != missing_mask.shape:
            raise RuntimeError("Station attention did not produce valid fusion weights.")
        weights = station_weights.to(features.dtype).masked_fill(missing_mask, 0.0)
        denominator = weights.sum(dim=1, keepdim=True)
        if (denominator <= 0).any():
            raise RuntimeError("No valid station-attention weights remain after masking.")
        return (
            features * (weights / denominator)[:, :, None, None]
        ).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        batch_size, stations, time_steps = x.shape
        valid_scale = self._valid_scale(missing_mask, x.dtype)
        y = self.stem(x.reshape(batch_size * stations, 1, time_steps))
        y = y.reshape(batch_size, stations, self.output_channels, y.shape[-1])
        y = y * valid_scale

        station_weights: torch.Tensor | None = None
        for index, block in enumerate(self.blocks):
            y = block(y.reshape(batch_size * stations, self.output_channels, -1))
            y = y.reshape(batch_size, stations, self.output_channels, y.shape[-1])
            if index == self.mid_attention_index:
                y = self.mid_station_attention(
                    y, station_key_padding_mask=missing_mask
                )
                y = y * valid_scale
            if index == len(self.blocks) - 1:
                y = self.station_attention(
                    y, station_key_padding_mask=missing_mask
                )
                station_weights = self.station_attention.last_station_weights
                y = y * valid_scale

        return self._fuse_stations(y, missing_mask, station_weights)


class MuSSEDTCN(nn.Module):
    """Single-query detector using a dilated-convolution temporal encoder."""

    def __init__(
        self,
        num_classes: int = 6,
        channels: int = 48,
        num_blocks: int = 10,
        num_queries: int = 1,
        query_dim: int = 128,
        query_head_channels: int = 256,
        query_head_kernel_size: int = 3,
        station_attn_heads: int = 4,
        station_attn_ff_mult: int = 2,
        station_mask_abs_sum_threshold: float = 1e1,
    ):
        super().__init__()
        self.encoder = MultiStationDilatedEncoder(
            channels=channels,
            num_blocks=num_blocks,
            station_attn_heads=station_attn_heads,
            station_attn_ff_mult=station_attn_ff_mult,
            station_mask_abs_sum_threshold=station_mask_abs_sum_threshold,
        )
        self.memory_projection = nn.Sequential(
            nn.Conv1d(channels, query_dim, kernel_size=1, bias=False),
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
        self.supports_temporal_attention = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        predictions = self.detection_head(self.memory_projection(features))
        predictions["encoder_features"] = features
        return predictions

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
