"""
UNet_MPNN - geometry-free multistation 1D U-Net with permutation-equivariant
station fusion and interpretable bottleneck station pooling.

Key ideas
---------
- Geometry-free: no station coordinates, edge attributes, or volcano routing.
- Early multiscale station fusion: cheap shared station-wise fusion blocks at
  selected encoder levels.
- Interpretable bottleneck pooling: optional temporal self-attention per station
  plus station-level attention weights for station importance.
- Mask-aware reductions: zero-padded stations are excluded from mean/max/softmax
  reductions via a validity mask.

Input:  [B, S, T]
Output: [B, C_out, T]
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal, Optional

import torch
import torch.nn as nn


class StationFusion(nn.Module):
    """
    Cheap permutation-equivariant station fusion.

    For each destination station, build a context tensor from OTHER stations using:
    - masked max over valid other stations
    - masked mean over valid other stations

    Then apply a shared temporal Conv1d over [own || max_ctx || mean_ctx], project
    back to the original channel count, and residual-add to own features.
    """

    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"fusion kernel_size must be odd for symmetric padding. Got {kernel_size}."
            )

        self.temporal = nn.Conv1d(
            in_channels=3 * channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm1d(channels)
        self.temporal_act = nn.ReLU(inplace=True)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    @staticmethod
    def _masked_other_mean(
        x: torch.Tensor,
        station_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid_f = (
            station_valid.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        )  # [B,S,1,1]
        sum_all = (x * valid_f).sum(dim=1, keepdim=True)  # [B,1,C,T]
        cnt_all = valid_f.sum(dim=1, keepdim=True)  # [B,1,1,1]

        other_sum = sum_all - (x * valid_f)
        other_cnt = cnt_all - valid_f
        has_other = other_cnt > 0

        safe_cnt = other_cnt.clamp_min(1.0)
        mean_other = other_sum / safe_cnt
        return torch.where(has_other, mean_other, torch.zeros_like(mean_other))

    @staticmethod
    def _masked_other_max(
        x: torch.Tensor,
        station_valid: torch.Tensor,
    ) -> torch.Tensor:
        B, S, C, T = x.shape
        valid = station_valid.bool()
        large_neg = x.new_tensor(-1e9)

        x_masked = x.masked_fill(~valid.unsqueeze(-1).unsqueeze(-1), large_neg)

        k = 2 if S >= 2 else 1
        top_vals, top_idx = torch.topk(x_masked, k=k, dim=1)
        max1 = top_vals[:, 0:1]  # [B,1,C,T]
        idx1 = top_idx[:, 0:1]  # [B,1,C,T]
        if k == 2:
            max2 = top_vals[:, 1:2]
        else:
            max2 = torch.full_like(max1, large_neg)

        stn_idx = torch.arange(S, device=x.device).view(1, S, 1, 1)
        owner_is_max = idx1.expand(B, S, C, T).eq(stn_idx)
        max_other = torch.where(
            owner_is_max,
            max2.expand(B, S, C, T),
            max1.expand(B, S, C, T),
        )

        valid_f = valid.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        cnt_all = valid_f.sum(dim=1, keepdim=True)
        other_cnt = cnt_all - valid_f
        has_other = other_cnt > 0
        return torch.where(has_other, max_other, torch.zeros_like(max_other))

    def forward(self, x: torch.Tensor, station_valid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, C, T]
            station_valid: [B, S] bool

        Returns:
            [B, S, C, T]
        """
        B, S, C, T = x.shape
        valid_f = station_valid.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)

        own = x
        max_ctx = self._masked_other_max(x, station_valid)
        mean_ctx = self._masked_other_mean(x, station_valid)

        fused_in = torch.cat([own, max_ctx, mean_ctx], dim=2)  # [B,S,3C,T]
        fused_in = fused_in.reshape(B * S, 3 * C, T)

        fused = self.temporal(fused_in)
        fused = self.temporal_act(self.temporal_bn(fused))
        fused = self.proj(fused).reshape(B, S, C, T)

        out = own + fused
        return out * valid_f


class UNet_MPNN(nn.Module):
    """
    Geometry-free 1D U-Net for multistation seismic segmentation.

    Architecture:
    - Per-station 1D UNet encoder/decoder.
    - Early encoder levels can run StationFusion blocks (cheap, mask-aware,
      permutation-equivariant station interaction).
    - Bottleneck applies optional temporal MHSA per station, then masked station
      pooling (mean/max/attention) to obtain a network-level decoder input.
    - Decoder uses masked station readouts from encoder skips.

    Interpretability:
    - temporal_weights: bottleneck temporal attention map from MHSA.
    - station_weights: bottleneck station attention scores (attention readout mode).
    - station_valid: station validity mask derived from padded inputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        init_features: int = 16,
        depth: int = 5,
        n_stations: int = 8,
        station_fusion_levels: Optional[list[int]] = None,
        fusion_kernel: int = 9,
        readout_mode: Literal["mean", "max", "attention"] = "attention",
        use_bottleneck_attention: bool = True,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_dropout: float = 0.2,
        bottleneck_attn_ff_mult: int = 2,
        feature_dropout: float = 0.2,
        return_attention: bool = False,
    ):
        super().__init__()

        if station_fusion_levels is None:
            station_fusion_levels = [0, 1, 2]

        if fusion_kernel % 2 == 0:
            raise ValueError(
                f"fusion_kernel must be odd for symmetric padding. Got {fusion_kernel}."
            )
        if readout_mode not in {"mean", "max", "attention"}:
            raise ValueError(
                f"readout_mode must be one of {{'mean', 'max', 'attention'}}. Got: {readout_mode}"
            )
        if feature_dropout < 0.0 or feature_dropout >= 1.0:
            raise ValueError(
                f"feature_dropout must be in [0, 1). Got: {feature_dropout}."
            )

        fusion_levels = sorted(set(int(lvl) for lvl in station_fusion_levels))
        bad_levels = [lvl for lvl in fusion_levels if lvl < 0 or lvl >= depth]
        if bad_levels:
            raise ValueError(
                f"station_fusion_levels must be in [0, {depth}). Invalid: {bad_levels}"
            )

        self.n_stations = int(n_stations)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.init_features = int(init_features)
        self.depth = int(depth)
        self.fusion_kernel = int(fusion_kernel)
        self.readout_mode = readout_mode
        self.use_bottleneck_attention = bool(use_bottleneck_attention)
        self.feature_dropout_p = float(feature_dropout)
        self.default_return_attention = bool(return_attention)

        self.encoder_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.station_fusion = nn.ModuleDict()

        feat_in = in_channels
        for idx in range(depth):
            feat_out = init_features * (2**idx)
            self.encoder_list.append(
                self._block_1d(
                    feat_in,
                    feat_out,
                    name=f"enc{idx}",
                    feature_dropout=self.feature_dropout_p,
                )
            )
            self.pool_list.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if idx in fusion_levels:
                self.station_fusion[str(idx)] = StationFusion(
                    channels=feat_out,
                    kernel_size=self.fusion_kernel,
                )
            feat_in = feat_out

        self.bottleneck_feat_channels = init_features * (2 ** (depth - 1))

        if self.use_bottleneck_attention:
            if self.bottleneck_feat_channels % bottleneck_attn_heads != 0:
                raise ValueError(
                    "bottleneck_feat_channels must be divisible by bottleneck_attn_heads. "
                    f"Got C={self.bottleneck_feat_channels}, heads={bottleneck_attn_heads}."
                )
            self.bottleneck_attn_norm1 = nn.LayerNorm(self.bottleneck_feat_channels)
            self.bottleneck_attn = nn.MultiheadAttention(
                embed_dim=self.bottleneck_feat_channels,
                num_heads=bottleneck_attn_heads,
                dropout=bottleneck_attn_dropout,
                batch_first=True,
            )
            self.bottleneck_attn_norm2 = nn.LayerNorm(self.bottleneck_feat_channels)
            self.bottleneck_ff = nn.Sequential(
                nn.Linear(
                    self.bottleneck_feat_channels,
                    self.bottleneck_feat_channels * bottleneck_attn_ff_mult,
                ),
                nn.GELU(),
                nn.Dropout(bottleneck_attn_dropout),
                nn.Linear(
                    self.bottleneck_feat_channels * bottleneck_attn_ff_mult,
                    self.bottleneck_feat_channels,
                ),
                nn.Dropout(bottleneck_attn_dropout),
            )
        else:
            self.bottleneck_attn_norm1 = None
            self.bottleneck_attn = None
            self.bottleneck_attn_norm2 = None
            self.bottleneck_ff = None

        self.station_score_mlp = nn.Sequential(
            nn.Linear(
                self.bottleneck_feat_channels, self.bottleneck_feat_channels // 2
            ),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_feat_channels // 2, 1),
        )

        self.decoder_list = nn.ModuleList()
        self.upconv_list = nn.ModuleList()

        current_channels = self.bottleneck_feat_channels
        for idx in range(depth):
            skip_channels = init_features * (2 ** (depth - 1 - idx))
            self.upconv_list.append(
                nn.ConvTranspose1d(
                    current_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoder_list.append(
                self._block_1d(
                    2 * skip_channels,
                    skip_channels,
                    name=f"dec{depth - idx}",
                    feature_dropout=self.feature_dropout_p,
                )
            )
            current_channels = skip_channels

        self.final_dropout = nn.Dropout(self.feature_dropout_p)
        self.conv_final = nn.Conv1d(init_features, out_channels, kernel_size=1)

    @staticmethod
    def _block_1d(
        in_channels: int,
        features: int,
        name: str,
        feature_dropout: float,
    ) -> nn.Sequential:
        dropout = (
            nn.Dropout(feature_dropout) if feature_dropout > 0.0 else nn.Identity()
        )
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "drop1", dropout),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            features,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "drop2", dropout),
                ]
            )
        )

    @staticmethod
    def _mask_stations(
        x: torch.Tensor,
        station_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid_f = station_valid.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        return x * valid_f

    @staticmethod
    def _masked_mean_stations(
        x: torch.Tensor,
        station_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid_f = station_valid.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        num = (x * valid_f).sum(dim=1)
        den = valid_f.sum(dim=1).clamp_min(1.0)
        return num / den

    @staticmethod
    def _masked_max_stations(
        x: torch.Tensor,
        station_valid: torch.Tensor,
    ) -> torch.Tensor:
        large_neg = x.new_tensor(-1e9)
        masked = x.masked_fill(~station_valid.unsqueeze(-1).unsqueeze(-1), large_neg)
        out = masked.max(dim=1).values
        has_any = station_valid.any(dim=1, keepdim=True).unsqueeze(-1)
        return torch.where(has_any, out, torch.zeros_like(out))

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int = 1,
    ) -> torch.Tensor:
        large_neg = logits.new_tensor(-1e9)
        masked_logits = logits.masked_fill(~mask, large_neg)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * mask.to(dtype=weights.dtype)
        z = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        return weights / z

    def _apply_bottleneck_attention(
        self,
        x_stations: torch.Tensor,
        want_temporal_weights: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Temporal MHSA per station.

        Args:
            x_stations: [B,S,C,T]
            want_temporal_weights: whether to return attention maps.

        Returns:
            x_out: [B,S,C,T]
            temporal_weights: [B,S,T,T] or None
        """
        if not self.use_bottleneck_attention:
            return x_stations, None

        B, S, C, T = x_stations.shape
        x_seq = x_stations.permute(0, 1, 3, 2).reshape(B * S, T, C)

        x_norm = self.bottleneck_attn_norm1(x_seq)
        x_attn, attn_w = self.bottleneck_attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=want_temporal_weights,
            average_attn_weights=True,
        )
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.bottleneck_ff(self.bottleneck_attn_norm2(x_seq))

        x_out = x_seq.reshape(B, S, T, C).permute(0, 1, 3, 2)

        temporal_weights = None
        if want_temporal_weights and attn_w is not None:
            temporal_weights = attn_w.reshape(B, S, T, T)

        return x_out, temporal_weights

    def _bottleneck_station_readout(
        self,
        x_stations: torch.Tensor,
        station_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Readout from station bottleneck tensor [B,S,C,T] to [B,C,T].

        Returns:
            pooled: [B,C,T]
            station_weights: [B,S] for attention mode, else None
        """
        if self.readout_mode == "mean":
            return self._masked_mean_stations(x_stations, station_valid), None

        if self.readout_mode == "max":
            return self._masked_max_stations(x_stations, station_valid), None

        station_summary = x_stations.mean(dim=-1)  # [B,S,C]
        scores = self.station_score_mlp(station_summary).squeeze(-1)  # [B,S]
        weights = self._masked_softmax(scores, station_valid, dim=1)  # [B,S]

        pooled = (x_stations * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return pooled, weights

    def _skip_readout(
        self, x_stations: torch.Tensor, station_valid: torch.Tensor
    ) -> torch.Tensor:
        # For attention mode, skips use masked mean for stability and lower cost.
        if self.readout_mode in {"mean", "attention"}:
            return self._masked_mean_stations(x_stations, station_valid)
        return self._masked_max_stations(x_stations, station_valid)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x: [B, S, T]
            return_attention: if True, return (out, attention_dict).

        Returns:
            out: [B, C_out, T]
            or (out, attn_dict)
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,S,T]. Got: {tuple(x.shape)}")

        B, S, T = x.shape
        if S != self.n_stations:
            raise ValueError(
                f"x has S={S} stations but model expects n_stations={self.n_stations}."
            )

        want_attn = bool(return_attention or self.default_return_attention)

        station_valid = x.abs().sum(dim=-1) > 0  # [B,S]

        x_flat = x.reshape(B * S, self.in_channels, T)
        encodings: list[torch.Tensor] = []

        for i in range(self.depth):
            x_flat = self.encoder_list[i](x_flat)
            C = x_flat.shape[1]
            Tl = x_flat.shape[2]

            x_stations = x_flat.reshape(B, S, C, Tl)
            x_stations = self._mask_stations(x_stations, station_valid)

            if str(i) in self.station_fusion:
                x_stations = self.station_fusion[str(i)](x_stations, station_valid)

            encodings.append(x_stations)

            x_flat = x_stations.reshape(B * S, C, Tl)
            x_flat = self.pool_list[i](x_flat)

        Cb, Tb = x_flat.shape[1], x_flat.shape[2]
        x_bottleneck = x_flat.reshape(B, S, Cb, Tb)
        x_bottleneck = self._mask_stations(x_bottleneck, station_valid)

        x_bottleneck, temporal_weights = self._apply_bottleneck_attention(
            x_bottleneck,
            want_temporal_weights=want_attn,
        )
        x_bottleneck = self._mask_stations(x_bottleneck, station_valid)

        x_dec, station_weights = self._bottleneck_station_readout(
            x_bottleneck,
            station_valid,
        )

        for i in range(self.depth):
            x_dec = self.upconv_list[i](x_dec)

            skip_stations = encodings[-(i + 1)]
            skip = self._skip_readout(skip_stations, station_valid)

            x_dec = torch.cat((x_dec, skip), dim=1)
            x_dec = self.decoder_list[i](x_dec)

        x_dec = self.final_dropout(x_dec)
        out = self.conv_final(x_dec)

        if not want_attn:
            return out

        attn: dict[str, torch.Tensor] = {
            "station_valid": station_valid,
        }
        if temporal_weights is not None:
            attn["temporal_weights"] = temporal_weights
        if station_weights is not None:
            attn["station_weights"] = station_weights

        return out, attn
