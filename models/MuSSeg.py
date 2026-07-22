import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.station_info import get_crater_coords, get_station_coords


class StationPairMessageBlock(nn.Module):
    """Permutation-equivariant station message passing block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        aggregation: str = "sum",
        dropout_p: float = 0.0,
        station_message_ratio: float = 1.0,
    ):
        super().__init__()
        if aggregation not in {"sum", "attention"}:
            raise ValueError(
                "station_message_aggregation must be 'sum' or 'attention'. "
                f"Got: {aggregation}."
            )
        if station_message_ratio <= 0.0 or station_message_ratio > 1.0:
            raise ValueError(
                "station_message_ratio must be in (0, 1]. "
                f"Got: {station_message_ratio}."
            )
        self.aggregation = aggregation
        self.station_message_ratio = float(station_message_ratio)
        self.station_message_channels = max(
            1, int(channels * self.station_message_ratio)
        )
        self.use_bottleneck = self.station_message_channels < channels

        if self.use_bottleneck:
            self.reduce_conv = nn.Conv1d(
                channels, self.station_message_channels, kernel_size=1, bias=False
            )
            self.reduce_bn = nn.BatchNorm1d(self.station_message_channels, eps=1e-3)
            self.expand_conv = nn.Conv1d(
                self.station_message_channels, channels, kernel_size=1, bias=False
            )
            self.expand_bn = nn.BatchNorm1d(channels, eps=1e-3)
        else:
            self.reduce_conv = nn.Identity()
            self.reduce_bn = nn.Identity()
            self.expand_conv = nn.Identity()
            self.expand_bn = nn.Identity()

        self.message_conv = nn.Conv1d(
            2 * self.station_message_channels,
            self.station_message_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        self.message_bn = nn.BatchNorm1d(self.station_message_channels, eps=1e-3)
        _ = dropout_p
        self.message_dropout = nn.Identity()

        if self.aggregation == "attention":
            self.score_conv = nn.Conv1d(
                self.station_message_channels,
                self.station_message_channels,
                kernel_size=1,
                bias=True,
            )
            self.score_fc = nn.Linear(self.station_message_channels, 1)

    def _reduce_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_bottleneck:
            return x
        return torch.relu(self.reduce_bn(self.reduce_conv(x)))

    def _expand_messages(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_bottleneck:
            return x
        return torch.relu(self.expand_bn(self.expand_conv(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, T]
        bsz, n_stations, channels, t_len = x.shape
        if n_stations == 1:
            return x

        x_reduced = self._reduce_features(x.reshape(bsz * n_stations, channels, t_len))
        x_reduced = x_reduced.reshape(
            bsz, n_stations, self.station_message_channels, t_len
        )

        aggregated = x.new_zeros((bsz, n_stations, channels, t_len))

        for i in range(n_stations):
            x_i = x_reduced[:, i, :, :]

            if self.aggregation == "sum":
                agg_i_reduced = x.new_zeros((bsz, self.station_message_channels, t_len))
                for j in range(n_stations):
                    if i == j:
                        continue
                    x_j = x_reduced[:, j, :, :]
                    msg_in = torch.cat([x_i, x_j], dim=1)
                    msg = self.message_conv(msg_in)
                    msg = self.message_bn(msg)
                    msg = torch.relu(msg)
                    msg = self.message_dropout(msg)
                    agg_i_reduced = agg_i_reduced + msg

                agg_i = self._expand_messages(agg_i_reduced)
            else:
                # Pass 1: compute attention logits only (small tensor [B, S]).
                large_neg = torch.finfo(x.dtype).min
                scores_i = x.new_full((bsz, n_stations), large_neg)
                for j in range(n_stations):
                    if i == j:
                        continue

                    x_j = x_reduced[:, j, :, :]
                    msg_in = torch.cat([x_i, x_j], dim=1)
                    msg = self.message_conv(msg_in)
                    msg = self.message_bn(msg)
                    msg = torch.relu(msg)
                    msg = self.message_dropout(msg)

                    score_feat = torch.relu(self.score_conv(msg))
                    score_feat = score_feat.mean(dim=-1)
                    scores_i[:, j] = self.score_fc(score_feat).squeeze(-1)

                weights_i = torch.softmax(scores_i, dim=1)

                # Pass 2: recompute messages and accumulate weighted sum without stacking.
                agg_i_reduced = x.new_zeros((bsz, self.station_message_channels, t_len))
                for j in range(n_stations):
                    if i == j:
                        continue

                    x_j = x_reduced[:, j, :, :]
                    msg_in = torch.cat([x_i, x_j], dim=1)
                    msg = self.message_conv(msg_in)
                    msg = self.message_bn(msg)
                    msg = torch.relu(msg)
                    msg = self.message_dropout(msg)

                    w_j = weights_i[:, j][:, None, None]
                    agg_i_reduced = agg_i_reduced + w_j * msg

                agg_i = self._expand_messages(agg_i_reduced)

            aggregated[:, i, :, :] = agg_i

        return x + aggregated


class StationAttentionBlock(nn.Module):
    """Optional global attention over stations (not over time)."""

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 2,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(
                "station-attention channels must be divisible by heads. "
                f"Got C={channels}, heads={heads}."
            )
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(channels * ff_mult, channels),
            nn.Identity(),
        )

    def forward(
        self, x: torch.Tensor, dist_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, S, C, T]
        pooled = x.mean(dim=-1)

        pooled_norm = self.norm1(pooled)
        attn_out, _ = self.attn(
            pooled_norm,
            pooled_norm,
            pooled_norm,
            attn_mask=dist_bias,
            need_weights=False,
        )
        pooled = pooled + attn_out
        pooled = pooled + self.ff(self.norm2(pooled))

        station_update = pooled[:, :, :, None]
        return x + station_update


class MuSSeg(nn.Module):
    """MuSSeg standalone model with optional shared-station permutation-invariant path."""

    def __init__(
        self,
        in_channels=3,
        classes=3,
        out_channels=None,
        depth=5,
        kernel_size=7,
        stride=4,
        filters_root=8,
        norm="std",
        feature_dropout=0.0,
        bottleneck_attention=False,
        shared_station_encoder=False,
        station_interaction="none",
        station_message_levels=None,
        station_message_aggregation="sum",
        station_message_ratio=1.0,
        station_attention_levels=None,
        pre_bottleneck_station_attn_merge=False,
        bottleneck_attn_heads=4,
        bottleneck_attn_dropout=0.0,
        bottleneck_attn_ff_mult=2,
        station_attn_heads=4,
        station_attn_dropout=0.0,
        station_attn_ff_mult=2,
        use_distance_attn_bias=False,
        use_distance_bottleneck_emb=False,
        volcano_name=None,
        **kwargs,
    ):

        super().__init__()

        if out_channels is not None:
            classes = out_channels

        if station_message_levels is None:
            station_message_levels = []
        if station_attention_levels is None:
            station_attention_levels = []

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters_root = filters_root
        self.bottleneck_attention = bool(bottleneck_attention)
        self.shared_station_encoder = bool(shared_station_encoder)
        self.station_interaction = str(station_interaction)
        self.use_distance_attn_bias = bool(use_distance_attn_bias)
        self.use_distance_bottleneck_emb = bool(use_distance_bottleneck_emb)
        self.volcano_name = volcano_name
        self.station_message_levels = sorted(
            set(int(level) for level in station_message_levels)
        )
        self.station_message_aggregation = station_message_aggregation
        self.station_message_ratio = float(station_message_ratio)
        self.station_attention_levels = sorted(
            set(int(level) for level in station_attention_levels)
        )
        self.pre_bottleneck_station_attn_merge = bool(
            pre_bottleneck_station_attn_merge
        )

        if feature_dropout < 0.0 or feature_dropout >= 1.0:
            raise ValueError(
                f"feature_dropout must be in [0, 1). Got: {feature_dropout}."
            )
        if self.station_message_ratio <= 0.0 or self.station_message_ratio > 1.0:
            raise ValueError(
                "station_message_ratio must be in (0, 1]. "
                f"Got: {self.station_message_ratio}."
            )

        late_level = self.depth - 1

        if self.station_interaction not in {
            "none",
            "late_station_message",
            "late_attention",
        }:
            raise ValueError(
                "station_interaction must be one of "
                "{'none', 'late_station_message', 'late_attention'}. "
                f"Got: {self.station_interaction}."
            )

        if self.station_interaction != "none" and not self.shared_station_encoder:
            raise ValueError(
                "station_interaction requires shared_station_encoder=True."
            )

        if (
            self.use_distance_attn_bias or self.use_distance_bottleneck_emb
        ) and not self.shared_station_encoder:
            raise ValueError(
                "distance-aware station features require shared_station_encoder=True."
            )
        if (
            self.use_distance_attn_bias or self.use_distance_bottleneck_emb
        ) and self.volcano_name is None:
            raise ValueError(
                "volcano_name is required when enabling distance-aware station "
                "features."
            )
        if self.use_distance_attn_bias and self.station_interaction != "late_attention":
            raise ValueError(
                "use_distance_attn_bias requires station_interaction='late_attention'."
            )

        if self.station_interaction == "late_station_message":
            if len(self.station_attention_levels) > 0:
                raise ValueError(
                    "station_interaction='late_station_message' cannot be combined "
                    "with "
                    "station_attention_levels."
                )
            self.station_message_levels = [late_level]
        elif self.station_interaction == "late_attention":
            if len(self.station_message_levels) > 0:
                raise ValueError(
                    "station_interaction='late_attention' cannot be combined with "
                    "station_message_levels."
                )
            self.station_attention_levels = [late_level]

        valid_pair_levels = set(range(self.depth))
        invalid_pair_levels = [
            level
            for level in self.station_message_levels
            if level not in valid_pair_levels
        ]
        if invalid_pair_levels:
            raise ValueError(
                f"Invalid station_message_levels={invalid_pair_levels}. "
                f"Allowed levels for depth={self.depth}: {sorted(valid_pair_levels)}."
            )

        if not self.shared_station_encoder and self.station_message_levels:
            raise ValueError(
                "station_message_levels require shared_station_encoder=True, because "
                "station-message interaction is defined on station embeddings "
                "[B, S, C, T]."
            )

        valid_station_attention_levels = set(range(self.depth))
        invalid_station_attention_levels = [
            level
            for level in self.station_attention_levels
            if level not in valid_station_attention_levels
        ]
        if invalid_station_attention_levels:
            raise ValueError(
                f"Invalid station_attention_levels={invalid_station_attention_levels}. "
                f"Allowed levels for depth={self.depth}: "
                f"{sorted(valid_station_attention_levels)}."
            )

        if not self.shared_station_encoder and self.station_attention_levels:
            raise ValueError(
                "station_attention_levels require shared_station_encoder=True."
            )

        if self.pre_bottleneck_station_attn_merge and not self.shared_station_encoder:
            raise ValueError(
                "pre_bottleneck_station_attn_merge requires "
                "shared_station_encoder=True."
            )

        if self.pre_bottleneck_station_attn_merge and not self.bottleneck_attention:
            raise ValueError(
                "pre_bottleneck_station_attn_merge requires "
                "bottleneck_attention=True."
            )

        self.feature_dropout_p = float(feature_dropout)
        self.activation = torch.relu
        self.feature_dropout = nn.Identity()
        self.final_dropout = nn.Identity()

        self.inc = nn.Conv1d(
            self.in_channels, self.filters_root, self.kernel_size, padding="same"
        )
        self.inc_shared = nn.Conv1d(
            1, self.filters_root, self.kernel_size, padding="same"
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root, eps=1e-3)
        self.in_bn_shared = nn.BatchNorm1d(self.filters_root, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root

        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        self.station_message_blocks = nn.ModuleDict()
        for level in self.station_message_levels:
            channels = int(2**level * self.filters_root)
            self.station_message_blocks[str(level)] = StationPairMessageBlock(
                channels=channels,
                kernel_size=self.kernel_size,
                aggregation=self.station_message_aggregation,
                dropout_p=0.0,
                station_message_ratio=self.station_message_ratio,
            )

        self.station_attention_blocks = nn.ModuleDict()
        for level in self.station_attention_levels:
            channels = int(2**level * self.filters_root)
            self.station_attention_blocks[str(level)] = StationAttentionBlock(
                channels=channels,
                heads=station_attn_heads,
                dropout=0.0,
                ff_mult=station_attn_ff_mult,
            )

        self.bottleneck_channels = int(2 ** (self.depth - 1) * self.filters_root)
        if self.bottleneck_channels % bottleneck_attn_heads != 0:
            raise ValueError(
                "bottleneck channels must be divisible by bottleneck_attn_heads. "
                f"Got C={self.bottleneck_channels}, heads={bottleneck_attn_heads}."
            )

        self.bottleneck_attn_norm1 = nn.LayerNorm(self.bottleneck_channels)
        self.bottleneck_attn = nn.MultiheadAttention(
            embed_dim=self.bottleneck_channels,
            num_heads=bottleneck_attn_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.bottleneck_attn_norm2 = nn.LayerNorm(self.bottleneck_channels)
        self.bottleneck_ff = nn.Sequential(
            nn.Linear(
                self.bottleneck_channels,
                self.bottleneck_channels * bottleneck_attn_ff_mult,
            ),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(
                self.bottleneck_channels * bottleneck_attn_ff_mult,
                self.bottleneck_channels,
            ),
            nn.Identity(),
        )

        self.station_merge_attn_norm = nn.LayerNorm(self.bottleneck_channels)
        self.station_merge_attn_score = nn.Linear(self.bottleneck_channels, 1)

        if self.use_distance_bottleneck_emb:
            self.dist_bottleneck_proj = nn.Linear(1, self.bottleneck_channels)
            nn.init.normal_(self.dist_bottleneck_proj.weight, std=1e-3)
            nn.init.zeros_(self.dist_bottleneck_proj.bias)
        else:
            self.dist_bottleneck_proj = None

        if self.use_distance_attn_bias:
            self.dist_attn_bias_proj = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.dist_attn_bias_proj.weight)
        else:
            self.dist_attn_bias_proj = None

        if self.use_distance_attn_bias or self.use_distance_bottleneck_emb:
            station_coords = list(get_station_coords(self.volcano_name).items())
            crater_coords = get_crater_coords(self.volcano_name)

            lat_mean = float(np.mean([coords[1] for _, coords in station_coords]))
            km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mean))
            km_per_deg_lat = 111.0

            coords_array = np.array(
                [
                    (
                        (lon - crater_coords[0]) * km_per_deg_lon,
                        (lat - crater_coords[1]) * km_per_deg_lat,
                    )
                    for _, (lon, lat) in station_coords
                ],
                dtype=np.float32,
            )
            dist_to_crater = np.linalg.norm(coords_array, axis=1).astype(np.float32)
            n_stations = int(dist_to_crater.shape[0])
            station_rank = np.empty(n_stations, dtype=np.float32)
            station_rank[np.argsort(dist_to_crater)] = np.arange(
                n_stations, dtype=np.float32
            )
            normalized = 1.0 - station_rank / n_stations + 1.0 / n_stations
            self.register_buffer(
                "station_dist",
                torch.from_numpy(normalized[:, None].astype(np.float32)),
                persistent=True,
            )
        else:
            self.station_dist = None

        for i in range(self.depth - 1):
            filters = int(2 ** (self.depth - 2 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def _apply_bottleneck_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Convert [N, C, T] -> [N, T, C] for batch_first attention.
        x_seq = x.transpose(1, 2)

        x_norm = self.bottleneck_attn_norm1(x_seq)
        x_attn, _ = self.bottleneck_attn(x_norm, x_norm, x_norm, need_weights=False)
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.bottleneck_ff(self.bottleneck_attn_norm2(x_seq))

        return x_seq.transpose(1, 2)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)

    @staticmethod
    def _station_max(x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, T]
        return x.max(dim=1).values

    def _station_attn_merge(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, T] -> merged [B, C, T]
        if x.shape[1] == 1:
            return x[:, 0, :, :]

        pooled = x.mean(dim=-1)
        pooled = self.station_merge_attn_norm(pooled)
        station_scores = self.station_merge_attn_score(pooled).squeeze(-1)
        station_weights = torch.softmax(station_scores, dim=1)
        return (x * station_weights[:, :, None, None]).sum(dim=1)

    def _apply_station_conv(
        self, x: torch.Tensor, conv: nn.Conv1d, bn: nn.BatchNorm1d
    ) -> torch.Tensor:
        # x: [B, S, C, T] -> apply Conv/BN station-wise with shared weights.
        bsz, n_stations, channels, t_len = x.shape
        y = x.reshape(bsz * n_stations, channels, t_len)
        y = self.activation(bn(conv(y)))
        y = self.feature_dropout(y)
        return y.reshape(bsz, n_stations, y.shape[1], y.shape[2])

    def _pad_shared_downsample(self, x: torch.Tensor, level: int) -> torch.Tensor:
        if level not in {1, 2, 3}:
            return x
        bsz, n_stations, channels, t_len = x.shape
        y = x.reshape(bsz * n_stations, channels, t_len)
        if level == 1:
            y = F.pad(y, (2, 3), "constant", 0)
        elif level == 2:
            y = F.pad(y, (1, 3), "constant", 0)
        elif level == 3:
            y = F.pad(y, (2, 3), "constant", 0)
        return y.reshape(bsz, n_stations, channels, y.shape[-1])

    def _forward_joint(self, x: torch.Tensor, logits: bool = True) -> torch.Tensor:
        x = self.activation(self.in_bn(self.inc(x)))
        x = self.feature_dropout(x)

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))
            x = self.feature_dropout(x)

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))
                x = self.feature_dropout(x)

        if self.bottleneck_attention:
            x = self._apply_bottleneck_attention(x)

        for (conv_up, bn1, conv_same, bn2), skip in zip(self.up_branch, skips[::-1]):
            x = self.activation(bn1(conv_up(x)))
            x = self.feature_dropout(x)
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))
            x = self.feature_dropout(x)

        x = self.final_dropout(x)
        x = self.out(x)
        if logits:
            return x
        return self.softmax(x)

    def _forward_shared(self, x: torch.Tensor, logits: bool = True) -> torch.Tensor:
        if x.ndim == 4 and x.shape[2] == 1:
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(
                "shared_station_encoder=True expects input shape [B, S, T] "
                f"or [B, S, 1, T]. Got shape: {tuple(x.shape)}"
            )
        if self.station_dist is not None and x.shape[1] != self.station_dist.shape[0]:
            raise ValueError(
                "Input station count does not match the configured volcano station "
                f"metadata. Got {x.shape[1]} stations, expected "
                f"{self.station_dist.shape[0]} for volcano {self.volcano_name}."
            )

        x = x[:, :, None, :]
        x = self._apply_station_conv(x, self.inc_shared, self.in_bn_shared)

        skips = []
        for level, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self._apply_station_conv(x, conv_same, bn1)

            if level in self.station_message_levels:
                x = self.station_message_blocks[str(level)](x)

            if level in self.station_attention_levels:
                dist_bias = None
                if self.use_distance_attn_bias:
                    station_dist = self.station_dist.squeeze(-1)
                    dist_diff = (
                        station_dist.unsqueeze(0) - station_dist.unsqueeze(1)
                    ).abs()
                    dist_bias = self.dist_attn_bias_proj(dist_diff.unsqueeze(-1)).squeeze(
                        -1
                    )
                x = self.station_attention_blocks[str(level)](x, dist_bias=dist_bias)

            if conv_down is not None:
                skips.append(self._station_max(x))
                x = self._pad_shared_downsample(x, level)
                x = self._apply_station_conv(x, conv_down, bn2)

        collapsed_before_bottleneck = False
        if self.bottleneck_attention:
            if self.pre_bottleneck_station_attn_merge:
                x = self._station_attn_merge(x)
                x = self._apply_bottleneck_attention(x)
                collapsed_before_bottleneck = True
            elif self.station_interaction == "none":
                # Shared-encoder baseline: fuse stations before temporal bottleneck attention.
                x = self._station_max(x)
                x = self._apply_bottleneck_attention(x)
                collapsed_before_bottleneck = True
            else:
                # Late interaction variants keep station-wise tensors until after interaction.
                bsz, n_stations, channels, t_len = x.shape
                x_flat = x.reshape(bsz * n_stations, channels, t_len)
                x_flat = self._apply_bottleneck_attention(x_flat)
                x = x_flat.reshape(bsz, n_stations, channels, t_len)

        if not collapsed_before_bottleneck:
            if self.use_distance_bottleneck_emb:
                dist_emb = self.dist_bottleneck_proj(self.station_dist)
                x = x + dist_emb[None, :, :, None]
            x = self._station_max(x)

        for (conv_up, bn1, conv_same, bn2), skip in zip(self.up_branch, skips[::-1]):
            x = self.activation(bn1(conv_up(x)))
            x = self.feature_dropout(x)
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))
            x = self.feature_dropout(x)

        x = self.final_dropout(x)
        x = self.out(x)
        if logits:
            return x
        return self.softmax(x)

    def forward(self, x, logits=True, **kwargs):
        if self.shared_station_encoder:
            return self._forward_shared(x, logits=logits)
        return self._forward_joint(x, logits=logits)

    def permute_stations(self, perm: torch.Tensor) -> None:
        """Reorder the station distance buffer to match a station permutation."""
        if self.station_dist is not None:
            self.station_dist = self.station_dist[perm]
