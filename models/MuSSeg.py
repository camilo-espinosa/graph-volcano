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
            attn_mask=dist_bias if dist_bias is not None else None,
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
        volcano_name: str | None = None,
        use_distance_attn_bias: bool = False,
        use_distance_bottleneck_emb: bool = False,
        use_station_weighted_skips: bool = False,
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
        self.shared_station_encoder = True
        self.station_interaction = str(station_interaction)
        self.station_message_levels = sorted(
            set(int(level) for level in station_message_levels)
        )
        self.station_message_aggregation = station_message_aggregation
        self.station_message_ratio = float(station_message_ratio)
        self.station_attention_levels = sorted(
            set(int(level) for level in station_attention_levels)
        )
        self.pre_bottleneck_station_attn_merge = bool(pre_bottleneck_station_attn_merge)
        self.use_distance_attn_bias = bool(use_distance_attn_bias)
        self.use_distance_bottleneck_emb = bool(use_distance_bottleneck_emb)
        self.use_station_weighted_skips = bool(use_station_weighted_skips)
        distance_features_enabled = (
            self.use_distance_attn_bias or self.use_distance_bottleneck_emb
        )
        self.volcano_name = volcano_name

        if distance_features_enabled and self.volcano_name is None:
            raise ValueError(
                "volcano_name is required when use_distance_attn_bias or "
                "use_distance_bottleneck_emb is True"
            )
        if self.use_distance_attn_bias and self.station_interaction != "late_attention":
            raise ValueError(
                "use_distance_attn_bias requires station_interaction='late_attention'"
            )

        if distance_features_enabled:
            station_items = list(get_station_coords(self.volcano_name).items())
            crater_lon, crater_lat = get_crater_coords(self.volcano_name)
            lat_mean = float(np.mean([lat for _, (_, lat) in station_items]))
            km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mean))
            km_per_deg_lat = 111.0
            raw_dists = np.array(
                [
                    np.sqrt(
                        ((lon - crater_lon) * km_per_deg_lon) ** 2
                        + ((lat - crater_lat) * km_per_deg_lat) ** 2
                    )
                    for _, (lon, lat) in station_items
                ],
                dtype=np.float32,
            )
            n_stations = int(raw_dists.shape[0])
            if n_stations == 1:
                normalized = np.ones((1,), dtype=np.float32)
            else:
                min_dist = float(np.min(raw_dists))
                max_dist = float(np.max(raw_dists))
                dist_span = max_dist - min_dist
                if dist_span <= 0.0:
                    raise ValueError(
                        "station distances must span a positive range when more than one station is present."
                    )
                min_score = 1.0 / float(n_stations)
                normalized = 1.0 - (1.0 - min_score) * (
                    (raw_dists - min_dist) / dist_span
                )
                normalized = normalized.astype(np.float32)
            if not np.all(np.isfinite(normalized)):
                raise ValueError(
                    "station distance scores must be finite for distance-aware features."
                )
            self.register_buffer(
                "station_dist",
                torch.from_numpy(normalized[:, None]),
            )
            closest_idx = int(np.argmin(raw_dists))
            farthest_idx = int(np.argmax(raw_dists))
            closest_name = str(station_items[closest_idx][0])
            farthest_name = str(station_items[farthest_idx][0])
            print(
                "[MuSSeg][station_dist] "
                f"volcano={self.volcano_name} "
                f"closest={closest_name} "
                f"km={float(raw_dists[closest_idx]):.3f} "
                f"score={float(normalized[closest_idx]):.3f} | "
                f"farthest={farthest_name} "
                f"km={float(raw_dists[farthest_idx]):.3f} "
                f"score={float(normalized[farthest_idx]):.3f}"
            )
            self.distance_merge_gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.station_dist = None

        if self.use_station_weighted_skips:
            self.skip_merge_alpha = nn.Parameter(torch.tensor(0.5))

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

        if self.use_station_weighted_skips:
            self.station_level_descriptor_dim = min(64, self.bottleneck_channels)
            level_attn_heads = 4
            while (
                self.station_level_descriptor_dim % level_attn_heads != 0
                and level_attn_heads > 1
            ):
                level_attn_heads //= 2
            if self.station_level_descriptor_dim % level_attn_heads != 0:
                level_attn_heads = 1

            self.station_level_descriptor_proj = nn.ModuleDict()
            for level in range(self.depth):
                channels = int(2**level * self.filters_root)
                self.station_level_descriptor_proj[str(level)] = nn.Linear(
                    channels,
                    self.station_level_descriptor_dim,
                )

            self.station_level_attn_norm1 = nn.LayerNorm(
                self.station_level_descriptor_dim
            )
            self.station_level_attn = nn.MultiheadAttention(
                embed_dim=self.station_level_descriptor_dim,
                num_heads=level_attn_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.station_level_attn_norm2 = nn.LayerNorm(
                self.station_level_descriptor_dim
            )
            self.station_level_attn_ff = nn.Sequential(
                nn.Linear(
                    self.station_level_descriptor_dim,
                    2 * self.station_level_descriptor_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    2 * self.station_level_descriptor_dim,
                    self.station_level_descriptor_dim,
                ),
            )
            self.station_level_score = nn.Linear(self.station_level_descriptor_dim, 1)

        if self.use_distance_bottleneck_emb:
            self.dist_bottleneck_proj = nn.Linear(1, self.bottleneck_channels)
            nn.init.normal_(self.dist_bottleneck_proj.weight, std=1e-3)
            nn.init.zeros_(self.dist_bottleneck_proj.bias)

        if self.use_distance_attn_bias:
            self.dist_attn_bias_proj = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.dist_attn_bias_proj.weight)

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

    def _station_distance_prior_logits(
        self,
        n_stations: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        station_dist = self._station_dist_for_count(
            n_stations,
            device=device,
            dtype=dtype,
        ).squeeze(-1)
        total = station_dist.sum()
        if float(total) <= 0.0:
            raise ValueError(
                "distance-based station merge requires positive station distance mass."
            )
        station_prior = station_dist / total
        return torch.log(station_prior.clamp_min(1e-6))

    def _merge_stations(
        self,
        x: torch.Tensor,
        learned_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, S, C, T] -> merged [B, C, T]
        if x.shape[1] == 1:
            return x[:, 0, :, :]

        if learned_logits is not None and learned_logits.shape != x.shape[:2]:
            raise ValueError(
                "learned station logits must have shape [B, S]. "
                f"Got {tuple(learned_logits.shape)} for station tensor "
                f"{tuple(x.shape)}."
            )

        if self.station_dist is None:
            if learned_logits is None:
                return self._station_max(x)
            station_weights = torch.softmax(learned_logits, dim=1)
            return (x * station_weights[:, :, None, None]).sum(dim=1)

        dist_logits = self._station_distance_prior_logits(
            int(x.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )
        gamma = self.distance_merge_gamma.to(device=x.device, dtype=x.dtype)
        dist_logits = gamma * dist_logits[None, :]
        if learned_logits is None:
            station_weights = torch.softmax(dist_logits, dim=1)
        else:
            station_weights = torch.softmax(learned_logits + dist_logits, dim=1)
        return (x * station_weights[:, :, None, None]).sum(dim=1)

    def _station_attn_merge(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, T] -> merged [B, C, T]
        pooled = x.mean(dim=-1)
        pooled = self.station_merge_attn_norm(pooled)
        station_logits = self.station_merge_attn_score(pooled).squeeze(-1)
        return self._merge_stations(x, learned_logits=station_logits)

    def _weighted_skip_station_logits_from_levels(
        self, descriptors_by_level: list[torch.Tensor]
    ) -> torch.Tensor:
        if len(descriptors_by_level) == 0:
            raise ValueError(
                "weighted skip station logits require at least one level descriptor."
            )

        stacked = torch.stack(descriptors_by_level, dim=2)
        bsz, n_stations, n_levels, d_model = stacked.shape

        level_tokens = stacked.reshape(bsz * n_stations, n_levels, d_model)
        level_tokens_norm = self.station_level_attn_norm1(level_tokens)
        level_attn_out, _ = self.station_level_attn(
            level_tokens_norm,
            level_tokens_norm,
            level_tokens_norm,
            need_weights=False,
        )
        level_tokens = level_tokens + level_attn_out
        level_tokens = level_tokens + self.station_level_attn_ff(
            self.station_level_attn_norm2(level_tokens)
        )

        station_features = level_tokens.mean(dim=1).reshape(bsz, n_stations, d_model)
        return self.station_level_score(station_features).squeeze(-1)

    def permute_stations(self, perm: torch.Tensor) -> None:
        if self.station_dist is not None:
            self.station_dist = self.station_dist[perm]

    def _station_dist_for_count(
        self,
        n_stations: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.station_dist is None:
            raise ValueError(
                "station_dist is required when distance-aware station features are enabled."
            )

        dist = self.station_dist
        current_n = int(dist.shape[0])
        target_n = int(n_stations)
        if current_n > target_n:
            dist = dist[:target_n]
        elif current_n < target_n:
            pad_n = target_n - current_n
            # Extra channels are padded traces and should carry no station information.
            pad_value = dist.new_tensor(0.0)
            pad = pad_value.expand(pad_n, 1)
            dist = torch.cat([dist, pad], dim=0)

        return dist.to(device=device, dtype=dtype)

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

    def forward(self, x: torch.Tensor, logits: bool = True) -> torch.Tensor:
        if x.ndim == 4 and x.shape[2] == 1:
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(
                "shared_station_encoder=True expects input shape [B, S, T] "
                f"or [B, S, 1, T]. Got shape: {tuple(x.shape)}"
            )

        x = x[:, :, None, :]
        x = self._apply_station_conv(x, self.inc_shared, self.in_bn_shared)

        skips = []
        station_level_descriptors = []
        store_station_skips = (
            self.use_station_weighted_skips or self.station_dist is not None
        )
        for level, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self._apply_station_conv(x, conv_same, bn1)

            if level in self.station_message_levels:
                x = self.station_message_blocks[str(level)](x)

            if level in self.station_attention_levels:
                if self.use_distance_attn_bias:
                    station_dist = self._station_dist_for_count(
                        int(x.shape[1]),
                        device=x.device,
                        dtype=x.dtype,
                    ).squeeze(-1)
                    dist_diff = (
                        station_dist.unsqueeze(0) - station_dist.unsqueeze(1)
                    ).abs()
                    dist_bias = self.dist_attn_bias_proj(
                        dist_diff.unsqueeze(-1)
                    ).squeeze(-1)
                    x = self.station_attention_blocks[str(level)](
                        x, dist_bias=dist_bias
                    )
                else:
                    x = self.station_attention_blocks[str(level)](x)

            if self.use_station_weighted_skips:
                descriptor = x.mean(dim=-1)
                descriptor = self.station_level_descriptor_proj[str(level)](descriptor)
                station_level_descriptors.append(descriptor)

            if conv_down is not None:
                skips.append(x if store_station_skips else self._station_max(x))
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
                x = self._merge_stations(x)
                x = self._apply_bottleneck_attention(x)
                collapsed_before_bottleneck = True
            else:
                # Late interaction variants keep station-wise tensors until after interaction.
                bsz, n_stations, channels, t_len = x.shape
                x_flat = x.reshape(bsz * n_stations, channels, t_len)
                x_flat = self._apply_bottleneck_attention(x_flat)
                x = x_flat.reshape(bsz, n_stations, channels, t_len)
                if self.use_distance_bottleneck_emb:
                    station_dist = self._station_dist_for_count(
                        n_stations,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    dist_emb = self.dist_bottleneck_proj(station_dist)
                    x = x + dist_emb[None, :, :, None]

        skip_logits = None
        if self.use_station_weighted_skips:
            skip_logits = self._weighted_skip_station_logits_from_levels(
                station_level_descriptors
            )

        if not collapsed_before_bottleneck:
            if self.use_station_weighted_skips:
                if skip_logits is None:
                    raise ValueError(
                        "skip logits are required for weighted station bottleneck merging."
                    )
                x = self._merge_stations(x, learned_logits=skip_logits)
            else:
                x = self._merge_stations(x)

        for (conv_up, bn1, conv_same, bn2), skip in zip(self.up_branch, skips[::-1]):
            x = self.activation(bn1(conv_up(x)))
            x = self.feature_dropout(x)
            x = x[:, :, 1:-2]

            if skip.ndim == 4:
                if self.use_station_weighted_skips:
                    if skip_logits is None:
                        raise ValueError(
                            "skip logits are required for weighted station skip merging."
                        )
                    weighted = self._merge_stations(skip, learned_logits=skip_logits)
                    base = self._merge_stations(skip)
                    alpha = self.skip_merge_alpha.clamp(0.0, 1.0)
                    skip = alpha * weighted + (1.0 - alpha) * base
                else:
                    skip = self._merge_stations(skip)

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))
            x = self.feature_dropout(x)

        x = self.final_dropout(x)
        x = self.out(x)
        if logits:
            return x
        return self.softmax(x)
