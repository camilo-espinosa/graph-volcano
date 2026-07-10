import torch
import torch.nn as nn
import torch.nn.functional as F


class PairConvBlock(nn.Module):
    """Permutation-equivariant station message passing block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        aggregation: str = "sum",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if aggregation not in {"sum", "attention"}:
            raise ValueError(
                "pairconv_aggregation must be 'sum' or 'attention'. "
                f"Got: {aggregation}."
            )
        self.aggregation = aggregation

        self.message_conv = nn.Conv1d(
            2 * channels, channels, kernel_size, padding="same", bias=False
        )
        self.message_bn = nn.BatchNorm1d(channels, eps=1e-3)
        self.message_dropout = nn.Dropout(dropout_p)

        if self.aggregation == "attention":
            self.score_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
            self.score_fc = nn.Linear(channels, 1)

    def _masked_softmax(
        self, scores: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        # scores: [B, S, S], valid_mask: [B, S, S]
        large_neg = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~valid_mask, large_neg)
        max_scores = masked_scores.max(dim=2, keepdim=True).values
        max_scores = torch.where(
            torch.isfinite(max_scores), max_scores, torch.zeros_like(max_scores)
        )
        exp_scores = torch.exp(masked_scores - max_scores) * valid_mask.float()
        denom = exp_scores.sum(dim=2, keepdim=True).clamp_min(1e-8)
        return exp_scores / denom

    def forward(self, x: torch.Tensor, station_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, T], station_mask: [B, S]
        bsz, n_stations, channels, t_len = x.shape

        aggregated = x.new_zeros((bsz, n_stations, channels, t_len))

        for i in range(n_stations):
            x_i = x[:, i, :, :]
            src_valid = station_mask.clone()
            src_valid[:, i] = False

            if self.aggregation == "sum":
                agg_i = x.new_zeros((bsz, channels, t_len))
                for j in range(n_stations):
                    if i == j:
                        continue
                    x_j = x[:, j, :, :]
                    msg_in = torch.cat([x_i, x_j], dim=1)
                    msg = self.message_conv(msg_in)
                    msg = self.message_bn(msg)
                    msg = torch.relu(msg)
                    msg = self.message_dropout(msg)

                    valid_j = station_mask[:, j].float()[:, None, None]
                    agg_i = agg_i + msg * valid_j
            else:
                # Pass 1: compute attention logits only (small tensor [B, S]).
                scores_i = x.new_zeros((bsz, n_stations))
                for j in range(n_stations):
                    if i == j:
                        continue
                    if not station_mask[:, j].any():
                        continue

                    x_j = x[:, j, :, :]
                    msg_in = torch.cat([x_i, x_j], dim=1)
                    msg = self.message_conv(msg_in)
                    msg = self.message_bn(msg)
                    msg = torch.relu(msg)
                    msg = self.message_dropout(msg)

                    score_feat = torch.relu(self.score_conv(msg))
                    score_feat = score_feat.mean(dim=-1)
                    scores_i[:, j] = self.score_fc(score_feat).squeeze(-1)

                weights_i = self._masked_softmax(
                    scores_i.unsqueeze(1), src_valid.unsqueeze(1)
                )
                weights_i = weights_i.squeeze(1)

                # Pass 2: recompute messages and accumulate weighted sum without stacking.
                agg_i = x.new_zeros((bsz, channels, t_len))
                for j in range(n_stations):
                    if i == j:
                        continue
                    if not station_mask[:, j].any():
                        continue

                    x_j = x[:, j, :, :]
                    msg_in = torch.cat([x_i, x_j], dim=1)
                    msg = self.message_conv(msg_in)
                    msg = self.message_bn(msg)
                    msg = torch.relu(msg)
                    msg = self.message_dropout(msg)

                    w_j = weights_i[:, j][:, None, None]
                    agg_i = agg_i + w_j * msg

            aggregated[:, i, :, :] = agg_i

        dest_valid = station_mask[:, :, None, None].float()
        return x + aggregated * dest_valid


class StationAttentionBlock(nn.Module):
    """Optional global attention over stations (not over time)."""

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        dropout: float = 0.2,
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
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ff_mult, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, station_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, T], station_mask: [B, S]
        pooled = x.mean(dim=-1)

        key_padding_mask = ~station_mask
        has_no_valid = ~station_mask.any(dim=1)
        if torch.any(has_no_valid):
            # Keep attention numerically stable for degenerate all-missing samples.
            key_padding_mask = key_padding_mask.clone()
            pooled = pooled.clone()
            key_padding_mask[has_no_valid, 0] = False
            pooled[has_no_valid, 0, :] = 0.0

        pooled_norm = self.norm1(pooled)
        attn_out, _ = self.attn(
            pooled_norm,
            pooled_norm,
            pooled_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = pooled + attn_out
        pooled = pooled + self.ff(self.norm2(pooled))

        station_update = pooled[:, :, :, None] * station_mask[:, :, None, None].float()
        return x + station_update


class PhaseNetPermutationInvariant(nn.Module):

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
        feature_dropout=0.2,
        bottleneck_attention=False,
        shared_station_encoder=False,
        pairconv_levels=None,
        pairconv_aggregation="sum",
        station_attention_levels=None,
        bottleneck_attn_heads=4,
        bottleneck_attn_dropout=0.2,
        bottleneck_attn_ff_mult=2,
        station_attn_heads=4,
        station_attn_dropout=0.2,
        station_attn_ff_mult=2,
        **kwargs,
    ):

        super().__init__()

        if out_channels is not None:
            classes = out_channels

        if pairconv_levels is None:
            pairconv_levels = []
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
        self.pairconv_levels = sorted(set(int(level) for level in pairconv_levels))
        self.pairconv_aggregation = pairconv_aggregation
        self.station_attention_levels = sorted(
            set(int(level) for level in station_attention_levels)
        )

        if feature_dropout < 0.0 or feature_dropout >= 1.0:
            raise ValueError(
                f"feature_dropout must be in [0, 1). Got: {feature_dropout}."
            )

        valid_pair_levels = set(range(self.depth - 1))
        invalid_pair_levels = [
            level for level in self.pairconv_levels if level not in valid_pair_levels
        ]
        if invalid_pair_levels:
            raise ValueError(
                f"Invalid pairconv_levels={invalid_pair_levels}. "
                f"Allowed levels for depth={self.depth}: {sorted(valid_pair_levels)}."
            )

        if not self.shared_station_encoder and self.pairconv_levels:
            raise ValueError(
                "pairconv_levels require shared_station_encoder=True, because "
                "PairConv is defined on station embeddings [B, S, C, T]."
            )

        # Requested scope for this first implementation: only [] or [0].
        if self.station_attention_levels not in ([], [0]):
            raise NotImplementedError(
                "This implementation supports station_attention_levels=[] or [0] only."
            )

        if self.station_attention_levels and 0 not in self.pairconv_levels:
            raise ValueError(
                "station_attention_levels=[0] requires PairConv at level 0 "
                "(pairconv_levels must include 0)."
            )

        self.feature_dropout_p = float(feature_dropout)
        self.activation = torch.relu
        self.feature_dropout = nn.Dropout(self.feature_dropout_p)
        self.final_dropout = nn.Dropout(self.feature_dropout_p)

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

        self.pairconv_blocks = nn.ModuleDict()
        for level in self.pairconv_levels:
            channels = int(2**level * self.filters_root)
            self.pairconv_blocks[str(level)] = PairConvBlock(
                channels=channels,
                kernel_size=self.kernel_size,
                aggregation=self.pairconv_aggregation,
                dropout_p=self.feature_dropout_p,
            )

        self.station_attention_blocks = nn.ModuleDict()
        for level in self.station_attention_levels:
            channels = int(2**level * self.filters_root)
            self.station_attention_blocks[str(level)] = StationAttentionBlock(
                channels=channels,
                heads=station_attn_heads,
                dropout=station_attn_dropout,
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
            dropout=bottleneck_attn_dropout,
            batch_first=True,
        )
        self.bottleneck_attn_norm2 = nn.LayerNorm(self.bottleneck_channels)
        self.bottleneck_ff = nn.Sequential(
            nn.Linear(
                self.bottleneck_channels,
                self.bottleneck_channels * bottleneck_attn_ff_mult,
            ),
            nn.GELU(),
            nn.Dropout(bottleneck_attn_dropout),
            nn.Linear(
                self.bottleneck_channels * bottleneck_attn_ff_mult,
                self.bottleneck_channels,
            ),
            nn.Dropout(bottleneck_attn_dropout),
        )

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
    def _masked_station_mean(
        x: torch.Tensor, station_mask: torch.Tensor
    ) -> torch.Tensor:
        # x: [B, S, C, T], station_mask: [B, S]
        weights = station_mask[:, :, None, None].float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

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
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected S=in_channels={self.in_channels} stations, got S={x.shape[1]}."
            )

        station_mask = x.abs().sum(dim=-1) > 0

        x = x[:, :, None, :]
        x = self._apply_station_conv(x, self.inc_shared, self.in_bn_shared)

        skips = []
        for level, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self._apply_station_conv(x, conv_same, bn1)

            if level in self.pairconv_levels:
                x = self.pairconv_blocks[str(level)](x, station_mask)

            if level in self.station_attention_levels:
                x = self.station_attention_blocks[str(level)](x, station_mask)

            if conv_down is not None:
                skips.append(self._masked_station_mean(x, station_mask))
                x = self._pad_shared_downsample(x, level)
                x = self._apply_station_conv(x, conv_down, bn2)

        if self.bottleneck_attention:
            bsz, n_stations, channels, t_len = x.shape
            x_flat = x.reshape(bsz * n_stations, channels, t_len)
            x_flat = self._apply_bottleneck_attention(x_flat)
            x = x_flat.reshape(bsz, n_stations, channels, t_len)

        x = self._masked_station_mean(x, station_mask)

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
