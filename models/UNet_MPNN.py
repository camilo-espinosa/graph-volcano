"""
UNet_MPNN — Edge-conditioned Message-Passing Neural Network backbone for
multistation seismic segmentation.

This model reuses the 1-D UNet skeleton, virtual-node readout, and bottleneck
temporal self-attention from UNet_GraphSAGE, but replaces the GraphSAGE
backbone with a single edge-conditioned MPNN. Edges carry spatial relational
attributes (delta position, distance, and optional offline cross-correlation
features), so the graph operation explicitly consumes pairwise station geometry.

Graph operation pattern (per time-bin):
    station embeddings -> mean-init virtual node -> edge-conditioned MPNN -> readout

Input:  [B, S, T]  where B=batch, S=stations (8), T=temporal samples (8192)
Output: [B, 6, T]  where 6 = number of classes (BG, VT, LP, TR, AV, IC)

Design notes
------------
- MPNN-only: no graphsage/mlp backend switch.
- Virtual node initialized by a plain mean over stations; message passing then
  updates it directly (no learned pooling module).
- Edge features are *architectural*: with `edge_feature_mode="none"` the message
  MLP input loses its edge slice entirely (a genuine ablation, not zeroed input).
- Optional `delta_pos_xcorr` dynamic edge features are computed OFFLINE and
  passed via `forward(edge_attr_dynamic=...)`; the model never runs FFTs.
- A single optional `graph_norm` (batchnorm) is available, default OFF.

Constructor highlights
----------------------
| Parameter | Controls |
|---|---|
| `graph_levels` | Encoder depths that run the MPNN (default [] = bottleneck only) |
| `use_skip_graph` / `skip_graph_levels` | Whether skip connections run the MPNN |
| `graph_topology` | `"fully_connected"` or `"star"` |
| `edge_feature_mode` | `"delta_pos"`, `"delta_pos_dist"`, `"none"`, `"delta_pos_xcorr"` |
| `node_feature_mode` | `"geometry"` or `"none"` (no spatial node info) |
| `use_rsam_node_feat` | Append precomputed per-station RSAM as a node feature |
| `mpnn_hidden_dim` / `mpnn_aggr` / `mpnn_layers` | MPNN edge-MLP width / aggregation / depth |
| `pairwise_conv_levels` / `pairwise_conv_kernel` / `pairwise_conv_aggr` | Optional PairwiseConv2d on selected encoder depths |
| `use_bottleneck_attention` | Toggle the bottleneck MHSA block |
| `graph_norm` | `"none"` or `"batchnorm"` post-op normalization |
"""

from collections import OrderedDict
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeMPNNLayer(MessagePassing):
    """
    One edge-conditioned message-passing step.

    message(x_i, x_j, edge_attr): MLP over concat([x_i, x_j, edge_attr]).
    update(aggr_out, x):          MLP over concat([x, aggr_out]) + residual.

    When `edge_dim == 0` (the `edge_feature_mode="none"` ablation) the message
    MLP input drops the edge slice entirely, so "no edge features" is a genuine
    architectural change rather than a silently zeroed input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        msg_in = 2 * in_channels + edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )
        self.res = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.edge_dim > 0 and edge_attr is not None:
            m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            m = torch.cat([x_i, x_j], dim=-1)
        return self.msg_mlp(m)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        upd = self.update_mlp(torch.cat([x, aggr_out], dim=-1))
        return upd + self.res(x)


class EdgeMPNN(nn.Module):
    """Stack of `num_layers` EdgeMPNNLayer modules (separate weights)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        aggr: str = "mean",
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else out_channels
            layers.append(
                EdgeMPNNLayer(
                    in_channels=layer_in,
                    out_channels=out_channels,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    aggr=aggr,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class PairwiseConv2d(nn.Module):
    """
    Cross-station message passing using a shared Conv2d over directed station pairs.

    For every ordered pair (i→j), stacks their feature sequences as two rows:
        row 0 = source station i   [C, T]
        row 1 = destination station j  [C, T]
    and applies a shared Conv2d kernel of shape [2, K] — spanning both stations
    (height) and K time samples (width).

    Because the kernel is shared across all 56 directed pairs, and every station
    appears as both source and destination, the kernel is forced to learn general
    directional waveform relationships (moveout, polarity, coherence) rather than
    station-specific or geometry-specific patterns. This makes it transferable
    across volcanoes with different network geometries.

    The pair outputs are aggregated back to per-station features by mean-pooling
    over all pairs where each station is the destination (7 incoming pairs per
    station for an 8-station network).

    No edge geometry is used here — geometry enters only at the bottleneck
    EdgeMPNN via edge_attr (Δpos).

    Args:
        in_channels  : C, feature channels per station per time step.
        out_channels : feature channels of the conv output and final station update.
        kernel_size  : temporal receptive field K in samples (must be odd).
                       Rule of thumb: K / sample_rate = max_lag_seconds.
                           K=9  at 100 Hz → 90 ms   (tight moveout)
                           K=51 at 100 Hz → 510 ms  (full moveout range)
        n_stations   : number of station nodes S (default 8 → 56 directed pairs).
        aggr         : aggregation over incoming pair messages: 'mean' or 'sum'.

    Input:
        x_flat : [B*S, C, T]   standard flat station layout used throughout UNet_MPNN.
        B      : batch size.
        S      : number of stations (must equal n_stations).

    Output:
        [B*S, C, T]   residual-ready output; add to x_flat in the caller.

    Note:
        out_channels is set equal to in_channels so the output can be added
        back as a residual without a projection. If you want out_channels != in_channels,
        add a 1x1 conv projection on x_flat before the residual add in the caller.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        n_stations: int = 8,
        aggr: str = "mean",
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for symmetric time padding. Got {kernel_size}."
            )
        if aggr not in {"mean", "sum"}:
            raise ValueError(f"aggr must be 'mean' or 'sum'. Got '{aggr}'.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_stations = n_stations
        self.aggr = aggr

        # --- directed pairs (i→j), i ≠ j ---
        # 56 pairs for 8 stations.
        # row 0 = source i, row 1 = destination j.
        # The shared kernel learns directional relationships:
        #   "what does source look like relative to destination"
        # without overfitting to specific station geometry, because every station
        # appears as source and destination across the full pair set.
        pairs = [(i, j) for i in range(n_stations) for j in range(n_stations) if i != j]
        self.n_pairs = len(pairs)  # S * (S-1) = 56

        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)  # [n_pairs]
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)  # [n_pairs]
        self.register_buffer("src_idx", src)
        self.register_buffer("dst_idx", dst)

        # For each destination station d, which pair indices have d as destination?
        # Each station receives from exactly (n_stations - 1) = 7 pairs.
        # Shape: [S, S-1]  — used to scatter pair outputs back to stations.
        incoming = [
            [idx for idx, (_, j) in enumerate(pairs) if j == d]
            for d in range(n_stations)
        ]
        incoming_tensor = torch.tensor(incoming, dtype=torch.long)  # [S, S-1]
        self.register_buffer("incoming_idx", incoming_tensor)
        self.n_incoming = n_stations - 1  # = 7

        # --- shared Conv2d ---
        # Input per pair:  [C, 2, T]
        #   channel dim C carries the feature maps
        #   height 2 = [source row, destination row]
        #   width  T = time samples
        # Kernel: [out_channels, in_channels, 2, K]
        #   height kernel = 2 → collapses the station dimension fully
        #   width  kernel = K → local temporal receptive field
        # Output per pair: [out_channels, 1, T] → squeezed to [out_channels, T]
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2, kernel_size),
            padding=(
                0,
                kernel_size // 2,
            ),  # no padding on station dim; symmetric on time
            bias=False,
        )
        # BatchNorm on out_channels; applied after reshaping to [B*n_pairs, out_channels, T]
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # --- per-timestep update MLP ---
        # Takes [x_i (in_channels) || aggregated_message (out_channels)]
        # and projects back to out_channels.
        # Applied as a 1x1 Conv1d so it operates independently at every time step
        # without reshaping to [B*S*T, ...].
        self.update_mlp = nn.Sequential(
            nn.Conv1d(
                in_channels + out_channels, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection for residual if channel widths differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x_flat: torch.Tensor, B: int, S: int) -> torch.Tensor:
        """
        Args:
            x_flat : [B*S, C, T]
            B      : batch size
            S      : number of stations (must equal self.n_stations)

        Returns:
            out : [B*S, out_channels, T]   — add to x_flat as residual in caller
        """
        assert (
            S == self.n_stations
        ), f"PairwiseConv2d expected S={self.n_stations} stations, got S={S}."

        C, T = x_flat.shape[1], x_flat.shape[2]

        # ── 1. reshape to [B, S, C, T] for station indexing ──────────────────
        x = x_flat.reshape(B, S, C, T)

        # ── 2. gather directed pairs ──────────────────────────────────────────
        x_src = x[:, self.src_idx]  # [B, n_pairs, C, T]  — source rows
        x_dst = x[:, self.dst_idx]  # [B, n_pairs, C, T]  — destination rows

        # Stack along a new height=2 dimension:
        #   dim 3 → [source, destination] rows
        # Result: [B, n_pairs, C, 2, T]
        x_pairs = torch.stack([x_src, x_dst], dim=3)

        # Merge batch and pair dims for Conv2d: [B*n_pairs, C, 2, T]
        x_pairs = x_pairs.reshape(B * self.n_pairs, C, 2, T)

        # ── 3. shared Conv2d ──────────────────────────────────────────────────
        # kernel [out_channels, C, 2, K] slides over height=2 and time=K
        # output: [B*n_pairs, out_channels, 1, T]  (height collapses to 1)
        pair_out = self.conv(x_pairs)  # [B*n_pairs, out_channels, 1, T]
        pair_out = pair_out.squeeze(2)  # [B*n_pairs, out_channels, T]
        pair_out = self.act(self.norm(pair_out))  # normalise + activate

        # ── 4. scatter pair outputs back to destination stations ──────────────
        # Reshape to [B, n_pairs, out_channels, T]
        pair_out = pair_out.reshape(B, self.n_pairs, self.out_channels, T)

        # incoming_idx: [S, S-1] — for each station, indices of its 7 incoming pairs
        # Index into pair_out: [B, S, S-1, out_channels, T]
        incoming_feats = pair_out[
            :, self.incoming_idx
        ]  # [B, S, n_incoming, out_channels, T]

        if self.aggr == "mean":
            aggregated = incoming_feats.mean(dim=2)  # [B, S, out_channels, T]
        else:
            aggregated = incoming_feats.sum(dim=2)  # [B, S, out_channels, T]

        # ── 5. update MLP ─────────────────────────────────────────────────────
        # Flatten back to [B*S, out_channels, T] for Conv1d (1x1 = per-timestep MLP)
        aggregated = aggregated.reshape(B * S, self.out_channels, T)

        # Concatenate station's own features with incoming aggregate along channel dim
        update_in = torch.cat([x_flat, aggregated], dim=1)  # [B*S, C+out_channels, T]
        updated = self.update_mlp(update_in)  # [B*S, out_channels, T]

        # ── 6. residual add ───────────────────────────────────────────────────
        return self.residual_proj(x_flat) + updated  # [B*S, out_channels, T]


class UNet_MPNN(nn.Module):
    """
    1-D UNet with an edge-conditioned MPNN at the bottleneck (always) and at
    optionally selected encoder/skip levels, plus bottleneck temporal
    self-attention.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        init_features: int = 16,
        depth: int = 5,
        n_stations: int = 8,
        station_coords: Optional[dict] = None,
        crater_coords: Optional[tuple] = None,
        graph_levels: Optional[List[int]] = None,
        use_skip_graph: bool = False,
        skip_graph_levels: Optional[List[int]] = None,
        graph_topology: Literal["fully_connected", "star"] = "fully_connected",
        edge_feature_mode: Literal[
            "delta_pos", "delta_pos_dist", "none", "delta_pos_xcorr"
        ] = "delta_pos",
        node_feature_mode: Literal["geometry", "none"] = "geometry",
        use_rsam_node_feat: bool = False,
        mpnn_hidden_dim: Optional[int] = None,
        mpnn_aggr: Literal["mean", "add", "max"] = "mean",
        mpnn_layers: int = 2,
        pairwise_conv_levels: Optional[List[int]] = None,
        pairwise_conv_kernel: int = 9,
        pairwise_conv_aggr: Literal["mean", "sum"] = "mean",
        xcorr_feat_dim: int = 2,
        use_bottleneck_attention: bool = True,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_dropout: float = 0.0,
        bottleneck_attn_ff_mult: int = 2,
        graph_norm: Literal["none", "batchnorm"] = "none",
        volcano_geom_nodes: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.n_stations = n_stations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.depth = depth
        self.n_nodes = n_stations + 1  # +1 virtual network node
        self.geom_feat_channels = 3  # x_km, y_km, dist_to_crater
        self.verbose = bool(verbose)

        # ----------------------------- validation -----------------------------
        if graph_topology not in {"fully_connected", "star"}:
            raise ValueError(
                "graph_topology must be one of {'fully_connected', 'star'}. "
                f"Got: {graph_topology}"
            )
        if edge_feature_mode not in {
            "delta_pos",
            "delta_pos_dist",
            "none",
            "delta_pos_xcorr",
        }:
            raise ValueError(
                "edge_feature_mode must be one of {'delta_pos', 'delta_pos_dist', "
                f"'none', 'delta_pos_xcorr'}}. Got: {edge_feature_mode}"
            )
        if node_feature_mode not in {"geometry", "none"}:
            raise ValueError(
                "node_feature_mode must be one of {'geometry', 'none'}. "
                f"Got: {node_feature_mode}"
            )
        if mpnn_aggr not in {"mean", "add", "max"}:
            raise ValueError(
                f"mpnn_aggr must be one of {{'mean', 'add', 'max'}}. Got: {mpnn_aggr}"
            )
        if graph_norm not in {"none", "batchnorm"}:
            raise ValueError(
                f"graph_norm must be one of {{'none', 'batchnorm'}}. Got: {graph_norm}"
            )
        if mpnn_layers < 1:
            raise ValueError(f"mpnn_layers must be >= 1. Got: {mpnn_layers}")
        if pairwise_conv_kernel % 2 == 0:
            raise ValueError(
                "pairwise_conv_kernel must be odd for symmetric time padding. "
                f"Got: {pairwise_conv_kernel}"
            )
        if pairwise_conv_aggr not in {"mean", "sum"}:
            raise ValueError(
                "pairwise_conv_aggr must be one of {'mean', 'sum'}. "
                f"Got: {pairwise_conv_aggr}"
            )
        if edge_feature_mode == "delta_pos_xcorr" and xcorr_feat_dim < 1:
            raise ValueError(
                "xcorr_feat_dim must be >= 1 when edge_feature_mode='delta_pos_xcorr'."
            )

        self.graph_topology = graph_topology
        self.edge_feature_mode = edge_feature_mode
        self.node_feature_mode = node_feature_mode
        self.use_rsam_node_feat = bool(use_rsam_node_feat)
        self.mpnn_aggr = mpnn_aggr
        self.mpnn_layers = int(mpnn_layers)
        self.bottleneck_mpnn_layers = max(3, self.mpnn_layers)
        self.pairwise_conv_kernel = int(pairwise_conv_kernel)
        self.pairwise_conv_aggr = pairwise_conv_aggr
        self.xcorr_feat_dim = int(xcorr_feat_dim)
        self.use_bottleneck_attention = bool(use_bottleneck_attention)
        self.graph_norm_type = graph_norm

        # --------------------------- node feature dims -------------------------
        self.node_feat_channels = (
            self.geom_feat_channels if node_feature_mode == "geometry" else 0
        ) + (1 if self.use_rsam_node_feat else 0)

        # --------------------------- edge feature dims -------------------------
        if edge_feature_mode == "none":
            self.edge_attr_dim_static = 0
        elif edge_feature_mode == "delta_pos":
            self.edge_attr_dim_static = 2
        elif edge_feature_mode == "delta_pos_dist":
            self.edge_attr_dim_static = 3
        else:  # delta_pos_xcorr
            self.edge_attr_dim_static = 2
        self.edge_attr_dim_total = self.edge_attr_dim_static + (
            self.xcorr_feat_dim if edge_feature_mode == "delta_pos_xcorr" else 0
        )

        # ------------------------------ graph levels ---------------------------
        if graph_levels is None:
            graph_levels = []
        self.graph_levels = sorted(
            [lvl for lvl in set(graph_levels) if 0 <= lvl < depth]
        )

        if pairwise_conv_levels is None:
            pairwise_conv_levels = []
        self.pairwise_conv_levels = sorted(
            [lvl for lvl in set(pairwise_conv_levels) if 0 <= lvl < depth]
        )

        self.use_skip_graph = bool(use_skip_graph)
        if not self.use_skip_graph:
            self.skip_graph_levels = []
        elif skip_graph_levels is None:
            self.skip_graph_levels = []
        else:
            self.skip_graph_levels = sorted(
                [lvl for lvl in set(skip_graph_levels) if 0 <= lvl < depth]
            )

        # ------------------------- geometry + topology -------------------------
        self._register_station_geometry(station_coords, crater_coords)
        self._register_volcano_geometry_bank(volcano_geom_nodes)

        # ------------------------------ encoder --------------------------------
        self.encoder_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.encoder_graph_op = nn.ModuleDict()
        self.encoder_graphnorm = nn.ModuleDict()
        self.pairwise_conv = nn.ModuleDict()

        for idx in self.pairwise_conv_levels:
            feat_channels = init_features * (2**idx)
            self.pairwise_conv[str(idx)] = PairwiseConv2d(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=self.pairwise_conv_kernel,
                n_stations=n_stations,
                aggr=self.pairwise_conv_aggr,
            )

        feat_in = in_channels
        for idx in range(depth):
            feat_out = init_features * (2**idx)
            self.encoder_list.append(
                self._block_1d(feat_in, feat_out, name=f"enc{idx}")
            )
            self.pool_list.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if idx in self.graph_levels:
                self.encoder_graph_op[str(idx)] = self._build_mpnn(
                    feat_out, self.mpnn_layers, mpnn_hidden_dim
                )
                self.encoder_graphnorm[str(idx)] = self._build_graph_norm(feat_out)
            feat_in = feat_out

        # ----------------------------- bottleneck ------------------------------
        self.bottleneck_feat_channels = init_features * (2 ** (depth - 1))
        self.graph_op_bottleneck = self._build_mpnn(
            self.bottleneck_feat_channels,
            self.bottleneck_mpnn_layers,
            mpnn_hidden_dim,
        )
        self.graphnorm_bottleneck = self._build_graph_norm(
            self.bottleneck_feat_channels
        )

        # Temporal self-attention over bottleneck sequence [B, T_bottleneck, C].
        if self.use_bottleneck_attention:
            if self.bottleneck_feat_channels % bottleneck_attn_heads != 0:
                raise ValueError(
                    "bottleneck_feat_channels must be divisible by "
                    "bottleneck_attn_heads. "
                    f"Got C={self.bottleneck_feat_channels}, "
                    f"heads={bottleneck_attn_heads}."
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

        # --------------------------- skip graph ops ----------------------------
        self.skip_graph_op = nn.ModuleDict()
        self.skip_graphnorm = nn.ModuleDict()
        for idx in range(depth):
            if idx in self.skip_graph_levels:
                skip_channels = init_features * (2**idx)
                self.skip_graph_op[str(idx)] = self._build_mpnn(
                    skip_channels, self.mpnn_layers, mpnn_hidden_dim
                )
                self.skip_graphnorm[str(idx)] = self._build_graph_norm(skip_channels)

        # ------------------------------ decoder --------------------------------
        self.decoder_list = nn.ModuleList()
        self.upconv_list = nn.ModuleList()

        current_channels = self.bottleneck_feat_channels
        for idx in range(depth):
            skip_channels = init_features * (2 ** (depth - 1 - idx))
            self.upconv_list.append(
                nn.ConvTranspose1d(
                    current_channels, skip_channels, kernel_size=2, stride=2
                )
            )
            self.decoder_list.append(
                self._block_1d(
                    2 * skip_channels, skip_channels, name=f"dec{depth - idx}"
                )
            )
            current_channels = skip_channels

        self.conv_final = nn.Conv1d(init_features, out_channels, kernel_size=1)

    # ============================ builders / norms =============================
    def _build_mpnn(
        self,
        channels: int,
        num_layers: int,
        mpnn_hidden_dim: Optional[int],
    ) -> EdgeMPNN:
        hidden = mpnn_hidden_dim if mpnn_hidden_dim is not None else channels
        return EdgeMPNN(
            in_channels=channels + self.node_feat_channels,
            out_channels=channels,
            edge_dim=self.edge_attr_dim_total,
            hidden_dim=hidden,
            num_layers=num_layers,
            aggr=self.mpnn_aggr,
        )

    def _build_graph_norm(self, channels: int) -> nn.Module:
        if self.graph_norm_type == "batchnorm":
            return nn.BatchNorm1d(channels)
        return nn.Identity()

    # ============================ geometry + edges =============================
    def _register_station_geometry(self, station_coords, crater_coords):
        if station_coords is None:
            station_coords = {
                "FRE": (-71.39, -36.87),
                "SHG": (-71.38, -36.88),
                "NBL": (-71.38, -36.82),
                "SHA": (-71.36, -36.80),
                "FU2": (-71.34, -36.90),
                "CHS": (-71.34, -36.87),
                "LBN": (-71.38, -36.85),
                "PLA": (-71.45, -36.83),
            }
        if crater_coords is None:
            crater_coords = (-71.37667, -36.86333)

        provided_station_items = list(station_coords.items())
        provided_count = len(provided_station_items)

        # Keep forward graph shape fixed by self.n_stations, while adapting
        # metadata availability by trimming/padding station coordinates.
        station_items = list(provided_station_items)
        if len(station_items) > self.n_stations:
            station_items = station_items[: self.n_stations]
        elif len(station_items) < self.n_stations:
            missing = self.n_stations - len(station_items)
            for idx in range(missing):
                station_items.append((f"PAD_{idx:02d}", crater_coords))
        station_coords = dict(station_items)

        if self.verbose:
            used_names = list(station_coords.keys())
            print(
                "[UNet_MPNN] station geometry configured "
                f"(provided={provided_count}, model_n_stations={self.n_stations}, "
                f"used={len(used_names)})"
            )
            print(
                "[UNet_MPNN] crater lon/lat="
                f"({float(crater_coords[0]):.5f}, {float(crater_coords[1]):.5f})"
            )
            print("[UNet_MPNN] station order used: " + ", ".join(used_names))

        lat_mean = float(np.mean([c[1] for c in station_coords.values()]))
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mean))
        km_per_deg_lat = 111.0

        coords_km = {}
        for stn, (lon, lat) in station_coords.items():
            coords_km[stn] = (
                (lon - crater_coords[0]) * km_per_deg_lon,
                (lat - crater_coords[1]) * km_per_deg_lat,
            )

        stn_names = list(station_coords.keys())
        coords_array = np.array([coords_km[stn] for stn in stn_names], dtype=np.float32)
        dist_to_crater = np.linalg.norm(coords_array, axis=1, keepdims=True).astype(
            np.float32
        )

        self.register_buffer("station_xy", torch.from_numpy(coords_array))
        self.register_buffer("dist_to_crater", torch.from_numpy(dist_to_crater))
        self.register_buffer("network_xy", torch.zeros(1, 2, dtype=torch.float32))
        self.register_buffer("network_dist", torch.zeros(1, 1, dtype=torch.float32))

        # Normalized geometry features for node inputs (stations + virtual node).
        xy_full = torch.cat([self.station_xy, self.network_xy], dim=0)
        dist_full = torch.cat([self.dist_to_crater, self.network_dist], dim=0)
        xy_norm = torch.linalg.norm(xy_full) + 1e-6
        dist_norm = torch.linalg.norm(dist_full) + 1e-6
        node_xy_norm = xy_full / xy_norm  # [n_nodes, 2], virtual node at origin
        geom = torch.cat([node_xy_norm, dist_full / dist_norm], dim=1)
        self.register_buffer("geom_nodes", geom)
        self.register_buffer("node_xy_norm", node_xy_norm)

        # --------------------------- edge topology ---------------------------
        # Ordering convention: station-station ordered pairs first (used for
        # dynamic xcorr alignment), then bidirectional station<->virtual edges.
        network_idx = self.n_stations
        station_pairs: List[List[int]] = []
        if self.graph_topology == "fully_connected":
            for i in range(self.n_stations):
                for j in range(self.n_stations):
                    if i != j:
                        station_pairs.append([i, j])
        self.n_station_pairs = len(station_pairs)

        virtual_edges: List[List[int]] = []
        for i in range(self.n_stations):
            virtual_edges.append([i, network_idx])
            virtual_edges.append([network_idx, i])

        edges = station_pairs + virtual_edges
        edge_index_base = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.register_buffer("edge_index_base", edge_index_base)

        # --------------------------- static edge attrs -----------------------
        num_edges = edge_index_base.shape[1]
        if self.edge_attr_dim_static > 0:
            attr_rows = []
            for src, dst in edges:
                delta = node_xy_norm[dst] - node_xy_norm[src]
                if self.edge_feature_mode == "delta_pos_dist":
                    dist = float(torch.linalg.norm(delta))
                    attr_rows.append([float(delta[0]), float(delta[1]), dist])
                else:  # delta_pos or static part of delta_pos_xcorr
                    attr_rows.append([float(delta[0]), float(delta[1])])
            edge_attr_base = torch.tensor(attr_rows, dtype=torch.float32)
        else:
            edge_attr_base = torch.zeros(num_edges, 0, dtype=torch.float32)
        self.register_buffer("edge_attr_base", edge_attr_base)

    def _build_batched_edge_index(self, num_graphs: int) -> torch.Tensor:
        """Vectorized edge batching for repeated graphs (graph-major order)."""
        edge_index = self.edge_index_base
        edges_per_graph = edge_index.shape[1]
        offsets = (
            torch.arange(num_graphs, device=edge_index.device, dtype=edge_index.dtype)
            * self.n_nodes
        ).view(num_graphs, 1, 1)
        edge_index_batch = edge_index.unsqueeze(0) + offsets
        return edge_index_batch.permute(1, 0, 2).reshape(
            2, num_graphs * edges_per_graph
        )

    def _build_batched_edge_attr_static(
        self,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
        B: Optional[int] = None,
        T_l: Optional[int] = None,
        volcano_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Repeat static edge attrs to [num_graphs*num_edges, static_dim]."""
        if (
            self.volcano_edge_attr_static.numel() > 0
            and volcano_idx is not None
            and B is not None
            and T_l is not None
        ):
            v_idx = volcano_idx.to(device=device, dtype=torch.long).view(B)
            if int(v_idx.min().item()) < 0 or int(v_idx.max().item()) >= int(
                self.volcano_edge_attr_static.shape[0]
            ):
                raise ValueError(
                    "volcano_idx contains out-of-range values for volcano_edge_attr_static "
                    f"(num_volcanoes={int(self.volcano_edge_attr_static.shape[0])})."
                )
            ea = self.volcano_edge_attr_static.to(device=device, dtype=dtype)[v_idx]
            num_edges, dim = int(ea.shape[1]), int(ea.shape[2])
            return (
                ea.unsqueeze(1)
                .expand(B, T_l, num_edges, dim)
                .reshape(B * T_l * num_edges, dim)
            )

        ea = self.edge_attr_base.to(device=device, dtype=dtype)
        num_edges, dim = ea.shape
        return (
            ea.unsqueeze(0)
            .expand(num_graphs, num_edges, dim)
            .reshape(num_graphs * num_edges, dim)
        )

    def _build_batched_edge_attr_dynamic(
        self,
        edge_attr_dynamic: torch.Tensor,
        B: int,
        T_l: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Expand dynamic per-station-pair features to align with the batched edges.

        Args:
            edge_attr_dynamic: [B, n_station_pairs, F_dyn]

        Returns:
            [num_graphs * num_edges, F_dyn], where num_graphs = B * T_l.
            Station<->virtual edges receive zero-padding for the dynamic part.
        """
        F_dyn = self.xcorr_feat_dim
        num_edges = self.edge_index_base.shape[1]
        out = torch.zeros(B, T_l, num_edges, F_dyn, device=device, dtype=dtype)
        if self.n_station_pairs > 0:
            ed = edge_attr_dynamic.to(device=device, dtype=dtype)
            out[:, :, : self.n_station_pairs, :] = ed.unsqueeze(1).expand(
                B, T_l, self.n_station_pairs, F_dyn
            )
        return out.reshape(B * T_l * num_edges, F_dyn)

    def _build_edge_attr(
        self,
        num_graphs: int,
        B: int,
        T_l: int,
        edge_attr_dynamic: Optional[torch.Tensor],
        volcano_idx: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.edge_attr_dim_total == 0:
            return None
        parts = []
        if self.edge_attr_dim_static > 0:
            parts.append(
                self._build_batched_edge_attr_static(
                    num_graphs,
                    device,
                    dtype,
                    B=B,
                    T_l=T_l,
                    volcano_idx=volcano_idx,
                )
            )
        if self.edge_feature_mode == "delta_pos_xcorr":
            if edge_attr_dynamic is None:
                raise ValueError(
                    "edge_feature_mode='delta_pos_xcorr' requires edge_attr_dynamic "
                    "of shape [B, n_station_pairs, F_dyn] to be passed to forward()."
                )
            parts.append(
                self._build_batched_edge_attr_dynamic(
                    edge_attr_dynamic, B, T_l, device, dtype
                )
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    # ============================== node features ==============================
    def _build_node_features(
        self,
        num_graphs: int,
        B: int,
        T_l: int,
        rsam: Optional[torch.Tensor],
        volcano_idx: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts = []
        if self.node_feature_mode == "geometry":
            if self.volcano_geom_nodes.numel() > 0 and volcano_idx is not None:
                v_idx = volcano_idx.to(device=device, dtype=torch.long).view(B)
                if int(v_idx.min().item()) < 0 or int(v_idx.max().item()) >= int(
                    self.volcano_geom_nodes.shape[0]
                ):
                    raise ValueError(
                        "volcano_idx contains out-of-range values for volcano_geom_nodes "
                        f"(num_volcanoes={int(self.volcano_geom_nodes.shape[0])})."
                    )
                geom_bt = self.volcano_geom_nodes.to(device=device, dtype=dtype)[v_idx]
                geom = (
                    geom_bt.unsqueeze(1)
                    .expand(B, T_l, self.n_nodes, self.geom_feat_channels)
                    .reshape(num_graphs, self.n_nodes, self.geom_feat_channels)
                )
            else:
                geom = self.geom_nodes.to(device=device, dtype=dtype)
                geom = geom.unsqueeze(0).expand(
                    num_graphs, self.n_nodes, self.geom_feat_channels
                )
            parts.append(geom)
        if self.use_rsam_node_feat:
            if rsam is None:
                rsam_nodes = torch.zeros(B, self.n_nodes, 1, device=device, dtype=dtype)
            else:
                r = rsam.to(device=device, dtype=dtype)
                stn = r.unsqueeze(-1)  # [B, S, 1]
                net = r.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
                rsam_nodes = torch.cat([stn, net], dim=1)  # [B, n_nodes, 1]
            parts.append(
                rsam_nodes.unsqueeze(1)
                .expand(B, T_l, self.n_nodes, 1)
                .reshape(num_graphs, self.n_nodes, 1)
            )
        if not parts:
            return torch.empty(num_graphs, self.n_nodes, 0, device=device, dtype=dtype)
        return torch.cat(parts, dim=2)

    def _register_volcano_geometry_bank(
        self,
        volcano_geom_nodes: Optional[torch.Tensor],
    ) -> None:
        if volcano_geom_nodes is None:
            self.register_buffer(
                "volcano_geom_nodes",
                torch.empty(
                    0, self.n_nodes, self.geom_feat_channels, dtype=torch.float32
                ),
                persistent=False,
            )
            self.register_buffer(
                "volcano_edge_attr_static",
                torch.empty(
                    0,
                    self.edge_index_base.shape[1],
                    self.edge_attr_dim_static,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            return

        geom_bank = torch.as_tensor(volcano_geom_nodes, dtype=torch.float32)
        if geom_bank.ndim != 3:
            raise ValueError(
                "volcano_geom_nodes must be a 3D tensor with shape "
                f"(num_volcanoes, {self.n_nodes}, {self.geom_feat_channels})."
            )
        if (
            geom_bank.shape[1] != self.n_nodes
            or geom_bank.shape[2] != self.geom_feat_channels
        ):
            raise ValueError(
                "volcano_geom_nodes has invalid shape. Expected (*, "
                f"{self.n_nodes}, {self.geom_feat_channels}), got {tuple(geom_bank.shape)}"
            )
        self.register_buffer("volcano_geom_nodes", geom_bank, persistent=False)

        if self.edge_attr_dim_static == 0:
            self.register_buffer(
                "volcano_edge_attr_static",
                torch.empty(
                    geom_bank.shape[0],
                    self.edge_index_base.shape[1],
                    0,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            return

        src = self.edge_index_base[0]
        dst = self.edge_index_base[1]
        xy = geom_bank[:, :, :2]
        delta = xy[:, dst, :] - xy[:, src, :]
        if self.edge_feature_mode == "delta_pos_dist":
            dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
            edge_attr_bank = torch.cat([delta, dist], dim=-1)
        else:
            edge_attr_bank = delta
        self.register_buffer(
            "volcano_edge_attr_static",
            edge_attr_bank.to(torch.float32),
            persistent=False,
        )

    # =============================== graph op ==================================
    def _apply_node_norm(
        self,
        x_nodes: torch.Tensor,
        graph_norm: nn.Module,
    ) -> torch.Tensor:
        if isinstance(graph_norm, nn.BatchNorm1d):
            return graph_norm(x_nodes)
        return x_nodes

    def _apply_mpnn(
        self,
        graph_op: EdgeMPNN,
        x_flat: torch.Tensor,
        B: int,
        S: int,
        graph_norm: Optional[nn.Module] = None,
        edge_attr_dynamic: Optional[torch.Tensor] = None,
        rsam: Optional[torch.Tensor] = None,
        volcano_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the edge-conditioned MPNN over station features at one level.

        Args:
            graph_op : EdgeMPNN stack for this level.
            x_flat   : [B*S, C, T_l] station features.

        Returns:
            station_out : [B*S, C, T_l] updated station features.
            network_out : [B, C, T_l] virtual-node readout (feeds the decoder).
        """
        _, C, T_l = x_flat.shape
        num_graphs = B * T_l
        device, dtype = x_flat.device, x_flat.dtype

        x_bst = x_flat.reshape(B, S, C, T_l)
        # Virtual node initialized by a plain mean over stations.
        network_feature = x_bst.mean(dim=1, keepdim=True)  # [B, 1, C, T_l]
        x_aug = torch.cat([x_bst, network_feature], dim=1)  # [B, n_nodes, C, T_l]
        x_nodes_in = x_aug.permute(0, 3, 1, 2).reshape(num_graphs, self.n_nodes, C)

        node_feats = self._build_node_features(
            num_graphs,
            B,
            T_l,
            rsam,
            volcano_idx,
            device,
            dtype,
        )
        if node_feats.shape[-1] > 0:
            x_in = torch.cat([x_nodes_in, node_feats], dim=2)
        else:
            x_in = x_nodes_in
        x_flat_nodes = x_in.reshape(
            num_graphs * self.n_nodes, C + self.node_feat_channels
        )

        edge_index_batch = self._build_batched_edge_index(num_graphs)
        edge_attr = self._build_edge_attr(
            num_graphs,
            B,
            T_l,
            edge_attr_dynamic,
            volcano_idx,
            device,
            dtype,
        )

        x_out_nodes = graph_op(x_flat_nodes, edge_index_batch, edge_attr)
        if graph_norm is not None:
            x_out_nodes = self._apply_node_norm(x_out_nodes, graph_norm)

        x_out = x_out_nodes.reshape(B, T_l, self.n_nodes, C).permute(0, 2, 3, 1)
        station_out = x_out[:, :S].reshape(B * S, C, T_l)
        network_out = x_out[:, S]
        return station_out, network_out

    def _apply_bottleneck_attention(self, x_dec: torch.Tensor) -> torch.Tensor:
        """Temporal MHSA block over bottleneck features [B, C, T_b] -> [B, C, T_b]."""
        if not self.use_bottleneck_attention:
            return x_dec
        x_seq = x_dec.transpose(1, 2)
        x_norm = self.bottleneck_attn_norm1(x_seq)
        x_attn, _ = self.bottleneck_attn(x_norm, x_norm, x_norm, need_weights=False)
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.bottleneck_ff(self.bottleneck_attn_norm2(x_seq))
        return x_seq.transpose(1, 2)

    @staticmethod
    def _virtual_node_init_readout(
        x_flat: torch.Tensor,
        B: int,
        S: int,
    ) -> torch.Tensor:
        """Mean-readout collapsing stations to a network-level representation."""
        _, C, T_l = x_flat.shape
        x_bst = x_flat.reshape(B, S, C, T_l)
        return x_bst.mean(dim=1)

    # ================================ forward ==================================
    def forward(
        self,
        x: torch.Tensor,
        edge_attr_dynamic: Optional[torch.Tensor] = None,
        rsam: Optional[torch.Tensor] = None,
        volcano_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, T]
            edge_attr_dynamic: [B, n_station_pairs, F_dyn], required when
                edge_feature_mode == 'delta_pos_xcorr'.
            rsam: [B, S], required when use_rsam_node_feat is True.

        Returns:
            out: [B, C_out, T]
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B, S, T]. Got shape: {tuple(x.shape)}")
        B, S, T = x.shape
        if S != self.n_stations:
            raise ValueError(
                f"x has S={S} stations but model n_stations={self.n_stations}."
            )

        if self.edge_feature_mode == "delta_pos_xcorr" and edge_attr_dynamic is None:
            raise ValueError(
                "edge_feature_mode='delta_pos_xcorr' requires edge_attr_dynamic."
            )
        if edge_attr_dynamic is not None:
            expected = (B, self.n_station_pairs, self.xcorr_feat_dim)
            if tuple(edge_attr_dynamic.shape) != expected:
                raise ValueError(
                    f"edge_attr_dynamic must be {expected}. "
                    f"Got: {tuple(edge_attr_dynamic.shape)}"
                )
        if self.use_rsam_node_feat and rsam is not None:
            if tuple(rsam.shape) != (B, S):
                raise ValueError(
                    f"rsam must be [B, S] = {(B, S)}. Got: {tuple(rsam.shape)}"
                )
        if volcano_idx is not None and tuple(volcano_idx.shape) not in {(B,), (B, 1)}:
            raise ValueError(
                f"volcano_idx must be [B] or [B,1] with B={B}. Got: {tuple(volcano_idx.shape)}"
            )

        x_flat = x.reshape(B * S, 1, T)

        encodings: list = []
        for i in range(self.depth):
            x_flat = self.encoder_list[i](x_flat)

            if str(i) in self.pairwise_conv:
                x_flat = self.pairwise_conv[str(i)](x_flat, B, S)

            if str(i) in self.encoder_graph_op:
                x_flat, _ = self._apply_mpnn(
                    self.encoder_graph_op[str(i)],
                    x_flat,
                    B,
                    S,
                    self.encoder_graphnorm[str(i)],
                    edge_attr_dynamic,
                    rsam,
                    volcano_idx,
                )
            encodings.append(x_flat)
            x_flat = self.pool_list[i](x_flat)

        _, x_dec = self._apply_mpnn(
            self.graph_op_bottleneck,
            x_flat,
            B,
            S,
            self.graphnorm_bottleneck,
            edge_attr_dynamic,
            rsam,
            volcano_idx,
        )
        if self.use_bottleneck_attention:
            x_dec = self._apply_bottleneck_attention(x_dec)

        for i in range(self.depth):
            x_dec = self.upconv_list[i](x_dec)

            skip = encodings[-(i + 1)]
            enc_level = self.depth - 1 - i
            if str(enc_level) in self.skip_graph_op:
                _, skip = self._apply_mpnn(
                    self.skip_graph_op[str(enc_level)],
                    skip,
                    B,
                    S,
                    self.skip_graphnorm[str(enc_level)],
                    edge_attr_dynamic,
                    rsam,
                    volcano_idx,
                )
            else:
                skip = self._virtual_node_init_readout(skip, B, S)

            x_dec = torch.cat((x_dec, skip), dim=1)
            x_dec = self.decoder_list[i](x_dec)

        return self.conv_final(x_dec)

    @staticmethod
    def _block_1d(in_channels: int, features: int, name: str) -> nn.Sequential:
        """Double Conv1d block: (Conv -> BN -> ReLU) x 2."""
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels, features, kernel_size=3, padding=1, bias=False
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            features, features, kernel_size=3, padding=1, bias=False
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
