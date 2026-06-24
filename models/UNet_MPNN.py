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
        xcorr_feat_dim: int = 2,
        use_bottleneck_attention: bool = True,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_dropout: float = 0.0,
        bottleneck_attn_ff_mult: int = 2,
        graph_norm: Literal["none", "batchnorm"] = "none",
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

        # ------------------------------ encoder --------------------------------
        self.encoder_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.encoder_graph_op = nn.ModuleDict()
        self.encoder_graphnorm = nn.ModuleDict()

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
    ) -> torch.Tensor:
        """Repeat static edge attrs to [num_graphs*num_edges, static_dim]."""
        ea = self.edge_attr_base.to(device=device, dtype=dtype)
        num_edges, dim = ea.shape
        return ea.unsqueeze(0).expand(num_graphs, num_edges, dim).reshape(
            num_graphs * num_edges, dim
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
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.edge_attr_dim_total == 0:
            return None
        parts = []
        if self.edge_attr_dim_static > 0:
            parts.append(
                self._build_batched_edge_attr_static(num_graphs, device, dtype)
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
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts = []
        if self.node_feature_mode == "geometry":
            geom = self.geom_nodes.to(device=device, dtype=dtype)
            parts.append(
                geom.unsqueeze(0).expand(
                    num_graphs, self.n_nodes, self.geom_feat_channels
                )
            )
        if self.use_rsam_node_feat:
            if rsam is None:
                rsam_nodes = torch.zeros(
                    B, self.n_nodes, 1, device=device, dtype=dtype
                )
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
            return torch.empty(
                num_graphs, self.n_nodes, 0, device=device, dtype=dtype
            )
        return torch.cat(parts, dim=2)

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
            num_graphs, B, T_l, rsam, device, dtype
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
            num_graphs, B, T_l, edge_attr_dynamic, device, dtype
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

        x_flat = x.reshape(B * S, 1, T)

        encodings: list = []
        for i in range(self.depth):
            x_flat = self.encoder_list[i](x_flat)
            if str(i) in self.encoder_graph_op:
                x_flat, _ = self._apply_mpnn(
                    self.encoder_graph_op[str(i)],
                    x_flat,
                    B,
                    S,
                    self.encoder_graphnorm[str(i)],
                    edge_attr_dynamic,
                    rsam,
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


# =============================== ablation configs ==============================
# Single source of truth for the MPNN ablation sweep. `edge_mpnn__bottleneck`
# is the default/baseline run.
MPNN_ABLATION_KWARGS = {
    # Defaults: fully_connected, delta_pos edges, bottleneck-only MPNN,
    # skip-graph off, bottleneck attention on, no graph norm. Run first.
    # "edge_mpnn__bottleneck": dict(
    #     graph_topology="fully_connected",
    #     edge_feature_mode="delta_pos",
    #     node_feature_mode="geometry",
    #     graph_levels=[],
    #     use_skip_graph=False,
    #     use_bottleneck_attention=True,
    #     graph_norm="none",
    # ),
    # THE key test: drop edge features architecturally (message MLP loses its
    # edge slice). Proves whether edge geometry matters.
    "edge_mpnn__early_l2": dict(
        graph_topology="fully_connected",
        edge_feature_mode="delta_pos",
        node_feature_mode="geometry",
        graph_levels=[2],           # level 2 only, no bottleneck graph
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    "edge_mpnn__early_l1": dict(
        graph_topology="fully_connected",
        edge_feature_mode="delta_pos",
        node_feature_mode="geometry",
        graph_levels=[1],           # level 1 only, no bottleneck graph
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    "edge_mpnn__both_l2_bottleneck": dict(
        graph_topology="fully_connected",
        edge_feature_mode="delta_pos",
        node_feature_mode="geometry",
        graph_levels=[2],           # early + bottleneck (bottleneck always present)
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),    
    "edge_mpnn__no_edge_feats": dict(
        graph_topology="fully_connected",
        edge_feature_mode="none",
        node_feature_mode="geometry",
        graph_levels=[],
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    # Shallow encoder MPNN + bottleneck. Only run if bottleneck ties baseline.
    "edge_mpnn__encoder": dict(
        graph_topology="fully_connected",
        edge_feature_mode="delta_pos",
        node_feature_mode="geometry",
        graph_levels=[1],
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    # Topology ablation: star graph (no station-station edges).
    "edge_mpnn__star_topology": dict(
        graph_topology="star",
        edge_feature_mode="delta_pos",
        node_feature_mode="geometry",
        graph_levels=[],
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    # Dynamic cross-correlation edge features (requires offline edge_attr_dynamic).
    "edge_mpnn__xcorr": dict(
        graph_topology="fully_connected",
        edge_feature_mode="delta_pos_xcorr",
        node_feature_mode="geometry",
        xcorr_feat_dim=2,
        graph_levels=[],
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    # Control: no spatial node info anywhere (pairs with edge none for a true
    # no-geometry run).
    "edge_mpnn__no_spatial_info": dict(
        graph_topology="fully_connected",
        edge_feature_mode="none",
        node_feature_mode="none",
        graph_levels=[],
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    # Append precomputed per-station RSAM as a node feature (run on the winner).
    "edge_mpnn__rsam": dict(
        graph_topology="fully_connected",
        edge_feature_mode="delta_pos",
        node_feature_mode="geometry",
        use_rsam_node_feat=True,
        graph_levels=[],
        use_skip_graph=False,
        use_bottleneck_attention=True,
        graph_norm="none",
    ),
    # Re-isolate the bottleneck attention in the MPNN context.
    "edge_mpnn__no_attention": dict(
        graph_topology="fully_connected",
        edge_feature_mode="delta_pos",
        node_feature_mode="geometry",
        graph_levels=[],
        use_skip_graph=False,
        use_bottleneck_attention=False,
        graph_norm="none",
    ),
}

