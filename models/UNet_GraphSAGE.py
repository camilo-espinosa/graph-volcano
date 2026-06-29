"""
UNet_GraphSAGE v5 — GraphSAGE + GraphNorm + Attention Pooling,
with bottleneck temporal self-attention for multistation seismic segmentation.

Extends v4 by replacing equal-weight virtual-node initialization with learned
attention pooling over station embeddings before each GraphSAGE operation.

Graph operation pattern:
    station embeddings -> attention pooling -> virtual node init -> GraphSAGE -> GraphNorm

Input:  [B, S, T]  where B=batch, S=stations (8), T=temporal samples (8192)
Output: [B, 6, T]  where 6 = number of classes (BG, VT, LP, TR, AV, IC)
"""

"""
**New structural components**

- **Star graph**: S station nodes + 1 virtual network node, with bidirectional edges between each station and the virtual node. Geometry features (x_km, y_km, dist_to_crater) or learned station embeddings are concatenated to node features before message passing.
- **GraphSAGE layers** applied at selected encoder levels and on skip connections, replacing independent per-station processing with cross-station message passing.
- **Virtual node readout**: the network node aggregates all station information and its embedding is used as the skip/bottleneck feature passed to the decoder, collapsing [B*S, C, T] → [B, C, T].
- **StationAttentionPool**: a small MLP that computes per-station attention scores and produces a weighted pooled representation of the virtual node (instead of simple mean pooling).
- **Bottleneck temporal self-attention (MHSA)**: a Transformer-style block (LayerNorm → MultiheadAttention + residual → FFN + residual) applied on the bottleneck sequence [B, T_bottleneck, C] after the graph readout.
- **GraphNorm / BatchNorm** applied after every graph operation.

**What can be configured**

| Parameter | Controls |
|---|---|
| `graph_levels` | Which encoder depths run GraphSAGE |
| `use_skip_graph` / `skip_graph_levels` | Whether skip connections also run GraphSAGE |
| `graphsage_layers` / `skip_graphsage_layers` | Depth of GraphSAGE stacks |
| `graph_backend` | `"graphsage"` (message passing) or `"mlp"` (per-station, no communication) |
| `use_message_passing` | Bypass GraphSAGE and use virtual-node readout only |
| `virtual_node_pool_mode` | `"learned"` (attention pool) or `"mean"` for all non-bottleneck levels |
| `bottleneck_virtual_node_pool_mode` | Override pooling specifically at bottleneck |
| `attention_pool_mode` | `"bottleneck_only"` or `"all_levels"` for learned pooling |
| `use_bottleneck_attention` | Toggle the MHSA block |
| `bottleneck_attn_heads/dropout/ff_mult` | MHSA hyperparameters |
| `graph_norm_type` | `"graphnorm"`, `"batchnorm"`, or `"none"` |
| `node_feature_mode` | `"geometry"`, `"learned_station_embedding"`, `"both"`, or `"none"` |
| `n_stations`, `station_coords`, `crater_coords` | Station count and spatial layout |

"""

from collections import OrderedDict
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm, GraphSAGE


class StationAttentionPool(nn.Module):
    """Learn attention weights over stations and produce pooled network feature."""

    def __init__(self, channels: int, hidden_ratio: float = 0.5, min_hidden: int = 8):
        super().__init__()
        hidden = max(int(channels * hidden_ratio), min_hidden)
        self.score_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        station_bt_s_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            station_bt_s_c: [B, T_l, S, C]

        Returns:
            pooled_bt_c: [B, T_l, C]
            alpha_bt_s: [B, T_l, S]
        """
        scores = self.score_mlp(station_bt_s_c).squeeze(-1)
        alpha = torch.softmax(scores, dim=2)
        pooled = (alpha.unsqueeze(-1) * station_bt_s_c).sum(dim=2)
        return pooled, alpha


class UNet_GraphSAGE(nn.Module):
    """
    1D UNet with GraphSAGE at selected encoder/skip levels, GraphNorm after
    every graph op, learned attention pooling for virtual-node initialization,
    and bottleneck temporal self-attention.

    Attention placement
    -------------------
    Bottleneck path:
        GraphSAGE + GraphNorm readout -> MHSA + residual -> FFN + residual

    Graph operation placement
    -------------------------
    - attention_pool_mode="all_levels":
        Encoder / bottleneck / skip graph ops use attention pooling.
    - attention_pool_mode="bottleneck_only":
        Only bottleneck graph op uses attention pooling; encoder/skip use mean init.

    Additional ablation switches
    ----------------------------
    - use_bottleneck_attention: toggle bottleneck MHSA block.
    - graph_norm_type: one of {"graphnorm", "batchnorm", "none"}.
    - node_feature_mode: one of {"geometry", "learned_station_embedding", "both", "none"}.
    - graph_backend: one of {"graphsage", "mlp"}.
    - use_message_passing: if False, bypass GraphSAGE message passing and use
      virtual-node readout only.
    - virtual_node_pool_mode: one of {"learned", "mean"}.
    - bottleneck_virtual_node_pool_mode: override bottleneck pooling mode only.
    - use_skip_graph / skip_graph_levels: control graph processing on skip paths.
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
        graphsage_layers: int = 2,
        skip_graphsage_layers: Optional[int] = None,
        attention_pool_mode: Literal[
            "all_levels", "bottleneck_only"
        ] = "bottleneck_only",
        use_bottleneck_attention: bool = True,
        graph_norm_type: Literal["graphnorm", "batchnorm", "none"] = "graphnorm",
        node_feature_mode: Literal[
            "geometry", "learned_station_embedding", "both", "none"
        ] = "geometry",
        station_embedding_dim: int = 8,
        graph_backend: Literal["graphsage", "mlp"] = "graphsage",
        use_message_passing: bool = True,
        virtual_node_pool_mode: Literal["learned", "mean"] = "learned",
        bottleneck_virtual_node_pool_mode: Optional[Literal["learned", "mean"]] = None,
        use_skip_graph: bool = True,
        skip_graph_levels: Optional[List[int]] = None,
        volcano_geom_nodes: Optional[torch.Tensor] = None,
        bottleneck_attn_heads: int = 4,
        bottleneck_attn_dropout: float = 0.0,
        bottleneck_attn_ff_mult: int = 2,
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

        if graph_norm_type not in {"graphnorm", "batchnorm", "none"}:
            raise ValueError(
                "graph_norm_type must be one of {'graphnorm', 'batchnorm', 'none'}. "
                f"Got: {graph_norm_type}"
            )
        if node_feature_mode not in {
            "geometry",
            "learned_station_embedding",
            "both",
            "none",
        }:
            raise ValueError(
                "node_feature_mode must be one of {'geometry', 'learned_station_embedding', 'both', 'none'}. "
                f"Got: {node_feature_mode}"
            )
        if graph_backend not in {"graphsage", "mlp"}:
            raise ValueError(
                "graph_backend must be one of {'graphsage', 'mlp'}. "
                f"Got: {graph_backend}"
            )
        if virtual_node_pool_mode not in {"learned", "mean"}:
            raise ValueError(
                "virtual_node_pool_mode must be one of {'learned', 'mean'}. "
                f"Got: {virtual_node_pool_mode}"
            )
        if (
            bottleneck_virtual_node_pool_mode is not None
            and bottleneck_virtual_node_pool_mode not in {"learned", "mean"}
        ):
            raise ValueError(
                "bottleneck_virtual_node_pool_mode must be one of {'learned', 'mean'} or None. "
                f"Got: {bottleneck_virtual_node_pool_mode}"
            )

        self.use_bottleneck_attention = bool(use_bottleneck_attention)
        self.graph_norm_type = graph_norm_type
        self.node_feature_mode = node_feature_mode
        self.graph_backend = graph_backend
        self.use_message_passing = bool(use_message_passing)
        self.virtual_node_pool_mode = virtual_node_pool_mode
        self.bottleneck_virtual_node_pool_mode = bottleneck_virtual_node_pool_mode
        self.use_skip_graph = bool(use_skip_graph)
        self.station_embedding_dim = int(station_embedding_dim)
        self.verbose = bool(verbose)
        if self.station_embedding_dim <= 0 and self.node_feature_mode in {
            "learned_station_embedding",
            "both",
        }:
            raise ValueError(
                "station_embedding_dim must be > 0 when using learned embeddings."
            )

        if attention_pool_mode not in {"all_levels", "bottleneck_only"}:
            raise ValueError(
                "attention_pool_mode must be one of {'all_levels', 'bottleneck_only'}. "
                f"Got: {attention_pool_mode}"
            )
        self.attention_pool_mode = attention_pool_mode
        self.attn_pool_all_levels = attention_pool_mode == "all_levels"

        if graph_levels is None:
            graph_levels = list(range(max(depth - 2, 0), depth))
        self.graph_levels = sorted(
            [lvl for lvl in set(graph_levels) if 0 <= lvl < depth]
        )

        if not self.use_skip_graph:
            self.skip_graph_levels = []
        elif skip_graph_levels is None:
            self.skip_graph_levels = list(self.graph_levels)
        else:
            self.skip_graph_levels = sorted(
                [lvl for lvl in set(skip_graph_levels) if 0 <= lvl < depth]
            )

        if self.node_feature_mode == "geometry":
            self.node_feat_channels = self.geom_feat_channels
            self.station_id_embedding = None
        elif self.node_feature_mode == "learned_station_embedding":
            self.node_feat_channels = self.station_embedding_dim
            self.station_id_embedding = nn.Embedding(
                self.n_nodes,
                self.station_embedding_dim,
            )
        elif self.node_feature_mode == "none":
            self.node_feat_channels = 0
            self.station_id_embedding = None
        else:
            self.node_feat_channels = (
                self.geom_feat_channels + self.station_embedding_dim
            )
            self.station_id_embedding = nn.Embedding(
                self.n_nodes,
                self.station_embedding_dim,
            )

        self.encoder_pool_mode = (
            "learned"
            if self.virtual_node_pool_mode == "learned" and self.attn_pool_all_levels
            else "mean"
        )
        self.skip_pool_mode = self.encoder_pool_mode
        self.bottleneck_pool_mode = (
            self.bottleneck_virtual_node_pool_mode
            if self.bottleneck_virtual_node_pool_mode is not None
            else self.virtual_node_pool_mode
        )

        self._register_station_geometry(station_coords, crater_coords)
        self._register_volcano_geometry_bank(volcano_geom_nodes)

        _skip_layers = (
            graphsage_layers if skip_graphsage_layers is None else skip_graphsage_layers
        )

        # Encoder
        self.encoder_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.encoder_graph_op = nn.ModuleDict()
        self.encoder_graphnorm = nn.ModuleDict()
        self.encoder_station_pool = nn.ModuleDict()

        feat_in = in_channels
        for idx in range(depth):
            feat_out = init_features * (2**idx)
            self.encoder_list.append(
                self._block_1d(feat_in, feat_out, name=f"enc{idx}")
            )
            self.pool_list.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if idx in self.graph_levels:
                if self.graph_backend == "graphsage":
                    self.encoder_graph_op[str(idx)] = GraphSAGE(
                        in_channels=feat_out + self.node_feat_channels,
                        hidden_channels=feat_out,
                        num_layers=graphsage_layers,
                        out_channels=feat_out,
                        dropout=0.0,
                    )
                else:
                    self.encoder_graph_op[str(idx)] = self._station_mlp(
                        feat_out,
                        name=f"enc{idx}_mlp",
                    )
                self.encoder_graphnorm[str(idx)] = self._build_graph_norm(feat_out)
                if self.encoder_pool_mode == "learned":
                    self.encoder_station_pool[str(idx)] = StationAttentionPool(feat_out)
            feat_in = feat_out

        # Bottleneck GraphSAGE + GraphNorm
        self.bottleneck_feat_channels = init_features * (2 ** (depth - 1))
        if self.graph_backend == "graphsage":
            self.graph_op_bottleneck = GraphSAGE(
                in_channels=self.bottleneck_feat_channels + self.node_feat_channels,
                hidden_channels=self.bottleneck_feat_channels,
                num_layers=3,
                out_channels=self.bottleneck_feat_channels,
                dropout=0.0,
            )
        else:
            self.graph_op_bottleneck = self._station_mlp(
                self.bottleneck_feat_channels,
                name="bottleneck_mlp",
            )
        self.graphnorm_bottleneck = self._build_graph_norm(
            self.bottleneck_feat_channels
        )
        self.station_pool_bottleneck = (
            StationAttentionPool(self.bottleneck_feat_channels)
            if self.bottleneck_pool_mode == "learned"
            else None
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

        # GraphSAGE modules for skip connections at selected levels.
        self.skip_graph_op = nn.ModuleDict()
        self.skip_graphnorm = nn.ModuleDict()
        self.skip_station_pool = nn.ModuleDict()
        for idx in range(depth):
            if idx in self.skip_graph_levels:
                skip_channels = init_features * (2**idx)
                if self.graph_backend == "graphsage":
                    self.skip_graph_op[str(idx)] = GraphSAGE(
                        in_channels=skip_channels + self.node_feat_channels,
                        hidden_channels=skip_channels,
                        num_layers=_skip_layers,
                        out_channels=skip_channels,
                        dropout=0.0,
                    )
                else:
                    self.skip_graph_op[str(idx)] = self._station_mlp(
                        skip_channels,
                        name=f"skip{idx}_mlp",
                    )
                self.skip_graphnorm[str(idx)] = self._build_graph_norm(skip_channels)
                if self.skip_pool_mode == "learned":
                    self.skip_station_pool[str(idx)] = StationAttentionPool(
                        skip_channels
                    )

        # Decoder
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

    def _station_mlp(self, channels: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (f"{name}_lin1", nn.Conv1d(channels, channels, kernel_size=1)),
                    (f"{name}_relu", nn.ReLU(inplace=True)),
                    (f"{name}_lin2", nn.Conv1d(channels, channels, kernel_size=1)),
                ]
            )
        )

    def _build_graph_norm(self, channels: int) -> nn.Module:
        if self.graph_norm_type == "graphnorm":
            return GraphNorm(channels)
        if self.graph_norm_type == "batchnorm":
            return nn.BatchNorm1d(channels)
        return nn.Identity()

    def _build_node_features(
        self,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
        B: Optional[int] = None,
        T_l: Optional[int] = None,
        volcano_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if self.node_feature_mode in {"geometry", "both"}:
            if (
                self.volcano_geom_nodes.numel() > 0
                and volcano_idx is not None
                and B is not None
                and T_l is not None
            ):
                v_idx = volcano_idx.to(device=device, dtype=torch.long).view(B)
                if int(v_idx.min().item()) < 0 or int(v_idx.max().item()) >= int(
                    self.volcano_geom_nodes.shape[0]
                ):
                    raise ValueError(
                        "volcano_idx contains out-of-range values for volcano_geom_nodes "
                        f"(num_volcanoes={int(self.volcano_geom_nodes.shape[0])})."
                    )
                geom_bt = self.volcano_geom_nodes.to(device=device, dtype=dtype)[v_idx]
                geom_batch = (
                    geom_bt.unsqueeze(1)
                    .expand(B, T_l, self.n_nodes, self.geom_feat_channels)
                    .reshape(num_graphs, self.n_nodes, self.geom_feat_channels)
                )
            else:
                geom_batch = self.geom_nodes.to(device=device, dtype=dtype)
                geom_batch = geom_batch.unsqueeze(0).expand(num_graphs, -1, -1)
            parts.append(geom_batch)
        if self.node_feature_mode in {"learned_station_embedding", "both"}:
            node_idx = torch.arange(self.n_nodes, device=device, dtype=torch.long)
            emb = self.station_id_embedding(node_idx).to(dtype=dtype)
            emb_batch = emb.unsqueeze(0).expand(num_graphs, -1, -1)
            parts.append(emb_batch)
        if not parts:
            return torch.empty(
                num_graphs,
                self.n_nodes,
                0,
                device=device,
                dtype=dtype,
            )
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
            return

        geom_bank = torch.as_tensor(volcano_geom_nodes, dtype=torch.float32)
        expected_shape = ("num_volcanoes", self.n_nodes, self.geom_feat_channels)
        if geom_bank.ndim != 3:
            raise ValueError(
                "volcano_geom_nodes must be a 3D tensor with shape "
                f"{expected_shape}. Got ndim={geom_bank.ndim}."
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

    def _compute_virtual_node_feature(
        self,
        x_flat: torch.Tensor,
        B: int,
        S: int,
        station_pool: Optional[StationAttentionPool],
    ) -> torch.Tensor:
        _, C, T_l = x_flat.shape
        x_bst = x_flat.reshape(B, S, C, T_l)
        if station_pool is None:
            return x_bst.mean(dim=1)
        station_bt_s_c = x_bst.permute(0, 3, 1, 2)
        pooled_bt_c, _ = station_pool(station_bt_s_c)
        return pooled_bt_c.permute(0, 2, 1)

    def _apply_node_norm(
        self,
        x_nodes: torch.Tensor,
        graph_norm: nn.Module,
        num_graphs: int,
    ) -> torch.Tensor:
        if isinstance(graph_norm, GraphNorm):
            batch = torch.arange(num_graphs, device=x_nodes.device).repeat_interleave(
                self.n_nodes
            )
            return graph_norm(x_nodes, batch=batch)
        if isinstance(graph_norm, nn.BatchNorm1d):
            return graph_norm(x_nodes)
        return x_nodes

    def _apply_station_norm(
        self,
        x_flat: torch.Tensor,
        graph_norm: nn.Module,
        B: int,
        S: int,
    ) -> torch.Tensor:
        if isinstance(graph_norm, nn.Identity):
            return x_flat
        _, C, T_l = x_flat.shape
        x_nodes = x_flat.reshape(B, S, C, T_l).permute(0, 3, 1, 2)
        x_nodes = x_nodes.reshape(B * T_l * S, C)

        if isinstance(graph_norm, GraphNorm):
            batch = torch.arange(B * T_l, device=x_nodes.device).repeat_interleave(S)
            x_nodes = graph_norm(x_nodes, batch=batch)
        elif isinstance(graph_norm, nn.BatchNorm1d):
            x_nodes = graph_norm(x_nodes)

        x_nodes = x_nodes.reshape(B, T_l, S, C).permute(0, 2, 3, 1)
        return x_nodes.reshape(B * S, C, T_l)

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
                "[UNet_GraphSAGE] station geometry configured "
                f"(provided={provided_count}, model_n_stations={self.n_stations}, "
                f"used={len(used_names)})"
            )
            print(
                "[UNet_GraphSAGE] crater lon/lat="
                f"({float(crater_coords[0]):.5f}, {float(crater_coords[1]):.5f})"
            )
            print("[UNet_GraphSAGE] station order used: " + ", ".join(used_names))

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

        # Cache normalized geometry features for all graph calls.
        xy_full = torch.cat([self.station_xy, self.network_xy], dim=0)
        dist_full = torch.cat([self.dist_to_crater, self.network_dist], dim=0)
        xy_norm = torch.linalg.norm(xy_full) + 1e-6
        dist_norm = torch.linalg.norm(dist_full) + 1e-6
        geom = torch.cat([xy_full / xy_norm, dist_full / dist_norm], dim=1)
        self.register_buffer("geom_nodes", geom)

        # Cache base star-graph topology (single graph, no batching offsets).
        edges = []
        network_idx = self.n_stations
        for i in range(self.n_stations):
            edges.append([i, network_idx])
            edges.append([network_idx, i])
        edge_index_base = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.register_buffer("edge_index_base", edge_index_base)

    def _build_station_graph(self):
        """Star graph: bidirectional station-network edges only."""
        return self.edge_index_base

    def _build_batched_edge_index(self, num_graphs: int) -> torch.Tensor:
        """Vectorized edge batching for repeated star graphs."""
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

    def _apply_graph_operation(
        self,
        graph_op: nn.Module,
        x_flat: torch.Tensor,
        B: int,
        S: int,
        graphnorm: Optional[nn.Module] = None,
        station_pool: Optional[StationAttentionPool] = None,
        volcano_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply configurable graph operation to station features.

        Args:
            graph_op : GraphSAGE or per-station MLP module for this level.
            x_flat : [B*S, C, T_l] station features for this level.
            B, S   : batch size and number of stations.
            graphnorm : optional normalization module applied after graph op.
            station_pool : optional station attention pooling module for virtual node init.

        Returns:
            station_out : [B*S, C, T_l] updated station features.
            network_out : [B, C, T_l] virtual node features (readout token).
        """
        _, C, T_l = x_flat.shape

        if self.graph_backend == "mlp":
            station_out = graph_op(x_flat)
            if graphnorm is not None:
                station_out = self._apply_station_norm(
                    station_out,
                    graphnorm,
                    B,
                    S,
                )
            network_out = self._compute_virtual_node_feature(
                station_out,
                B,
                S,
                station_pool,
            )
            return station_out, network_out

        if not self.use_message_passing:
            station_out = x_flat
            if graphnorm is not None:
                station_out = self._apply_station_norm(
                    station_out,
                    graphnorm,
                    B,
                    S,
                )
            network_out = self._compute_virtual_node_feature(
                station_out,
                B,
                S,
                station_pool,
            )
            return station_out, network_out

        num_graphs = B * T_l
        x_bst = x_flat.reshape(B, S, C, T_l)
        network_feature = self._compute_virtual_node_feature(
            x_flat,
            B,
            S,
            station_pool,
        ).unsqueeze(1)

        x_aug = torch.cat([x_bst, network_feature], dim=1)
        x_nodes_in = x_aug.permute(0, 3, 1, 2).reshape(num_graphs, self.n_nodes, C)
        node_feats = self._build_node_features(
            num_graphs,
            device=x_nodes_in.device,
            dtype=x_nodes_in.dtype,
            B=B,
            T_l=T_l,
            volcano_idx=volcano_idx,
        )
        x_with_node_feats = torch.cat([x_nodes_in, node_feats], dim=2)
        x_flat_nodes = x_with_node_feats.reshape(
            num_graphs * self.n_nodes,
            C + self.node_feat_channels,
        )

        edge_index_batch = self._build_batched_edge_index(num_graphs)
        x_out_nodes = graph_op(x_flat_nodes, edge_index_batch)
        if graphnorm is not None:
            x_out_nodes = self._apply_node_norm(
                x_out_nodes,
                graphnorm,
                num_graphs,
            )

        x_out = x_out_nodes.reshape(B, T_l, self.n_nodes, C).permute(0, 2, 3, 1)
        station_out = x_out[:, :S].reshape(B * S, C, T_l)
        network_out = x_out[:, S]
        return station_out, network_out

    def _apply_bottleneck_attention(self, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Temporal MHSA block over bottleneck features.

        Args:
            x_dec: [B, C, T_b]

        Returns:
            [B, C, T_b]
        """
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
        """Readout equivalent to virtual-node initialization without message passing."""
        _, C, T_l = x_flat.shape
        x_bst = x_flat.reshape(B, S, C, T_l)
        return x_bst.mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        volcano_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, T]

        Returns:
            out: [B, C_out, T]
        """
        B, S, T = x.shape

        x_flat = x.reshape(B * S, 1, T)

        encodings: list = []
        for i in range(self.depth):
            x_flat = self.encoder_list[i](x_flat)

            if str(i) in self.encoder_graph_op:
                encoder_station_pool = (
                    self.encoder_station_pool[str(i)]
                    if str(i) in self.encoder_station_pool
                    else None
                )
                x_flat, _ = self._apply_graph_operation(
                    self.encoder_graph_op[str(i)],
                    x_flat,
                    B,
                    S,
                    self.encoder_graphnorm[str(i)],
                    encoder_station_pool,
                    volcano_idx,
                )

            encodings.append(x_flat)
            x_flat = self.pool_list[i](x_flat)

        _, x_dec = self._apply_graph_operation(
            self.graph_op_bottleneck,
            x_flat,
            B,
            S,
            self.graphnorm_bottleneck,
            self.station_pool_bottleneck,
            volcano_idx,
        )
        if self.use_bottleneck_attention:
            x_dec = self._apply_bottleneck_attention(x_dec)

        for i in range(self.depth):
            x_dec = self.upconv_list[i](x_dec)

            skip = encodings[-(i + 1)]
            enc_level = self.depth - 1 - i
            if str(enc_level) in self.skip_graph_op:
                skip_station_pool = (
                    self.skip_station_pool[str(enc_level)]
                    if str(enc_level) in self.skip_station_pool
                    else None
                )
                _, skip = self._apply_graph_operation(
                    self.skip_graph_op[str(enc_level)],
                    skip,
                    B,
                    S,
                    self.skip_graphnorm[str(enc_level)],
                    skip_station_pool,
                    volcano_idx,
                )
            else:
                skip = self._virtual_node_init_readout(
                    skip,
                    B,
                    S,
                )

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
