# Cross-Volcano Model Requirements (Temporary)

Scope:
- Source registry: utils/model_registry.py
- Source data-flow script: scripts/04_cross-volcano.py
- Source model forward signatures: models/UNet.py, models/UNet_bottleneck_attention.py, models/UNet_GraphSAGE.py, models/UNet_MPNN.py
- Source dataset/eval plumbing: utils/train_utils.py

Legend:
- Waveforms: raw multistation waveform tensor x [B,S,T]
- Volcano geometry: per-volcano node/edge geometry selected via volcano_idx and volcano_geom_nodes
- Extra payload: descriptor payload and/or precomputed edge_attr_dynamic

## Active MODEL_REGISTRY entries and requirements

| Model key | Family | Needs waveforms | Needs volcano geometry | Needs extra payload | Why |
|---|---|---|---|---|---|
| unet | unet | Yes | No | No | Plain 2D UNet forward(x) only. |
| unet_bottleneck_attention | unet | Yes | No | No | UNet + bottleneck MHSA, still forward(x) only. |
| ablation_2_mlp_backend | graphsage | Yes | No (effective) | No | graph_backend=mlp bypasses graph message passing and node features. |
| ablation_3_no_message_passing | graphsage | Yes | No (effective) | No | use_message_passing=False bypasses graph op and node features. |
| ablation_4_no_bottleneck_attention | graphsage | Yes | Yes | No | Graph operations still active; geometry-based node features used. |
| ablation_5_no_norm | graphsage | Yes | Yes | No | Same as base GraphSAGE with graph_norm_type=none only. |
| only_graph_no_attention | graphsage | Yes | Yes | No | graph_levels=[] but bottleneck graph op still runs with geometry features. |
| ablation_11_no_node_features | graphsage | Yes | No | No | node_feature_mode=none disables geometry/embedding node features. |
| edge_mpnn__bottleneck | mpnn | Yes | Yes | No | Bottleneck MPNN uses static edge attrs from volcano geometry + node geometry. |
| edge_mpnn__xcorr | mpnn | Yes | Yes | Yes (edge_attr_dynamic precomputed) | Requires fold-specific edge_attr_dynamic [B, n_station_pairs, xcorr_feat_dim] plus volcano-specific geometry. |
| pairwise_conv2d__l0 | mpnn | Yes | Yes | No | Pairwise conv at level 0 + bottleneck MPNN still uses geometry-derived edge attrs. |
| edge_mpnn__early_l0 | mpnn | Yes | Yes | No | Early level MPNN + bottleneck MPNN use geometry-derived attrs. |
| edge_mpnn__aggr_max | mpnn | Yes | Yes | No | Same requirements; aggregation changes only. |
| edge_mpnn__layers_4 | mpnn | Yes | Yes | No | Same requirements; depth changes only. |
| edge_mpnn__early_l2 | mpnn | Yes | Yes | No | Same requirements; graph level placement changes only. |
| edge_mpnn__early_l1 | mpnn | Yes | Yes | No | Same requirements; graph level placement changes only. |
| edge_mpnn__both_l2_bottleneck | mpnn | Yes | Yes | No | Same requirements; graph level placement changes only. |
| edge_mpnn__no_edge_feats | mpnn | Yes | Yes | No | edge_feature_mode=none, but node_feature_mode=geometry still needs per-volcano node geometry. |
| edge_mpnn__encoder | mpnn | Yes | Yes | No | Multiple encoder levels + bottleneck MPNN use geometry-derived attrs. |
| edge_mpnn__star_topology | mpnn | Yes | Yes | No | Star topology still uses geometry (delta_pos) on star edges. |
| edge_mpnn__no_spatial_info | mpnn | Yes | No | No | edge_feature_mode=none and node_feature_mode=none removes spatial inputs. |
| edge_mpnn__rsam | mpnn | Yes | Yes | RSAM computed on the fly | RSAM node feature required; script computes RSAM inside wrapper. |
| edge_mpnn__no_attention | mpnn | Yes | Yes | No | Bottleneck temporal attention disabled only; MPNN geometry inputs still required. |

## Cross-volcano script expectations to satisfy requirements

- Graph/MPNN models are instantiated with volcano_geom_nodes=volcano_geom_bank.
- CrossVolcanoLOODataset is created with return_volcano_idx=True for all graph/MPNN models.
- Batch extraction must pass volcano_idx into model forward kwargs.
- For RSAM ablation, GraphForwardWrapper injects rsam into forward kwargs.
- For xcorr-style models (if enabled), edge_attr_dynamic must be forwarded from descriptor payload to model forward kwargs.

## Quick conclusion

- Current active registry entries are fully satisfiable with waveforms + volcano_idx + volcano_geom_nodes, except that dedicated verification logs should explicitly prove per-sample volcano_idx routing and geometry lookup.
- There is a latent forwarding gap for edge_attr_dynamic if xcorr ablations are re-enabled.
