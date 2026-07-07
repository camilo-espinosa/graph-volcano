Here's the task text you can hand to your agent.

---

# Task: Refactor `models/UNet_MPNN.py` into a clean, geometry-free, interpretable multistation segmentation model

## Goal

Rewrite `models/UNet_MPNN.py` as a new, clean model. Remove all the geometry/edge/ablation machinery we have already tested and concluded on. The new model keeps the 1D U-Net skeleton but replaces the expensive per-time-bin bottleneck-only message passing with **cheap, permutation-equivariant station fusion in the early encoder levels** plus an **interpretable station-attention bottleneck**. Output stays `[B, 6, T]`.

Do not preserve backward compatibility. Do not keep any of the removed switches "just in case." Produce a clean file.

## Core design principles (why the refactor)

1. Station operations must be **permutation-invariant / equivariant** (no fixed station order, no per-station-index weights). Achieve this only via **shared functions over stations/pairs + symmetric aggregation** (mean/max/attention). Never via a convolution kernel that has a distinct weight per station row.
2. Station fusion should happen **early and multi-scale** (like the 2D U-Net's strength), but implemented cheaply as a shared Conv2d over station pairs + symmetric pooling, NOT as PyG `MessagePassing` at high temporal resolution.
3. The model must be **geometry-free**: no coordinates, no edge features, no per-volcano geometry. Our results showed geometry hurts cross-volcano transfer.
4. The model must expose **interpretability artifacts**: temporal attention weights and a single station-importance scalar per station per window.
5. Handle zero-padded stations correctly via a **validity mask** everywhere stations are reduced.

---

## PART 1 — REMOVE the following entirely

Remove all of this code and every reference to it (constructor args, buffers, helper methods, forward-path branches, validation blocks):

### Geometry / coordinates
- Remove constructor args: `station_coords`, `crater_coords`, `volcano_geom_nodes`.
- Remove `_register_station_geometry` entirely.
- Remove `_register_volcano_geometry_bank` entirely.
- Remove all geometry buffers: `station_xy`, `dist_to_crater`, `network_xy`, `network_dist`, `geom_nodes`, `node_xy_norm`, `volcano_geom_nodes`, `volcano_edge_attr_static`.
- Remove `geom_feat_channels`.

### Edge features / edge machinery
- Remove constructor args: `graph_topology`, `edge_feature_mode`, `xcorr_feat_dim`.
- Remove all edge construction: `edge_index_base`, `edge_attr_base`, `n_station_pairs`, the virtual-node edge logic, and everything building station-station / station-virtual edges.
- Remove `_build_batched_edge_index`, `_build_batched_edge_attr_static`, `_build_batched_edge_attr_dynamic`, `_build_edge_attr`.
- Remove the `edge_attr_dynamic` argument from `forward` and from all internal calls.
- Remove all xcorr / dynamic-edge handling.

### Node features
- Remove constructor args: `node_feature_mode`, `use_rsam_node_feat`.
- Remove `_build_node_features` entirely.
- Remove `node_feat_channels` and the RSAM node-feature path.
- Remove the `rsam` argument from `forward`.

### Per-volcano routing
- Remove the `volcano_idx` argument from `forward` and from every internal method. The model no longer knows about volcanoes.

### Old graph plumbing we are replacing
- Remove the PyG `EdgeMPNN` / `EdgeMPNNLayer` **edge-conditioned** message passing as the encoder/bottleneck mechanism. (See PART 2 for what replaces it. If you keep any MPNN at all, it is only an optional geometry-free bottleneck variant — but by default the bottleneck uses station-attention pooling, not MPNN. Prefer removing PyG entirely unless the optional bottleneck-MPNN switch is implemented cleanly.)
- Remove `edge_dim` / edge_attr paths from any retained message code.
- Remove the virtual-node concept entirely (`n_nodes`, virtual-node init, virtual-node readout). Station pooling replaces it.

### Old switches / ablation knobs (all removed)
- Remove: `graph_levels`, `use_skip_graph`, `skip_graph_levels`, `mpnn_aggr`, `mpnn_layers`, `mpnn_hidden_dim`, `graph_norm`, `pairwise_conv_levels`, `pairwise_conv_kernel`, `pairwise_conv_aggr`.
- Remove `encoder_graph_op`, `encoder_graphnorm`, `skip_graph_op`, `skip_graphnorm`, `graph_op_bottleneck`, `graphnorm_bottleneck`, `_build_mpnn`, `_build_graph_norm`, `_apply_node_norm`, `_apply_mpnn`, `_virtual_node_init_readout`.
- Remove the existing `PairwiseConv2d` class as-is (we re-implement a cleaner equivariant fusion block in PART 2; you may reuse the directional-pair idea but not the current memory-heavy 56-pair materialization as the default).

### Imports
- Remove `numpy` if no longer used, remove `from torch_geometric.nn import MessagePassing` (unless the optional bottleneck-MPNN switch is implemented — otherwise drop PyG completely).

---

## PART 2 — ADD the following

### New constructor signature (keep it small)

```
UNet_MPNN(
    in_channels: int = 1,
    out_channels: int = 6,
    init_features: int = 16,
    depth: int = 5,
    n_stations: int = 8,
    station_fusion_levels: list[int] = [0, 1, 2],   # encoder levels that run cheap station fusion
    fusion_kernel: int = 9,                          # temporal kernel for the pair conv (odd)
    readout_mode: str = "attention",                 # {"mean", "max", "attention"} station pooling at bottleneck
    use_bottleneck_attention: bool = True,
    bottleneck_attn_heads: int = 4,
    bottleneck_attn_dropout: float = 0.0,
    bottleneck_attn_ff_mult: int = 2,
    feature_dropout: float = 0.0,                     # dropout in conv blocks + before conv_final
    return_attention: bool = False,                  # if True, forward also returns interpretability dict
)
```

Validate: `fusion_kernel` odd; `readout_mode` in the allowed set; `station_fusion_levels` all in `[0, depth)`; bottleneck channels divisible by heads.

### Validity mask (compute once in forward)
- From input `x` of shape `[B, S, T]`, compute `station_valid = (x.abs().sum(dim=-1) > 0)` → `[B, S]` boolean.
- Thread this mask into every station-reduction: the fusion aggregation, the bottleneck station pooling, and the decoder skip readout.
- Padded stations: excluded from mean; set to `-inf` before max and before softmax so they get zero weight.

### Permutation-equivariant station fusion block (the new early-level primitive)
Add a module (e.g. `StationFusion`) applied at each level in `station_fusion_levels`, operating on `[B, S, C, T]`:

- For each station, build `[own_features ∥ context_features]` where `context` is a **symmetric pool over the OTHER stations** using both max and mean (concat both contexts). Use the validity mask so padded stations never contribute to the pool and padded stations' own outputs are left at zero.
  - Preferred cheap form: context = masked max over other stations AND masked mean over other stations, concatenated → per station input is `[C (own) + C (max-ctx) + C (mean-ctx)]`.
  - Optional (more expressive) form: keep the directional-pair idea from the old `PairwiseConv2d` (shared height-2 Conv2d over each source→destination pair, then symmetric aggregate over incoming pairs). If implemented, do it memory-efficiently and keep it permutation-equivariant (shared kernel across all pairs + symmetric aggregation). This can be a second selectable fusion type, but the masked max+mean-context form is the default.
- Apply a shared temporal conv (kernel = `fusion_kernel`, padded to preserve T) + BatchNorm + ReLU, then a `1x1` conv to project back to C. Residual-add to the station's own features.
- Must be permutation-equivariant: no weight may depend on station index. Same weights for every station.
- Cost target: this must be cheap enough to run at level 0 (T=8192). It is a conv-cost operation, not a per-time-bin scatter/gather.

### Encoder
- Keep the existing `_block_1d` double-Conv1d encoder blocks.
- After each encoder block whose level is in `station_fusion_levels`, apply `StationFusion`.
- Keep `MaxPool1d` downsampling between levels.
- Add `feature_dropout` (as `nn.Dropout`) inside `_block_1d` after the ReLUs (or right after each block) so it applies at every level.

### Bottleneck (the interpretable core)
Operate on bottleneck station features `[B, S, C, T_b]` (T_b = 512 at depth 5):

1. **Temporal self-attention** (existing MHSA block), applied per station over the time axis. Keep the pre-norm + FF residual structure. Change `need_weights=False` to `need_weights=True` and capture the temporal attention weights when `return_attention=True`.
2. **Station pooling → network vector `[B, C, T_b]`**, controlled by `readout_mode`:
   - `"mean"`: masked mean over stations.
   - `"max"`: masked max over stations.
   - `"attention"` (default and the interpretable one):
     - Summarize each station over time (masked mean or the temporal-attention output) → `[B, S, C]`.
     - Small MLP `C → 1` producing **one scalar score per station** → `[B, S]`.
     - Mask padded stations to `-inf`, softmax over S → **one weight per station per window** → `[B, S]`.
     - Weighted sum of station vectors (broadcast weight over time) → `[B, C, T_b]`.
     - Save these `[B, S]` station weights for interpretability.
- The pooled network vector `[B, C, T_b]` is what feeds the decoder (replacing the old virtual-node readout).

### Decoder
- Keep the existing ConvTranspose1d + double-conv decoder with U-Net skips.
- The skip at each level currently came from station-wise encoder features; collapse them to `[B, C, T]` with a **masked symmetric readout** consistent with `readout_mode` (masked mean/max; for attention mode you may reuse masked mean for skips to keep it simple — document the choice). Apply `feature_dropout` in decoder blocks too.
- Final `conv_final` (1x1 Conv1d) → `[B, 6, T]`.

### Interpretability return
- When `return_attention=True`, `forward` returns `(out, attn)` where `attn` is a dict containing at least:
  - `station_weights`: `[B, S]` (the bottleneck station-attention scalars; present only when `readout_mode="attention"`).
  - `temporal_weights`: the bottleneck temporal attention map (as returned by MHSA).
  - `station_valid`: `[B, S]` mask.
- When `return_attention=False`, `forward` returns just `out` `[B, 6, T]` (default, so training loops are unaffected).

---

## PART 3 — forward signature (final)

```
forward(x: [B, S, T], return_attention: bool = False)
    -> out [B, 6, T]
    or (out, attn_dict) if return_attention=True
```

No `edge_attr_dynamic`, no `rsam`, no `volcano_idx`. Those are gone.

---

## PART 4 — constraints / acceptance

- Output shape `[B, 6, T]` unchanged; default `forward(x)` returns only the tensor so existing training code keeps working.
- No geometry, no edges, no volcano indices anywhere.
- Every station reduction respects `station_valid` (mean excludes padded, max/softmax use `-inf` for padded).
- All station operations are permutation-invariant/equivariant; verify by a quick internal check (shuffling the station dim of a valid all-ones input permutes `station_weights` correspondingly and leaves the pooled output unchanged up to that permutation).
- Keep it lightweight: target ~3M params at `init_features=16, depth=5`. The early fusion must be conv-cost (runnable at T=8192 with a normal batch size, not batch_size=4).
- Clean module docstring describing: geometry-free, permutation-invariant early fusion, interpretable station-attention bottleneck, masking, dropout. Remove the old docstring's switch table and replace with the new one.
- Update the registry (`utils/model_registry.py`) minimally so the model still constructs (new kwargs only), but you can leave the registry ablation entries for a later task — this task is the model file itself.

---

## Notes for the agent
- Do not keep removed switches as no-ops. Delete them.
- Do not reintroduce fixed-order station convolutions (no `Conv2d((S, kT))` full-height kernel with per-row weights).
- If unsure whether to keep the optional bottleneck MPNN, DROP it and drop PyG. Default bottleneck is station-attention pooling.
- Prefer clarity over cleverness; this file will be read for a paper.

---

