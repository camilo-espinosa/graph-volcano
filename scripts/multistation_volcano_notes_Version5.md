# Multi-Station Volcano-Seismic Models — Working Notes

## 0. TL;DR
- On NChVC the graph machinery is inert: **attention carries all in-domain gain**; message passing without edge
  information does nothing (`no_message_passing` ≈ `mlp_backend` ≈ star-GraphSAGE, all ~0.88).
- **Best in-domain graph model = `no_norm` config: ties baseline (0.90±0.010) at 3.02M params (39% of baseline 7.78M).**
  Removing GraphNorm — not adding capacity — is the fix (it normalized over only ~9 redundant nodes).
- Retrained baselines still win cross-volcano **classification** (`unet_attention` best on CAU/LDM/VCA).
- Root cause: intra-edifice stations are near-redundant; the discriminative signal lives in **station-pair (edge)
  relations** (moveout, amplitude gradient, polarity) that node-only message passing cannot see.
- **Decision: skip further GraphSAGE; go to edge-conditioned layers.** Lead = `edge_mpnn`; the key test is whether
  edge features rescue the graph. Reframe the science story around representation learning + unlabeled archive.

## 1. Context & Scope
- Replace patch-stacking 2-D representation (UNet baseline; paper F1 0.91 / IoU 0.88 on NChVC) with a representation
  honoring stations as an unordered, geometry-bearing set.
- **Volcano (network)-level evaluation only** (no per-station labels yet); readout collapses [B*S,C,T] -> [B,C,T].
- Tests mirror the reference: NChVC fit ✓, zero-shot cross-volcano ✓, fine-tuned (TODO), continuous (TODO).
- 5-fold CV; cross-volcano per fold; means ± std. Training loop/augmentation changed vs paper, so UNet and
  UNet+attention were **retrained** — always compare vs these, NOT the paper's 0.91/0.88.

## 2. Core hypothesis
- Node-only multi-station fusion is redundant at intra-edifice apertures (proven: message passing inert).
- **The only fusion likely to help is edge-conditioned**, putting station-pair relationships (starting with
  Δposition) on the edges. An MPNN (`NNConv`/edge-MLP) is the right tool; GATv2 (edge-weighting) is a weaker test;
  a station-Transformer is the "is a graph needed at all?" foil.
- IoU is secondary: report it, but build no claim on the small in-domain edge.

## 3. Decisions locked
- **Skip further GraphSAGE/star work**; move directly to edge-conditioned layers. ✔
- Keep `no_norm` config and `v5_full` only as reference points (best graph in-domain; and proof GraphNorm hurts). ✔
- Skip IoU-artifact verification; skip dedicated noise experiment (conditional only). ✔
- No new data. Aperture is a *prediction for edge models only* → test as a free post-hoc **only if an edge model works**
  (existing apertures: NVCHVC ~5 km, CAU ~11 km, LDM ~13 km, VCA ~16 km). ✔
- Edge features: **Δposition is the default (free, the point of the layer).** Cross-correlation lag/coherence is
  **optional**, tried only if Δposition-based `edge_mpnn` shows life (one FFT op/pair, precomputed & cached → cheap).

## 4. Where to put the graph op (applies to all new backbones)
- **Bottleneck only**, to start. Your ablations showed extra encoder/skip graph levels add cost, not benefit, and
  bottleneck-only already matched full placement. It's also cheapest (T=512 at depth 5 vs 8192 at input).
- **One caveat for edge models only:** moveout needs high time resolution, which the bottleneck (~2.5 s/sample) may
  have lost. So run a shallow-level placement **only if** the bottleneck version ties the baseline. Don't sweep levels.

## 5. FINAL ablation set
### Group A — Baselines & anchors (rename)
| Final name | Was | Role |
|---|---|---|
| `unet_baseline` | Unet (baseline) | reference architecture (retrained) |
| `unet_attention` | Unet + bottleneck attention | matched control + honest best (cross-volcano classification leader) |
| `graphsage_star` | `no_norm` config | naive-graph anchor (best graph in-domain, 0.90 @ 3.02M) |
| `graphsage_star_withnorm` | `v5_full` | kept only to show GraphNorm hurts (0.88) |

### Group B — Mechanism ablations (rename)
| Final name | Was | Question it answers |
|---|---|---|
| `no_attention` | ablation_4_no_bottleneck_attention | is attention necessary? (yes) |
| `no_message_passing` | ablation_3_no_message_passing | does message passing help beyond pooling? (no) |
| `mlp_backend` | ablation_2_mlp_backend | does any cross-station coupling help? (no) |
| `graph_only` | only_graph_no_attention | graph-alone lower bound |

### Group C — Controls (one run each)
| Final name | Role |
|---|---|
| `no_spatial_info` | node features zeroed/random — if it ties `graphsage_star`, geometry is proven unused |

### Group D — New backbones (lead → secondary → foil)
| Backbone | Graph | Layer | Edge features | Role |
|---|---|---|---|---|
| `edge_mpnn` | fully-connected | edge-conditioned (NNConv/edge-MLP) | Δposition (+ optional lag/coherence) | **LEAD — can edges rescue the graph?** |
| `gat_fc` | fully-connected | GATv2 | edge-weighting only | secondary (weaker edge test) |
| `station_transformer` | — | self-attention over ≤8 station tokens | relative position | optional foil ("is a graph needed?") |

### Group E — Targeted ablations per backbone (≤3 runs each, not a full sweep)
| Backbone | Ablation | Isolates |
|---|---|---|
| `edge_mpnn` | `__bottleneck` (run first) | cheap default placement |
| `edge_mpnn` | `__encoder` (only if bottleneck ties baseline) | does high-res moveout matter? |
| `edge_mpnn` | `__no_edge_feats` | **THE key test: do edges add anything?** |
| `edge_mpnn` | `__xcorr` (optional) | does lag/coherence beat Δposition alone? |
| `gat_fc` | `__no_edge_feats` | does edge-weighting help GAT? |
| `station_transformer` | `__no_relpos` | does spatial encoding help attention? |
| winner | `+rsam_node_feat` | does per-station RSAM (signal-derived, label-free) help? |

### Dropped (supplement only)
All remaining internal GraphSAGE-design ablations (norm type, pooling mode, skip-graph, learned-embedding, all-levels,
bigger/with-level-2 kept only as a capacity reference). `ablation_8` == `only_bottleneck_attention` — merge.

## 6. Scientific reframing (Paper 2)
Frame the model as a **scientific instrument**, not a better classifier ("within noise of baseline at <45% params, AND
interpretable in ways patch stacking is not"). Must deliver ≥2 concrete results: station-attention weights vs
distance/SNR; learned station embeddings recover geometry (unsupervised); temporal attention differs by class;
(core) cluster/anomaly-detect over the **unlabeled Chilean archive**.

## 7. Long-term: 1-D detection (start/end/class) instead of segmentation
Removes the reference paper's heavy heuristics (station-sum, binarize, BG-transition, <2.5 s merge, duration filter,
RSAM). Not trivial — it's temporal action detection. Risks: set matching (anchors → DETR-1D later), end-time regression
is hard (evaluate with IoP), small/rare events need focal loss, overlaps need overlap labels, sparse supervision vs
dense masks (~7,500 windows). De-risked path: segmentation backbone + anchor detection head + auxiliary segmentation
loss → IoP eval → DETR-1D. Reuse the graph/attention front-end. Synergy: per-station + overlap-capable detection is
where the graph finally becomes indispensable.

## 8. Target papers
### Paper 1 — model-centric / technical (IEEE TGRS) [write first]
Controlled comparison of fusion strategies under identical 5-fold CV + cross-volcano + continuous.
- Optimistic: "edge-conditioned fusion improves cross-volcano detection robustness and matches/exceeds patch stacking
  at a fraction of params; benefit scales with aperture."
- Pessimistic (std-backed today): "intra-edifice station graphs are redundant (topology-agnostic); patch-stacking's
  gain is achieved more cheaply with lightweight temporal attention, and removing GraphNorm ties the baseline at 39% params."
- Requirements: cross-fold std ✓; matched UNet+attention control ✓; `no_spatial_info` control; continuous eval;
  conditional aperture post-hoc.
### Paper 2 — science / representation-learning (JGR: Solid Earth optimistic; SRL pessimistic; also Computers & Geosciences, AI in Geosciences)
"Deep multi-station models as scientific instruments" + scaling to the unlabeled Chilean archive. Cites Paper 1 for validation.

## 9. Roadmap (no deadline; gated — stop anytime with a complete story)
**Phase 0 — solidify current study (cheap):**
- [ ] Promote `no_norm` to `graphsage_star`; keep `v5_full` as `graphsage_star_withnorm` (GraphNorm-hurts result).
- [ ] Add `no_spatial_info` control; apply Group A/B renames.
- GATE: current work = complete (pessimistic, std-backed) TGRS submission.

**Phase 1 — edge-conditioned graphs (MAIN INTEREST):**
- [ ] `edge_mpnn__bottleneck` with Δposition edges (run first).
- [ ] `edge_mpnn__no_edge_feats` (THE key test) and `edge_mpnn__encoder` (only if bottleneck ties baseline).
- [ ] `gat_fc` (+ `__no_edge_feats`) as secondary; `station_transformer` (+ `__no_relpos`) as optional foil.
- [ ] `__xcorr` and `+rsam_node_feat` only on whatever shows life.
- GATE: any edge-aware variant beats `graphsage_star`/`unet_attention` somewhere → carry forward; else lock the
  pessimistic framing and lean on Phase 3 (interpretability) for Paper 2.

**Phase 2 — finish reference battery (best model + baselines):**
- [ ] Mixed fine-tuning 1/5/10/20% (+ class completion).
- [ ] Continuous 10-h trace, class-specific + class-agnostic, IoP. (Most likely place a graph genuinely wins — high-IoU profile.)

**Phase 3 — conditional analyses (free):**
- [ ] Aperture post-hoc (per-model performance vs per-volcano aperture) — only if an edge model works.
- [ ] Noise sweep — only if an edge model wins (coherence is expected to be more noise-robust).

**Phase 4 — interpretability + unlabeled (bridge to Paper 2):**
- [ ] Station-attention vs distance/SNR; embeddings recover geometry; temporal attention by class.
- [ ] (Stretch) cluster/anomaly over an unlabeled archive slice — seed of Paper 2.

## 10. Status
- [x] NChVC fit — best graph = `no_norm` (0.90 / 3.02M); attention carries the gain; GraphNorm hurts.
- [x] Zero-shot cross-volcano — baseline best classifier; graph shows high-IoU/low-F1; no aperture benefit for naive graph.
- [ ] Phase 0: promote `no_norm`, add `no_spatial_info`, renames.
- [ ] Phase 1: edge-conditioned backbones (start `edge_mpnn__bottleneck`).
- [ ] Phase 2: fine-tuned + continuous.
- [ ] Phase 3: conditional aperture + noise.
- [ ] Phase 4: interpretability + unlabeled demo.