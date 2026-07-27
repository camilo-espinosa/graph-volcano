# MuSSeg Late-Attention Ablation Grid

This note explains the baseline MuSSeg model first, then shows how each ablation changes one assumption in that baseline.

## Baseline: `musseg_pi_se_lsa_ba`

`musseg_pi_se_lsa_ba` is the baseline shared-station MuSSeg model with late station attention at the bottleneck.

The input is a multistation 1D signal shaped like `[B, S, T]`, where `B` is the batch size, `S` is the number of stations, and `T` is the time axis. The model processes every station with the same encoder weights, so the early feature extraction is shared across stations rather than learned separately for each one.

### One Forward Pass

The baseline forward pass works like this:

1. The input is reshaped to `[B, S, 1, T]` so each station can be treated as its own channel-group.
2. A shared convolutional encoder is applied station by station. Each station gets the same filters, so the model learns the same seismic feature extractor for all stations.
3. Through the down path, the model keeps the station axis, so intermediate features stay shaped like `[B, S, C, T]`.
4. At the bottleneck, the model compresses time with a mean, turning `[B, S, C, T]` into `[B, S, C]`.
5. The station-merge modules score each station and convert those scores into weights `w` with shape `[B, S]`.
6. The model collapses the station axis into a single `[B, C, T]` bottleneck tensor.
7. The decoder upsamples that merged representation back to the original temporal resolution.

### What The Station-Merge Modules Do

The bottleneck merge is not a generic self-attention block. It is a simple station-ranking mechanism made from two existing layers:

- `station_merge_attn_norm`: a `LayerNorm` that normalizes each station’s pooled bottleneck vector.
- `station_merge_attn_score`: a linear layer that converts each station vector into one scalar score.

Concretely, if the bottleneck tensor is `[B, S, C, T]`, the model first averages over time to get `[B, S, C]`. Then it normalizes each station vector, scores each station independently, and softmaxes the scores across stations. The result is `w`, a per-example station weighting vector of shape `[B, S]`.

The meaning of `w` is: which stations the model believes matter most for this example, after it has already built a station-wise bottleneck representation.

The baseline assumption is simple: stations should be merged once, at the bottleneck, after the model has extracted station-wise features. All skip connections are collapsed in the simplest possible way, by taking the max over stations before they enter the decoder.

## A More Concrete View Of The Baseline

Think of one example with three stations. The model first learns station-specific representations, but with shared filters. If one station is clearly more informative for a given event, the bottleneck merge can give that station a higher weight. If another station is mostly noise, it gets downweighted.

That is the main idea of `lsa_ba`: learn station importance at the bottleneck, then decode a single fused representation.

## What The New Ablations Change

The new models keep the same backbone. Each ablation changes only one additional assumption, so the comparison stays clean.

### Distance-Aware Ablations

These variants add geographic information about the volcano station layout.

The model already computes a per-station distance score from the station coordinates and crater coordinates. The idea is simple: stations that are geographically closer to the crater should be treated differently from stations that are farther away.

The distance signal is built from the station metadata in `get_station_coords(volcano_name)` and the crater location from `get_crater_coords(volcano_name)`. For each station, the code:

1. Reads its longitude and latitude.
2. Computes an approximate kilometer offset from the crater using a local flat-earth approximation.
3. Converts longitude degrees to kilometers with `111 * cos(mean_latitude)` and latitude degrees to kilometers with `111`.
4. Combines the two offsets with Euclidean distance.
5. Converts that raw distance into a continuous closeness score with a normalized distance fraction:

$$
u_i = \frac{d_i - d_{\min}}{d_{\max} - d_{\min}}, \quad
station\_dist_i = 1 - \left(1 - \frac{1}{s}\right) u_i
$$

where `d_i` is the station-to-crater distance for station $i$, `d_{\min}` is the closest station distance, `d_{\max}` is the farthest station distance, and $s$ is the number of stations.

This keeps the endpoints exact by construction: if $u_i=0$ (closest station), then $station\_dist_i=1$; if $u_i=1$ (farthest station), then $station\_dist_i=1/s$.

That makes the stored `station_dist` buffer a closeness score rather than a rank. The closest station gets score `1.0`, and the farthest station gets score `1 / s`. Every other station is placed proportionally between those two endpoints using its actual distance, so the score depends on geometry rather than ordinal rank.

That score is used in three places when a distance variant is active (that is, when `use_distance_attn_bias=True` or `use_distance_bottleneck_emb=True`):

- `use_distance_attn_bias=True` adds a distance-based bias inside late station attention, so stations that are geographically closer or farther can influence each other differently during the attention step.
- `use_distance_bottleneck_emb=True` adds a learned embedding of station distance at the bottleneck, after the late station-wise bottleneck attention but before the final station collapse.
- All station merges use a distance prior mixed with learned station logits through a learnable scalar `distance_merge_gamma`:

$$
merge\_logits = learned\_logits + \gamma \cdot \log(distance\_prior)
$$

and then a softmax over stations.

When distance attention bias is enabled, the model does one extra step at the station-attention levels: it forms pairwise station distance differences from the normalized station scores, runs those differences through a tiny learned linear projection, and uses the result as the attention mask/bias. That means the attention module does not just see station features; it also sees how far apart the stations are relative to the crater-based ordering.

The assumption here is that station geography carries useful prior structure in both interaction and merging. The shared encoder itself is unchanged.

### Weighted-Skip Ablations

`use_station_weighted_skips=True` changes how skip connections are merged.

In the baseline, each skip tensor is collapsed with a max over stations before being passed to the decoder. In distance-enabled variants, full `[B, S, C, T]` skips are kept and merged with the same distance-aware merge rule used at the bottleneck.

At the bottleneck, the model computes station logits from pooled features using the same station-merge modules described above:

1. `pooled = x.mean(dim=-1)` gives `[B, S, C]`.
2. `pooled = station_merge_attn_norm(pooled)`.
3. `scores = station_merge_attn_score(pooled).squeeze(-1)` gives `[B, S]`.
4. `w = softmax(scores + gamma * log(distance_prior), dim=1)` for distance-enabled variants, and `w = softmax(scores, dim=1)` otherwise.

This is the same station ranking logic used by the bottleneck merge. The new part is that weighted-skip variants reuse this bottleneck-derived signal for skip fusion, while distance-enabled variants inject distance priors at every merge point.

In weighted-skip variants, each decoder skip is collapsed with a learned merge and a base merge:

$$
weighted = \sum_s w_s \cdot skip_s
$$

$$
base = distance\_aware\_merge(skip)
$$

$$
skip\_merged = \alpha \cdot weighted + (1 - \alpha) \cdot base
$$

where `alpha` is the learnable scalar `skip_merge_alpha`, clamped to `[0, 1]` at use time.

The assumption here is that bottleneck station importance should also control skip fusion when weighted skips are enabled. In distance-enabled variants, station geometry also enters every merge through the learnable `gamma`-scaled distance prior.

## Model Grid

The comparison set is:

| model key | display name | dist_attn | dist_emb | weighted_skip |
|---|---|---:|---:|---:|
| `musseg_pi_se_lsa_ba` | `MuSSeg_PI_SE_LateStationAttention_BA` | - | - | - |
| `musseg_pi_se_lsa_ba_dist_attn` | `MuSSeg_PI_SE_LSA_BA_DistAttn` | ✓ | - | - |
| `musseg_pi_se_lsa_ba_dist_emb` | `MuSSeg_PI_SE_LSA_BA_DistEmb` | - | ✓ | - |
| `musseg_pi_se_lsa_ba_dist_both` | `MuSSeg_PI_SE_LSA_BA_DistBoth` | ✓ | ✓ | - |
| `musseg_pi_se_lsa_ba_wskip` | `MuSSeg_PI_SE_LSA_BA_WeightedSkip` | - | - | ✓ |
| `musseg_pi_se_lsa_ba_dist_attn_wskip` | `MuSSeg_PI_SE_LSA_BA_DistAttn_WeightedSkip` | ✓ | - | ✓ |
| `musseg_pi_se_lsa_ba_dist_emb_wskip` | `MuSSeg_PI_SE_LSA_BA_DistEmb_WeightedSkip` | - | ✓ | ✓ |
| `musseg_pi_se_lsa_ba_dist_both_wskip` | `MuSSeg_PI_SE_LSA_BA_DistBoth_WeightedSkip` | ✓ | ✓ | ✓ |

This gives a simple ablation ladder:

1. Start with the baseline `lsa_ba` model.
2. Add only distance attention bias.
3. Add only distance bottleneck embeddings.
4. Add both distance features together.
5. Add only weighted skip fusion.
6. Combine weighted skips with each distance setting.

## Why This Grid Is Useful

The grid is structured to isolate two questions:

1. Does station geography help late station attention?
2. Does weighted skip fusion help more than the baseline max-collapse skip path?

The biggest implementation risk is memory usage. Distance-enabled variants and weighted-skip variants keep full `[B, S, C, T]` skip tensors alive until merge time, which is more expensive than collapsing skips immediately to `[B, C, T]`. That cost grows with station count, temporal length, and the number of encoder levels.

### Potential Problems

- Distance-enabled and weighted-skip paths can trigger OOM earlier than the baseline because they delay skip collapse.
- `skip_merge_alpha` is clamped instead of reparameterized, so it can hit flat gradients if optimization pushes it outside `[0, 1]`.
- One bottleneck-derived `w` is reused for every decoder skip level, which is clean for ablation purposes but may be too rigid if different resolutions need different station emphasis.

### Proposed Fixes If Needed

- If memory becomes a problem, use mixed precision or gradient checkpointing around the encoder blocks first.
- If that is still not enough, recompute or collapse only the skip levels that need weighting instead of storing every full station-wise skip tensor.
- If `alpha` becomes hard to optimize, replace the clamp with a bounded parameterization only after confirming that the clamp is the bottleneck.
- If the shared `w` is too coarse, consider a follow-up ablation with a per-level skip weighting head rather than changing this one.

## Bottom Line

`musseg_pi_se_lsa_ba` is the clean baseline: shared station encoding, late attention at the bottleneck, max-collapsed skips. The new variants add station geography through distance-aware interaction and distance-aware merging, and optionally add weighted skip fusion.

If you read only one thing: distance-enabled variants now inject station distance priors at every station merge via a learnable `gamma`, while weighted-skip variants additionally reuse bottleneck-derived station logits for skip fusion. The distance and weighted-skip versions are the most likely to hit memory limits first, so that is the main practical risk to watch during training.
