## MuSSeg Encoder Ablations

**Status:** Complete (best ablation: musseg_d4_r16_s222_k127)

| Scientific Question | Switch |
|---|---|
| Does global temporal context help? | Bottleneck Attention |
| Is fixed station identity necessary? | Shared Station Encoder |
| Do early station interactions help? | Early PairConv |
| Do repeated station interactions help? | Hierarchical PairConv |
| Does learned message weighting improve PairConv? | Sum vs Attention aggregation |
| Does station attention improve PairConv? | Station Attention |

---

## MuSSED Decoder Ablations

**Status:** In progress (decoder lightweight optimization phase)

**Recommended Baseline:** num_decoder_layers=1, num_decoder_heads=4, hidden_dim=128, num_queries=(data-dependent), use_temporal_projection=False, constrain_intervals=True

| Hyperparameter | Ablation Values | Rationale |
|---|---|---|
| **num_decoder_layers** | {1, 2, 3} | Trade-off between model capacity and inference speed. Lightweight baseline: 1 layer. |
| **num_decoder_heads** | {4, 8} | Attention head multiplicity. Lightweight: 4. Baseline (current): 4. |
| **hidden_dim (FFN)** | {128, 256, 512} | Feed-forward network width. Lightweight: 128. Baseline (current): 256. |
| **num_queries** | {3, 5, 8, data_floor+2} | Event prediction slots. Must be ≥ max events per training window. Baseline: 3. |
| **decoder_dropout** | {0.0, 0.1} | Regularization. Baseline (current): 0.1. |
| **use_temporal_projection** | {False, True} | Whether to project encoder output before decoder. False (current): more efficient. True: adds learnable projection. |
| **constrain_intervals** | {True} | Enforce 0 ≤ start ≤ center ≤ end ≤ 1. Currently: True (recommended). |
| **interval_parameterization** | {center+width (current), direct} | Regress (center, half_width) vs (start, center, end). Current: center+width (guarantees ordering). |

### Phase 1: Lightweight Baseline
**Goal:** Establish smallest decoder that maintains >90% of full-capacity F1.

```python
MuSSED_LIGHTWEIGHT = {
    # Encoder (frozen, from best MuSSeg ablation)
    "depth": 4,
    "kernel_size": 127,
    "stride": [2, 2, 2],
    "dilation": [1, 1, 1, 1],
    "filters_root": 16,  # output_channels = 128
    
    # Lightweight decoder
    "num_decoder_layers": 1,  # ← MINIMAL (was 2)
    "num_decoder_heads": 4,
    "hidden_dim": 128,  # ← REDUCED (was 256)
    "decoder_dropout": 0.1,
    "use_temporal_projection": False,  # ← SKIP projection
    
    # Detection head
    "num_queries": 3,  # Set based on max events per window in training data
    "constrain_intervals": True,
}
```

**Expected result:** ~40-50% parameter reduction in decoder, <5% F1 loss.

### Phase 2: Query Count Optimization
**Goal:** Find optimal num_queries (insufficient queries cause missed events; excess queries add computational cost).

1. Inspect training data to find max events per window → set baseline to `max_events + 2`
2. Test {baseline, baseline+2, baseline+4} to find diminishing returns

### Phase 3: Projection Trade-off
**Goal:** Decide whether learnable projection is worth the cost.

Test `use_temporal_projection=True` with hidden_dim∈{128,256} against lightweight baseline to measure:
- Latency (critical for real-time seismic monitoring)
- Accuracy (should improve slightly)
- Parameter count

### Phase 4: Attention Head Scaling
**Goal:** Minimal attention heads for robust multi-scale temporal reasoning.

Test `num_decoder_heads∈{2, 4, 8}` on lightweight baseline. Expected: 4 is sweet spot.

---

## Training & Evaluation Notes

- **Metrics:** Detection precision/recall (IoU threshold ~0.5 for temporal overlap), event center MAE, interval coverage
- **Baseline comparison:** Not comparable to segmentation F1/IoU (different output space)
- **Missing components:** Hungarian matcher ✓, DETR loss ✓, but training script (07_detection_heads.py) is placeholder
- **Hardware target:** Real-time deployment on Raspberry Pi (ARM, <100ms latency for 4096-sample window)


