# MuSSED Implementation Plan

## Overview

MuSSED (MuSSeg Set Detector) is a DETR-inspired event detection model built on top of MuSSeg. Instead of per-timestep segmentation labels, it predicts a set of events with their properties (class, temporal location, duration).

**Main architectural concept:**
- Reuse MuSSeg's encoder to extract station-aware temporal features [B, C, T']
- Add a DETR-style transformer decoder with learnable event queries
- Each query predicts: class, event center, start time, end time, objectness

---

## Implementation Phases

### Phase 1: Core Model Architecture ✓ **TARGET FOR THIS SESSION**

#### 1.1 MuSSED Model (`models/MuSSED.py`)
**Status: To implement**

Minimal v1 implementation with:

```
MuSSED
├── encoder
│   ├── Reused MuSSeg encoder (station interaction + distance priors)
│   └── Output: [B, C, T'] temporal feature map
├── temporal_projection
│   └── Optional: project encoder features to fixed dimension
├── event_queries
│   └── Learnable queries [Nq, C_query]
├── detr_decoder
│   ├── Multi-head self-attention on queries
│   ├── Multi-head cross-attention (queries × temporal features)
│   └── Feed-forward per query
└── detection_heads
    ├── class_head: [Nq] → [Nq, n_classes]
    ├── center_head: [Nq] → [Nq, 1] (normalized time)
    ├── start_head: [Nq] → [Nq, 1] (normalized time)
    ├── end_head: [Nq] → [Nq, 1] (normalized time)
    └── confidence_head: [Nq] → [Nq, 1] (objectness)
```

**Input:** `x` with shape [B, S, T]
- B: batch size
- S: number of stations (variable)
- T: sequence length (variable)

**Output:** Dictionary with:
- `class_logits`: [B, Nq, n_classes] - per-query class logits
- `center`: [B, Nq, 1] - normalized event center time
- `start`: [B, Nq, 1] - normalized event start time
- `end`: [B, Nq, 1] - normalized event end time
- `confidence`: [B, Nq, 1] - objectness score
- `encoder_features`: [B, C, T'] - for interpretability

**Key design decisions:**
- Keep encoder frozen initially (for this session)
- Use learnable positional embeddings for temporal features
- Use cross-attention to attend temporal features
- No Hungarian matching yet (that's Phase 2)

---

### Phase 2: Training Infrastructure (FUTURE)

#### 2.1 Hungarian Matcher (`utils/hungarian_matcher_1d.py`)
**Status: To implement later**

- Match predicted queries to ground-truth events
- Compute cost matrix: classification cost + L1 regression cost
- Return matched indices and unmatched query indices

#### 2.2 DETR Loss (`utils/detr_event_loss.py`)
**Status: To implement later**

- Classification loss: Focal or CE (weighted for class imbalance)
- Bounding box regression: L1 on (center, start, end)
- IoU/temporal GIoU loss
- Combined weighted loss

#### 2.3 Event Target Converter (`utils/event_targets.py`)

- Convert segmentation labels [C, T] → event intervals

  
**Input:** One-hot segmentation [C, T]
**Output:** List of (class_id, start_norm, end_norm, center_norm)

#### 2.4 Evaluation Metrics (`utils/event_detection_metrics.py`)
**Status: To implement later**

- mAP@tIoU: mean average precision at temporal IoU thresholds
- F1@tIoU: F1 score at various IoU thresholds
- Per-class metrics

---

### Phase 3: Training Loop (FUTURE)

#### 3.1 DETR Trainer (`utils/trainer_detr_1d.py`)
**Status: To implement later**

- Data loading (segmentation → event targets)
- Forward pass with mixed precision
- Loss computation with Hungarian matching
- Backward pass and optimization
- Validation with event-based metrics

#### 3.2 Training Script (`scripts/02c_test_mussed.py`)
**Status: To implement now (inference/testing only)**

- Load MuSSED model
- Test with real data + synthetic inputs
- Verify output shapes and values
- No gradient computation

---

## Data Format

### Inputs
- **x**: [B, S, T] float32 (normalized waveforms)
  - B ∈ [1, 16]: batch size
  - S ∈ [1, 10]: number of stations
  - T ∈ [2048, 16384]: time samples

### Segmentation Labels (current MuSSeg format)
- **y_onehot**: [B, C, T] float32 (one-hot)
  - C = 6 (5 event classes + background)
  - Background class (index 0) is implicit or explicit

### Event Labels (future MuSSED format)
- **events**: List[Dict] per sample
  ```python
  {
    'class_id': 1-5,  # VT, LP, TR, AV, IC
    'start_norm': 0.0-1.0,
    'end_norm': 0.0-1.0,
    'center_norm': 0.0-1.0,
  }
  ```

---

## Configuration Parameters

### Model Hyperparameters

```python
MuSSED_KWARGS = {
    # Encoder (MuSSeg reused)
    'in_channels': 3,
    'encoder_classes': 6,  # inherited from MuSSeg encoder
    'depth': 5,
    'filters_root': 8,
    'kernel_size': 7,
    'stride': 4,
    
    # Detection head
    'num_queries': 3,  # Nq, start small (2-5)
    'query_dim': 256,  # latent dimension
    'hidden_dim': 512,  # transformer FF dimension
    'num_heads': 4,  # attention heads
    'num_decoder_layers': 3,  # transformer decoder depth
    'dropout': 0.1,
    
    # Station interaction (from MuSSeg)
    'station_interaction': 'late_attention',
    'use_distance_attn_bias': True,  # optional
    'volcano_name': 'NVCHVC',  # if distance bias enabled
}
```

### Inference Parameters

- **top_k**: keep top-K non-empty queries (default: 3)
- **confidence_threshold**: filter queries below threshold (default: 0.3)
- **temporal_nms**: optional light NMS (default: off initially)

---

## Testing Strategy (Phase 1)

### Test Inputs

#### Real Data Test
- Load a real sample from NVCHVC training data
- Single batch with variable stations (e.g., 5 stations)
- Time length from actual data (~8000-12000 samples)

#### Synthetic Data Tests

1. **Batch size variation**: [1, 4, 8, 16]
2. **Station variation**: [1, 2, 5, 10]
3. **Time length variation**: [2048, 4096, 8192, 16384]
4. **Random combinations**: Verify no shape mismatches

### Validation Checks

✓ Forward pass completes without error
✓ Output shapes are correct:
  - `class_logits`: [B, Nq, 6]
  - `center/start/end`: [B, Nq, 1]
  - `confidence`: [B, Nq, 1]
  - `encoder_features`: [B, C, T']

✓ Output values are valid:
  - class_logits: unrestricted (pre-softmax)
  - center/start/end: approximately in [0, 1] (can exceed slightly due to network)
  - confidence: unrestricted (pre-sigmoid)

✓ Gradient flow (backward pass):
  - No NaN or Inf
  - Gradients propagate to all parameters

✓ Memory usage is reasonable:
  - No excessive GPU memory spike
  - Batch processing works correctly

### Output Inspection

For each test case, print:
- Input shapes
- Encoder output shape
- Query attention weights (if accessible)
- Per-query predicted events
- Timing (forward pass time)

---

## File Structure

```
models/
  MuSSED.py              ← New (Phase 1)
  
utils/
  detr_event_loss.py     ← Future (Phase 2)
  hungarian_matcher_1d.py ← Future (Phase 2)
  event_targets.py       ← Future (Phase 2)
  event_detection_metrics.py ← Future (Phase 2)
  trainer_detr_1d.py     ← Future (Phase 3)
  
scripts/
  02c_test_mussed.py     ← New (Phase 1, inference only)
```

---

## Success Criteria for Phase 1

1. ✓ MuSSED model loads and runs inference
2. ✓ Outputs have correct shapes and valid values
3. ✓ Works with variable input sizes
4. ✓ Gradients flow correctly (for future training)
5. ✓ Test script demonstrates flexibility
6. ✓ Interpretability outputs (encoder features, attention) are accessible

After Phase 1 approval, proceed to Phase 2 (losses, matching, evaluation) and Phase 3 (training loop).

---

## Next Steps

1. **Immediate**: Implement MuSSED.py + 02c_test_mussed.py (this session)
2. **Review**: Check output quality, interpretability
3. **Future**: Implement Phase 2 (losses, Hungarian, eval metrics)
4. **Future**: Implement Phase 3 (training loop, integrate with existing pipeline)
5. **Future**: Evaluate integration difficulty with current pipeline
