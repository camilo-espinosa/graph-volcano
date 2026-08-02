# MuSSED Phases 2 & 3 - Complete Implementation Summary

## 🎯 Objective Complete

Implemented **Phases 2 and 3** for MuSSED (Multi-Station Seismic Event Detection) training infrastructure. The model can now be **fully trained** end-to-end using the existing MuSSeg segmentation data format.

---

## 📦 What Was Implemented

### Phase 2: Training Infrastructure ✓

**5 new utility modules** for the training pipeline:

#### 1. **event_targets.py** - Segmentation to Events Converter
- Converts one-hot segmentation labels `[C, T]` to event intervals
- Extracts event boundaries, merges consecutive same-class frames
- Normalizes temporal coordinates to `[0, 1]` range
- Key classes:
  - `EventInterval`: Represents a single event (class_id, start, end, center, frame indices)
  - `segmentation_to_events()`: Convert single sample
  - `batch_segmentation_to_events()`: Convert batch

#### 2. **hungarian_matcher_1d.py** - Query-to-Event Matching
- Assigns predicted queries to ground-truth events using Hungarian algorithm
- Computes multi-factor cost matrix:
  - **Classification cost**: Focal-weighted negative log probability
  - **Temporal cost**: L1 distance on (center, start, end)
  - **IoU cost**: Temporal GIoU (1 - GIoU)
- Solves optimal assignment via `scipy.optimize.linear_sum_assignment`
- Returns matched/unmatched indices for loss computation

#### 3. **detr_event_loss.py** - DETR-Style Combined Loss
- **Focal Loss** (classification): Handles class imbalance, focuses on hard examples
- **L1 Loss** (regression): Temporal coordinate accuracy (center, start, end)
- **BCE Loss** (confidence): Objectness score for matched/unmatched queries
- **Unmatched penalty**: Encourages background predictions for unmatched queries
- Customizable loss weights and focal parameters

#### 4. **event_detection_metrics.py** - Event Detection Evaluation
- Computes standard event detection metrics:
  - **mAP@tIoU**: Mean average precision at multiple IoU thresholds (0.1 to 0.9)
  - **Per-class AP**: Individual class performance
  - **F1@tIoU**: F1 score at specific IoU threshold
- Temporal IoU computation for event-level matching
- Confidence-based prediction filtering

#### 5. **trainer_detr_1d.py** - Training Orchestrator
- `DETRTrainer` class: Full training pipeline
- Core methods:
  - `setup_optimizer()`: Initialize AdamW with learning rate scheduler
  - `train_epoch()`: Train for one epoch, return loss metrics
  - `evaluate()`: Validate on dev set, compute event-level metrics
  - `train()`: Full training loop with checkpointing and validation
  - `save_checkpoint()` / `load_checkpoint()`: Model persistence
- Features:
  - Linear warmup + cosine decay learning rate schedule
  - Gradient clipping for training stability
  - Automatic best model tracking (highest mAP)
  - Periodic checkpoint saving
  - JSON history logging

### Phase 3: Training Loop ✓

**Updated training script** - `scripts/02c_test_mussed.py`:
- Replaced inference-only testing with **full training pipeline**
- Complete CLI with argparse:
  - `--train-npz`: Path to training data (MuSSeg format)
  - `--val-npz`: Path to validation data (optional)
  - `--epochs`: Number of training epochs
  - `--batch-size`: Training batch size
  - `--learning-rate`: Initial LR
  - `--weight-decay`: L2 regularization
  - `--warmup-steps`: LR warmup duration
  - `--checkpoint-interval`: Save checkpoint every N epochs
  - `--confidence-threshold`: Evaluation confidence threshold
  - `--output-dir`: Output directory for results

---

## 📊 Data Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MuSSeg Data (NPZ)                            │
│  - Segmentation labels: [C, T] one-hot (C=6 classes)               │
│  - Station waveforms: [S, T]                                       │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  DataLoader & Collate Function                     │
│  Output: MuSSegBatch(x: [B,S,T], y_onehot: [B,C,T], ...)          │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MuSSED Forward Pass                              │
│  - Temporal Encoder: [B,S,T] → [B,C',T']                           │
│  - Learnable Queries: Nq queries [B, Nq, d_model]                  │
│  - Transformer Decoder: self+cross-attention                       │
│  - Heads: class, center, start, end, confidence                    │
│  Output: {class_logits, center, start, end, confidence}            │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│            Segmentation → Events Conversion                         │
│  y_onehot [B,C,T] → List[List[EventInterval]]                      │
│  - Extract event boundaries per class                              │
│  - Normalize to [0,1] time range                                   │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Hungarian Matching                                     │
│  - Cost matrix: classification + L1 + GIoU                         │
│  - Solve: min cost assignment                                      │
│  - Output: matched & unmatched indices                             │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Loss Computation                                       │
│  - Classification: Focal loss (matched + unmatched)                │
│  - Regression: L1 on temporal coords                               │
│  - Confidence: BCE on matched/unmatched                            │
│  - Total: Weighted sum of losses                                   │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│            Backward Pass & Optimization                             │
│  - Gradient computation & clipping                                 │
│  - AdamW update with LR scheduler                                  │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  Validation (Optional)                              │
│  - Event-level metrics: mAP@tIoU, F1, per-class AP                 │
│  - Best model tracking & checkpointing                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prepare Data (MuSSeg Format)

Ensure you have training/validation data in the MuSSeg NPZ format with:
- Segmentation labels `[C, T]` (C=6 classes, including background)
- Station waveforms `[S, T]`

See `data/NVCHVC/` for examples.

### 2. Run Training

```bash
python scripts/02c_test_mussed.py \
    --train-npz data/NVCHVC/train.npz \
    --val-npz data/NVCHVC/val.npz \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --output-dir results/mussed_baseline
```

### 3. Monitor Progress

Training outputs go to the specified `--output-dir`:
```
results/mussed_baseline/
├── config.json              # Training configuration
├── history.json             # Metrics per epoch
├── best_model.pt            # Best model (highest mAP)
├── final_model.pt           # Final model
└── checkpoint_epoch_5.pt    # Periodic checkpoints
```

Track training with:
```bash
cat results/mussed_baseline/history.json | jq .
```

---

## 🧪 Testing

An integration test is provided to verify all components work together:

```bash
python tests/test_mussed_pipeline.py
```

Tests include:
1. ✓ Event target conversion (segmentation → events)
2. ✓ Hungarian matching (queries → targets)
3. ✓ DETR loss computation
4. ✓ Event detection metrics
5. ✓ Trainer instantiation and forward pass

---

## 📋 Key Design Decisions

### Loss Weights (Customizable)
```python
{
    "class_loss": 2.0,          # Classification accuracy
    "bbox_loss": 5.0,           # Temporal precision (emphasized)
    "conf_loss": 2.0,           # Objectness prediction
    "unmatched_query": 0.1,     # Penalty for unmatched background
}
```

### Cost Matrix Components
- **Classification**: Negative log probability of target class (focal-weighted)
- **L1 Regression**: Distance on (center, start, end) coordinates
- **GIoU**: Temporal generalized IoU (addresses scale sensitivity)

### Event Extraction Strategy
- Extract events via **argmax per timestep** (not soft predictions)
- Merge **consecutive same-class frames** into single event
- Normalize to **[0, 1]** for scale invariance

### Optimization
- **Optimizer**: AdamW (weight decay for regularization)
- **LR Schedule**: Linear warmup (500 steps) + cosine decay
- **Gradient Clipping**: max_norm=1.0 for stability
- **Batch Normalization**: Standard (inherited from MuSSeg encoder)

---

## 🔧 Advanced Usage

### Custom Loss Weights

```python
from utils.detr_event_loss import DETREventLoss

loss_fn = DETREventLoss(
    num_classes=6,
    loss_weights={
        'class_loss': 3.0,      # Emphasize classification
        'bbox_loss': 10.0,      # Emphasize temporal accuracy
        'conf_loss': 1.0,
        'unmatched_query': 0.05,
    },
    focal_alpha=0.25,
    focal_gamma=2.0,
)
```

### Custom Metrics

```python
from utils.event_detection_metrics import EventDetectionMetrics

metrics_fn = EventDetectionMetrics(
    iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    num_classes=6
)

metrics = metrics_fn.evaluate_batch(predictions, targets, confidence_threshold=0.5)
```

### Evaluate Only Mode

```python
from utils.trainer_detr_1d import DETRTrainer
from models.MuSSED import MuSSED

model = MuSSED(**config)
trainer = DETRTrainer(model, device='cuda')
trainer.load_checkpoint('results/mussed/best_model.pt')

val_metrics = trainer.evaluate(test_loader, confidence_threshold=0.5)
print(val_metrics)
```

---

## 📁 Files Created/Modified

### Created (Phase 2 & 3)
| File | Lines | Purpose |
|------|-------|---------|
| `utils/event_targets.py` | ~250 | Segmentation to events conversion |
| `utils/hungarian_matcher_1d.py` | ~350 | Query-to-event matching |
| `utils/detr_event_loss.py` | ~300 | Combined training loss |
| `utils/event_detection_metrics.py` | ~350 | Event detection metrics |
| `utils/trainer_detr_1d.py` | ~400 | Training orchestration |
| `tests/test_mussed_pipeline.py` | ~500 | Integration tests |
| `MUSSED_TRAINING_GUIDE.md` | - | Detailed usage guide |

### Modified
| File | Changes |
|------|---------|
| `scripts/02c_test_mussed.py` | Replaced inference-only testing with full training pipeline |

---

## ✅ Verification Checklist

- [x] All Python files compile without syntax errors
- [x] Import chain verified (no circular dependencies)
- [x] Hungarian algorithm integrated correctly
- [x] Loss computation tested with dummy data
- [x] Metrics evaluation working
- [x] Trainer class fully functional
- [x] Training script has complete CLI
- [x] Config saving to JSON
- [x] Checkpoint saving/loading implemented
- [x] Integration test provided

---

## 🎓 Recommended Next Steps

1. **Prepare Data**: Ensure training/validation NPZ files are ready
2. **Tune Hyperparameters**: Start with defaults, adjust if loss plateaus
3. **Run Training**: Execute training script with 50-100 epochs
4. **Monitor Metrics**: Track mAP, per-class AP, and F1 scores
5. **Analyze Results**: Check which event types have lowest AP
6. **Evaluate Test Set**: Use best model for final evaluation
7. **Save Predictions**: Export event predictions for downstream analysis

---

## 📝 Notes

- **Backward Compatibility**: Fully compatible with existing MuSSeg data format
- **GPU Support**: Automatic CUDA detection; CPU fallback available
- **Checkpointing**: Best model based on validation mAP automatically saved
- **Logging**: JSON history for plotting training curves
- **Extensibility**: Loss weights, metrics thresholds all customizable

---

## 🔗 Related Files

- Implementation plan: `MUSSED_IMPLEMENTATION_PLAN.md`
- Detailed guide: `MUSSED_TRAINING_GUIDE.md`
- Integration test: `tests/test_mussed_pipeline.py`
- Model architecture: `models/MuSSED.py` (Phase 1)

---

**Status**: ✅ Phase 2 & 3 Complete - Ready for Training
