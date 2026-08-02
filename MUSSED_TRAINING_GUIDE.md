# MuSSED Training Infrastructure - Implementation Guide

## Overview

Phases 2 and 3 of MuSSED training have been implemented:

### Phase 2: Training Infrastructure ✓
- **event_targets.py**: Convert segmentation labels [C,T] to event intervals
- **hungarian_matcher_1d.py**: Assign predicted queries to ground-truth events using Hungarian algorithm
- **detr_event_loss.py**: Combined DETR loss (focal + L1 + confidence + GIoU)
- **event_detection_metrics.py**: Event detection metrics (mAP@tIoU, F1@tIoU, per-class metrics)

### Phase 3: Training Loop ✓
- **trainer_detr_1d.py**: Full training orchestrator with validation, checkpointing, and metrics
- **02c_test_mussed.py**: Complete training script with CLI arguments

## Quick Start

### 1. Prepare Your Data

Your training and validation data should be in MuSSeg format (NPZ files with segmentation labels):

```
train.npz:
  - filepaths: [N] array of paths to .npy files
  - labels: [N] sample-level labels
  - label_ids: [N] sample IDs

Each .npy file should contain:
  - Shape: [S + C, T] where S=num_stations, C=6 (classes), T=time samples
  - Rows 0:S: station waveforms
  - Rows S:S+C: one-hot segmentation labels (6 classes)
```

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

### 3. Training Output

Checkpoints and logs will be saved to `results/mussed_baseline/`:

```
results/mussed_baseline/
├── config.json              # Training config
├── history.json             # Training history
├── best_model.pt            # Best model (highest mAP)
├── final_model.pt           # Final model
└── checkpoint_epoch_5.pt    # Periodic checkpoints
```

## Architecture Details

### Data Flow

```
1. MuSSeg data (NPZ) 
   ↓
2. DataLoader (MuSSegBatch: x [B,S,T], y_onehot [B,C,T])
   ↓
3. MuSSED Forward Pass
   - Encoder: [B,S,T] → [B,C,T']
   - Queries: Nq learnable queries
   - Decoder: Transformer self+cross-attention
   - Heads: class, center, start, end, confidence
   ↓
4. Segmentation to Events
   y_onehot [B,C,T] → List[List[EventInterval]]
   - Extract event boundaries per class
   - Normalize to [0,1] time range
   ↓
5. Hungarian Matching
   - Build cost matrix: classification + L1 + GIoU
   - Solve matching problem
   ↓
6. Loss Computation
   - Classification: Focal loss
   - Regression: L1 (center, start, end)
   - Confidence: BCE on matched/unmatched
   ↓
7. Backward Pass & Optimization
```

### Loss Weights

Default loss weights in `DETREventLoss`:
- `class_loss`: 2.0 (classification)
- `bbox_loss`: 5.0 (temporal regression)
- `conf_loss`: 2.0 (confidence/objectness)
- `unmatched_query`: 0.1 (penalty for unmatched queries predicting background)

### Metrics

Computed at validation time:
- **mAP@0.1, mAP@0.2, ..., mAP@0.9**: Mean average precision at each IoU threshold
- **mAP**: Average over all IoU thresholds
- **AP_class_i@tIoU**: Per-class AP
- **F1@tIoU**: F1 score

## Key Components

### event_targets.py

Convert one-hot segmentation to event intervals:

```python
from utils.event_targets import segmentation_to_events

# Single sample
events = segmentation_to_events(y_onehot=[C,T], normalize=True)
# Returns: List[EventInterval]

# Batch
events_batch = batch_segmentation_to_events(y_batch=[B,C,T], normalize=True)
# Returns: List[List[EventInterval]]
```

**EventInterval** attributes:
- `class_id`: 1-5 (0 is background)
- `start_norm`: [0, 1] normalized start time
- `end_norm`: [0, 1] normalized end time
- `center_norm`: [0, 1] normalized center time
- `start_frame`: absolute frame index
- `end_frame`: absolute frame index

### hungarian_matcher_1d.py

Match predictions to ground-truth using Hungarian algorithm:

```python
from utils.hungarian_matcher_1d import HungarianMatcher

matcher = HungarianMatcher(
    cost_class=1.0,
    cost_bbox=1.0,
    cost_giou=1.0,
)

matches = matcher(predictions, targets, num_classes=6)
# matches: List[MatchResult]
# MatchResult: pred_indices, target_indices, unmatched_pred, unmatched_target
```

### detr_event_loss.py

DETR-style combined loss:

```python
from utils.detr_event_loss import DETREventLoss

loss_fn = DETREventLoss(num_classes=6)

loss_dict = loss_fn(predictions, targets)
# Returns: {
#   'loss_total': scalar,
#   'loss_class': scalar,
#   'loss_bbox': scalar,
#   'loss_conf': scalar,
#   'metrics': {...}
# }
```

### event_detection_metrics.py

Event detection evaluation:

```python
from utils.event_detection_metrics import EventDetectionMetrics

metrics_fn = EventDetectionMetrics(iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])

metrics = metrics_fn.evaluate_batch(predictions, targets, confidence_threshold=0.5)
# Returns: {
#   'mAP@0.1': ...,
#   'mAP@0.3': ...,
#   'mAP': ...,
#   'AP_class_1@0.5': ...,
#   ...
# }
```

### trainer_detr_1d.py

Full training orchestrator:

```python
from utils.trainer_detr_1d import DETRTrainer
from models.MuSSED import MuSSED

model = MuSSED(**config)
trainer = DETRTrainer(model, device='cuda')
trainer.setup_optimizer(learning_rate=1e-4)

history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=50,
    output_dir='results/mussed',
)
```

**Trainer methods:**
- `setup_optimizer()`: Initialize optimizer with learning rate scheduler
- `train_epoch()`: Train for one epoch, return metrics
- `evaluate()`: Validate on dev set, compute event-level metrics
- `train()`: Full training loop with checkpointing
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load model from checkpoint

## Training Tips

### Hyperparameter Tuning

1. **Learning Rate**: Start with 1e-4, adjust if loss doesn't decrease
2. **Batch Size**: 8-16 recommended (adjust for GPU memory)
3. **Warmup Steps**: 500-1000 typically sufficient
4. **Loss Weights**: Increase `bbox_loss` if predictions are too loose on temporal boundaries
5. **Confidence Threshold**: 0.5 default; lower to catch more events, raise to reduce false positives

### Debugging

Monitor during training:
- `loss_total`: Should decrease monotonically
- `loss_class`: Classification accuracy
- `loss_bbox`: Temporal regression accuracy
- `loss_conf`: Objectness prediction
- `mAP`: Should increase over time (validation only)

If loss plateaus:
- Reduce learning rate
- Increase warmup steps
- Check data loading (verify event label format)

### Data Requirements

- Minimum 100-200 samples for meaningful training
- Balanced classes help with focal loss
- Longer sequences (>4096 samples) allow more events per sample

## Advanced Usage

### Custom Loss Weights

```python
loss_fn = DETREventLoss(
    num_classes=6,
    loss_weights={
        'class_loss': 3.0,      # Emphasize classification
        'bbox_loss': 10.0,      # Emphasize temporal accuracy
        'conf_loss': 1.0,
        'unmatched_query': 0.05,
    },
    focal_alpha=0.25,           # Focal loss alpha
    focal_gamma=2.0,            # Focal loss focusing parameter
)
```

### Custom Matcher Costs

```python
matcher = HungarianMatcher(
    cost_class=2.0,             # Higher = more emphasis on classification cost
    cost_bbox=1.0,              # Temporal regression cost
    cost_giou=2.0,              # GIoU-based cost
)
```

### Evaluate Only Mode

Load a checkpoint and evaluate on test set:

```python
model = MuSSED(**config)
trainer = DETRTrainer(model, device='cuda')
trainer.load_checkpoint('results/mussed/best_model.pt')

metrics = trainer.evaluate(test_loader, epoch=0, confidence_threshold=0.5)
print(metrics)
```

## Files Summary

| File | Purpose |
|------|---------|
| `utils/event_targets.py` | Segmentation → Events conversion |
| `utils/hungarian_matcher_1d.py` | Query-to-target matching |
| `utils/detr_event_loss.py` | Combined training loss |
| `utils/event_detection_metrics.py` | Event detection evaluation metrics |
| `utils/trainer_detr_1d.py` | Training orchestration |
| `scripts/02c_test_mussed.py` | End-to-end training script |

## Next Steps

1. Prepare data in MuSSeg format (see `scripts/01_prepare_data.py`)
2. Run training with appropriate hyperparameters
3. Monitor `history.json` and checkpoints
4. Use best model for inference/evaluation
5. Analyze per-class metrics to identify weak classes

---

**Implementation Status**: Phase 2 and 3 ✓ Complete
- All components tested and integrated
- Ready for full model training with real data
