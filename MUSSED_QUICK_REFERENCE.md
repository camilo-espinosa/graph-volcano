# MuSSED Training - Quick Reference

## 📂 Project Structure

```
graph-volcano/
├── models/
│   ├── MuSSED.py                    ✓ Model architecture (Phase 1)
│   └── ...
├── utils/
│   ├── event_targets.py             ✓ NEW: Segmentation → Events
│   ├── hungarian_matcher_1d.py      ✓ NEW: Query matching
│   ├── detr_event_loss.py           ✓ NEW: Training loss
│   ├── event_detection_metrics.py   ✓ NEW: Evaluation metrics
│   ├── trainer_detr_1d.py           ✓ NEW: Training orchestrator
│   ├── musseg_utils.py              ✓ Data loading (existing)
│   └── ...
├── scripts/
│   ├── 02c_test_mussed.py           ✓ UPDATED: Training script
│   └── ...
├── tests/
│   └── test_mussed_pipeline.py      ✓ NEW: Integration tests
├── data/
│   ├── NVCHVC/
│   │   ├── train.npz
│   │   └── val.npz
│   └── ...
├── results/
│   └── mussed/                      (Training outputs)
│       ├── config.json
│       ├── history.json
│       ├── best_model.pt
│       └── checkpoint_epoch_*.pt
├── MUSSED_IMPLEMENTATION_PLAN.md    (Original plan)
├── MUSSED_PHASES_2_3_SUMMARY.md     ✓ NEW: This summary
├── MUSSED_TRAINING_GUIDE.md         ✓ NEW: Detailed guide
└── README.md
```

## 🎬 One-Liner Training Example

```bash
python scripts/02c_test_mussed.py --train-npz data/NVCHVC/train.npz --val-npz data/NVCHVC/val.npz --epochs 50 --batch-size 8 --output-dir results/mussed
```

## 📊 CLI Arguments Reference

```bash
REQUIRED:
  --train-npz PATH              Path to training data (MuSSeg format)

OPTIONAL:
  --val-npz PATH                Path to validation data (default: None)
  --epochs N                    Number of epochs (default: 50)
  --batch-size N                Batch size (default: 8)
  --learning-rate LR            Initial learning rate (default: 1e-4)
  --weight-decay WD             L2 regularization (default: 1e-4)
  --warmup-steps N              LR warmup steps (default: 500)
  --checkpoint-interval N       Save every N epochs (default: 5)
  --confidence-threshold T      Eval threshold (default: 0.5)
  --output-dir PATH             Output directory (default: results/mussed)
```

## 🔑 Core Classes & Functions

### Event Targets (event_targets.py)
```python
from utils.event_targets import EventInterval, segmentation_to_events, batch_segmentation_to_events

# Single sample
events = segmentation_to_events(y_onehot=[C,T])  # → List[EventInterval]

# Batch
events_batch = batch_segmentation_to_events(y_batch=[B,C,T])  # → List[List[EventInterval]]

# EventInterval attributes:
#   class_id (int): 1-5 event class
#   start_norm (float): [0,1] start time
#   end_norm (float): [0,1] end time
#   center_norm (float): [0,1] center time
#   start_frame (int): absolute frame
#   end_frame (int): absolute frame
```

### Hungarian Matcher (hungarian_matcher_1d.py)
```python
from utils.hungarian_matcher_1d import HungarianMatcher

matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)
matches = matcher(predictions, targets, num_classes=6)

# matches[b].pred_indices     → matched query indices
# matches[b].target_indices   → matched target indices
# matches[b].unmatched_pred   → unmatched query indices
# matches[b].unmatched_target → unmatched target indices
```

### DETR Loss (detr_event_loss.py)
```python
from utils.detr_event_loss import DETREventLoss

loss_fn = DETREventLoss(num_classes=6)
loss_dict = loss_fn(predictions, targets)

# loss_dict keys:
#   loss_total (torch.Tensor): scalar loss
#   loss_class: classification component
#   loss_bbox: regression component
#   loss_conf: confidence component
#   metrics: matching statistics
```

### Metrics (event_detection_metrics.py)
```python
from utils.event_detection_metrics import EventDetectionMetrics

metrics_fn = EventDetectionMetrics(iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
metrics = metrics_fn.evaluate_batch(predictions, targets)

# metrics keys:
#   mAP@0.1, mAP@0.3, ..., mAP@0.9: per-threshold mAP
#   mAP: average over all thresholds
#   AP_class_1@0.5, ..., AP_class_5@0.5: per-class AP
```

### Trainer (trainer_detr_1d.py)
```python
from utils.trainer_detr_1d import DETRTrainer
from models.MuSSED import MuSSED

model = MuSSED(**config)
trainer = DETRTrainer(model, device='cuda')

# Setup
trainer.setup_optimizer(learning_rate=1e-4, warmup_steps=500)

# Train one epoch
train_metrics = trainer.train_epoch(train_loader)

# Evaluate
val_metrics = trainer.evaluate(val_loader)

# Full training
history = trainer.train(
    train_loader, 
    val_loader,
    num_epochs=50,
    output_dir='results/mussed'
)

# Checkpointing
trainer.save_checkpoint('model.pt')
trainer.load_checkpoint('model.pt')
```

## 📈 Output Files Reference

### config.json
```json
{
  "model_config": {
    "in_channels": 8,
    "num_classes": 6,
    "num_queries": 3,
    "query_dim": 128,
    ...
  },
  "training_config": {
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 0.0001,
    ...
  },
  "timestamp": "2024-08-02T10:30:45.123456"
}
```

### history.json
```json
{
  "train_loss": [0.5234, 0.4123, 0.3456, ...],
  "train_loss_class": [...],
  "train_loss_bbox": [...],
  "train_loss_conf": [...],
  "val_metrics": [
    {
      "mAP": 0.45,
      "mAP@0.1": 0.67,
      "mAP@0.5": 0.42,
      "AP_class_1@0.1": 0.55,
      ...
    },
    ...
  ]
}
```

### best_model.pt
```
Contains:
  - model_state: Model weights
  - optimizer_state: Optimizer state (optional)
  - scheduler_state: LR scheduler state (optional)
  - timestamp: When saved
```

## 🐛 Debugging Tips

**Problem**: Loss doesn't decrease
- **Solution**: Check learning rate (try 1e-3 or 1e-5), increase warmup steps

**Problem**: Low validation mAP
- **Solution**: Increase `bbox_loss` weight (more emphasis on temporal accuracy)

**Problem**: Too many false positives
- **Solution**: Increase `confidence_threshold` during evaluation

**Problem**: CUDA out of memory
- **Solution**: Reduce batch_size or sequence length

**Problem**: Unmatched queries (all going to background)
- **Solution**: Increase `unmatched_query` penalty weight in loss function

## 📚 Documentation

- **Full Guide**: `MUSSED_TRAINING_GUIDE.md` (comprehensive)
- **Summary**: `MUSSED_PHASES_2_3_SUMMARY.md` (this document)
- **Implementation Plan**: `MUSSED_IMPLEMENTATION_PLAN.md` (original design)
- **Model Code**: `models/MuSSED.py` (architecture)

## ✅ Verification

All components working:
```bash
# Check syntax
python -m py_compile utils/event_targets.py utils/hungarian_matcher_1d.py utils/detr_event_loss.py utils/event_detection_metrics.py utils/trainer_detr_1d.py scripts/02c_test_mussed.py

# Run integration tests
python tests/test_mussed_pipeline.py
```

## 🚀 Next Steps

1. **Prepare data**: Convert to MuSSeg NPZ format
2. **Run training**: Execute training script
3. **Monitor**: Check history.json for progress
4. **Evaluate**: Use best_model.pt on test set
5. **Analyze**: Review per-class AP to identify weak classes

---

**Phases 2 & 3**: ✅ Complete and Ready
**Status**: Production-Ready for Training
