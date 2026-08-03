# MuSSED Ablation Compatibility Assessment

**Date**: 2025-08-03  
**Status**: Assessment Document (Not yet implemented)

## Executive Summary

This document outlines the changes required to integrate MuSSED (event detection) training and evaluation into the existing ablation testing infrastructure. Currently, the ablation framework supports:
- **2D models**: UNet, UNetBottleneckAttention
- **1D models**: PhaseNet, PhaseNetBottleneck, MuSSeg (and variants)

We need to add support for:
- **Event detector models**: MuSSED

The key challenge is **unifying outputs** across three distinct model families while maintaining clean, intuitive code that clearly separates loss components and metrics.

### MINIMAL CHANGES Principle

**CRITICAL**: This implementation adds ONLY the minimal infrastructure needed to train and evaluate MuSSED models within the existing ablation framework:

- **NO changes to existing directory structure** — All outputs go to the same `fold_XX/` folders
- **NO new directories** — Attention visualizations are embedded in validation plots (not separate files)
- **NO changes to main script architecture** — Scripts 02-05 remain structurally the same
- **Only add**: 4 new trainer/utility modules (`trainer_segmentation.py`, `trainer_detection.py`, `validation_plots.py`, `metrics_reporter.py`)
- **Only modify**: Scripts 02-05 with if/else trainer selection based on `trainer_kind`
- **Only update**: Model registry with MuSSED entries

This keeps the codebase clean, maintainable, and focused on the core task: detection model support.

---

## Part 1: Model Family Architecture Overview

### Current Model Types and Output Formats

#### 2D Segmentation Models (Trainer: UNet)
- **Models**: UNet, UNetBottleneckAttention
- **Input**: [B, 1, H, W] (batch, 1-channel image, height, width)
- **Output**: [B, C, H, W] logits (C=6 classes)
- **Loss Components**: 
  - Dice loss (per-class, then averaged)
  - Cross-entropy loss
  - Combined: `dice_weight * dice_loss + ce_weight * ce_loss`
- **Validation Metrics**:
  - Per-class F1 scores
  - Per-class IoU scores
  - Confusion matrices
- **Validation Plots**: Segmentation-based (raw predictions, post-processed predictions)
- **Trainer Function**: `train_one_unet_fold()` in `utils/train_utils.py`

#### 1D Segmentation Models (Trainer: Ablation/PhaseNet)
- **Models**: PhaseNet, PhaseNetBottleneck, MuSSeg (and dilation/kernel variants)
- **Input**: [B, S, T] where S=8 stations, T=4096 time samples
  - Expanded internally to: [B*S, 1, T] or [B, 1, S, T]
- **Output**: [B, C, T] logits (C=6 classes, at original resolution or reduced by stride)
- **Loss Components**:
  - Dice loss (per-class segmentation)
  - Cross-entropy loss
  - Combined: `dice_weight * dice_loss + ce_weight * ce_loss`
- **Validation Metrics**:
  - Per-class F1 scores (from segmentation thresholding)
  - Per-class IoU scores
  - Confusion matrices
- **Validation Plots**: Segmentation-based (stations stacked + raw/post-processed activations)
- **Trainer Function**: `train_one_ablation_fold()` in `utils/train_utils.py`

#### Event Detector Models (Trainer: DETR)
- **Models**: MuSSED
- **Input**: [B, S, T] where S=8 stations, T=4096 time samples
- **Output**: Dictionary with predictions for:
  - `class_logits`: [B, Nq, C] where Nq=10 queries
  - `centers`: [B, Nq] normalized center times [0, 1]
  - `starts`: [B, Nq] normalized start times [0, 1]
  - `ends`: [B, Nq] normalized end times [0, 1]
  - `confidence`: [B, Nq] objectness scores
- **Loss Components** (from `DETREventLoss`):
  - Classification loss (focal loss)
  - Bounding-box loss (L1 on center, start, end)
  - Confidence loss (BCE on objectness)
  - Unmatched query penalty
- **Validation Metrics** (from `EventDetectionMetrics`):
  - mAP @ various IoU thresholds (0.1, 0.3, 0.5, 0.7, 0.9)
  - mAP (average over all thresholds)
  - Per-class AP @ specific IoU
  - F1 @ IoU=0.5
- **Validation Plots**: Event-based (temporal intervals with confidence, classes, attention heatmaps)
- **Trainer Function**: `train()` in `utils/trainer_detr_1d.py` (standalone implementation)

---

## Part 2: Unified Training Pipeline Architecture

### Design Goal: Clean and Simple

Direct conditional logic in training scripts using if/else blocks to select the correct trainer. No factory patterns, no abstract base classes—just straightforward code selection.

### Trainer Selection via If/Else

In training scripts (e.g., `02_ablation_tests.py`), select trainer based on `spec["trainer_kind"]`:

```python
trainer_kind = spec["trainer_kind"]

if trainer_kind in ("2d", "1d"):
    # Use segmentation trainer
    train_one_segmentation_fold(
        model=model,
        fold_data_dir=fold_data_dir,
        fold_out_dir=fold_out_dir,
        device=device,
        config=config,
        trainer_kind=trainer_kind,
        ...
    )
elif trainer_kind == "detr":
    # Use event detection trainer
    train_one_event_detection_fold(
        model=model,
        fold_data_dir=fold_data_dir,
        fold_out_dir=fold_out_dir,
        device=device,
        config=config,
        ...
    )
else:
    raise ValueError(f"Unknown trainer_kind: {trainer_kind}")
```

### Loss Component Separation

For each trainer type, we track and report loss components at training time:

**Segmentation Models** (both 2D and 1D):
```python
{
    "loss_total": 0.4567,      # Combined weighted loss
    "loss_dice": 0.2345,       # Dice component
    "loss_ce": 0.3421          # CrossEntropy component
}
```

**Event Detection (MuSSED)**:
```python
{
    "loss_total": 0.8234,      # Combined weighted loss
    "loss_class": 0.2156,      # Focal classification loss
    "loss_bbox": 0.4321,       # L1 temporal regression loss
    "loss_conf": 0.1234,       # Confidence/objectness loss
    "loss_unmatched": 0.0523   # Penalty for unmatched queries
}
```

### Batch Progress Output

Standardize console output across both trainer types:

```
  Epoch 003 batch 0100/1250 | loss=0.4567 [dice=0.2345 ce=0.3421]
  Epoch 003 batch 0200/1250 | loss=0.8234 [class=0.2156 bbox=0.4321 conf=0.1234 umatch=0.0523]
```

---

## Part 3: Unified Validation Output

### Validation Metrics Dictionary

**ALL MODELS** (Segmentation and Detection) — Main CSV has **IDENTICAL COLUMNS**:
```python
{
    "val_loss": 0.3456,           # Total validation loss
    "val_loss_dice": 0.1234,      # Dice component (seg: actual dice, det: mapped from loss_class)
    "val_loss_ce": 0.2222,        # CE component (seg: actual CE, det: mapped from loss_bbox+loss_conf)
    # Per-class metrics
    "f1_VT": 0.89, "f1_LP": 0.87, "f1_TR": 0.85, "f1_AV": 0.83, "f1_IC": 0.81,
    "mean_f1": 0.85,              # UNIFIED: F1@0.5 for detection, F1 for segmentation
    "iou_VT": 0.79, "iou_LP": 0.77, "iou_TR": 0.75, "iou_AV": 0.73, "iou_IC": 0.71,
    "mean_iou": 0.75,             # UNIFIED: computed from event IoU for detection
}
```

**Event Detection Models (MuSSED) — ADDITIONAL CSV ONLY**:

`detection_metrics.csv` contains event-specific metrics:
```python
{
    "mAP@0.1": 0.92,
    "mAP@0.3": 0.90,
    "mAP@0.5": 0.85,
    "mAP@0.7": 0.72,
    "mAP@0.9": 0.45,
    "mAP": 0.77,              # Average over all thresholds
    "F1@0.5": 0.83,           # Same as mean_f1 in main CSV
    "AP_VT@0.5": 0.87, "AP_LP@0.5": 0.86, "AP_TR@0.5": 0.84, "AP_AV@0.5": 0.82, "AP_IC@0.5": 0.81,
}
```

### Epoch Summary Output Format

**IDENTICAL for all models** — No changes to existing format:

```
==============================================================================
EPOCH 003 SUMMARY
==============================================================================
Train Loss:  0.4567  [dice=0.2345 ce=0.3421]
Val Loss:    0.3456  [dice=0.1234 ce=0.2222]
Metrics:     mean_f1=0.8512 mean_iou=0.7643 best_epoch=002 no_improve=1/20
Plots Saved: 5 event plots
Best Metric: mean_f1 (improved to 0.8512)
==============================================================================
```

**Note for Detection Models**: The format is identical. `mean_f1` shows F1@0.5 from detection, loss components are mapped: `loss_dice` ← `loss_class`, `loss_ce` ← `loss_bbox + loss_conf`. No additional epoch summary lines needed.

---

## Part 4: Validation Event Visualization

### Current Approach (Segmentation Models)

For 1D models, validation plots show:
1. Waveforms for all 8 stations
2. Raw model predictions (softmax activations per class)
3. Post-processed predictions (hardmax one-hot)
4. Ground truth labels (one-hot segmentation)

**File**: `save_validation_event_plot()` in `utils/train_utils.py`

### Proposed Approach (Event Detector Models)

For MuSSED, validation plots should show **detected events** rather than segmentation:

#### Plot Structure (One plot per validation sample)

```
Panel Layout:
  [0-7]     8 station waveforms (stacked vertically)
  [8]       Ground-truth events (temporal intervals with class colors)
  [9]       Predicted events (temporal intervals with confidence scores)
  [10]      Station attention heatmap (per-station relevance scores)
  [11]      Temporal attention heatmap (time-localized attention weights)
```

#### Ground-Truth Events Panel
- Extract events from segmentation labels (use `segmentation_to_events()`)
- Draw horizontal bars for each event:
  - X-axis: normalized time [0, 1]
  - Y-axis: class (with color-coded class labels: VT, LP, TR, AV, IC)
  - Width: event duration
  - Height: thin line per class
  - Annotation: class name on the bar

#### Predicted Events Panel
- Draw detected events from MuSSED predictions:
  - X-axis: normalized time [0, 1]
  - Y-axis: class
  - Width: event interval (start to end)
  - Height: height proportional to **confidence score**
  - Color opacity: confidence (0.3 to 1.0 based on score)
  - Annotation: confidence score (`0.95`) and class name
  - Marker: center point marked with ▼ (triangle marker)

#### Station Attention Heatmap (Encoded as Panel Opacity)
- Extract station attention weights from **`StationAttentionBlock`** (late attention module in MuSSED encoder, after bottleneck)
- Extract attention matrix: [B, num_heads, S, S]
- Average over batch and attention heads → [S] importance scores
- Normalize to [0, 1]
- **Encode as transparency/opacity** of each station panel:
  - Low importance (0.0): very faint, semi-transparent (~alpha 0.3-0.4)
  - High importance (1.0): fully opaque (alpha 1.0)
  - Applied uniformly across entire time dimension for that station
  - Result: Important stations "pop out" visually, unimportant ones fade into background

#### Temporal Attention Heatmap (Encoded as Color Saturation)
- Extract temporal attention weights from **`TemporalBottleneckAttention`** (bottleneck attention layer)
- Extract attention matrix: [B, num_heads, T, T]
- Average over batch and attention heads → [T] importance scores
- Normalize to [0, 1]
- **Encode as color saturation** of background fill behind all waveforms:
  - Low attention at time t: desaturated (nearly white/light gray, ~10-20% saturation)
  - High attention at time t: saturated (deep blue, ~80-100% saturation)
  - Varies smoothly across time steps
  - Applied uniformly across all stations at each time point
  - Result: Attention peaks appear as bright blue bands, low-attention zones are pale/whitish
- Overlay event boundaries on top:
  - Dashed red lines at ground-truth event boundaries
  - Solid lighter lines at predicted event boundaries

#### Complete Plot Information

**Plot File Naming**:
```
validation_events_epoch_003_sample_015_f1_0.87.png
```

**Plot Annotations** (title section; ensure the title fits the plot wi \n):
```
Epoch 3 | Fold 01 | Sample ID: NVCHVC_001_seg_015
Model: mussed_d4_r16_s222_k127
Ground Truth: 3 events | Predicted: 2 events
mAP@0.5: 0.85 | F1@0.5: 0.87
```

---

## Part 5: Attention Weight Visualization Details (Heatmaps Over Waveforms)

### Key Principle: Heatmaps Overlaid on Input Data

Attention visualizations are **NOT separate subplots**, but **heatmaps overlaid directly on the waveform plots**. This allows direct comparison between attention weights and actual signal content.

### A. Station Attention (Encoded as Panel Opacity)

**What it represents**: Which stations are most important for the model's event predictions.

**Extraction method**:
1. Hook into `StationAttentionBlock` in MuSSED encoder
2. Extract attention weights [B, num_heads, S, S]
3. Average over batch and attention heads → [S]
4. Normalize to [0, 1]
5. Result: per-station importance scores

**Visualization** (Dual Encoding Part 1: Opacity):
- Apply attention score as **alpha/transparency** of each station's waveform panel
- Low importance (0.0-0.3): very faint, alpha ~0.3-0.4 (almost invisible, fades into background)
- Medium importance (0.3-0.7): moderate, alpha ~0.6-0.7 (visible but subtle)
- High importance (0.7-1.0): fully opaque, alpha ~0.95-1.0 (prominent, clear)
- Applied uniformly across entire time dimension for that station
- Result: Important stations visually "pop out", unimportant ones recede into background
- Example:
  ```
  Station S1 (attn=0.9): [OPAQUE █] ▔▔▔╱▔▔╲▔▔▔  ← clear, prominent
  Station S2 (attn=0.6): [MEDIUM ░] ▔▔╱▔▔▔╲▔▔   ← moderately visible
  Station S3 (attn=0.3): [FAINT  ·] ▔╱▔▔▔▔╲▔    ← barely visible
  ```

### B. Temporal Attention (Encoded as Color Saturation)

**What it represents**: Which time windows are most attended to during event detection.

**Extraction method**:
1. Hook into `TemporalBottleneckAttention` layer in MuSSED encoder
2. Extract attention weights [B, num_heads, T_out, T_in]
3. Average over batch and attention heads → [T]
4. Normalize to [0, 1]
5. Result: temporal attention scores across time window

**Visualization** (Dual Encoding Part 2: Color Saturation):
- Apply attention score as **background color saturation** (blue hue, varying saturation)
- Low attention (0.0-0.3): desaturated, nearly white/light gray (~10-15% saturation) — barely visible tint
- Medium attention (0.3-0.7): moderate saturation, light to medium blue (~40-60% saturation)
- High attention (0.7-1.0): saturated, deep rich blue (~80-100% saturation) — very prominent
- Varies smoothly across time steps
- Applied as **semi-transparent fill behind all waveforms** (alpha ~0.2-0.3 for visibility)
- Overlay event boundaries for reference:
  - Dashed vertical red lines at ground-truth event boundaries
  - Solid lighter blue lines at predicted event boundaries
- Example visualization:
  ```
  Temporal:  ░░░░▒▒▒▓▓▓▓████████████████▓▓▓▒▒░░░░░░░░░░░▒▒▒▒▓▓████
             LOW      MEDIUM    HIGH ATTENTION     LOW    MED   HIGH
  Waveform:  ▔▔▔▔┬────┬──────────────┬─────┬─────┬──────────────┬
             GT: ┴event 1┴            ┴event 2┴
             Pred:──event 1.5──────────event 2──
  ```

### Dual Encoding: Combined Visualization Effect

Both attention mechanisms are **embedded in a single validation plot** using separate encoding dimensions:

1. **Station Importance** (opacity dimension):
   - Each station panel has variable transparency based on attention score
   - Important stations clearly visible and opaque
   - Unimportant stations fade into background (low alpha)

2. **Temporal Dynamics** (color saturation dimension):
   - Blue-colored background fill varies by time step
   - Low attention zones: pale/whitish (low saturation)
   - High attention zones: deep blue (high saturation)
   - Creates temporal "heat map" effect without competing with station opacity

3. **Event Overlays**:
   - Ground-truth events: dashed red lines
   - Predicted events: solid lighter lines with confidence annotations

4. **Combined Result**:
   - No visual clash between the two attention dimensions
   - Station importance at a glance (transparency)
   - Temporal attention dynamics immediately visible (color saturation)
   - Both interpretable independently and together
   - Professional, clean appearance

This encoding keeps the output directory structure unchanged while providing rich, interpretable attention visualization.

---

## Part 6: Script-by-Script Analysis

### Script 02: `02_ablation_tests.py` (Main Training)

**Current Status**: Handles 2D (UNet) and 1D (PhaseNet/MuSSeg) models only.

**Changes Required**:

1. **Model Registry Update**:
   - Add MuSSED entry to `MODEL_SPECS` with `trainer_kind="detr"`
   - Specify loss weights and detection head parameters

2. **Training Function Polymorphism**:
   - Maintain conditional logic:
     ```python
     if spec["trainer_kind"] == "2d":
         train_one_unet_fold(...)
     else:
         train_one_ablation_fold(...)
     ```
     - But update with detection models


3. **Data Loading Adaptation**:
   - Current: `MultiStation1DDataset` + `UNetPatchDataset`
   - For MuSSED: Already compatible (uses `MuSSegWindowDataset`)
   - Need: Detect model type and load appropriate dataset

4. **Output Directories**:
   - **MINIMAL CHANGE**: Keep EXACT same directory structure for all model types
   - Current structure: `checkpoints/`, `reports/`, `confusion_matrices/`, `validation_event_plots/`
   - For MuSSED: Use same folders, with detection-specific visualizations in validation plots (no new directories)

5. **Metrics Collection**:
   - **UNCHANGED for all models**: `training_metrics.csv` has identical columns for segmentation and detection models
   - Columns: `lr`, `epoch`, `train_loss`, `train_loss_dice`, `train_loss_ce`, `val_loss`, `val_loss_dice`, `val_loss_ce`, `f1_VT`, ..., `mean_f1`, `mean_iou`, etc.
   - For MuSSED: Map detection loss components to standard columns:
     - `train_loss_dice` ← `loss_class`
     - `train_loss_ce` ← `loss_bbox + loss_conf + loss_unmatched`
     - `mean_f1` ← `F1@0.5`
   - **ADDITIONAL for MuSSED only**: Create `detection_metrics.csv` with event-specific columns
     - Columns: `epoch`, `mAP@0.1`, `mAP@0.3`, `mAP@0.5`, `mAP@0.7`, `mAP@0.9`, `mAP`, `F1@0.5`, `AP_VT@0.5`, etc.

6. **Best Checkpointing Criterion**:
   - **UNIFIED**: All models use `max(mean_f1)` for best model selection
   - Segmentation: max(mean_f1) from segmentation F1 scores
   - Detection: max(mean_f1) which equals max(F1@0.5) from detection
   - Early stopping metric is identical: validation `mean_f1`

7. **Console Output Unification**:
   - **IDENTICAL format for all models** — No changes to existing epoch summary format
   - Train/Val loss and metrics reported identically
   - Detection models map their specific metrics to standard columns automatically
   - No model-specific lines added to console output, everything fits existing format

---

### Script 03: `03_evaluate_nvchvc_station_scramble.py` (Station Shuffling Robustness)

**Current Status**: Evaluates trained checkpoints with scrambled station ordering.

**Changes Required**:

1. **Checkpoint Loading**:
   - Current: `load_checkpoint_into_model()` and `evaluate_multistation_checkpoint()`
   - For MuSSED: Need MuSSED-compatible loading (already has `load_checkpoint_into_model()`)
   - No changes needed, but must ensure station shuffling is compatible

2. **Station Scrambling Logic**:
   - Current: Shuffles station order in [B, S, T] → [B, S_shuffled, T]
   - For MuSSED: Same approach works (permutation-invariant by design)
   - Confirm: MuSSED output invariant to station permutation

3. **Evaluation Function**:
   - Current: `evaluate_multistation_checkpoint()` computes segmentation metrics
   - For MuSSED: Need event detection evaluation
   - Create: `evaluate_event_detection_checkpoint(model, test_loader, device)`
   - Returns: mAP, F1@0.5, per-class AP, etc.

4. **Output Format**:
   - Current: CSV with columns for per-class F1 and IoU
   - For MuSSED: CSV with columns for mAP@various_thresholds, F1@0.5
   - Use model family to determine columns

5. **Visualization**:
   - Current: Saves confusion matrices
   - For MuSSED: Save per-class AP bar charts, mAP trend plots
   - Optional: Event detection examples with predictions vs. ground truth

---

### Script 04: `04_zero_shot_cross_volcano.py` (Cross-Volcano Transfer)

**Current Status**: Evaluates models trained on NVCHVC on other volcanoes.

**Changes Required**:

1. **Data Compatibility**:
   - Current: Uses `progressive_finetuning` dataset format
   - For MuSSED: Must convert segmentation labels to events using same function
   - Ensure: Event conversion is consistent across all evaluation points

2. **Evaluation**:
   - Same changes as Script 03
   - Use `evaluate_event_detection_checkpoint()` for MuSSED
   - Use `evaluate_multistation_checkpoint()` for segmentation models

3. **Metrics Reporting**:
   - Track which volcanoes benefit most from NVCHVC pretraining
   - For segmentation: F1 improvement by volcano
   - For events: mAP improvement by volcano
   - Create: Comparison plots (segmentation vs. event detection transfer)

4. **Early Stopping**:
   - No changes, but criteria metric differs by model type

---

### Script 05: `05_progressive_finetuning.py` (Target-Volcano Fine-tuning)

**Current Status**: Fine-tunes models from NVCHVC on limited target-volcano data.

**Changes Required**:

1. **Model Loading**:
   - Current: Uses `load_checkpoint_into_model()`
   - For MuSSED: Same function works
   - Confirm: State dict compatibility for fine-tuning

2. **Fine-tuning Loops**:
   - Current: Separate loops for UNet and PhaseNet
   - For MuSSED: May need separate loop (different loss function, metrics)
   - Or: Integrate into unified trainer abstraction

3. **Learning Rate Scheduling**:
   - Current: Different LR for different subset sizes (see `SUBSET_FIXED_LR`)
   - For MuSSED: May need different LR scaling
   - Recommendation: Make LR scheduling configurable per model family

4. **Validation During Fine-tuning**:
   - Use model family to determine evaluation function
   - Report appropriate metrics per family

5. **Output Format**:
   - Same as Script 02, but tagged by target volcano and subset size

---

## Part 7: Implementation Recommendations

### Code Organization

```
utils/
├── train_utils.py (existing, keep all functions, add new training functions)
├── trainer_segmentation.py (NEW)
│   └── train_one_segmentation_fold() (unified 2D+1D segmentation)
├── trainer_detection.py (NEW)
│   └── train_one_event_detection_fold() (MuSSED event detection)
├── validation_plots.py (NEW)
│   ├── plot_segmentation_validation() (refactor from existing)
│   ├── plot_event_validation() (new for MuSSED)
│   └── extract_attention_weights() (new for both)
├── metrics_reporter.py (NEW)
│   ├── format_epoch_summary() (unified output)
│   └── save_training_history_*() (conditional columns)
└── model_registry.py (update with MuSSED entries)
```

### Key Design Principles

1. **Simplicity**: Direct if/else selection of trainer based on `trainer_kind`
   - No abstract base classes
   - No factory functions
   - Clear, readable control flow

2. **Fail-Fast**: Explicit error checking
   - If model spec missing `trainer_kind`, raise clear error
   - If trainer_kind unknown, raise ValueError

3. **Separation of Concerns**:
   - Segmentation training: `trainer_segmentation.py`
   - Event detection training: `trainer_detection.py`
   - Metrics/reporting: `metrics_reporter.py`
   - Validation plots: `validation_plots.py`

4. **Backward Compatibility**:
   - Keep existing functions unchanged in `train_utils.py`
   - Add new training functions (don't replace old ones)
   - Scripts call new functions

---

## Part 8: File-by-File Checklist

### utils/train_utils.py
- [ ] Keep existing `train_one_unet_fold()` unchanged
- [ ] Keep existing `train_one_ablation_fold()` unchanged
- [ ] Add import: `from utils.trainer_segmentation import train_one_segmentation_fold`
- [ ] Add import: `from utils.trainer_detection import train_one_event_detection_fold`

### utils/trainer_segmentation.py (NEW)
- [ ] Implement `train_one_segmentation_fold(model, fold_data_dir, fold_out_dir, device, config, trainer_kind, ...)`
  - [ ] Handles both 2D (UNet) and 1D (PhaseNet/MuSSeg) segmentation
  - [ ] Trains with combined Dice + CE loss
  - [ ] Tracks: `loss_dice`, `loss_ce` components
  - [ ] Validation: compute mean_f1, mean_iou, per-class metrics
  - [ ] Early stopping on `max(mean_f1)`
  - [ ] Saves: `training_metrics.csv` (standard format)

### utils/trainer_detection.py (NEW)
- [ ] Implement `train_one_event_detection_fold(model, fold_data_dir, fold_out_dir, device, config, ...)`
  - [ ] Handles MuSSED event detection
  - [ ] Trains with DETR-style loss (focal + L1 + confidence)
  - [ ] Tracks: `loss_class`, `loss_bbox`, `loss_conf`, `loss_unmatched` components
  - [ ] Validation: compute event-level metrics (mAP, F1@0.5, per-class AP)
  - [ ] Compute mean_f1 and mean_iou for main CSV compatibility
  - [ ] Early stopping on `max(F1@0.5)` (mapped to mean_f1)
  - [ ] Saves: `training_metrics.csv` (compatible format) + `detection_metrics.csv` (event-specific)

### utils/validation_plots.py (NEW)
- [ ] Implement `plot_segmentation_validation()` - refactored from existing code
- [ ] Implement `plot_event_validation()` with:
  - [ ] Random sample selection logic (N=15 default, configurable)
  - [ ] Event interval visualization (GT vs Pred)
  - [ ] Station attention bar chart (from actual attention weights)
  - [ ] Temporal attention line plot with event overlays
- [ ] Implement `extract_station_attention_weights(model, data, device)`
  - [ ] Hook into StationAttentionBlock
  - [ ] Extract from [B, num_heads, S, S] → [S] importance
  - [ ] Return normalized [0, 1] importance scores
- [ ] Implement `extract_temporal_attention_weights(model, data, device)`
  - [ ] Hook into TemporalBottleneckAttention
  - [ ] Extract from [B, num_heads, T, T] → [T] importance
  - [ ] Return normalized [0, 1] importance scores

### utils/metrics_reporter.py (NEW)
- [ ] Implement `format_epoch_summary_segmentation()` - segmentation output
  - [ ] Format: `train_loss=X [dice=Y ce=Z] val_loss=A [dice=B ce=C] mean_f1=D mean_iou=E ...`
- [ ] Implement `format_epoch_summary_detection()` - detection output
  - [ ] Format: `train_loss=X [class=Y bbox=Z conf=W umatch=V] val_loss=A mean_f1=B mean_iou=C (F1@0.5=D mAP=E)`
- [ ] Implement `save_training_history_segmentation()` - existing CSV format
  - [ ] Same columns as current: `lr`, `epoch`, `train_loss`, `train_loss_dice`, `train_loss_ce`, `val_loss`, `val_loss_dice`, `val_loss_ce`, `mean_f1`, per-class metrics, etc.
- [ ] Implement `save_training_history_detection()` - writes both CSVs
  - [ ] Main CSV (compatible): same columns as segmentation (with mapped values)
  - [ ] Detection CSV (event-specific): `epoch`, `mAP@0.1`, ..., `mAP`, `F1@0.5`, per-class AP

### utils/model_registry.py
- [ ] Add MuSSED entries with `trainer_kind="detr"`
- [ ] Ensure all required model_kwargs are specified
  - [ ] `batch_size` (recommend: 16-24)
  - [ ] `best_metric_name="mAP"` or `"F1@0.5"`

### scripts/02_ablation_tests.py (REFACTOR)
- [ ] Replace conditional training logic with unified approach
- [ ] Add `trainer_kind` detection from spec
- [ ] Dispatch to unified trainer
- [ ] Update console output to unified format
- [ ] Add attention heatmap directory creation
- [ ] Update metrics CSV generation

### scripts/03_evaluate_nvchvc_station_scramble.py (REFACTOR)
- [ ] Add event detection evaluation path
- [ ] Create `evaluate_event_detection_checkpoint()` if not exists
- [ ] Update output CSV columns based on model type
- [ ] Add event detection visualization (optional)

### scripts/04_zero_shot_cross_volcano.py (REFACTOR)
- [ ] Add event detection evaluation path
- [ ] Update metrics tracking for event detection
- [ ] Add cross-family comparison plots (optional)

### scripts/05_progressive_finetuning.py (REFACTOR)
- [ ] Add event detection fine-tuning path
- [ ] Update validation metrics based on model type
- [ ] Ensure learning rate scheduling appropriate for events

---

## Part 9: Specific MuSSED Considerations

### Event Conversion Consistency

**Critical**: All evaluation scripts must use the same `segmentation_to_events()` function:

```python
# utils/event_targets.py (existing)
from utils.event_targets import segmentation_to_events, batch_segmentation_to_events
```

Ensure:
- Ground-truth event normalization: use `normalize=True` always
- Time range: [0, 1] across all scripts
- Class filtering: background (class 0) excluded from events

### MuSSED Configuration for Ablations

Recommend adding these MuSSED variants to registry:

```python
"mussed_base": {
    "trainer_kind": "detr",
    "display_name": "MuSSED (Base)",
    "model_cls": MuSSED,
    "model_kwargs": {
        "num_classes": 6,
        "depth": 4,
        "kernel_size": 127,
        "stride": [2, 2, 2],
        "filters_root": 16,
        "bottleneck_attention": True,
        "num_queries": 10,
        "query_dim": 128,
        "hidden_dim": 256,
        "num_decoder_heads": 4,
        "num_decoder_layers": 2,
        "use_temporal_projection": False,
        "constrain_intervals": True,
    },
    "batch_size": 16,
    "best_metric_name": "mAP",
    ...
}
```

### Validation Plot Configuration

Add to MuSSED training config:

```python
MUSSED_CONFIG = {
    # ... model params ...
    "validation_plot_config": {
        "max_plots_per_epoch": 5,
        "show_station_attention": True,
        "show_temporal_attention": True,
        "attention_colormap": "viridis",
        "event_colors": {
            "VT": "#df8d5e",
            "LP": "#2ca02c",
            "TR": "#d62728",
            "AV": "#9467bd",
            "IC": "#8c564b",
        },
    }
}
```

---

## Part 10: Proposed Timeline

### Phase 1: Foundation (Week 1)
- [ ] Create `trainer_segmentation.py` with segmentation training function
- [ ] Create `trainer_detection.py` with event detection training function
- [ ] Create `validation_plots.py` with both plot types + attention extraction
- [ ] Create `metrics_reporter.py` for unified output formatting

### Phase 2: Integration (Week 2)
- [ ] Update `02_ablation_tests.py` with if/else trainer selection
- [ ] Update model registry with MuSSED entries
- [ ] Test on small NVCHVC subset (1 fold, 10 epochs)

### Phase 3: Evaluation Scripts (Week 3)
- [ ] Update `03_evaluate_nvchvc_station_scramble.py` with detection evaluation
- [ ] Update `04_zero_shot_cross_volcano.py` with detection evaluation
- [ ] Add event detection visualization (optional)

### Phase 4: Fine-tuning (Week 4)
- [ ] Update `05_progressive_finetuning.py` with detection fine-tuning
- [ ] Test cross-volcano fine-tuning
- [ ] Full integration test on all 5 folds

### Phase 5: Documentation & Polish (Week 5)
- [ ] Add comprehensive docstrings
- [ ] Create usage guide for MuSSED ablations
- [ ] Performance profiling and optimization
- [ ] Code review and cleanup

---

## Part 11: APPROVED DECISIONS (User Input)

### 1. Attention Visualization Method
**Decision**: Use **actual attention weights** from attention modules (not gradient-based saliency)
- Extract from `StationAttentionBlock` for per-station importance
- Extract from `TemporalBottleneckAttention` for temporal dynamics
- More interpretable and consistent across runs

### 2. Validation Plot Frequency
**Decision**: Save **random N samples per epoch** (default N=15)
- Not just best samples, but representative random sample
- Allows inspection of failure modes and edge cases
- More memory-efficient than saving all plots
- Configurable via `val_plot_count` in training config

### 3. Metrics CSV Format
**Decision**: **Two-CSV approach**
- **Main CSV** (`training_metrics.csv`): Keep exact same columns as current segmentation models
  - Columns: `lr`, `epoch`, `train_loss`, `train_loss_dice`, `train_loss_ce`, `val_loss`, `val_loss_dice`, `val_loss_ce`, `mean_f1`, `mean_iou`, etc.
  - For detection models: Calculate mean_f1 and mean_iou from event-level predictions (to maintain compatibility)
  - This ensures backward compatibility with existing analysis pipelines and aggregation scripts

- **Detection-specific CSV** (`detection_metrics.csv`): Only created for MuSSED
  - Columns: `epoch`, `mAP@0.1`, `mAP@0.3`, `mAP@0.5`, `mAP@0.7`, `mAP@0.9`, `mAP`, `F1@0.5`, `AP_VT@0.5`, `AP_LP@0.5`, etc.
  - Side-by-side with main CSV for detailed event detection analysis

### 4. Best Model Selection Criterion
**Decision**: All model types use **max(mean_f1)** for best checkpoint selection
- Segmentation models: Best model chosen by `max(mean_f1)` (from segmentation F1 scores)
- Detection models: Best model chosen by `max(mean_f1)` (where mean_f1 = F1@0.5 from events)
- Single `best_f1.pt` checkpoint naming for all model types
- Identical early stopping logic across all families: Stop when mean_f1 plateaus

### 5. Backward Compatibility: Clarification
**Question Asked**: "Keep old functions or migrate fully?"

**Clarification**: This refers to the training entry points in `utils/train_utils.py`

**Decision**: **Keep old functions, add new unified wrapper**
- Keep existing `train_one_unet_fold()` and `train_one_ablation_fold()` functions
  - These remain unchanged to preserve any existing call sites
  - Mark with deprecation notices in docstrings (but no runtime warnings, fail-fast principle)
- Add new unified function `train_unified_fold()` 
  - Routes to correct trainer based on `spec["trainer_kind"]`
  - Used by all scripts going forward (02, 03, 04, 05)
- Scripts call new function, but codebase is not "broken" if someone uses old functions directly
- This approach:
  - Maintains backward compatibility for custom training pipelines
  - Keeps codebase clean (new code uses unified path)
  - Avoids mysterious `ImportError` for users who clone older branches

### 6. Early Stopping Criterion
**Decision**: Use **validation mean_f1** for all model types (identical criterion)
- Segmentation models: Stop when `max(mean_f1)` plateaus (no improvement for N epochs)
- Detection models: Stop when `max(mean_f1)` plateaus, where mean_f1 = F1@0.5 from events
- **Unified**: Same metric, same logic, works across all model families
- Configuration: `early_stop_patience` parameter controls plateau detection (e.g., 20 epochs)

---

## Part 9: Specific MuSSED Considerations

#### Event Conversion Consistency

**Critical**: All evaluation scripts must use the same `segmentation_to_events()` function:

```python
# utils/event_targets.py (existing)
from utils.event_targets import segmentation_to_events, batch_segmentation_to_events
```

Ensure:
- Ground-truth event normalization: use `normalize=True` always
- Time range: [0, 1] across all scripts
- Class filtering: background (class 0) excluded from events

#### MuSSED Configuration for Ablations

Add to model registry:

```python
"mussed_base": {
    "family": "detr",
    "trainer_kind": "detr",
    "display_name": "MuSSED (Base)",
    "model_cls": MuSSED,
    "model_kwargs": {
        "num_classes": 6,
        "depth": 4,
        "kernel_size": 127,
        "stride": [2, 2, 2],
        "filters_root": 16,
        "bottleneck_attention": True,
        "bottleneck_attn_heads": 4,
        "bottleneck_attn_ff_mult": 2,
        "station_attn_heads": 4,
        "station_attn_ff_mult": 2,
        "num_queries": 10,
        "query_dim": 128,
        "hidden_dim": 256,
        "num_decoder_heads": 4,
        "num_decoder_layers": 2,
        "decoder_dropout": 0.1,
        "use_temporal_projection": False,
        "constrain_intervals": True,
    },
    "batch_size": 16,
    "enabled": True,
    "aliases": (),
}
```

#### Output Directories Structure

**UNCHANGED** for all model types (including MuSSED):
```
fold_XX/
├── checkpoints/
│   ├── best_f1.pt           # Best model (by mean_f1 for seg, F1@0.5 for detection)
│   ├── best_val_loss.pt
│   └── best_train_loss.pt
├── reports/
│   ├── training_metrics.csv     # Same columns for all models (backward compatible)
│   ├── detection_metrics.csv    # ONLY for MuSSED (event-level metrics)
│   └── fold_summary.json
├── confusion_matrices/          # For segmentation models
├── validation_event_plots/      # Visualizations (both segmentation and detection)
│   ├── epoch_000/
│   │   ├── sample_001_events.png (or seg.png for segmentation)
│   │   ├── sample_001_station_attn.png (MuSSED only, embedded in plot)
│   │   ├── sample_001_temporal_attn.png (MuSSED only, embedded in plot)
│   │   ├── sample_042_events.png
│   │   └── ...  (random N=15 samples)
│   └── epoch_001/
```

**Note**: For MuSSED, attention visualizations (station + temporal) are **heatmaps overlaid on the waveform data**, not separate files. See Part 5 for details.

---

## Summary: Required Changes by Script

| Script | Changes | Priority | Complexity |
|--------|---------|----------|------------|
| 02_ablation_tests.py | Unify training logic, add MuSSED path | HIGH | MEDIUM |
| 02b_aggregate_results.py | Support detection_metrics.csv aggregation | MEDIUM | LOW |
| 03_station_scramble.py | Add event detection evaluation | MEDIUM | LOW |
| 04_zero_shot.py | Add event detection evaluation | MEDIUM | LOW |
| 05_progressive_finetuning.py | Add event detection fine-tuning | MEDIUM | MEDIUM |

---

## Notes for Implementation

- **Code Simplicity**: Use direct if/else blocks for trainer selection based on `trainer_kind`
- **Clear Control Flow**: No abstract base classes or factory patterns—straightforward conditional logic
- **Documentation**: Each function should have clear docstrings explaining inputs/outputs/shapes
- **Testing**: Create unit tests for loss component extraction and metrics computation
- **Performance**: Profile attention weight extraction (may be slow for large batches)
- **CSV Compatibility**: Main CSV must have identical columns for segmentation and detection models
  - Detection models map their unique metrics to "compatible" columns for backward compatibility
  - Analysis scripts that read main CSV will work seamlessly across all model types
- **Attention Weight Extraction**: Use actual weights from hooks, not gradients for reproducibility
- **Validation Plot Storage**: Implement efficient random sampling to avoid excessive disk usage

---

**Document Status**: ✅ APPROVED, SIMPLIFIED (no polymorphism), and READY for implementation.

