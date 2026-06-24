# Detailed Script Reference

This document explains what each script in `scripts/` does, the main functions it uses, and the expected inputs/outputs. It is intended as a review aid to confirm that each script is doing what it is supposed to do.

## Shared Utilities

These helpers are used across multiple scripts after the refactor.

- `utils/script_common.py`
  - `resolve_project_path(path, project_root)` - resolves relative paths from the repository root.
  - `parse_csv_selection(raw_csv, available, name)` - validates comma-separated selections.
  - `discover_targets(cross_data_root, required_files=None)` - finds target folders that contain the required artifacts.
- `utils/fold_io_utils.py`
  - `checkpoint_path_for_fold(root, fold_id, checkpoint_name="best_f1.pt")` - builds the standard checkpoint path.
  - `load_fold_summary(root, fold_id, candidate_relative_paths=None)` - loads fold summary JSON from a fold folder.
  - `load_training_fold_summary(root, fold_id)` - convenience wrapper for training summaries.
  - `append_row_csv(csv_path, row, fieldnames)` - appends a row to a semicolon-separated CSV.
  - `load_completed_keys(csv_path, id_columns)` - reads completed evaluation keys for resumable runs.
- `utils/metrics_report_utils.py`
  - `compute_iou_from_cm(cm)` - computes per-class and mean IoU from a confusion matrix.
  - `compute_per_class_summary(per_fold_values, class_names)` - computes summary statistics per class.
- `utils/finetune_utils.py`
  - `split_indices_stratified(label_ids, val_frac, seed)` - reproducible stratified train/validation split.
  - `apply_finetune_protocol(model, protocol_key)` - freezes/unfreezes parameters for finetuning protocols.
  - `trainable_parameter_count(model)` - counts trainable and total parameters.

## 01_prepare_data.py

### Purpose
Builds the prepared dataset structure and manifests used by the training and evaluation scripts.

### Main Function
- `main()` - coordinates the data preparation workflow.

### Inputs
- Raw/prepared data under `data/`.

### Outputs
- Prepared fold data and metadata used by the downstream scripts.

### Review Notes
- This is the preprocessing entry point. It should be the first script run before any training or evaluation.

## 02_ablation_tests.py

### Purpose
Runs 5-fold cross-validation training for the UNet_GraphSAGE ablation set and aggregates the fold results.

### Main Functions
- `parse_args()` - parses mode, experiment root, ablation list, and aggregation options.
- `select_ablations(raw_ablations)` - validates selected ablation names.
- `write_ablation_aggregate(aggregate_dir, ablation_name, fold_summaries)` - writes the per-ablation CSV and JSON summaries.
- `write_global_comparisons(experiment_root, leaderboard_rows)` - writes the cross-ablation ranking tables.
- `run_train_mode(device, selected, experiment_root, experiment_name)` - trains all selected ablations across folds.
- `run_aggregate_only_mode(selected, experiment_root, require_all_folds)` - rebuilds summaries from saved fold outputs.
- `main()` - dispatches to training or aggregation-only mode.

### Inputs
- Fold data from `data/prepared_data/NVCHVC/cv_5fold/fold_XX/`.

### Outputs
- Training outputs under `results/experiments/<experiment_name>/ablations/<ablation>/fold_XX/`.
- Aggregates under each ablation folder and global leaderboards under `comparisons/`.

### Review Notes
- This script is the main ablation training driver.
- It now uses shared helpers for path resolution, fold summary loading, and per-class summaries.

## 03_unet_5fold.py

### Purpose
Runs 5-fold training for the baseline 2D UNet model and writes fold-level and aggregate reports.

### Main Functions
- `parse_args()` - parses the experiment root argument.
- `compute_iou_from_cm(cm)` - shared IoU helper, now imported from `utils/metrics_report_utils.py`.
- `evaluate_unet(...)` - evaluates the UNet on a dataloader and returns loss/F1/IoU metrics.
- `train_one_unet_fold(...)` - trains a single fold with early stopping and checkpointing.
- `write_aggregate(experiment_root, fold_summaries)` - writes the combined fold summaries and leaderboards.
- `main()` - orchestrates the 5-fold UNet training run.

### Inputs
- Fold data from `data/prepared_data/NVCHVC/cv_5fold/fold_XX/`.

### Outputs
- Fold checkpoints, reports, and aggregate summaries under the experiment folder.

### Review Notes
- This is the baseline UNet counterpart to the ablation training workflow.

## 04_zero_shot_cross_volcano.py

### Purpose
Evaluates ablation checkpoints on cross-volcano test data without additional fine-tuning.

### Main Functions
- `parse_args()` - parses experiment root, cross-data root, output directory, and filtering options.
- `discover_ablations(ablations_root)` - returns the ablation folders in canonical experiment order.
- `evaluate_checkpoint_on_target(...)` - runs inference and computes event-level metrics for one target volcano.
- `load_checkpoint_into_model(...)` - loads the checkpoint while preserving geometry-specific buffers.
- `append_progress_rows(...)` - writes one evaluation row to the resumable CSV outputs.
- `write_aggregate_reports(out_dir)` - builds summary tables and rankings from all evaluation rows.
- `main()` - orchestrates the zero-shot evaluation.

### Inputs
- Ablation checkpoints under `results/experiments/complete_experiment/ablations/<ablation>/fold_XX/checkpoints/best_f1.pt`.
- Cross-volcano test artifacts under `data/prepared_data/cross_volcano/<VOLCANO>/test_80.npz`.

### Outputs
- Zero-shot reports under `<experiment_root>/zero_shot_cross_volcano/`.

### Review Notes
- This script is intended to evaluate the ablations trained in script 02.
- It uses shared helpers for path resolution, target discovery, checkpoint path construction, and CSV I/O.

## 05_zero_shot_unet_cross_volcano.py

### Purpose
Evaluates UNet checkpoints on cross-volcano test data without additional fine-tuning.

### Main Functions
- `parse_args()` - parses experiment root, cross-data root, model filters, and output options.
- `load_unet_shape_and_loss(experiment_root)` - reads model shape and loss weights from the experiment manifest.
- `evaluate_checkpoint_on_target(...)` - computes loss, F1, IoU, and confusion matrix metrics for one target volcano.
- `load_checkpoint_into_model(...)` - loads the model checkpoint into the UNet instance.
- `append_progress_rows(...)` - writes resumable evaluation rows to CSV files.
- `write_aggregate_reports(out_dir)` - groups results by model and target and writes summary tables.
- `main()` - orchestrates the UNet zero-shot evaluation.

### Inputs
- UNet checkpoints under the experiment root.
- Cross-volcano test artifacts under `data/prepared_data/cross_volcano/<VOLCANO>/test_80.npz`.

### Outputs
- Zero-shot reports under `<experiment_root>/zero_shot_cross_volcano_unet/`.

### Review Notes
- This is the UNet counterpart to script 04.
- Its default source experiment folder now points to `complete_experiment` for result loading.

## 06_ablations_finetune_cross_volcano.py

### Purpose
Fine-tunes ablation checkpoints on cross-volcano small-train splits and evaluates the fine-tuned models on `test_80`.

### Main Functions
- `parse_args()` - parses experiment root, cross-data root, target/protocol filters, and optional matrix saving.
- `load_base_manifest(experiment_root)` - loads the original training manifest for config and ablation metadata.
- `apply_finetune_protocol(model, protocol_key)` - now imported from `utils/finetune_utils.py`; switches between all-weights and decoder-only training.
- `split_indices_stratified(label_ids, val_frac, seed)` - imported helper for the 85/15 train/validation split.
- `trainable_parameter_count(model)` - imported helper for logging model size.
- `evaluate_graphsage(...)` - wraps GraphSAGE evaluation and metric extraction.
- `write_reports(output_dir, rows)` - writes fold-level and grouped summary CSVs.
- `main()` - orchestrates the full fine-tuning matrix.

### Inputs
- Base ablation checkpoints under `results/experiments/complete_experiment/ablations/...`.
- Cross-volcano train artifacts: `train_01pct.npz`, `train_05pct.npz`, `train_10pct.npz`, `train_20pct.npz`.
- Target test artifacts: `test_80.npz`.

### Outputs
- Fine-tuning outputs under `<experiment_root>/finetune_cross_volcano/`.

### Review Notes
- This script is the most configuration-heavy workflow in the repository.
- It now falls back to local ablation kwargs for entries that were added after the original manifest was created.

## 07_finetune_unets_cross_volcano.py

### Status
Currently empty placeholder file.

## 08_ablations_continuous.py

### Status
Currently empty placeholder file.

## 09_unet_continuous.py

### Status
Currently empty placeholder file.

## 10_train_obj_detection.py

### Status
Currently empty placeholder file.

## EXAMPLE_TRAIN.py

### Purpose
Legacy example training script and utility reference.

### Main Functions
- `load_legacy_swin_transformer()` - dynamically loads the legacy SwinTransformer implementation.
- `count_trainable_parameters(model)` - counts trainable parameters.
- `free_gpu_memory()` - clears CUDA cache and Python references.
- `print_time(t_i, t_f)` - formats elapsed time.
- `f1_score_from_confusion_matrix(confusion_matrix_)` - computes class-wise F1 scores.
- `dice_loss_2D(pred, target)` - computes legacy 2D Dice loss.
- `model_selector(arch, N=256)` - chooses a model architecture.
- `create_dataset(...)` - builds a dataframe from a folder structure.
- `img_to_trace_y(...)` and `img_to_trace_X(...)` - transform between patch-level and trace-level representations.
- `longest_event(BG_diff)` - finds the longest non-background event segment.
- `predicted_from_output(...)` - converts model output to event prediction.
- `cm_eval(...)` - event-level confusion-matrix evaluation.
- `cm_save(...)` - saves confusion matrix images.
- `initialize_weights(model)` - initializes weights for model layers.

### Review Notes
- This file is legacy and partially overlaps with the modern utilities in `utils/`.
- It is useful as a reference, but the newer scripts should rely on `utils/train_utils.py` and the shared helpers added in this refactor.
