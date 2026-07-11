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
Builds the NVCHVC prepared dataset structure and 5-fold manifests used by baseline ablation training.

### Main Function
- `main()` - coordinates the data preparation workflow.

### Inputs
- Raw/prepared data under `data/`.

### Outputs
- Prepared NVCHVC fold data and metadata used by the downstream scripts.

## 01b_prepare_cross-volcano_data.py

### Purpose
Builds the leave-one-out cross-volcano manifests used by the new cross-volcano protocols.

### Main Functions
- `main()` - configures source volcanoes, target-per-class sampling, augmentation policy, and writes fold artifacts.
- `generate_cross_volcano_leave_one_out_manifests(...)` (from `utils/data_utils.py`) - generates train/val/test manifests per held-out volcano.

### Inputs
- Prepared volcano data under `data/`.

### Outputs
- `data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/train.npz`
- `data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/val.npz`
- `data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/test.npz`

### Review Notes
- `test.npz` contains the full held-out volcano set for each fold.
- Manifests include extra metadata fields like `held_out_volcano`, `volcano_idx`, and `descriptor_paths`.

### Review Notes
- This is the preprocessing entry point. It should be the first script run before any training or evaluation.

## 02_ablation_tests.py

### Purpose
Runs 5-fold cross-validation training for the active registry model set on NVCHVC.

### Main Functions
- `parse_args()` - parses selected model keys and output root.
- `select_model_keys(raw_models)` - validates selected model keys against the active registry.
- `main()` - trains all selected models across the 5 NVCHVC folds.

### Inputs
- Fold data from `data/prepared_data/NVCHVC/cv_5fold/fold_XX/`.

### Outputs
- Training outputs under `results/experiments/<experiment_name>/ablations/<model_key>/fold_XX/`.

### Review Notes
- This script is the main ablation training driver.
- It now defaults to the active registry entries and does not use family-based selection.

## 03_evaluate_nvchvc_station_scramble.py

### Purpose
Evaluates the trained ablation checkpoints on each NVCHVC fold test split after randomly scrambling station order per sample.

### Main Functions
- `parse_args()` - parses experiment root, NVCHVC fold root, model filters, and scramble options.
- `evaluate_multistation_checkpoint_on_test_fold(...)` - computes 1D multistation metrics on scrambled fold tests.
- `evaluate_unet_checkpoint_on_test_fold(...)` - computes UNet metrics on scrambled fold tests.
- `write_aggregate_reports(out_dir)` - writes summary and ranking CSVs for the scramble sensitivity comparison.
- `main()` - orchestrates the fold-aligned checkpoint evaluation matrix.

### Inputs
- Checkpoints under `results/experiments/complete_experiment/ablations/<model_key>/fold_XX/checkpoints/best_f1.pt`.
- NVCHVC fold tests under `data/prepared_data/NVCHVC/cv_5fold/fold_XX/test.npz`.

### Outputs
- Reports under `<experiment_root>/nvchvc_station_scramble_eval/`.

### Review Notes
- This script isolates station-order sensitivity on the in-distribution NVCHVC test splits.
- Station permutations are deterministic for a given `--station-scramble-seed`.

## 04_zero_shot_cross_volcano.py

### Purpose
Evaluates model checkpoints in zero-shot mode on full held-out volcano test sets generated by script `01b_prepare_cross-volcano_data.py`.

### Main Functions
- `parse_args()` - parses experiment root, cross-data root, model filters, and output options.
- `discover_loo_target_test_paths(cross_data_root)` - resolves each held-out volcano to its `test.npz` path from `cross_volcano_loo`.
- `evaluate_multistation_checkpoint_on_target(...)` - computes 1D multistation metrics.
- `evaluate_unet_checkpoint_on_target(...)` - computes UNet metrics.
- `write_aggregate_reports(out_dir)` - writes summary and ranking CSVs.
- `main()` - orchestrates the zero-shot matrix across model/fold/target combinations.

### Inputs
- Checkpoints under `results/experiments/complete_experiment/ablations/<model_key>/fold_XX/checkpoints/best_f1.pt`.
- Cross-volcano held-out tests under `data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/test.npz`.

### Outputs
- Zero-shot reports under `<experiment_root>/zero_shot_cross_volcano/`.

### Review Notes
- This script now evaluates on full held-out volcano sets from the leave-one-out generator, not `test_80.npz`.
- New active outputs are model-centric and no longer include a `family` field in the fold/summary CSVs.

## 04b_zero_shot_cross_volcano_scrambled.py

### Purpose
Evaluates the same zero-shot cross-volcano checkpoints as script `04`, but with randomized station ordering per sample.

### Main Functions
- Same evaluation flow as `04_zero_shot_cross_volcano.py`, with station scrambling enabled by default.

### Inputs
- Checkpoints under `results/experiments/complete_experiment/ablations/<model_key>/fold_XX/checkpoints/best_f1.pt`.
- Cross-volcano held-out tests under `data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/test.npz`.

### Outputs
- Reports under `<experiment_root>/zero_shot_cross_volcano_scrambled/`.

### Review Notes
- This is the zero-shot analogue of script `03`, intended to quantify how much station-order dependence transfers out of distribution.

## cross_volcano_leave_one_out.py

### Purpose
Runs leave-one-out cross-volcano training and evaluation for all supported model families.

### Main Functions
- `parse_args()` - parses model/fold selection and training settings.
- `CrossVolcanoLOODataset` (from `utils/train_utils.py`) - mixed-volcano dataset with `volcano_idx` and optional descriptors.
- Graph wrappers/extractors - inject optional dynamic edge features and RSAM when required.
- `main()` - orchestrates fold discovery, training, checkpointing, and held-out evaluation.

### Inputs
- Leave-one-out manifests under `data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/`.
- Model definitions and registry entries from `utils/model_registry.py`.

### Outputs
- Per-fold checkpoints and metrics under the configured experiment output.

### Review Notes
- This preserved script is no longer part of the numbered analysis sequence, but it keeps the full leave-one-out training workflow available.

## cross_volcano_leave_one_out_eval_only.py

### Purpose
Loads existing leave-one-out checkpoints and evaluates them on their corresponding held-out test sets without retraining.

### Main Functions
- `parse_args()` - parses data root, weights root, output location, and model filters.
- Graph wrappers/extractors - preserve optional descriptor and volcano-index routing for eval-only runs.
- `main()` - orchestrates per-fold checkpoint loading and held-out evaluation.

### Inputs
- Leave-one-out manifests under `data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/test.npz`.
- Trained checkpoints under the selected weights root.

### Outputs
- Eval-only reports under the configured output folder.

### Review Notes
- This preserved script is also outside the numbered analysis sequence.

## 05_progressive_finetuning.py

### Status
Current placeholder for the progressive finetuning workflow.

## 06_continuous_tests.py

### Status
Current placeholder for the continuous tests workflow.

## 07_detection_heads.py

### Status
Current placeholder for the detection heads workflow.
- `initialize_weights(model)` - initializes weights for model layers.

### Review Notes
- This file is legacy and partially overlaps with the modern utilities in `utils/`.
- It is useful as a reference, but the newer scripts should rely on `utils/train_utils.py` and the shared helpers added in this refactor.
