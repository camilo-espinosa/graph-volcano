# Script Refactor Phase 1 Notes

## Scope
- scripts/01_prepare_data.py
- scripts/01b_prepare_cross-volcano_data.py

## Objective
Expose the full data-preparation protocol directly in each script so readers can follow the pipeline without jumping to a large orchestration function in utils.

## Changes Applied
- Inlined NVCHVC 5-fold generation flow in scripts/01_prepare_data.py main.
- Inlined cross-volcano leave-one-out generation flow in scripts/01b_prepare_cross-volcano_data.py main.
- Removed dependency on high-level orchestrators:
  - generate_nvchvc_manifests
  - generate_cross_volcano_leave_one_out_manifests

## Keep Local (Script-owned)
These represent protocol orchestration and should stay visible in scripts:
- Fold loop ordering and split writing sequence.
- Manifest field assembly specific to each protocol.
- Target-per-class allocation logic for cross-volcano train pool.
- Per-fold augmentation trigger conditions and output naming conventions.

## Keep Shared (Utils)
These are generic and reusable across scripts:
- collect_volcano_samples
- build_stratified_kfold_specs
- _stratified_train_val_split_from_train
- save_manifest
- expand_training_set_with_augmentation
- augment_trace_for_storage
- CLASS_TO_ID and VALID_CLASSES

## Rationale
- Script-level clarity improves when the experiment protocol is explicit in main.
- Utility-level reuse remains for low-level stable operations (I/O, stratified split helpers, augmentation primitives).
- This reduces "orchestration hiding" in utils while avoiding duplicated low-level code.

## Validation
- Both refactored scripts pass diagnostics after inlining.
