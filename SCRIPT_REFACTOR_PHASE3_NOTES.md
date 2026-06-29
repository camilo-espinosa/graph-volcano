# Script Refactor Phase 3 Notes

## Scope
- scripts/03_zero_shot_cross_volcano.py
- scripts/04_cross-volcano.py

## Objective
Continue improving script intuitiveness by moving orchestration glue into `main` while keeping heavy reusable/evaluation primitives as dedicated helpers.

## Changes Applied
### scripts/03_zero_shot_cross_volcano.py
- Inlined model folder discovery and model selection logic into `main`.
- Inlined row-field assembly into `main`.
- Inlined per-row CSV append routing (global/by_target/by_fold) into `main`.
- Removed thin wrappers:
  - `discover_model_folders(...)`
  - `select_model_keys(...)`
  - `build_row_fieldnames(...)`
  - `append_progress_rows(...)`

### scripts/04_cross-volcano.py
- Inlined fold discovery/validation into `main`.
- Inlined model selection into `main`.
- Inlined held-out volcano extraction from test manifest into fold loop.
- Inlined output fieldnames assembly into `main`.
- Inlined final summary report generation into `main`.
- Removed thin wrappers:
  - `discover_fold_dirs(...)`
  - `select_model_keys(...)`
  - `held_out_from_test_manifest(...)`
  - `build_row_fieldnames(...)`
  - `write_summary(...)`

## Keep Shared (Utils)
- Reusable low-level/engine functions remain in shared modules:
  - training/evaluation engines
  - data loading datasets
  - confusion matrix utilities
  - metric/statistics primitives

## Validation
- Diagnostics: no errors in both scripts.
- CLI smoke tests pass:
  - scripts/03_zero_shot_cross_volcano.py --help
  - scripts/04_cross-volcano.py --help
