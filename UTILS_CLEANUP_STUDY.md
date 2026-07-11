# Utils Cleanup Study

## Scope
This study covers two surfaces:

- repository utility modules under `utils/`
- helper functions/classes still defined inside `scripts/*.py`

This is a study only. No code changes are proposed here beyond staged recommendations.

## High-Level Findings

The current utility surface has three distinct states:

1. Active and cohesive
- `active_eval_utils.py`
- `fold_io_utils.py`
- `script_common.py`
- `station_info.py`

2. Active but overloaded / overlapping
- `train_utils.py`
- `data_utils.py`

3. Dormant or likely removable after verification
- `finetune_utils.py`
- `model_utils.py`

The biggest structural problem is not the number of utility files. It is that `train_utils.py` still contains too many unrelated responsibilities:

- losses
- confusion-matrix and event metrics
- plotting helpers
- dataset classes
- sampler classes
- training loops
- augmentation helpers
- CSV summary helpers

That file is currently the main cleanup bottleneck.

## Current Usage Map

### Clearly active in current workflow

- `utils/model_registry.py`
- `utils/train_utils.py`
- `utils/active_eval_utils.py`
- `utils/fold_io_utils.py`
- `utils/script_common.py`
- `utils/data_utils.py`
- `utils/metrics_report_utils.py`
- `utils/station_info.py`

### Preserved / historical workflows still reading them

- `utils/station_info.py`
- `utils/train_utils.py`
- `utils/model_registry.py`
- `utils/fold_io_utils.py`
- `utils/script_common.py`

These are still used by:
- `scripts/cross_volcano_leave_one_out.py`
- `scripts/cross_volcano_leave_one_out_eval_only.py`

### No in-repo imports found

- `utils/finetune_utils.py`
- `utils/model_utils.py`

These are strong delete candidates, but only after confirming they are not invoked manually from notebooks or ad hoc local scripts outside this repo surface.

## File-by-File Study

## `utils/train_utils.py`

### Current role
This is effectively a mixed kitchen-sink module.

### What is active and should stay somewhere
- loss functions
- confusion matrix / event metrics
- `UNetPatchDataset`
- `MultiStation1DDataset`
- `CrossVolcanoLOODataset`
- `BalancedBatchSampler`
- training loops for script `02`
- visualization helpers used during training/eval
- `cleanup_gpu_cache`
- `compute_summary`

### Clear overlap / duplication
- augmentation helpers are duplicated with `utils/data_utils.py`
  - `random_time_shift`
  - `amplitude_scaling`
  - `add_noise`
- `compute_iou_from_cm` exists both here and in `utils/metrics_report_utils.py`

### Likely unused inside repo
- `event_iou_like_score`

### Recommendation
Do not keep growing this file. It should be split by responsibility rather than by model family.

## `utils/data_utils.py`

### Current role
Data preparation, patch stacking, manifest generation, and storage-time augmentation.

### Issue
It overlaps with `train_utils.py` on augmentation helpers.

### Recommendation
This should become the single home for augmentation primitives and preprocessing transforms. Training/eval utilities should import from here instead of maintaining parallel augmentation helpers.

## `utils/active_eval_utils.py`

### Current role
Good direction. It already centralizes active-script evaluation behavior.

### Recommendation
Keep it. Expand it slightly if needed for active-script evaluation/report orchestration, but do not let it turn into a second `train_utils.py`.

## `utils/fold_io_utils.py`

### Current role
Small and coherent:
- checkpoint path resolution
- fold summary loading
- append CSV row
- resumable key loading

### Recommendation
Keep as-is or merge with `script_common.py` only if you want a single tiny “run_io” module. It is already reasonably scoped.

## `utils/script_common.py`

### Current role
Small and coherent:
- path resolution
- CSV selection parsing
- target discovery

### Recommendation
Keep as-is unless you explicitly want to merge tiny path/selection helpers into a more general run helper module.

## `utils/metrics_report_utils.py`

### Current role
Very small reporting/stat helper module.

### Issue
Contains functionality that overlaps conceptually with report code still embedded in scripts and with `compute_iou_from_cm` duplicated in `train_utils.py`.

### Recommendation
This should either:
- grow into a true reporting module and absorb script-level aggregate writers, or
- be folded into a broader `report_utils.py`

## `utils/station_info.py`

### Current role
Station/crater metadata and geometry helpers.

### Recommendation
Keep. This is cohesive and understandable.

## `utils/model_registry.py`

### Current role
Model definition registry.

### Recommendation
Keep. It is no longer the main cleanup problem after the recent simplification.

## `utils/model_utils.py`

### Current role
Legacy model builder and checkpoint helpers.

### Study result
No in-repo imports found.

### Recommendation
Candidate for deletion after one manual confirmation step:
- verify no notebooks, local scratch scripts, or external entry points still import it

If retained, it should be explicitly marked legacy.

## `utils/finetune_utils.py`

### Current role
Finetune split/protocol helpers.

### Study result
No in-repo imports found.

### Recommendation
Do not delete immediately if script `05` is expected to use it soon.

Best path:
- keep for now
- mark as dormant/planned for `05`
- delete later only if script `05` takes a different shape

## Script-Embedded Helper Study

## `scripts/02_ablation_tests.py`

### Current state
Good now. Only lightweight script-local helpers remain:
- `parse_args()`
- `select_model_keys()`

### Recommendation
Keep local. They are script-specific and intuitive.

## `scripts/02b_aggregate_ablation_results.py`

### Current embedded helpers
- `write_ablation_aggregate(...)`
- `write_global_comparisons(...)`

### Recommendation
These are good extraction candidates.
They are not just small script glue; they are reusable report writers.

Best destination:
- a dedicated `report_utils.py` or expanded `metrics_report_utils.py`

## `scripts/03_evaluate_nvchvc_station_scramble.py`

### Current embedded helpers
- `parse_args()`
- `write_aggregate_reports(...)`

### Recommendation
- `parse_args()` should stay local
- `write_aggregate_reports(...)` is a merge candidate with the zero-shot report writer pattern in `04` and `04b`

## `scripts/04_zero_shot_cross_volcano.py`

### Current embedded helpers
- `parse_args()`
- `discover_loo_target_test_paths(...)`
- `write_aggregate_reports(...)`

### Recommendation
- `parse_args()` should stay local
- `discover_loo_target_test_paths(...)` is duplicated with `04b` and should be shared
- `write_aggregate_reports(...)` should likely merge with the corresponding function in `04b`, and possibly share a report-building core with `03`

## `scripts/04b_zero_shot_cross_volcano_scrambled.py`

### Current embedded helpers
Same pattern as `04`.

### Recommendation
This should not own separate copies long-term.
Use shared report/discovery helpers with only scramble behavior parameterized.

## Preserved scripts

### `scripts/cross_volcano_leave_one_out.py`
### `scripts/cross_volcano_leave_one_out_eval_only.py`

These still contain many embedded helpers.

### Recommendation
Do not prioritize them in the first cleanup pass.
They are preserved workflows, not the active intuitive script surface.

Only touch them later if:
- they are brought back into active use
- or their helpers block cleanup in shared modules

## Concrete Merge / Delete Candidates

## Strong merge candidates

1. Merge duplicated augmentation primitives into one place
Files involved:
- `utils/data_utils.py`
- `utils/train_utils.py`

2. Merge report-writing logic
Files involved:
- `scripts/02b_aggregate_ablation_results.py`
- `scripts/03_evaluate_nvchvc_station_scramble.py`
- `scripts/04_zero_shot_cross_volcano.py`
- `scripts/04b_zero_shot_cross_volcano_scrambled.py`
- `utils/metrics_report_utils.py`

3. Merge path/CSV run helpers only if you want fewer tiny modules
Files involved:
- `utils/fold_io_utils.py`
- `utils/script_common.py`

This is optional, not necessary.

## Strong delete candidates after verification

1. `utils/model_utils.py`
Reason:
- no in-repo imports found

2. `event_iou_like_score` in `utils/train_utils.py`
Reason:
- no in-repo callers found

## Hold, do not delete yet

1. `utils/finetune_utils.py`
Reason:
- currently dormant, but likely relevant for planned script `05`

2. preserved-script helpers in:
- `scripts/cross_volcano_leave_one_out.py`
- `scripts/cross_volcano_leave_one_out_eval_only.py`

Reason:
- preserved workflow, not active cleanup priority

## Proposed Target Grouping

If you want a cleaner end-state, the utility modules should move toward this shape:

- `utils/data_utils.py`
  - data preparation
  - patch stacking
  - augmentation primitives
  - manifest generation

- `utils/dataset_utils.py`
  - `UNetPatchDataset`
  - `MultiStation1DDataset`
  - `CrossVolcanoLOODataset`
  - `BalancedBatchSampler`

- `utils/eval_utils.py`
  - multistation evaluation
  - UNet evaluation
  - confusion/F1/IoU helpers
  - checkpoint loading for eval

- `utils/report_utils.py`
  - aggregate writers for `02b`, `03`, `04`, `04b`
  - CSV append/load helpers
  - per-class summary helpers

- `utils/train_utils.py`
  - training loops only
  - no datasets
  - no report writers
  - minimal plotting helpers only if training-specific

- `utils/script_common.py`
  - keep or merge into `report_utils.py` / `run_io_utils.py`

This grouping is more intuitive than the current mixed responsibilities.

## Proposed Stages

## Stage U1: Usage Confirmation Pass

Goal:
Classify utilities into active, preserved, dormant, and delete-candidate.

Actions:
- confirm whether `utils/model_utils.py` is used outside the repo scripts
- confirm whether `utils/finetune_utils.py` is intended for script `05`
- confirm whether preserved cross-volcano scripts must remain runnable during cleanup

Deliverable:
- final keep/delete list

## Stage U2: Extract Script Report Writers

Goal:
Remove non-trivial report-writing logic from active scripts while keeping script entry points intuitive.

Actions:
- extract `write_ablation_aggregate(...)` and `write_global_comparisons(...)` from `02b`
- extract/merge `write_aggregate_reports(...)` from `03`, `04`, `04b`
- extract/merge `discover_loo_target_test_paths(...)` from `04` and `04b`

Deliverable:
- new `utils/report_utils.py` or expanded `metrics_report_utils.py`
- active scripts left with `parse_args()`, `main()`, and small local orchestration only

## Stage U3: Split `train_utils.py` By Responsibility

Goal:
Break the current monolithic utility file into intuitive modules.

Actions:
- move dataset classes and sampler out
- move evaluation metrics/helpers out
- keep only training-loop-specific logic in `train_utils.py`

Deliverable:
- smaller `train_utils.py`
- separate dataset/eval utility modules

## Stage U4: Deduplicate Augmentation Helpers

Goal:
Resolve duplicated augmentation primitives between `data_utils.py` and `train_utils.py`.

Actions:
- choose one canonical home, preferably `data_utils.py`
- replace duplicate definitions with imports

Deliverable:
- single source of truth for trace augmentation helpers

## Stage U5: Delete Verified Unused Utilities

Goal:
Remove dormant/dead code only after usage is confirmed.

Actions:
- delete `utils/model_utils.py` if confirmed unused
- delete orphan helpers such as `event_iou_like_score`
- keep `finetune_utils.py` only if script `05` will use it

Deliverable:
- reduced utility surface

## Stage U6: Optional Tiny-Module Consolidation

Goal:
Decide whether very small modules should stay separate or merge.

Actions:
- evaluate `fold_io_utils.py` + `script_common.py`
- decide whether the current split is clearer than a single `run_io_utils.py`

Deliverable:
- either keep the current tiny modules intentionally
- or merge them into one clearer small module

## Recommended Order

1. Stage U1
2. Stage U2
3. Stage U3
4. Stage U4
5. Stage U5
6. Stage U6

Reason:
- first confirm what can actually be deleted
- then extract script-local helpers before moving lower-level utility pieces
- then split the monolithic utility file
- then remove duplication
- only then delete dormant code
- leave tiny-module consolidation for last because it is stylistic, not structural

## Guardrails

- Do not delete `finetune_utils.py` before deciding the shape of script `05`.
- Do not delete `model_utils.py` before one explicit confirmation step.
- Do not refactor preserved cross-volcano scripts first.
- Keep active scripts intuitive: local argument parsing and `main()` orchestration should stay in the script files.
- Prefer extracting only the helpers that are substantial or duplicated across scripts.
