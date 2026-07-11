# Script Cleanup Plan

## Goal
Simplify the active analysis scripts so they are focused on:

- running the models that are actually available in the target experiment folder
- the current PhaseNet and UNet workflow only
- minimal CLI surface
- minimal dead branching and helper code

## Current Observations

The current active scripts still carry structure for older workflows that are no longer used:

- `scripts/02_ablation_tests.py`
- `scripts/03_evaluate_nvchvc_station_scramble.py`
- `scripts/04_zero_shot_cross_volcano.py`
- `scripts/04b_zero_shot_cross_volcano_scrambled.py`

The main sources of unnecessary complexity are:

- `--family` filtering even though runs are not separated by family anymore
- branching for UNet vs graph vs PhaseNet evaluation
- graph-specific imports and evaluation helpers in scripts that should now target only PhaseNet and UNet
- CSV/report fields that keep family metadata even when it does not drive execution
- model selection logic that starts from registry/family groupings instead of from the current target folder or current active model set
- legacy training wording and defaults in `02_ablation_tests.py` that still refer to older graph-centric workflows

The active workflow should keep PhaseNet and UNet support. The graph-based paths are the legacy structure that should be removed from the active scripts.

## Cleanup Principles

- Keep each stage independently runnable.
- Prefer deleting dead branches over adding new abstraction.
- Start from the scripts you actually run now: `02`, `03`, `04`, `04b`.
- Select models from the target folder first, then validate against the registry.
- Only remove helpers after the active scripts no longer depend on them.
- Preserve outputs and filenames unless a stage explicitly changes them.

## Proposed Stages

## Stage 1: Simplify Script Entry Points

Scope:

- `scripts/02_ablation_tests.py`
- `scripts/03_evaluate_nvchvc_station_scramble.py`
- `scripts/04_zero_shot_cross_volcano.py`
- `scripts/04b_zero_shot_cross_volcano_scrambled.py`

Changes:

- remove `--family`
- remove family-based selection logic
- make model selection reflect the actual active model set for each script
- for evaluation scripts, select models from the available folders under the target experiment root
- keep optional `--models` override, but validate only against the current valid model set for that script
- keep current outputs unchanged

Acceptance criteria:

- the four scripts still run
- evaluation model discovery is driven by the target folder contents
- training model discovery/default selection in `02` is consistent with the current active model set
- no `--family` argument remains in the active scripts

## Stage 2: Remove Graph-Based Branching From Active Scripts

Scope:

- same four active scripts, with graph-branch removal applied wherever it still exists

Changes:

- remove graph-specific evaluation branches
- keep only the PhaseNet and UNet evaluation paths
- keep the current 1D multistation dataset path used by PhaseNet-style models (`MultiStation1DDataset`)
- remove no-longer-used imports tied to graph-only evaluation logic

Acceptance criteria:

- each script has only the PhaseNet and UNet model-loading/training paths it still needs
- scripts no longer dispatch through graph-only model branches
- the 1D multistation dataset path remains available for PhaseNet-style models
- outputs remain compatible with current downstream analysis where possible

## Stage 3: Extract Shared PhaseNet/UNet Evaluation Utilities

Scope:

- `utils/train_utils.py` or a new focused helper module if clearly smaller
- scripts `02`, `03`, `04`, `04b`

Changes:

- consolidate duplicated active-model loading and evaluation logic shared by `02`, `03`, `04`, and `04b`
- keep scramble support configurable but centralized
- keep target-specific discovery local to the scripts

Acceptance criteria:

- duplicated PhaseNet and UNet active-workflow code between `02`, `03`, `04`, and `04b` is materially reduced
- scramble behavior is implemented once, not repeated in multiple scripts

## Stage 4: Remove Dead Registry and Metadata Surface Used Only By Old Families

Scope:

- `utils/model_registry.py`
- active scripts only

Changes:

- remove metadata that only existed to dispatch by family in active workflows if it is no longer needed
- simplify selection helpers around the current model set
- keep aliases and checkpoint compatibility where useful

Acceptance criteria:

- active scripts do not depend on family dispatch metadata
- registry remains the single source of available model definitions

## Stage 5: Remove Unused Helper Functions and Methods

Scope:

- active scripts
- utility modules touched by those scripts

Changes:

- delete helpers that become unused after stages 1 to 4
- remove stale docstrings and comments referring to graph-family execution or family-filter-driven usage
- keep historical scripts untouched unless they block cleanup

Acceptance criteria:

- no dead local functions remain in `02`, `03`, `04`, `04b`
- helper modules do not keep orphaned support code for removed active paths

## Stage 6: Documentation and Validation Pass

Scope:

- `README.md`
- `README_SCRIPTS_DETAILED.md`
- any notes that describe the active script flow

Changes:

- update docs to reflect folder-driven model discovery and the active PhaseNet plus UNet evaluation workflow
- document the simplified CLI for `02`, `03`, `04`, `04b`
- run focused validation

Acceptance criteria:

- docs match the actual script behavior
- `py_compile` passes for touched scripts
- no stale references to family-based execution remain in the active workflow docs

## Recommended Order

1. Stage 1
2. Stage 2
3. Stage 3
4. Stage 5
5. Stage 4
6. Stage 6

Reason for this order:

- first simplify the user-facing script behavior
- then collapse the execution paths
- then extract shared code only after the simplified shape is clear
- then delete dead code safely
- then shrink registry metadata once the callers are stable
- finish with docs and validation

## Notes For Execution

- Do not clean the preserved historical scripts first.
- Do not remove report columns unless they are confirmed unused downstream.
- Prefer one small stage at a time with validation after each stage.
- If a helper is shared by active and historical scripts, only remove it after checking the historical script impact.
