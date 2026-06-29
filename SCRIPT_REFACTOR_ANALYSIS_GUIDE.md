# Script Refactor Analysis Guide

## Goal
Make scripts easier to read and run by keeping script-specific logic inside each script and moving only truly reusable pieces to shared utilities.

## Scope
Apply this analysis progressively, starting with:
- scripts/01_prepare_data.py
- scripts/01b_prepare_cross-volcano_data.py
Then continue with:
- scripts/02_ablation_tests.py
- scripts/02b_aggregate_ablation_results.py
- scripts/03_zero_shot_cross_volcano.py
- scripts/04_cross-volcano.py

## Principles
1. Prefer linear, top-to-bottom script flow for pipeline steps.
2. Keep helper functions local to a script if they are only used by that script.
3. Keep shared utilities only when used by two or more scripts, or when domain logic is large and stable.
4. Avoid utility growth that hides core experiment logic.
5. Preserve output formats and folder structure while refactoring.

## Decision Rules (Keep Local vs Shared)
For each imported helper, answer:
1. Is it used by only one script?
2. Is it central to understanding this script's experiment protocol?
3. Is it short or medium in length and safe to inline?
4. Does inlining improve readability for new collaborators?

If most answers are yes, keep it local (inline into script or define small local helper).

Keep in shared utils when:
- It is used in multiple scripts.
- It is low-level generic logic (for example, tensor transforms, reusable metrics math, generic file I/O helpers).
- It would cause duplication risk if copied.

## Refactor Workflow Per Script
1. Identify the script objective in one sentence.
2. List all imported utility functions.
3. Tag each function as Local Candidate or Shared Utility.
4. Inline Local Candidates into script flow (prefer main pipeline sections over many nested functions).
5. Keep Shared Utilities imported.
6. Run diagnostics and one smoke test command.
7. Confirm outputs are unchanged in structure and naming.

## Preferred Script Structure
1. Header docstring with purpose and run command.
2. Config/constants block.
3. Main pipeline with explicit numbered sections.
4. Minimal local helpers only when needed for readability.
5. if __name__ == "__main__": main()

## Review Checklist
- Script objective is obvious from top 30 lines.
- Core protocol steps are visible in main without jumping across files.
- Only reusable/generic logic remains in utils.
- No behavioral changes to data splits, folds, metrics, or output files.
- CLI flags still match expected usage.

## Progressive Plan
Phase 1:
- Analyze and refactor 01 and 01b.

Phase 2:
- Apply same method to 02 and 02b.

Phase 3:
- Apply same method to 03 and 04.

At the end of each phase, record:
- What moved into script.
- What stayed in utils and why.
- Any behavior-preservation checks run.
