# Graph Volcano

Graph Volcano contains the training and evaluation pipelines for volcano-seismic segmentation experiments, including NVCHVC 5-fold experiments and leave-one-out cross-volcano protocols.

## Repository Layout

- `data/`: prepared datasets, manifests, and fold artifacts.
- `models/`: model architectures.
- `scripts/`: experiment entry points.
- `utils/`: shared data, training, and metrics helpers.
- `results/`: checkpoints, reports, and experiment outputs.

## Requirements

Install dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Script Reference (Current)

- `scripts/01_prepare_data.py`: builds NVCHVC prepared data and 5-fold manifests.
- `scripts/01b_prepare_cross-volcano_data.py`: builds leave-one-out cross-volcano manifests under `data/prepared_data/cross_volcano_loo/`.
- `scripts/01c_edge_data.py`: precomputes dynamic edge features and RSAM artifacts for graph workflows.
- `scripts/02_ablation_tests.py`: runs 5-fold ablation training and per-ablation aggregation.
- `scripts/02b_aggregate_ablation_results.py`: re-aggregates already completed ablation folds into summary/comparison tables.
- `scripts/03_zero_shot_cross_volcano.py`: evaluates trained checkpoints in zero-shot mode on held-out volcano test sets.
- `scripts/04_cross-volcano.py`: leave-one-out cross-volcano training/evaluation protocol.
- `scripts/05_continuous-tests.py`: placeholder (currently empty).
- `scripts/06_detect_head.py`: placeholder (currently empty).
- `scripts/ablation_param_counts.py`: prints parameter counts for ablation variants.

## Typical Workflow

Run scripts from the repository root.

### 1. Prepare datasets

```bash
python scripts/01_prepare_data.py
python scripts/01b_prepare_cross-volcano_data.py
python scripts/01c_edge_data.py
```

### 2. Train ablation models (5-fold)

```bash
python scripts/02_ablation_tests.py --mode train
```

### 3. Aggregate ablation outputs (optional if already trained)

```bash
python scripts/02b_aggregate_ablation_results.py
```

### 4. Evaluate zero-shot cross-volcano performance

```bash
python scripts/03_zero_shot_cross_volcano.py
```

### 5. Run leave-one-out cross-volcano training/evaluation

```bash
python scripts/04_cross-volcano.py
```

## Outputs

- Main experiment outputs are written to `results/experiments/<experiment_name>/`.
- Many evaluation/aggregation scripts default to `results/experiments/complete_experiment`.
- `results/latest/` stores pointers to most recent artifacts.

## Notes

- Use `README_SCRIPTS_DETAILED.md` for a script-by-script technical breakdown.
- Script status in this README reflects the current repository files.