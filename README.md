# Graph Volcano

This repository contains the code used for the volcano-seismic experiments in the article. It includes NVCHVC data preparation, leave-one-out cross-volcano data generation, multi-family model training, and zero-shot/fine-tuning evaluation pipelines.

## Repository Layout

- `data/` - prepared datasets and fold artifacts.
- `models/` - model definitions.
- `scripts/` - experiment entry points.
- `utils/` - reusable training, data, and metric helpers.
- `results/` - generated experiment outputs.

## Main Scripts

- `scripts/01_prepare_data.py` - prepares NVCHVC 5-fold manifests.
- `scripts/01b_prepare_cross-volcano_data.py` - builds leave-one-out cross-volcano manifests under `data/prepared_data/cross_volcano_loo/`.
- `scripts/02_ablation_tests.py` - runs 5-fold ablation training.
- `scripts/03_zero_shot_cross_volcano.py` - zero-shot evaluation of checkpoints on full held-out volcano test sets from `cross_volcano_loo`.
- `scripts/04_cross-volcano.py` - leave-one-out train/val/test protocol over target volcanoes.
- `scripts/06_ablations_finetune_cross_volcano.py` - fine-tunes ablation checkpoints on cross-volcano data.
- `scripts/07_finetune_unets_cross_volcano.py` - placeholder for UNet fine-tuning on cross-volcano data.
- `scripts/08_ablations_continuous.py` - placeholder for continuous ablation experiments.
- `scripts/09_unet_continuous.py` - placeholder for continuous UNet experiments.
- `scripts/10_train_obj_detection.py` - placeholder for object-detection training.
- `scripts/EXAMPLE_TRAIN.py` - legacy training example and utility reference.

## Default Experiment Folder

Most scripts read results from `results/experiments/complete_experiment` by default when evaluating or aggregating previously generated outputs.

## Requirements

Install the Python dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Typical Usage

### 1. Prepare data

```bash
python scripts/01_prepare_data.py
python scripts/01b_prepare_cross-volcano_data.py
```

### 2. Train ablations

```bash
python scripts/02_ablation_tests.py --mode train
```

### 3. Run zero-shot cross-volcano evaluation

```bash
python scripts/03_zero_shot_cross_volcano.py
```

### 4. Run leave-one-out cross-volcano training/evaluation

```bash
python scripts/04_cross-volcano.py
```

### 5. Fine-tune ablations on cross-volcano data

```bash
python scripts/06_ablations_finetune_cross_volcano.py
```

## Outputs

Experiment outputs are written under `results/experiments/<experiment_name>/`. The `results/latest/` folder stores pointer files to the most recent experiment artifacts.

## Notes

- The scripts are designed to be run from the repository root.
- `scripts/07` to `scripts/10` are currently placeholders.
- See `README_SCRIPTS_DETAILED.md` for a script-by-script technical breakdown.