# Graph Volcano

This repository contains the code used for the volcano-seismic experiments in the article. It includes data preparation, 5-fold training, zero-shot cross-volcano evaluation, ablation studies, and fine-tuning pipelines for UNet and UNet_GraphSAGE models.

## Repository Layout

- `data/` - prepared datasets and fold artifacts.
- `models/` - model definitions.
- `scripts/` - experiment entry points.
- `utils/` - reusable training, data, and metric helpers.
- `results/` - generated experiment outputs.

## Main Scripts

- `scripts/01_prepare_data.py` - prepares fold manifests and dataset metadata.
- `scripts/02_ablation_tests.py` - runs 5-fold ablation training for UNet_GraphSAGE.
- `scripts/03_unet_5fold.py` - runs 5-fold training for the baseline UNet.
- `scripts/04_zero_shot_cross_volcano.py` - evaluates ablation checkpoints on cross-volcano test data.
- `scripts/05_zero_shot_unet_cross_volcano.py` - evaluates UNet checkpoints on cross-volcano test data.
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
```

### 2. Train ablations

```bash
python scripts/02_ablation_tests.py --mode train
```

### 3. Train the baseline UNet

```bash
python scripts/03_unet_5fold.py
```

### 4. Run zero-shot evaluation

```bash
python scripts/04_zero_shot_cross_volcano.py
python scripts/05_zero_shot_unet_cross_volcano.py
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