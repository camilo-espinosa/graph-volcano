# Graph Volcano

Training and evaluation pipelines for volcano-seismic waveform segmentation and event detection, with 5-fold NVCHVC experiments and leave-one-out cross-volcano protocols.

## Models

### MuSSeg
Multi-station seismic segmentation model. U-Net-based temporal encoder operating on multi-station waveform inputs `[B, S, T]`. Produces per-timestep class probabilities via pixel-wise segmentation. Supports station-level message passing and attention mechanisms to capture inter-station relationships.

**Objective**: assign a seismic class label (noise, tremor, explosion, etc.) to every time step across all stations.

### MuSSED
Multi-Station Seismic Event Detection. DETR-inspired model built on top of the MuSSeg encoder. A transformer decoder with a fixed set of learnable event queries attends to the encoded temporal features and predicts a set of events, each with class, center time, start time, end time, and confidence.

**Objective**: detect and localize a fixed number of discrete seismic events from continuous multi-station waveform streams.

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

- `scripts/01_prepare_data.py`: generates NVCHVC 5-fold dataset with inner validation and train-only augmentation.
- `scripts/01b_prepare_cross-volcano_data.py`: prepares progressive finetuning splits (80/20) for CAU, VCA, and LDM target volcanoes.
- `scripts/02_ablation_tests.py`: runs 5-fold training for the active registry models on NVCHVC.
- `scripts/02b_aggregate_ablation_results.py`: re-aggregates already completed ablation folds into summary/comparison tables.
- `scripts/03_evaluate_nvchvc_station_scramble.py`: evaluates NVCHVC fold test sets with randomly scrambled station ordering.
- `scripts/04_zero_shot_cross_volcano.py`: evaluates trained checkpoints in zero-shot mode on the progressive-finetuning held-out test sets.
- `scripts/04b_zero_shot_cross_volcano_scrambled.py`: zero-shot evaluation on the same test sets with randomly scrambled station ordering.
- `scripts/05_progressive_finetuning.py`: progressive finetuning workflow for target volcano splits.
- `scripts/06_continuous_tests.py`: placeholder for the continuous tests workflow.
- `scripts/ablation_param_counts.py`: instantiates all registry models and prints parameter counts.

## Typical Workflow

Run scripts from the repository root.

### 1. Prepare datasets

```bash
python scripts/01_prepare_data.py
python scripts/01b_prepare_cross-volcano_data.py
```

### 2. Train ablation models (5-fold)

```bash
python scripts/02_ablation_tests.py
```

### 3. Aggregate ablation outputs (optional if already trained)

```bash
python scripts/02b_aggregate_ablation_results.py
```

### 4. Evaluate NVCHVC station-order sensitivity

```bash
python scripts/03_evaluate_nvchvc_station_scramble.py
```

### 5. Evaluate zero-shot progressive finetuning performance

```bash
python scripts/04_zero_shot_cross_volcano.py
python scripts/04b_zero_shot_cross_volcano_scrambled.py
```

## Outputs

- Main experiment outputs are written to `results/experiments/<experiment_name>/`.
- Many evaluation/aggregation scripts default to `results/experiments/complete_experiment`.
- `results/latest/` stores pointers to most recent artifacts.
- The active evaluation scripts now write model-centric outputs without a `family` column in their new fold/summary CSVs.


