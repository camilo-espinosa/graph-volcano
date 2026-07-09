"""
5-fold ablation runner for UNet_MPNN and UNet_GraphSAGE.

How to run:
    # Run script defaults (see MODEL_KEYS_TO_RUN below)
    python scripts/02_ablation_tests.py

    # Run all ablations currently defined in the registry
    python scripts/02_ablation_tests.py --models all

    # Run one specific ablation
    python scripts/02_ablation_tests.py --models <model_key>

    # Run multiple specific ablations
    python scripts/02_ablation_tests.py --models <model_key_1>,<model_key_2>

This script trains and evaluates selected ablations across stratified 5-fold CV:
- Folds: 1..5
- Currently configured for UNet_MPNN ablations (UNet_GraphSAGE configs commented out)

Fold data is read from:
    data/prepared_data/NVCHVC/cv_5fold/fold_XX/{train_aug,val,test}.npz

Results are written under:
    results/experiments/EXP_<timestamp>_NVCHVC_5fold/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.train_utils import (
    cleanup_gpu_cache,
    ensure_fold_data_exists,
    train_one_unet_fold,
    train_one_ablation_fold,
)
from utils.model_registry import MODEL_REGISTRY, MODEL_SPECS, get_model_spec
from utils.model_registry import MODEL_REGISTRY, MODEL_SPECS, get_model_spec
from utils.script_common import resolve_project_path

# Default run set. Override with --models if needed.
MODEL_KEYS_TO_RUN = list(MODEL_REGISTRY.keys())
# MODEL_KEYS_TO_RUN.reverse()
# ------------------------------- HYPERPARAMETERS --------------------------------
CONFIG = {
    "volcano": "NVCHVC",
    "arch": "UNet_MPNN",  # Informational; actual arch per ablation
    "batch_size": 16,
    "epochs": 100,
    "early_stop_patience": 20,
    "lr": 1e-4,
    "lr_final": 1e-6,
    "dice_weight": 0.7,
    "ce_weight": 0.3,
    "val_plot_events": 5,
    "save_confusion_matrix_each_epoch": True,
    "seed": 42,
}
FOLDS = range(1, 6)


# ------------------------------ PATHS AND OUTPUTS -------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / CONFIG["volcano"] / "cv_5fold"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"EXP_{TIMESTAMP}_{CONFIG['volcano']}_5fold"
EXPERIMENT_ROOT = EXPERIMENTS_ROOT / EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-fold ablation training and save fold outputs for selected models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated model keys to process. " "Default: edge_mpnn__no_attention"
        ),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help=(
            "Experiment root directory (relative paths are resolved from project root). "
            "Default: a new timestamped folder under results/experiments/."
        ),
    )
    return parser.parse_args()


def select_model_keys(raw_models: str | None) -> list[str]:
    if raw_models is None:
        candidate_names = list(MODEL_KEYS_TO_RUN)
    else:
        if raw_models.strip().lower() == "all":
            candidate_names = list(MODEL_SPECS.keys())
        else:
            candidate_names = [x.strip() for x in raw_models.split(",") if x.strip()]

    selected = []
    for name in candidate_names:
        if name not in MODEL_SPECS:
            raise ValueError(
                f"Unknown model key '{name}'. Available: {sorted(MODEL_SPECS.keys())}"
            )
        selected.append(name)
    return selected


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    selected = select_model_keys(args.models)

    experiment_root = resolve_project_path(
        args.experiment_root or EXPERIMENT_ROOT, PROJECT_ROOT
    )
    experiment_root.mkdir(parents=True, exist_ok=True)

    selected_specs = {name: get_model_spec(name) for name in selected}
    run_manifest = {
        "experiment_name": experiment_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "results_root": str(experiment_root),
        "device": str(device),
        "config": CONFIG,
        "models_to_run": selected,
        "model_specs": {
            name: {
                "display_name": spec["display_name"],
                "family": spec["family"],
                "trainer_kind": spec["trainer_kind"],
                "batch_size": spec["batch_size"],
                "model_kwargs": spec["model_kwargs"],
            }
            for name, spec in selected_specs.items()
        },
        "folds": [{"fold": int(f)} for f in FOLDS],
    }

    with (experiment_root / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    env_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": int(torch.cuda.device_count()),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }
    with (experiment_root / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    print(f"Experiment root: {experiment_root}")
    print(f"Device: {device}")
    print(f"Models to run ({len(selected)}): {selected}")

    for model_key in selected:
        spec = selected_specs[model_key]
        model_root = experiment_root / "ablations" / model_key

        model_config = dict(CONFIG)
        model_config["batch_size"] = int(spec["batch_size"] or CONFIG["batch_size"])

        for fold_id in FOLDS:
            fold_data_dir = DATA_ROOT / f"fold_{fold_id:02d}"
            ensure_fold_data_exists(fold_data_dir)

            fold_out_dir = model_root / f"fold_{fold_id:02d}"
            if spec["trainer_kind"] == "2d":
                train_one_unet_fold(
                    model_key=model_key,
                    fold_id=fold_id,
                    fold_data_dir=fold_data_dir,
                    fold_out_dir=fold_out_dir,
                    device=device,
                    config=model_config,
                )
            else:
                train_one_ablation_fold(
                    ablation_name=model_key,
                    model_kwargs={
                        "_model_cls": spec["model_cls"],
                        **spec["model_kwargs"],
                    },
                    fold_id=fold_id,
                    fold_data_dir=fold_data_dir,
                    fold_out_dir=fold_out_dir,
                    device=device,
                    config=model_config,
                )
        cleanup_gpu_cache()

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "pointer.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": experiment_root.name,
                "experiment_root": str(experiment_root),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("Model-family 5-fold training complete")
    print(f"Experiment folder: {experiment_root}")
    print("Run script 02b to aggregate fold outputs and build comparison reports.")
    print("=" * 80)


if __name__ == "__main__":
    main()
