"""
5-fold ablation runner for the active model registry.

How to run:
    # Run script defaults (see MODEL_KEYS_TO_RUN below)
    python scripts/02_ablation_tests.py

    # Run all ablations currently defined in the registry
    python scripts/02_ablation_tests.py --models all

    # Run one specific ablation
    python scripts/02_ablation_tests.py --models <model_key>

    # Run multiple specific ablations
    python scripts/02_ablation_tests.py --models <model_key_1>,<model_key_2>

This script trains and evaluates selected models across stratified 5-fold CV.

Fold data is read from:
    data/prepared_data/NVCHVC/cv_5fold/fold_XX/{train_aug,val,test}.npz

Results are written under:
    results/experiments/EXP_<timestamp>_NVCHVC_5fold/

If you point --experiment-root at an existing experiment folder, completed
folds are skipped automatically by looking for reports/fold_summary.json.
Use --rerun-completed-folds to force all folds to run again.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.train_utils import (
    cleanup_gpu_cache,
    ensure_fold_data_exists,
)
from utils.trainer_segmentation import train_one_segmentation_fold
from utils.trainer_detection import train_one_event_detection_fold
from utils.fold_io_utils import is_training_fold_complete
from utils.model_registry import EVENT_DETECTION_EVAL_DEFAULTS, MODEL_SPECS, get_model_spec
from utils.script_common import resolve_project_path

# Default run set. Override with --models if needed.
MODEL_KEYS_TO_RUN = list(MODEL_SPECS.keys())
# MODEL_KEYS_TO_RUN.reverse()
# ------------------------------- HYPERPARAMETERS --------------------------------
CONFIG = {
    "volcano": "NVCHVC",
    "batch_size": 16,
    "epochs": 100,
    "early_stop_patience": 15,
    "lr": 5e-4,
    "lr_final": 1e-5,
    "dice_weight": 0.7,
    "ce_weight": 0.3,
    "val_plot_events": 5,
    "save_confusion_matrix_each_epoch": False,
    "seed": 42,
    "val_plot_samples_per_class": 2,
    "val_plot_misdetected_events_per_class": 10,
    "val_plot_forward_batch_size": 2,
    "best_epoch_attention_mode": "station",
    "final_attention_mode": "full",
    "event_confidence_threshold": 0.5,
    "match_iou_threshold": float(EVENT_DETECTION_EVAL_DEFAULTS["match_iou_threshold"]),
    "matching_strategy": str(EVENT_DETECTION_EVAL_DEFAULTS["matching_strategy"]),
    "overlap_recall_threshold": float(EVENT_DETECTION_EVAL_DEFAULTS["overlap_recall_threshold"]),
}
FOLDS = range(1, 6)

# Learning-rate scaling defaults.
# Effective LR per model is computed as:
#   lr_eff = lr_base * (batch_size / lr_scale_ref_batch) ** lr_scale_alpha
LR_SCALE_REF_BATCH = 16
LR_SCALE_ALPHA = -0.25

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
        help="Comma-separated model keys to process. Default: all active registry entries.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help=(
            "Comma-separated folds to run (e.g. '1,3' or '01,03'). "
            "Use 'all' for all folds. Default: all folds."
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
    parser.add_argument(
        "--rerun-completed-folds",
        action="store_true",
        help="Do not skip folds that already have reports/fold_summary.json.",
    )
    parser.add_argument(
        "--lr-scale-ref-batch",
        type=float,
        default=LR_SCALE_REF_BATCH,
        help=(
            "Reference batch size used for LR scaling. "
            "Effective lr = base_lr * (batch/ref_batch)^alpha."
        ),
    )
    parser.add_argument(
        "--lr-scale-alpha",
        type=float,
        default=1,  # LR_SCALE_ALPHA,
        help=(
            "Exponent for LR scaling by batch size. "
            "Use 0 to disable scaling, 0.5 for sqrt scaling."
        ),
    )
    return parser.parse_args()


def compute_scaled_lr(
    base_lr: float, batch_size: int, ref_batch: float, alpha: float
) -> float:
    if ref_batch <= 0:
        raise ValueError("--lr-scale-ref-batch must be > 0.")
    if batch_size <= 0:
        raise ValueError("Batch size must be > 0 for LR scaling.")
    scale = (float(batch_size) / float(ref_batch)) ** float(alpha)
    return float(base_lr) * scale


def select_model_keys(raw_models: str | None) -> list[str]:
    available_keys = list(MODEL_SPECS.keys())
    if raw_models is None:
        candidate_names = list(MODEL_KEYS_TO_RUN)
    else:
        if raw_models.strip().lower() == "all":
            candidate_names = list(available_keys)
        else:
            candidate_names = [x.strip() for x in raw_models.split(",") if x.strip()]

    selected = []
    for name in candidate_names:
        if name not in available_keys:
            raise ValueError(
                f"Unknown model key '{name}'. Available: {sorted(available_keys)}"
            )
        selected.append(name)
    return selected


def select_folds(raw_folds: str | None) -> list[int]:
    available_folds = [int(f) for f in FOLDS]
    available_fold_set = set(available_folds)

    if raw_folds is None or raw_folds.strip().lower() == "all":
        return list(available_folds)

    candidate_values = [x.strip() for x in raw_folds.split(",") if x.strip()]
    if not candidate_values:
        raise ValueError("--folds was provided but no fold values were parsed.")

    selected: list[int] = []
    for value in candidate_values:
        try:
            fold_id = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid fold '{value}'. Expected integers from {available_folds}."
            ) from exc

        if fold_id not in available_fold_set:
            raise ValueError(
                f"Unsupported fold '{fold_id}'. Available folds: {available_folds}."
            )

        if fold_id not in selected:
            selected.append(fold_id)

    return selected


def resolve_event_detection_loss_weights(spec: dict) -> dict[str, float]:
    """Return explicit model-specific event-detection loss weights, if defined."""
    if spec.get("trainer_kind") != "event_detection":
        return {}

    raw = spec.get("loss_weights") or {}
    weights: dict[str, float] = {}
    for key, value in raw.items():
        weights[str(key)] = float(value)
    return weights


def resolve_event_detection_loss_config(spec: dict) -> dict[str, str]:
    """Return explicit model-specific event-detection loss config, if defined."""
    if spec.get("trainer_kind") != "event_detection":
        return {}
    return {}


def summarize_manifest_classes(manifest_path: Path) -> dict[str, object]:
    with np.load(manifest_path) as data:
        label_ids = np.asarray(data["label_ids"], dtype=np.int64)
        labels = (
            np.asarray(data["labels"], dtype=str)
            if "labels" in data
            else np.asarray([], dtype=str)
        )
    class_ids = sorted(int(x) for x in np.unique(label_ids).tolist())
    id_counts = {
        int(class_id): int(np.sum(label_ids == class_id)) for class_id in class_ids
    }
    label_counts: dict[str, int] = {}
    if labels.size == label_ids.size and labels.size > 0:
        unique_labels = sorted(np.unique(labels).tolist())
        label_counts = {
            str(label): int(np.sum(labels == label)) for label in unique_labels
        }
    return {
        "samples": int(label_ids.shape[0]),
        "class_count": int(len(class_ids)),
        "class_ids": class_ids,
        "id_counts": id_counts,
        "label_counts": label_counts,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    selected = select_model_keys(args.models)
    selected_folds = select_folds(args.folds)

    experiment_root = resolve_project_path(
        args.experiment_root or EXPERIMENT_ROOT, PROJECT_ROOT
    )
    experiment_root.mkdir(parents=True, exist_ok=True)

    selected_specs = {name: get_model_spec(name) for name in selected}
    train_manifest_name = "train_aug.npz"
    val_manifest_name = "val.npz"
    test_manifest_name = "test.npz"

    run_manifest = {
        "experiment_name": experiment_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "results_root": str(experiment_root),
        "device": str(device),
        "train_manifest_name": train_manifest_name,
        "config": CONFIG,
        "models_to_run": selected,
        "folds_to_run": [int(f) for f in selected_folds],
        "model_specs": {
            name: {
                "display_name": name,
                "trainer_kind": spec["trainer_kind"],
                "batch_size": spec["batch_size"],
                "model_kwargs": spec["model_kwargs"],
                "loss_weights": resolve_event_detection_loss_weights(spec),
                "loss_config": resolve_event_detection_loss_config(spec),
                "eval_matching": spec.get("eval_matching", {}),
            }
            for name, spec in selected_specs.items()
        },
        "folds": [{"fold": int(f)} for f in selected_folds],
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
    print(f"Folds to run ({len(selected_folds)}): {selected_folds}")
    first_fold = selected_folds[0]
    first_fold_dir = DATA_ROOT / f"fold_{first_fold:02d}"
    print("Data source:")
    print(f"  dataset_root={DATA_ROOT}")
    print(f"  volcano={CONFIG['volcano']}")
    print(
        "  manifests="
        f"train:{train_manifest_name}, val:{val_manifest_name}, test:{test_manifest_name}"
    )
    print(
        "  example_paths="
        f"{first_fold_dir / train_manifest_name}, "
        f"{first_fold_dir / val_manifest_name}, "
        f"{first_fold_dir / test_manifest_name}"
    )
    for manifest_name in (train_manifest_name, val_manifest_name, test_manifest_name):
        manifest_path = first_fold_dir / manifest_name
        if not manifest_path.exists():
            print(f"  {manifest_name}: MISSING at {manifest_path}")
            continue
        summary = summarize_manifest_classes(manifest_path)
        print(
            f"  {manifest_name}: samples={summary['samples']} "
            f"class_count={summary['class_count']} "
            f"class_ids={summary['class_ids']}"
        )
        print(f"    id_counts={summary['id_counts']}")
        if summary["label_counts"]:
            print(f"    label_counts={summary['label_counts']}")
    print(
        "LR scaling: "
        f"base_lr={CONFIG['lr']:.3e}, "
        f"base_lr_final={CONFIG['lr_final']:.3e}, "
        f"ref_batch={float(args.lr_scale_ref_batch):g}, "
        f"alpha={float(args.lr_scale_alpha):g}"
    )
    print(f"Train manifest selected: {train_manifest_name}")
    if args.rerun_completed_folds:
        print("Completed folds will be rerun.")
    else:
        print("Completed folds will be skipped when fold summaries already exist.")

    for model_key in selected:
        spec = selected_specs[model_key]
        model_root = experiment_root / "ablations" / model_key

        model_config = dict(CONFIG)
        model_batch_size = int(spec["batch_size"] or CONFIG["batch_size"])
        model_config["batch_size"] = model_batch_size

        scaled_lr = compute_scaled_lr(
            base_lr=float(CONFIG["lr"]),
            batch_size=model_batch_size,
            ref_batch=float(args.lr_scale_ref_batch),
            alpha=float(args.lr_scale_alpha),
        )
        scaled_lr_final = compute_scaled_lr(
            base_lr=float(CONFIG["lr_final"]),
            batch_size=model_batch_size,
            ref_batch=float(args.lr_scale_ref_batch),
            alpha=float(args.lr_scale_alpha),
        )
        model_config["lr"] = float(scaled_lr)
        model_config["lr_final"] = float(scaled_lr_final)
        model_config["train_manifest_name"] = train_manifest_name
        explicit_loss_weights = resolve_event_detection_loss_weights(spec)
        explicit_loss_config = resolve_event_detection_loss_config(spec)
        if explicit_loss_weights:
            model_config["loss_weights"] = explicit_loss_weights
        if explicit_loss_config:
            model_config["loss_config"] = explicit_loss_config

        print(
            f"[{model_key}] batch_size={model_batch_size} | "
            f"lr={model_config['lr']:.3e} | lr_final={model_config['lr_final']:.3e}"
        )
        if spec["trainer_kind"] == "event_detection":
            if "loss_weights" in model_config:
                print(f"[{model_key}] loss_weights=" f"{model_config['loss_weights']}")
            else:
                print(f"[{model_key}] loss_weights=<trainer defaults>")
            if "loss_config" in model_config:
                print(f"[{model_key}] loss_config={model_config['loss_config']}")

        completed_folds = []
        remaining_folds = []
        for fold_id in selected_folds:
            if not args.rerun_completed_folds and is_training_fold_complete(
                model_root,
                fold_id,
            ):
                completed_folds.append(fold_id)
            else:
                remaining_folds.append(fold_id)

        if completed_folds:
            print(
                f"[{model_key}] Skipping completed folds: "
                f"{[int(x) for x in completed_folds]}"
            )
        if remaining_folds:
            print(
                f"[{model_key}] Remaining folds to run: "
                f"{[int(x) for x in remaining_folds]}"
            )
        else:
            print(f"[{model_key}] All folds already completed; nothing to run.")
            continue

        for fold_id in remaining_folds:
            fold_data_dir = DATA_ROOT / f"fold_{fold_id:02d}"
            ensure_fold_data_exists(fold_data_dir)
            selected_train_manifest = fold_data_dir / train_manifest_name
            if not selected_train_manifest.exists():
                raise FileNotFoundError(
                    f"Missing {train_manifest_name} for fold {fold_id:02d}: {selected_train_manifest}. "
                    "Run scripts/01_prepare_data.py first."
                )

            fold_out_dir = model_root / f"fold_{fold_id:02d}"
            trainer_kind = spec["trainer_kind"]

            if trainer_kind in ("2d", "1d"):
                # Unified segmentation trainer for both 2D and 1D models
                train_one_segmentation_fold(
                    trainer_kind=trainer_kind,
                    model_key_or_kwargs=model_key,
                    fold_id=fold_id,
                    fold_data_dir=fold_data_dir,
                    fold_out_dir=fold_out_dir,
                    device=device,
                    config=model_config,
                )
            elif trainer_kind == "event_detection":
                # Event detection trainer for MuSSED
                train_one_event_detection_fold(
                    model_key_or_kwargs=model_key,
                    fold_id=fold_id,
                    fold_data_dir=fold_data_dir,
                    fold_out_dir=fold_out_dir,
                    device=device,
                    config=model_config,
                )
            else:
                raise ValueError(
                    f"Unknown trainer_kind '{trainer_kind}' for model {model_key}. "
                    f"Expected one of: '2d', '1d', 'event_detection'."
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
    print("Model 5-fold training complete")
    print(f"Experiment folder: {experiment_root}")
    print("Run script 02b to aggregate fold outputs and build comparison reports.")
    print("=" * 80)


if __name__ == "__main__":
    main()
