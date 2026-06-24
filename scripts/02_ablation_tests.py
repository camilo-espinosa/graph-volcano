"""
5-fold ablation runner for UNet_MPNN and UNet_GraphSAGE.

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

import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.train_utils import (
    cleanup_gpu_cache,
    compute_summary,
    ensure_fold_data_exists,
    train_one_ablation_fold,
)
from utils.script_common import resolve_project_path
from utils.fold_io_utils import load_fold_summary
from utils.metrics_report_utils import compute_per_class_summary
from models.UNet_MPNN import MPNN_ABLATION_KWARGS

# ----------------------- ABLATION CONFIG (CUSTOMIZE HERE) -----------------------
GRAPH_LEVELS = [3, 4]

V5_FULL_KWARGS = {
    "graph_levels": GRAPH_LEVELS,
    "attention_pool_mode": "bottleneck_only",
    "use_bottleneck_attention": True,
    "graph_norm_type": "none",
    "node_feature_mode": "geometry",
    "graph_backend": "graphsage",
    "use_message_passing": True,
    "virtual_node_pool_mode": "learned",
    "bottleneck_virtual_node_pool_mode": "learned",
    "use_skip_graph": True,
    "init_features": 16,
    "depth": 5,
}
# ============================================================================
# GRAPHSAGE ABLATIONS (commented out, kept for reference)
# ============================================================================
# ABLATION_MODEL_KWARGS = {
#     # "ablation_5_no_norm": { #the real v5_full
#     #     **V5_FULL_KWARGS,
#     #     "graph_norm_type": "none",
#     # },
#     # "ablation_2_mlp_backend": {
#     #     **V5_FULL_KWARGS,
#     #     "graph_backend": "mlp",
#     # },
#     # "ablation_3_no_message_passing": {
#     #     **V5_FULL_KWARGS,
#     #     "use_message_passing": False,
#     # },
#     # "ablation_4_no_bottleneck_attention": {
#     #     **V5_FULL_KWARGS,
#     #     "use_bottleneck_attention": False,
#     # },
#     # "ablation_11_no_node_features": {
#     #     **V5_FULL_KWARGS,
#     #     "node_feature_mode": "none",
#     # },
#     # "only_graph_no_attention": {
#     #     **V5_FULL_KWARGS,
#     #     "graph_levels": [],
#     #     "use_bottleneck_attention": False,
#     #     "graph_norm_type": "none",
#     # },
# }

# ============================================================================
# MPNN ABLATIONS (active set)
# ============================================================================
ABLATION_MODEL_KWARGS = {
    name: {"_model_class": "UNet_MPNN", **kwargs}
    for name, kwargs in MPNN_ABLATION_KWARGS.items()
}


# GraphSAGE batch sizes (commented out, for reference)
# batch_sizes = {
#     "ablation_11_no_node_features": 20,
#     "ablation_2_mlp_backend": 20,
#     "ablation_3_no_message_passing": 20,
#     "ablation_4_no_bottleneck_attention": 14,
#     "ablation_5_no_norm": 18,
#     "ablation_6_batchnorm": 18,
#     "ablation_7_mean_virtual_node_pool": 14,
#     "ablation_8_graph_only_bottleneck": 24,
#     "ablation_9_no_skip_graph": 18,
# }

# MPNN batch sizes
batch_sizes = {
    "edge_mpnn__early_l2": 6,
    "edge_mpnn__early_l1": 6,
    "edge_mpnn__both_l2_bottleneck": 6,
    "edge_mpnn__bottleneck": 12,
    "edge_mpnn__no_edge_feats": 12,
    "edge_mpnn__encoder": 12,
    "edge_mpnn__star_topology": 12,
    "edge_mpnn__xcorr": 12,
    "edge_mpnn__no_spatial_info": 12,
    "edge_mpnn__rsam": 12,
    "edge_mpnn__no_attention": 12,
}
# By default, run all listed ablations. Customize this list as needed.
ABLATIONS_TO_RUN = list(ABLATION_MODEL_KWARGS.keys())

# ------------------------------- HYPERPARAMETERS --------------------------------
CONFIG = {
    "volcano": "NVCHVC",
    "arch": "UNet_MPNN",  # Informational; actual arch per ablation
    "batch_size": 24,
    "epochs": 100,
    "early_stop_patience": 20,
    "lr": 1e-4,
    "lr_final": 1e-6,
    "dice_weight": 0.7,
    "ce_weight": 0.3,
    "val_plot_events": 20,
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
CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation training or aggregate already generated fold results."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "aggregate-only"],
        default="train",
        help="Execution mode. Use 'aggregate-only' to skip training and only build summaries.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help=(
            "Experiment root directory (relative paths are resolved from project root). "
            "In 'train' mode, defaults to a new timestamped folder. "
            "In 'aggregate-only' mode, defaults to results/experiments/complete_experiment."
        ),
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=None,
        help="Comma-separated ablation names to process. Defaults to all configured ablations.",
    )
    parser.add_argument(
        "--require-all-folds",
        action="store_true",
        help="Fail if any fold summary is missing for a selected ablation.",
    )
    return parser.parse_args()


def select_ablations(raw_ablations: str | None) -> list[str]:
    if raw_ablations is None:
        candidate_names = list(ABLATIONS_TO_RUN)
    else:
        candidate_names = [x.strip() for x in raw_ablations.split(",") if x.strip()]

    selected = []
    for name in candidate_names:
        if name not in ABLATION_MODEL_KWARGS:
            raise ValueError(
                f"Unknown ablation '{name}'. Available: {sorted(ABLATION_MODEL_KWARGS.keys())}"
            )
        selected.append(name)
    return selected


def write_ablation_aggregate(
    aggregate_dir: Path,
    ablation_name: str,
    fold_summaries: list[dict],
) -> dict:
    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(
        aggregate_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    val_f1_values = [float(x["best_val_mean_f1"]) for x in fold_summaries]
    test_f1_values = [float(x["test_mean_f1"]) for x in fold_summaries]
    test_iou_values = [float(x["test_mean_iou"]) for x in fold_summaries]
    test_iou_all_values = [float(x["test_mean_iou_all"]) for x in fold_summaries]
    best_epoch_values = [int(x["best_epoch"]) for x in fold_summaries]
    test_f1_per_class_values = [
        [float(v) for v in x["test_f1_per_class"]] for x in fold_summaries
    ]
    test_iou_per_class_values = [
        [float(v) for v in x["test_iou_per_class"]] for x in fold_summaries
    ]

    for row in test_f1_per_class_values:
        if len(row) != len(CLASS_NAMES):
            raise ValueError(
                f"Invalid test_f1_per_class length for '{ablation_name}': expected {len(CLASS_NAMES)}, got {len(row)}"
            )
    for row in test_iou_per_class_values:
        if len(row) != len(CLASS_NAMES):
            raise ValueError(
                f"Invalid test_iou_per_class length for '{ablation_name}': expected {len(CLASS_NAMES)}, got {len(row)}"
            )

    test_f1_per_class_summary = compute_per_class_summary(
        test_f1_per_class_values,
        CLASS_NAMES,
    )
    test_iou_per_class_summary = compute_per_class_summary(
        test_iou_per_class_values,
        CLASS_NAMES,
    )

    ablation_summary = {
        "ablation": ablation_name,
        "n_folds": len(fold_summaries),
        "best_epoch": compute_summary(best_epoch_values),
        "val_mean_f1": compute_summary(val_f1_values),
        "test_mean_f1": compute_summary(test_f1_values),
        "test_mean_iou": compute_summary(test_iou_values),
        "test_mean_iou_all": compute_summary(test_iou_all_values),
        "test_f1_per_class": test_f1_per_class_summary,
        "test_iou_per_class": test_iou_per_class_summary,
    }

    with (aggregate_dir / "cv5fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    pd.DataFrame([ablation_summary]).to_csv(
        aggregate_dir / "cv5fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    best_epoch_by_fold_rows = [
        {
            "ablation": ablation_name,
            "fold": int(x["fold"]),
            "best_epoch": int(x["best_epoch"]),
            "best_val_mean_f1": float(x["best_val_mean_f1"]),
        }
        for x in fold_summaries
    ]
    pd.DataFrame(best_epoch_by_fold_rows).to_csv(
        aggregate_dir / "best_epoch_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    best_epoch_summary_row = {
        "ablation": ablation_name,
        "best_epoch_mean": float(ablation_summary["best_epoch"]["mean"]),
        "best_epoch_std": float(ablation_summary["best_epoch"]["std"]),
        "best_epoch_min": float(ablation_summary["best_epoch"]["min"]),
        "best_epoch_max": float(ablation_summary["best_epoch"]["max"]),
    }
    pd.DataFrame([best_epoch_summary_row]).to_csv(
        aggregate_dir / "best_epoch_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    per_class_f1_row = {"ablation": ablation_name}
    per_class_iou_row = {"ablation": ablation_name}
    for class_name in CLASS_NAMES:
        per_class_f1_row[f"{class_name}_mean"] = float(
            test_f1_per_class_summary[class_name]["mean"]
        )
        per_class_f1_row[f"{class_name}_std"] = float(
            test_f1_per_class_summary[class_name]["std"]
        )
        per_class_iou_row[f"{class_name}_mean"] = float(
            test_iou_per_class_summary[class_name]["mean"]
        )
        per_class_iou_row[f"{class_name}_std"] = float(
            test_iou_per_class_summary[class_name]["std"]
        )

    pd.DataFrame([per_class_f1_row]).to_csv(
        aggregate_dir / "cv5fold_summary_per_class_f1.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    pd.DataFrame([per_class_iou_row]).to_csv(
        aggregate_dir / "cv5fold_summary_per_class_iou.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    leaderboard_row = {
        "ablation": ablation_name,
        "best_epoch_mean": float(ablation_summary["best_epoch"]["mean"]),
        "best_epoch_std": float(ablation_summary["best_epoch"]["std"]),
        "best_epoch_min": float(ablation_summary["best_epoch"]["min"]),
        "best_epoch_max": float(ablation_summary["best_epoch"]["max"]),
        "val_mean_f1_mean": float(ablation_summary["val_mean_f1"]["mean"]),
        "val_mean_f1_std": float(ablation_summary["val_mean_f1"]["std"]),
        "test_mean_f1_mean": float(ablation_summary["test_mean_f1"]["mean"]),
        "test_mean_f1_std": float(ablation_summary["test_mean_f1"]["std"]),
        "test_mean_iou_mean": float(ablation_summary["test_mean_iou"]["mean"]),
        "test_mean_iou_std": float(ablation_summary["test_mean_iou"]["std"]),
        "test_mean_iou_all_mean": float(ablation_summary["test_mean_iou_all"]["mean"]),
        "test_mean_iou_all_std": float(ablation_summary["test_mean_iou_all"]["std"]),
    }

    for class_name in CLASS_NAMES:
        leaderboard_row[f"test_f1_{class_name}_mean"] = float(
            test_f1_per_class_summary[class_name]["mean"]
        )
        leaderboard_row[f"test_f1_{class_name}_std"] = float(
            test_f1_per_class_summary[class_name]["std"]
        )
        leaderboard_row[f"test_iou_{class_name}_mean"] = float(
            test_iou_per_class_summary[class_name]["mean"]
        )
        leaderboard_row[f"test_iou_{class_name}_std"] = float(
            test_iou_per_class_summary[class_name]["std"]
        )

    return leaderboard_row


def write_global_comparisons(
    experiment_root: Path, leaderboard_rows: list[dict]
) -> None:
    comparisons_dir = experiment_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    leaderboard_df = leaderboard_df.sort_values(by="test_mean_f1_mean", ascending=False)
    leaderboard_df.to_csv(
        comparisons_dir / "ablation_leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    rank_f1_df = leaderboard_df[["ablation", "test_mean_f1_mean", "test_mean_f1_std"]]
    rank_f1_df.to_csv(
        comparisons_dir / "ablation_rank_by_mean_f1.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    rank_iou_df = leaderboard_df[
        ["ablation", "test_mean_iou_mean", "test_mean_iou_std"]
    ].sort_values(by="test_mean_iou_mean", ascending=False)
    rank_iou_df.to_csv(
        comparisons_dir / "ablation_rank_by_mean_iou.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    per_class_f1_cols = ["ablation"]
    for class_name in CLASS_NAMES:
        per_class_f1_cols.extend(
            [f"test_f1_{class_name}_mean", f"test_f1_{class_name}_std"]
        )
    per_class_f1_df = leaderboard_df.sort_values(
        by="test_mean_f1_mean",
        ascending=False,
    )[per_class_f1_cols].copy()
    per_class_f1_df.to_csv(
        comparisons_dir / "ablation_mean_f1_per_class.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    per_class_iou_cols = ["ablation"]
    for class_name in CLASS_NAMES:
        per_class_iou_cols.extend(
            [f"test_iou_{class_name}_mean", f"test_iou_{class_name}_std"]
        )
    per_class_iou_df = leaderboard_df.sort_values(
        by="test_mean_iou_mean",
        ascending=False,
    )[per_class_iou_cols].copy()
    per_class_iou_df.to_csv(
        comparisons_dir / "ablation_mean_iou_per_class.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    best_epoch_df = leaderboard_df[
        [
            "ablation",
            "best_epoch_mean",
            "best_epoch_std",
            "best_epoch_min",
            "best_epoch_max",
        ]
    ].sort_values(by="best_epoch_mean", ascending=True)
    best_epoch_df.to_csv(
        comparisons_dir / "ablation_best_epoch_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )


def run_train_mode(
    device: torch.device,
    selected: list[str],
    experiment_root: Path,
    experiment_name: str,
) -> None:
    run_manifest = {
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "results_root": str(experiment_root),
        "device": str(device),
        "config": CONFIG,
        "ablations_to_run": selected,
        "ablation_model_kwargs": {
            name: ABLATION_MODEL_KWARGS[name] for name in selected
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
    print(f"Ablations to run ({len(selected)}): {selected}")

    leaderboard_rows = []

    for ablation_name in selected:
        CONFIG["batch_size"] = int(batch_sizes.get(ablation_name, CONFIG["batch_size"]))
        ablation_root = experiment_root / "ablations" / ablation_name
        aggregate_dir = ablation_root / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        fold_summaries = []
        for fold_id in FOLDS:
            fold_data_dir = DATA_ROOT / f"fold_{fold_id:02d}"
            ensure_fold_data_exists(fold_data_dir)

            fold_out_dir = ablation_root / f"fold_{fold_id:02d}"
            fold_summary = train_one_ablation_fold(
                ablation_name=ablation_name,
                model_kwargs=ABLATION_MODEL_KWARGS[ablation_name],
                fold_id=fold_id,
                fold_data_dir=fold_data_dir,
                fold_out_dir=fold_out_dir,
                device=device,
                config=CONFIG,
            )
            fold_summaries.append(fold_summary)

        leaderboard_rows.append(
            write_ablation_aggregate(
                aggregate_dir=aggregate_dir,
                ablation_name=ablation_name,
                fold_summaries=fold_summaries,
            )
        )
        cleanup_gpu_cache()

    write_global_comparisons(
        experiment_root=experiment_root, leaderboard_rows=leaderboard_rows
    )

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "pointer.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "experiment_root": str(experiment_root),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("Ablation 5-fold run complete")
    print(f"Experiment folder: {experiment_root}")
    print(
        f"Leaderboard: {experiment_root / 'comparisons' / 'ablation_leaderboard.csv'}"
    )
    print("=" * 80)


def discover_ablations_from_folder(ablations_root: Path) -> list[str]:
    """Discover all ablation folders in the ablations directory."""
    if not ablations_root.exists():
        return []

    ablations = []
    for item in ablations_root.iterdir():
        if item.is_dir():
            ablations.append(item.name)

    return sorted(ablations)


def run_aggregate_only_mode(
    selected: list[str],
    experiment_root: Path,
    require_all_folds: bool,
) -> None:
    print(f"Aggregate-only mode")
    print(f"Experiment root: {experiment_root}")

    leaderboard_rows = []
    ablations_root = experiment_root / "ablations"
    if not ablations_root.exists():
        raise FileNotFoundError(f"Ablations root not found: {ablations_root}")

    # Auto-discover ablations from folder if none exist in selected or if selected ablations not found
    discovered_ablations = discover_ablations_from_folder(ablations_root)

    # Check if selected ablations exist in the folder
    existing_selected = [a for a in selected if (ablations_root / a).exists()]

    if not existing_selected and discovered_ablations:
        # If requested ablations don't exist but we found some, use discovered ones
        print(f"Requested ablations not found. Auto-discovering from folder...")
        ablations_to_process = discovered_ablations
    else:
        ablations_to_process = existing_selected if existing_selected else selected

    print(f"Ablations to process ({len(ablations_to_process)}): {ablations_to_process}")

    for ablation_name in ablations_to_process:
        ablation_root = ablations_root / ablation_name
        if not ablation_root.exists():
            print(
                f"[WARN] Skipping '{ablation_name}': folder not found at {ablation_root}"
            )
            continue

        aggregate_dir = ablation_root / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        fold_summaries = []
        missing_folds = []
        for fold_id in FOLDS:
            fold_summary = load_fold_summary(root=ablation_root, fold_id=fold_id)
            if fold_summary is None:
                missing_folds.append(fold_id)
                continue
            fold_summaries.append(fold_summary)

        if missing_folds:
            msg = (
                f"Ablation '{ablation_name}' is missing fold summaries: "
                f"{[int(x) for x in missing_folds]}"
            )
            if require_all_folds:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg}")

        if not fold_summaries:
            print(f"[WARN] Skipping '{ablation_name}': no fold summaries were found.")
            continue

        leaderboard_rows.append(
            write_ablation_aggregate(
                aggregate_dir=aggregate_dir,
                ablation_name=ablation_name,
                fold_summaries=fold_summaries,
            )
        )

    if not leaderboard_rows:
        raise RuntimeError(
            "No ablations could be aggregated. Check folders and fold_summary.json files."
        )

    write_global_comparisons(
        experiment_root=experiment_root, leaderboard_rows=leaderboard_rows
    )

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
    print("Ablation aggregation complete")
    print(f"Experiment folder: {experiment_root}")
    print(
        f"Leaderboard: {experiment_root / 'comparisons' / 'ablation_leaderboard.csv'}"
    )
    print("=" * 80)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    selected = select_ablations(args.ablations)

    if args.mode == "aggregate-only":
        experiment_root = resolve_project_path(
            args.experiment_root or (EXPERIMENTS_ROOT / "complete_experiment"),
            PROJECT_ROOT,
        )
        run_aggregate_only_mode(
            selected=selected,
            experiment_root=experiment_root,
            require_all_folds=bool(args.require_all_folds),
        )
        return

    experiment_root = resolve_project_path(
        args.experiment_root or EXPERIMENT_ROOT, PROJECT_ROOT
    )
    experiment_root.mkdir(parents=True, exist_ok=True)
    run_train_mode(
        device=device,
        selected=selected,
        experiment_root=experiment_root,
        experiment_name=experiment_root.name,
    )


if __name__ == "__main__":
    main()
