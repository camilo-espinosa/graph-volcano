"""
5-fold ablation runner for UNet_GraphSAGE.

This script trains and evaluates selected ablations across stratified 5-fold CV:
- Folds: 1..5

Fold data is read from:
    data/prepared_data/NVCHVC/cv_5fold/fold_XX/{train_aug,val,test}.npz

Results are written under:
    results/experiments/EXP_<timestamp>_NVCHVC_5fold/
"""

from __future__ import annotations

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

# ----------------------- ABLATION CONFIG (CUSTOMIZE HERE) -----------------------
GRAPH_LEVELS = [3, 4]

V5_FULL_KWARGS = {
    "graph_levels": GRAPH_LEVELS,
    "attention_pool_mode": "bottleneck_only",
    "use_bottleneck_attention": True,
    "graph_norm_type": "graphnorm",
    "node_feature_mode": "geometry",
    "graph_backend": "graphsage",
    "use_message_passing": True,
    "virtual_node_pool_mode": "learned",
    "bottleneck_virtual_node_pool_mode": "learned",
    "use_skip_graph": True,
}

ABLATION_MODEL_KWARGS = {
    "v5_full": {
        **V5_FULL_KWARGS,
    },
    "v5_full_all_levels": {
        **V5_FULL_KWARGS,
        "attention_pool_mode": "all_levels",
    },
    "ablation_2_mlp_backend": {
        **V5_FULL_KWARGS,
        "graph_backend": "mlp",
    },
    "ablation_3_no_message_passing": {
        **V5_FULL_KWARGS,
        "use_message_passing": False,
    },
    "ablation_4_no_bottleneck_attention": {
        **V5_FULL_KWARGS,
        "use_bottleneck_attention": False,
    },
    "ablation_5_no_norm": {
        **V5_FULL_KWARGS,
        "graph_norm_type": "none",
    },
    "ablation_6_batchnorm": {
        **V5_FULL_KWARGS,
        "graph_norm_type": "batchnorm",
    },
    "ablation_7_mean_virtual_node_pool": {
        **V5_FULL_KWARGS,
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
    },
    "ablation_8_graph_only_bottleneck": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
    },
    "ablation_9_no_skip_graph": {
        **V5_FULL_KWARGS,
        "use_skip_graph": False,
    },
    "ablation_10_learned_station_embedding_only": {
        **V5_FULL_KWARGS,
        "node_feature_mode": "learned_station_embedding",
        "station_embedding_dim": 3,
    },
}

# By default, run all listed ablations. Customize this list as needed.
ABLATIONS_TO_RUN = list(ABLATION_MODEL_KWARGS.keys())


# ------------------------------- HYPERPARAMETERS --------------------------------
CONFIG = {
    "volcano": "NVCHVC",
    "arch": "UNet_GraphSAGE",
    "batch_size": 12,
    "epochs": 1,
    "early_stop_patience": 20,
    "lr": 5e-4,
    "lr_final": 5e-6,
    "dice_weight": 0.7,
    "ce_weight": 0.3,
    "val_plot_events": 1,
    "save_confusion_matrix_each_epoch": True,
    "seed": 42,
}


# ------------------------------ PATHS AND OUTPUTS -------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / CONFIG["volcano"] / "cv_5fold"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"EXP_{TIMESTAMP}_{CONFIG['volcano']}_5fold"
EXPERIMENT_ROOT = EXPERIMENTS_ROOT / EXPERIMENT_NAME

FOLDS = range(1, 6)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    selected = []
    for name in ABLATIONS_TO_RUN:
        if name not in ABLATION_MODEL_KWARGS:
            raise ValueError(
                f"Unknown ablation '{name}'. Available: {sorted(ABLATION_MODEL_KWARGS.keys())}"
            )
        selected.append(name)

    run_manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "results_root": str(EXPERIMENT_ROOT),
        "device": str(device),
        "config": CONFIG,
        "ablations_to_run": selected,
        "ablation_model_kwargs": {
            name: ABLATION_MODEL_KWARGS[name] for name in selected
        },
        "folds": [{"fold": int(f)} for f in FOLDS],
    }

    with (EXPERIMENT_ROOT / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    env_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": int(torch.cuda.device_count()),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }
    with (EXPERIMENT_ROOT / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    print(f"Experiment root: {EXPERIMENT_ROOT}")
    print(f"Device: {device}")
    print(f"Ablations to run ({len(selected)}): {selected}")

    leaderboard_rows = []

    for ablation_name in selected:
        ablation_root = EXPERIMENT_ROOT / "ablations" / ablation_name
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

        ablation_summary = {
            "ablation": ablation_name,
            "n_folds": len(fold_summaries),
            "val_mean_f1": compute_summary(val_f1_values),
            "test_mean_f1": compute_summary(test_f1_values),
            "test_mean_iou": compute_summary(test_iou_values),
            "test_mean_iou_all": compute_summary(test_iou_all_values),
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

        leaderboard_rows.append(
            {
                "ablation": ablation_name,
                "val_mean_f1_mean": float(ablation_summary["val_mean_f1"]["mean"]),
                "val_mean_f1_std": float(ablation_summary["val_mean_f1"]["std"]),
                "test_mean_f1_mean": float(ablation_summary["test_mean_f1"]["mean"]),
                "test_mean_f1_std": float(ablation_summary["test_mean_f1"]["std"]),
                "test_mean_iou_mean": float(ablation_summary["test_mean_iou"]["mean"]),
                "test_mean_iou_std": float(ablation_summary["test_mean_iou"]["std"]),
                "test_mean_iou_all_mean": float(
                    ablation_summary["test_mean_iou_all"]["mean"]
                ),
                "test_mean_iou_all_std": float(
                    ablation_summary["test_mean_iou_all"]["std"]
                ),
            }
        )

        cleanup_gpu_cache()

    comparisons_dir = EXPERIMENT_ROOT / "comparisons"
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

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "pointer.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": EXPERIMENT_NAME,
                "experiment_root": str(EXPERIMENT_ROOT),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("Ablation 5-fold run complete")
    print(f"Experiment folder: {EXPERIMENT_ROOT}")
    print(f"Leaderboard: {comparisons_dir / 'ablation_leaderboard.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
