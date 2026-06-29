"""
Aggregate previously generated 5-fold ablation results and build comparisons.

Run:
    python scripts/02b_aggregate_ablation_results.py

Default experiment root:
    results/experiments/complete_experiment

The script scans <experiment_root>/ablations/* and aggregates all discovered ablation folders.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.train_utils import compute_summary
from utils.script_common import resolve_project_path
from utils.fold_io_utils import load_fold_summary
from utils.metrics_report_utils import compute_per_class_summary

FOLDS = range(1, 6)
CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "complete_experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ablation fold outputs and produce comparison tables."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help=(
            "Experiment root directory (relative paths are resolved from project root). "
            "Default: results/experiments/complete_experiment."
        ),
    )
    parser.add_argument(
        "--require-all-folds",
        action="store_true",
        help="Fail if any discovered ablation is missing one or more fold summaries.",
    )
    return parser.parse_args()


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
    test_iou_all_values = [
        float(x["test_mean_iou_all"])
        for x in fold_summaries
        if "test_mean_iou_all" in x and x["test_mean_iou_all"] is not None
    ]
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
        "test_f1_per_class": test_f1_per_class_summary,
        "test_iou_per_class": test_iou_per_class_summary,
    }
    if len(test_iou_all_values) > 0:
        ablation_summary["test_mean_iou_all"] = compute_summary(test_iou_all_values)

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
    }
    if "test_mean_iou_all" in ablation_summary:
        leaderboard_row["test_mean_iou_all_mean"] = float(
            ablation_summary["test_mean_iou_all"]["mean"]
        )
        leaderboard_row["test_mean_iou_all_std"] = float(
            ablation_summary["test_mean_iou_all"]["std"]
        )

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


def main() -> None:
    args = parse_args()
    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)

    print("Aggregate-only mode")
    print(f"Experiment root: {experiment_root}")

    leaderboard_rows: list[dict] = []
    ablations_root = experiment_root / "ablations"
    if not ablations_root.exists():
        raise FileNotFoundError(f"Ablations root not found: {ablations_root}")

    ablations_to_process = sorted(
        [p.name for p in ablations_root.iterdir() if p.is_dir()]
    )
    print(f"Discovered ablations ({len(ablations_to_process)}): {ablations_to_process}")

    for ablation_name in ablations_to_process:
        ablation_root = ablations_root / ablation_name
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
            if args.require_all_folds:
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


if __name__ == "__main__":
    main()
