"""
Zero-shot cross-volcano evaluation for all ablation checkpoints.

This script evaluates every ablation/fold checkpoint found under:
    results/experiments/complete_experiment/ablations/<ablation>/fold_XX/checkpoints/best_f1.pt

On cross-volcano test artifacts:
    data/prepared_data/cross_volcano/<VOLCANO>/test_80.npz

Outputs are written to:
    results/experiments/complete_experiment/zero_shot_cross_volcano/
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.UNet_GraphSAGE import UNet_GraphSAGE
from utils.fold_io_utils import (
    append_row_csv,
    checkpoint_path_for_fold,
    load_completed_keys,
    load_training_fold_summary,
)
from utils.script_common import discover_targets, parse_csv_selection, resolve_project_path
from utils.station_info import get_crater_coords, get_station_coords
from utils.train_utils import (
    GraphSAGEDataset,
    cleanup_gpu_cache,
    compute_event_f1_iou_graphsage,
    compute_summary,
    save_confusion_matrix_image,
)


CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
ALL_CLASS_NAMES = ["BG", "VT", "LP", "TR", "AV", "IC"]
FOLDS = range(1, 6)
EXPECTED_INPUT_STATIONS = 8

RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "complete_experiment"
DEFAULT_CROSS_DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / "cross_volcano"
DEFAULT_OUTPUT_NAME = "zero_shot_cross_volcano"

# Use the same ablation definitions used for training in script 02.
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
    "init_features": 16,
    "depth": 5,
}


ABLATION_MODEL_KWARGS = {
    "v5_full_with_level_2": {
        **V5_FULL_KWARGS,
        "graph_levels":[2, 3, 4]
    },
    "v5_full_bigger_model": {
        **V5_FULL_KWARGS,
          "init_features": 24,
    },
    "v5_full_all_levels": {
        **V5_FULL_KWARGS,
        "attention_pool_mode": "all_levels",
    },
    "v5_full": {
        **V5_FULL_KWARGS,
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
    "only_bottleneck_attention": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "use_bottleneck_attention": True,
        "use_skip_graph": False,
        "use_message_passing": False,
    },
    "only_graph_no_attention": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "use_bottleneck_attention": False,
        "graph_norm_type": "none",
    },
    "leaner_model": {
    **V5_FULL_KWARGS,
        "graph_levels": [],
        "graph_norm_type": "none",
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
        "use_skip_graph": False,
    },
    "leanest_model": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "graph_norm_type": "none",
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
        "use_skip_graph": False,
        "node_feature_mode": "learned_station_embedding",
    },

}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-shot cross-volcano performance for all ablation/fold "
            "checkpoints in complete_experiment."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Root folder containing 'ablations' (default: complete_experiment).",
    )
    parser.add_argument(
        "--cross-data-root",
        type=Path,
        default=DEFAULT_CROSS_DATA_ROOT,
        help="Cross-volcano dataset root containing <VOLCANO>/test_80.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for zero-shot reports. Defaults to "
            "<experiment-root>/zero_shot_cross_volcano."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=None,
        help="Comma-separated ablation names to evaluate. Default: all discovered.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated target volcano names. Default: all with test_80.npz.",
    )
    parser.add_argument(
        "--allow-missing-folds",
        action="store_true",
        help="Skip missing fold checkpoints instead of failing.",
    )
    parser.add_argument(
        "--save-confusion-matrices",
        action="store_true",
        help="Save confusion matrix image per evaluated ablation/fold/target.",
    )
    parser.add_argument(
        "--verbose-model-stations",
        action="store_true",
        help="Print station metadata/order used by UNet_GraphSAGE on model creation.",
    )
    parser.add_argument(
        "--no-verbose-model-stations",
        action="store_false",
        dest="verbose_model_stations",
        help="Disable station metadata/order logs from UNet_GraphSAGE.",
    )
    parser.set_defaults(verbose_model_stations=True)
    return parser.parse_args()


def discover_ablations(ablations_root: Path) -> list[str]:
    if not ablations_root.exists():
        raise FileNotFoundError(f"Ablations root not found: {ablations_root}")

    available = {p.name for p in ablations_root.iterdir() if p.is_dir()}

    # Preserve the experiment's canonical ablation order as defined in this script.
    ordered = [name for name in ABLATION_MODEL_KWARGS.keys() if name in available]

    # Keep any extra folder names deterministic while leaving known ablations first.
    extras = sorted([name for name in available if name not in ABLATION_MODEL_KWARGS])
    return ordered + extras


def evaluate_checkpoint_on_target(
    model: torch.nn.Module,
    test_npz_path: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[
    list[float],
    float,
    list[float],
    float,
    list[float],
    float,
    float,
    np.ndarray,
    int,
    list[int],
]:
    ds = GraphSAGEDataset(test_npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    active_event_ids = sorted(
        [int(x) for x in np.unique(ds.label_ids).tolist() if int(x) in [1, 2, 3, 4, 5]]
    )
    active_class_indices = [x - 1 for x in active_event_ids]

    model.eval()

    with torch.inference_mode():
        (
            f1_per_class,
            mean_f1,
            iou_per_class,
            mean_iou,
            iou_all_classes,
            mean_iou_all,
            eval_loss,
            cm,
        ) = compute_event_f1_iou_graphsage(
            model,
            loader,
            device,
            return_cm=True,
            return_val_loss=True,
            return_event_plot_payloads=False,
            save_event_plots=False,
            max_event_plots=0,
            epoch=None,
        )

    if len(active_class_indices) > 0:
        mean_f1 = float(np.mean([f1_per_class[i] for i in active_class_indices]))
        mean_iou = float(np.mean([iou_per_class[i] for i in active_class_indices]))

    n_samples = int(len(ds))

    del ds, loader
    cleanup_gpu_cache()

    return (
        [float(x) for x in f1_per_class],
        float(mean_f1),
        [float(x) for x in iou_per_class],
        float(mean_iou),
        [float(x) for x in iou_all_classes],
        float(mean_iou_all),
        float(eval_loss),
        cm,
        n_samples,
        active_event_ids,
    )


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_state = ckpt["model_state_dict"]

    # Geometry buffers are target-volcano specific in this script; keep current model buffers.
    geometry_buffer_keys = {
        "station_xy",
        "dist_to_crater",
        "network_xy",
        "network_dist",
        "geom_nodes",
        "edge_index_base",
    }
    state = {
        k: v
        for k, v in ckpt_state.items()
        if k not in geometry_buffer_keys
    }

    # If station count changes, resize learned station embeddings by copying overlap.
    emb_key = "station_id_embedding.weight"
    if emb_key in state and emb_key in model.state_dict():
        src = state[emb_key]
        dst = model.state_dict()[emb_key]
        if src.shape != dst.shape:
            patched = dst.clone()
            rows = min(int(src.shape[0]), int(dst.shape[0]))
            cols = min(int(src.shape[1]), int(dst.shape[1]))
            patched[:rows, :cols] = src[:rows, :cols]
            state[emb_key] = patched

    model.load_state_dict(state, strict=False)
    del ckpt
    cleanup_gpu_cache()


def build_row_fieldnames() -> list[str]:
    fieldnames = [
        "ablation",
        "fold",
        "target_volcano",
        "n_test",
        "checkpoint",
        "train_best_epoch",
        "train_best_val_mean_f1",
        "test_loss",
        "test_mean_f1",
        "test_mean_iou",
        "test_mean_iou_all",
        "n_active_classes",
        "active_classes",
    ]

    for class_name in CLASS_NAMES:
        fieldnames.append(f"test_f1_{class_name}")
        fieldnames.append(f"test_iou_{class_name}")

    for class_name in ALL_CLASS_NAMES:
        fieldnames.append(f"test_iou_all_{class_name}")

    return fieldnames


def append_progress_rows(out_dir: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    append_row_csv(out_dir / "zero_shot_fold_metrics.csv", row=row, fieldnames=fieldnames)

    target_name = str(row["target_volcano"])
    append_row_csv(
        out_dir / "by_target" / target_name / "fold_metrics.csv",
        row=row,
        fieldnames=fieldnames,
    )

    ablation_name = str(row["ablation"])
    fold_id = int(row["fold"])
    append_row_csv(
        out_dir / "by_fold" / ablation_name / f"fold_{fold_id:02d}.csv",
        row=row,
        fieldnames=fieldnames,
    )


def write_aggregate_reports(out_dir: Path) -> None:
    fold_metrics_path = out_dir / "zero_shot_fold_metrics.csv"
    if not fold_metrics_path.exists():
        raise FileNotFoundError(f"Missing fold metrics file: {fold_metrics_path}")

    fold_df = pd.read_csv(
        fold_metrics_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    if len(fold_df) == 0:
        raise RuntimeError(f"Fold metrics file is empty: {fold_metrics_path}")

    summary_rows = []
    per_class_f1_rows = []
    per_class_iou_rows = []

    grouped = fold_df.groupby(["ablation", "target_volcano"], sort=True)
    for (ablation, target), grp in grouped:
        mean_f1_summary = compute_summary(grp["test_mean_f1"].astype(float).tolist())
        mean_iou_summary = compute_summary(grp["test_mean_iou"].astype(float).tolist())
        mean_iou_all_summary = compute_summary(
            grp["test_mean_iou_all"].astype(float).tolist()
        )
        loss_summary = compute_summary(grp["test_loss"].astype(float).tolist())

        summary_row = {
            "ablation": str(ablation),
            "target_volcano": str(target),
            "n_folds": int(len(grp)),
            "test_mean_f1_mean": float(mean_f1_summary["mean"]),
            "test_mean_f1_std": float(mean_f1_summary["std"]),
            "test_mean_iou_mean": float(mean_iou_summary["mean"]),
            "test_mean_iou_std": float(mean_iou_summary["std"]),
            "test_mean_iou_all_mean": float(mean_iou_all_summary["mean"]),
            "test_mean_iou_all_std": float(mean_iou_all_summary["std"]),
            "test_loss_mean": float(loss_summary["mean"]),
            "test_loss_std": float(loss_summary["std"]),
        }

        per_class_f1_row = {
            "ablation": str(ablation),
            "target_volcano": str(target),
        }
        per_class_iou_row = {
            "ablation": str(ablation),
            "target_volcano": str(target),
        }

        for class_name in CLASS_NAMES:
            f1_col = f"test_f1_{class_name}"
            iou_col = f"test_iou_{class_name}"
            f1_summary = compute_summary(grp[f1_col].astype(float).tolist())
            iou_summary = compute_summary(grp[iou_col].astype(float).tolist())

            summary_row[f"{f1_col}_mean"] = float(f1_summary["mean"])
            summary_row[f"{f1_col}_std"] = float(f1_summary["std"])
            summary_row[f"{iou_col}_mean"] = float(iou_summary["mean"])
            summary_row[f"{iou_col}_std"] = float(iou_summary["std"])

            per_class_f1_row[f"{class_name}_mean"] = float(f1_summary["mean"])
            per_class_f1_row[f"{class_name}_std"] = float(f1_summary["std"])
            per_class_iou_row[f"{class_name}_mean"] = float(iou_summary["mean"])
            per_class_iou_row[f"{class_name}_std"] = float(iou_summary["std"])

        summary_rows.append(summary_row)
        per_class_f1_rows.append(per_class_f1_row)
        per_class_iou_rows.append(per_class_iou_row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["target_volcano", "test_mean_f1_mean"],
        ascending=[True, False],
    )
    summary_df.to_csv(
        out_dir / "zero_shot_summary_by_ablation_target.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_f1_rows).sort_values(
        by=["target_volcano", "ablation"],
        ascending=[True, True],
    ).to_csv(
        out_dir / "zero_shot_per_class_f1.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_iou_rows).sort_values(
        by=["target_volcano", "ablation"],
        ascending=[True, True],
    ).to_csv(
        out_dir / "zero_shot_per_class_iou.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    comparisons_dir = out_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    for target in sorted(summary_df["target_volcano"].unique().tolist()):
        target_df = summary_df[summary_df["target_volcano"] == target].copy()
        target_dir = comparisons_dir / str(target)
        target_dir.mkdir(parents=True, exist_ok=True)

        target_df.sort_values(by="test_mean_f1_mean", ascending=False).to_csv(
            target_dir / "rank_by_mean_f1.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )
        target_df.sort_values(by="test_mean_iou_mean", ascending=False).to_csv(
            target_dir / "rank_by_mean_iou.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )


def main() -> None:
    args = parse_args()

    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    cross_data_root = resolve_project_path(args.cross_data_root, PROJECT_ROOT)

    out_dir = (
        resolve_project_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir is not None
        else (experiment_root / DEFAULT_OUTPUT_NAME)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations_root = experiment_root / "ablations"
    discovered_ablations = discover_ablations(ablations_root)
    selected_ablations = parse_csv_selection(args.ablations, discovered_ablations, "ablations")

    discovered_targets = discover_targets(cross_data_root)
    selected_targets = parse_csv_selection(args.targets, discovered_targets, "target volcanoes")

    # Validate station/crater metadata availability up front.
    for target_name in selected_targets:
        _ = get_station_coords(target_name)
        _ = get_crater_coords(target_name)

    missing_model_kwargs = [
        name for name in selected_ablations if name not in ABLATION_MODEL_KWARGS
    ]
    if len(missing_model_kwargs) > 0:
        raise KeyError(
            "Missing model kwargs for ablations in Script 4: "
            f"{missing_model_kwargs}. Add them to ABLATION_MODEL_KWARGS."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "experiment_root": str(experiment_root),
        "cross_data_root": str(cross_data_root),
        "output_dir": str(out_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "selected_ablations": selected_ablations,
        "selected_targets": selected_targets,
        "expected_input_stations": int(EXPECTED_INPUT_STATIONS),
        "verbose_model_stations": bool(args.verbose_model_stations),
        "target_station_count": {
            target: int(len(get_station_coords(target))) for target in selected_targets
        },
        "folds": [int(f) for f in FOLDS],
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 80)
    print("Zero-shot cross-volcano evaluation")
    print(f"Experiment root: {experiment_root}")
    print(f"Cross data root: {cross_data_root}")
    print(f"Output dir: {out_dir}")
    print(f"Ablations ({len(selected_ablations)}): {selected_ablations}")
    print(f"Targets ({len(selected_targets)}): {selected_targets}")
    print(f"Device: {device}")
    print("=" * 80)

    fieldnames = build_row_fieldnames()
    completed_keys = load_completed_keys(
        out_dir / "zero_shot_fold_metrics.csv",
        id_columns=["ablation", "fold", "target_volcano"],
    )
    if len(completed_keys) > 0:
        print(f"Resuming run: found {len(completed_keys)} completed evaluations in fold metrics CSV.")

    newly_completed = 0

    for ablation_name in selected_ablations:
        print(f"\n[ABLATION] {ablation_name}")
        ablation_root = ablations_root / ablation_name
        model_kwargs = ABLATION_MODEL_KWARGS[ablation_name]

        for fold_id in FOLDS:
            ckpt_path = checkpoint_path_for_fold(ablation_root, fold_id)
            if not ckpt_path.exists():
                msg = f"Missing checkpoint: {ckpt_path}"
                if args.allow_missing_folds:
                    print(f"[WARN] {msg}")
                    continue
                raise FileNotFoundError(msg)

            train_fold_summary = load_training_fold_summary(ablation_root, fold_id)
            train_best_epoch = (
                int(train_fold_summary["best_epoch"])
                if train_fold_summary is not None and "best_epoch" in train_fold_summary
                else None
            )
            train_best_val_mean_f1 = (
                float(train_fold_summary["best_val_mean_f1"])
                if train_fold_summary is not None
                and "best_val_mean_f1" in train_fold_summary
                else None
            )

            for target_name in selected_targets:
                eval_key = (ablation_name, int(fold_id), target_name)
                if eval_key in completed_keys:
                    print(f"  [SKIP] fold={fold_id:02d} target={target_name} already completed")
                    continue

                station_coords = get_station_coords(target_name)
                crater_coords = get_crater_coords(target_name)
                real_station_count = int(len(station_coords))
                if real_station_count != EXPECTED_INPUT_STATIONS:
                    print(
                        f"  [INFO] target={target_name}: metadata has {real_station_count} stations; "
                        f"model uses {EXPECTED_INPUT_STATIONS} (metadata padding/trimming applied in model)."
                    )
                model = UNet_GraphSAGE(
                    in_channels=1,
                    out_channels=6,
                    n_stations=int(EXPECTED_INPUT_STATIONS),
                    station_coords=station_coords,
                    crater_coords=crater_coords,
                    verbose=bool(args.verbose_model_stations),
                    **model_kwargs,
                ).to(device)
                load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=ckpt_path,
                    device=device,
                )

                test_npz_path = cross_data_root / target_name / "test_80.npz"
                if not test_npz_path.exists():
                    raise FileNotFoundError(f"Missing test artifact: {test_npz_path}")

                (
                    f1_per_class,
                    mean_f1,
                    iou_per_class,
                    mean_iou,
                    iou_all_classes,
                    mean_iou_all,
                    eval_loss,
                    cm,
                    n_samples,
                    active_event_ids,
                ) = evaluate_checkpoint_on_target(
                    model=model,
                    test_npz_path=test_npz_path,
                    batch_size=int(args.batch_size),
                    device=device,
                )

                row = {
                    "ablation": ablation_name,
                    "fold": int(fold_id),
                    "target_volcano": target_name,
                    "n_test": int(n_samples),
                    "checkpoint": str(ckpt_path),
                    "train_best_epoch": train_best_epoch,
                    "train_best_val_mean_f1": train_best_val_mean_f1,
                    "test_loss": float(eval_loss),
                    "test_mean_f1": float(mean_f1),
                    "test_mean_iou": float(mean_iou),
                    "test_mean_iou_all": float(mean_iou_all),
                    "n_active_classes": int(len(active_event_ids)),
                    "active_classes": ",".join([CLASS_NAMES[x - 1] for x in active_event_ids]),
                }

                for idx, class_name in enumerate(CLASS_NAMES):
                    row[f"test_f1_{class_name}"] = float(f1_per_class[idx])
                    row[f"test_iou_{class_name}"] = float(iou_per_class[idx])

                for idx, class_name in enumerate(ALL_CLASS_NAMES):
                    row[f"test_iou_all_{class_name}"] = float(iou_all_classes[idx])

                append_progress_rows(out_dir=out_dir, row=row, fieldnames=fieldnames)
                completed_keys.add(eval_key)
                newly_completed += 1

                if args.save_confusion_matrices:
                    cm_dir = out_dir / "confusion_matrices" / ablation_name / f"fold_{fold_id:02d}"
                    cm_dir.mkdir(parents=True, exist_ok=True)
                    cm_path = cm_dir / f"cm_{target_name}_test80_best_f1.png"
                    save_confusion_matrix_image(
                        cm=cm,
                        labels=CLASS_NAMES,
                        out_path=cm_path,
                        title=(
                            f"Zero-shot CM | {ablation_name} | fold {fold_id:02d} | "
                            f"target {target_name}"
                        ),
                    )

                print(
                    f"  fold={fold_id:02d} target={target_name} "
                    f"mean_f1={mean_f1:.4f} mean_iou={mean_iou:.4f}"
                )

                del cm
                del model
                cleanup_gpu_cache()

                gc.collect()
        gc.collect()

    if newly_completed == 0 and len(completed_keys) == 0:
        raise RuntimeError("No evaluations were executed. Check ablations/folds/targets.")

    write_aggregate_reports(out_dir=out_dir)

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "pointer_zero_shot_cross_volcano.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(out_dir),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("Zero-shot evaluation complete")
    print(f"Fold metrics: {out_dir / 'zero_shot_fold_metrics.csv'}")
    print(f"Summary: {out_dir / 'zero_shot_summary_by_ablation_target.csv'}")
    print(f"Comparisons: {out_dir / 'comparisons'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
