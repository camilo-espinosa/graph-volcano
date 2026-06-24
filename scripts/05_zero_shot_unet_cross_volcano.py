"""
Zero-shot cross-volcano evaluation for UNet model-family checkpoints.

This script evaluates each model/fold checkpoint found under:
    results/experiments/<UNET_EXPERIMENT>/<model_key>/fold_XX/checkpoints/best_f1.pt

On cross-volcano test artifacts:
    data/prepared_data/cross_volcano/<VOLCANO>/test_80.npz

Outputs are written to:
    <experiment_root>/zero_shot_cross_volcano_unet/
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

from models.UNet import UNet
from models.UNet_bottleneck_attention import UNetBottleneckAttention
from utils.fold_io_utils import (
    append_row_csv,
    checkpoint_path_for_fold,
    load_completed_keys,
    load_training_fold_summary,
)
from utils.metrics_report_utils import compute_iou_from_cm
from utils.script_common import discover_targets, parse_csv_selection, resolve_project_path
from utils.train_utils import (
    UNetPatchDataset,
    cleanup_gpu_cache,
    cm_eval,
    combined_dice_ce_loss_2d,
    compute_summary,
    f1_score_from_confusion_matrix,
    save_confusion_matrix_image,
)


CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
FOLDS = range(1, 6)

RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = (
    EXPERIMENTS_ROOT / "complete_experiment"
)
DEFAULT_CROSS_DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / "cross_volcano"
DEFAULT_OUTPUT_NAME = "zero_shot_cross_volcano_unet"

LEN_WINDOW = 8192
IM_SIZE = 256
DEFAULT_DICE_WEIGHT = 0.7
DEFAULT_CE_WEIGHT = 0.3

MODEL_SPECS = {
    "unet_bottleneck_attention": {
        "display_name": "UNetBottleneckAttention",
        "model_cls": UNetBottleneckAttention,
        "model_kwargs": {
            "bottleneck_attn_heads": 4,
            "bottleneck_attn_dropout": 0.0,
            "bottleneck_attn_ff_mult": 2,
        },
    },
    "unet": {
        "display_name": "UNet",
        "model_cls": UNet,
        "model_kwargs": {},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-shot cross-volcano performance for UNet model/fold checkpoints."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help=(
            "UNet experiment root containing model folders. "
            "Default: complete_experiment"
        ),
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
            "<experiment-root>/zero_shot_cross_volcano_unet."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to evaluate. Default: all MODEL_SPECS keys.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated target volcano names. Default: all with test_80.npz.",
    )
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Skip missing model folders instead of failing.",
    )
    parser.add_argument(
        "--allow-missing-folds",
        action="store_true",
        help="Skip missing fold checkpoints instead of failing.",
    )
    parser.add_argument(
        "--save-confusion-matrices",
        action="store_true",
        help="Save confusion matrix image per evaluated model/fold/target.",
    )
    return parser.parse_args()


def load_unet_shape_and_loss(experiment_root: Path) -> tuple[int, int, float, float]:
    init_features = 16
    depth = 5
    dice_weight = DEFAULT_DICE_WEIGHT
    ce_weight = DEFAULT_CE_WEIGHT

    run_manifest_path = experiment_root / "run_manifest.json"
    if run_manifest_path.exists():
        with run_manifest_path.open("r", encoding="utf-8") as f:
            run_manifest = json.load(f)
        config = run_manifest.get("config", {})
        init_features = int(config.get("init_features", init_features))
        depth = int(config.get("depth", depth))
        dice_weight = float(config.get("dice_weight", dice_weight))
        ce_weight = float(config.get("ce_weight", ce_weight))

    return init_features, depth, dice_weight, ce_weight


def evaluate_checkpoint_on_target(
    model: torch.nn.Module,
    test_npz_path: Path,
    batch_size: int,
    device: torch.device,
    dice_weight: float,
    ce_weight: float,
) -> tuple[list[float], float, list[float], float, float, np.ndarray, int, list[int]]:
    ds = UNetPatchDataset(test_npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    active_event_ids = sorted(
        [int(x) for x in np.unique(ds.label_ids).tolist() if int(x) in [1, 2, 3, 4, 5]]
    )
    active_class_indices = [x - 1 for x in active_event_ids]

    model.eval()

    total_loss = 0.0
    n_batches = 0
    with torch.inference_mode():
        for xb, y_onehot, _ in loader:
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)
            out = model(xb)
            loss, _, _ = combined_dice_ce_loss_2d(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=dice_weight,
                ce_weight=ce_weight,
            )
            total_loss += float(loss.item())
            n_batches += 1
            del xb, y_onehot, out, loss

    mean_loss = float(total_loss / n_batches) if n_batches > 0 else 0.0

    cm = cm_eval(
        model=model,
        dataloader=loader,
        device=device,
        len_window=LEN_WINDOW,
        im_size=IM_SIZE,
        clases_list={1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"},
        t_bg=0,
        t_cl=0,
    )
    f1_scores, _, _ = f1_score_from_confusion_matrix(cm)
    f1_scores = [float(x) for x in f1_scores]
    mean_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
    iou_per_class, mean_iou = compute_iou_from_cm(cm)

    if len(active_class_indices) > 0:
        mean_f1 = float(np.mean([f1_scores[i] for i in active_class_indices]))
        mean_iou = float(np.mean([iou_per_class[i] for i in active_class_indices]))

    n_samples = int(len(ds))

    del ds, loader
    cleanup_gpu_cache()

    return (
        [float(x) for x in f1_scores],
        float(mean_f1),
        [float(x) for x in iou_per_class],
        float(mean_iou),
        float(mean_loss),
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
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    cleanup_gpu_cache()


def build_row_fieldnames() -> list[str]:
    fieldnames = [
        "model_key",
        "model_display_name",
        "fold",
        "target_volcano",
        "n_test",
        "checkpoint",
        "train_best_epoch",
        "train_best_val_mean_f1",
        "test_loss",
        "test_mean_f1",
        "test_mean_iou",
        "n_active_classes",
        "active_classes",
    ]
    for class_name in CLASS_NAMES:
        fieldnames.append(f"test_f1_{class_name}")
        fieldnames.append(f"test_iou_{class_name}")
    return fieldnames


def append_progress_rows(out_dir: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    append_row_csv(out_dir / "zero_shot_fold_metrics.csv", row=row, fieldnames=fieldnames)

    target_name = str(row["target_volcano"])
    append_row_csv(
        out_dir / "by_target" / target_name / "fold_metrics.csv",
        row=row,
        fieldnames=fieldnames,
    )

    model_key = str(row["model_key"])
    fold_id = int(row["fold"])
    append_row_csv(
        out_dir / "by_fold" / model_key / f"fold_{fold_id:02d}.csv",
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

    grouped = fold_df.groupby(["model_key", "target_volcano"], sort=True)
    for (model_key, target), grp in grouped:
        mean_f1_summary = compute_summary(grp["test_mean_f1"].astype(float).tolist())
        mean_iou_summary = compute_summary(grp["test_mean_iou"].astype(float).tolist())
        loss_summary = compute_summary(grp["test_loss"].astype(float).tolist())

        display_names = grp["model_display_name"].dropna().unique().tolist()
        display_name = str(display_names[0]) if len(display_names) > 0 else str(model_key)

        summary_row = {
            "model_key": str(model_key),
            "model_display_name": display_name,
            "target_volcano": str(target),
            "n_folds": int(len(grp)),
            "test_mean_f1_mean": float(mean_f1_summary["mean"]),
            "test_mean_f1_std": float(mean_f1_summary["std"]),
            "test_mean_iou_mean": float(mean_iou_summary["mean"]),
            "test_mean_iou_std": float(mean_iou_summary["std"]),
            "test_loss_mean": float(loss_summary["mean"]),
            "test_loss_std": float(loss_summary["std"]),
        }

        per_class_f1_row = {
            "model_key": str(model_key),
            "model_display_name": display_name,
            "target_volcano": str(target),
        }
        per_class_iou_row = {
            "model_key": str(model_key),
            "model_display_name": display_name,
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
        out_dir / "zero_shot_summary_by_model_target.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_f1_rows).sort_values(
        by=["target_volcano", "model_key"],
        ascending=[True, True],
    ).to_csv(
        out_dir / "zero_shot_per_class_f1.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_iou_rows).sort_values(
        by=["target_volcano", "model_key"],
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

    selected_models = parse_csv_selection(
        args.models,
        sorted(list(MODEL_SPECS.keys())),
        "model keys",
    )

    discovered_targets = discover_targets(cross_data_root)
    selected_targets = parse_csv_selection(args.targets, discovered_targets, "target volcanoes")

    init_features, depth, dice_weight, ce_weight = load_unet_shape_and_loss(experiment_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "experiment_root": str(experiment_root),
        "cross_data_root": str(cross_data_root),
        "output_dir": str(out_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "model_shape": {
            "init_features": int(init_features),
            "depth": int(depth),
        },
        "loss_weights": {
            "dice_weight": float(dice_weight),
            "ce_weight": float(ce_weight),
        },
        "selected_models": selected_models,
        "selected_targets": selected_targets,
        "folds": [int(f) for f in FOLDS],
        "model_specs": {
            key: {
                "display_name": MODEL_SPECS[key]["display_name"],
                "model_kwargs": MODEL_SPECS[key]["model_kwargs"],
            }
            for key in selected_models
        },
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 80)
    print("Zero-shot cross-volcano evaluation for UNet family")
    print(f"Experiment root: {experiment_root}")
    print(f"Cross data root: {cross_data_root}")
    print(f"Output dir: {out_dir}")
    print(f"Models ({len(selected_models)}): {selected_models}")
    print(f"Targets ({len(selected_targets)}): {selected_targets}")
    print(f"Device: {device}")
    print("=" * 80)

    fieldnames = build_row_fieldnames()
    completed_keys = load_completed_keys(
        out_dir / "zero_shot_fold_metrics.csv",
        id_columns=["model_key", "fold", "target_volcano"],
    )
    if len(completed_keys) > 0:
        print(f"Resuming run: found {len(completed_keys)} completed evaluations in fold metrics CSV.")

    newly_completed = 0

    for model_key in selected_models:
        model_root = experiment_root / model_key
        if not model_root.exists():
            msg = f"Missing model folder: {model_root}"
            if args.allow_missing_models:
                print(f"[WARN] {msg}")
                continue
            raise FileNotFoundError(msg)

        model_spec = MODEL_SPECS[model_key]
        print(f"\n[MODEL] {model_key} ({model_spec['display_name']})")

        model = model_spec["model_cls"](
            in_channels=1,
            out_channels=6,
            init_features=int(init_features),
            depth=int(depth),
            **model_spec["model_kwargs"],
        ).to(device)

        for fold_id in FOLDS:
            ckpt_path = checkpoint_path_for_fold(model_root, fold_id)
            if not ckpt_path.exists():
                msg = f"Missing checkpoint: {ckpt_path}"
                if args.allow_missing_folds:
                    print(f"[WARN] {msg}")
                    continue
                raise FileNotFoundError(msg)

            load_checkpoint_into_model(model=model, checkpoint_path=ckpt_path, device=device)

            train_fold_summary = load_training_fold_summary(model_root, fold_id)
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
                eval_key = (model_key, int(fold_id), target_name)
                if eval_key in completed_keys:
                    print(f"  [SKIP] fold={fold_id:02d} target={target_name} already completed")
                    continue

                test_npz_path = cross_data_root / target_name / "test_80.npz"
                if not test_npz_path.exists():
                    raise FileNotFoundError(f"Missing test artifact: {test_npz_path}")

                (
                    f1_per_class,
                    mean_f1,
                    iou_per_class,
                    mean_iou,
                    eval_loss,
                    cm,
                    n_samples,
                    active_event_ids,
                ) = evaluate_checkpoint_on_target(
                    model=model,
                    test_npz_path=test_npz_path,
                    batch_size=int(args.batch_size),
                    device=device,
                    dice_weight=float(dice_weight),
                    ce_weight=float(ce_weight),
                )

                row = {
                    "model_key": model_key,
                    "model_display_name": str(model_spec["display_name"]),
                    "fold": int(fold_id),
                    "target_volcano": target_name,
                    "n_test": int(n_samples),
                    "checkpoint": str(ckpt_path),
                    "train_best_epoch": train_best_epoch,
                    "train_best_val_mean_f1": train_best_val_mean_f1,
                    "test_loss": float(eval_loss),
                    "test_mean_f1": float(mean_f1),
                    "test_mean_iou": float(mean_iou),
                    "n_active_classes": int(len(active_event_ids)),
                    "active_classes": ",".join([CLASS_NAMES[x - 1] for x in active_event_ids]),
                }

                for idx, class_name in enumerate(CLASS_NAMES):
                    row[f"test_f1_{class_name}"] = float(f1_per_class[idx])
                    row[f"test_iou_{class_name}"] = float(iou_per_class[idx])

                append_progress_rows(out_dir=out_dir, row=row, fieldnames=fieldnames)
                completed_keys.add(eval_key)
                newly_completed += 1

                if args.save_confusion_matrices:
                    cm_dir = out_dir / "confusion_matrices" / model_key / f"fold_{fold_id:02d}"
                    cm_dir.mkdir(parents=True, exist_ok=True)
                    cm_path = cm_dir / f"cm_{target_name}_test80_best_f1.png"
                    save_confusion_matrix_image(
                        cm=cm,
                        labels=CLASS_NAMES,
                        out_path=cm_path,
                        title=(
                            f"Zero-shot CM | {model_key} | fold {fold_id:02d} | "
                            f"target {target_name}"
                        ),
                    )

                print(
                    f"  fold={fold_id:02d} target={target_name} "
                    f"mean_f1={mean_f1:.4f} mean_iou={mean_iou:.4f}"
                )

                del cm
                cleanup_gpu_cache()

            gc.collect()

        del model
        cleanup_gpu_cache()
        gc.collect()

    if newly_completed == 0 and len(completed_keys) == 0:
        raise RuntimeError("No evaluations were executed. Check models/folds/targets.")

    write_aggregate_reports(out_dir=out_dir)

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "pointer_zero_shot_cross_volcano_unet.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "output_dir": str(out_dir),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("Zero-shot UNet evaluation complete")
    print(f"Fold metrics: {out_dir / 'zero_shot_fold_metrics.csv'}")
    print(f"Summary: {out_dir / 'zero_shot_summary_by_model_target.csv'}")
    print(f"Comparisons: {out_dir / 'comparisons'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
