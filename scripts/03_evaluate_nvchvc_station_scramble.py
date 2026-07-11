"""
Evaluate trained ablation checkpoints on the NVCHVC fold test sets with scrambled station ordering.

This script mirrors the test-time evaluation used after script 02 training, but it
randomly permutes station order per sample before inference so the metrics quantify
how much each model depends on fixed station identity.

Run:
    python scripts/03_evaluate_nvchvc_station_scramble.py

Inputs:
    results/experiments/complete_experiment/ablations/<model_key>/fold_XX/checkpoints/best_f1.pt
    data/prepared_data/NVCHVC/cv_5fold/fold_XX/test.npz

Outputs:
    results/experiments/complete_experiment/nvchvc_station_scramble_eval/
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.fold_io_utils import (
    append_row_csv,
    checkpoint_path_for_fold,
    load_completed_keys,
    load_training_fold_summary,
)
from utils.model_registry import MODEL_SPECS
from utils.script_common import parse_csv_selection, resolve_project_path
from utils.active_eval_utils import (
    evaluate_multistation_checkpoint as evaluate_multistation_checkpoint_on_test_fold,
    evaluate_unet_checkpoint as evaluate_unet_checkpoint_on_test_fold,
    load_checkpoint_into_model,
    load_unet_shape_and_loss,
)
from utils.train_utils import (
    cleanup_gpu_cache,
    compute_summary,
    save_confusion_matrix_image,
)

CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
ALL_CLASS_NAMES = ["BG", "VT", "LP", "TR", "AV", "IC"]
FOLDS = range(1, 6)
LEN_WINDOW = 8192
IM_SIZE = 256
DEFAULT_SCRAMBLE_STATIONS = True

RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "complete_experiment"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / "NVCHVC" / "cv_5fold"
DEFAULT_OUTPUT_NAME = "nvchvc_station_scramble_eval"
DEFAULT_POINTER_NAME = "pointer_nvchvc_station_scramble_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ablation checkpoints on their NVCHVC fold test sets with "
            "scrambled station ordering."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Root folder containing 'ablations' (default: complete_experiment).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="NVCHVC 5-fold root containing fold_XX/test.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for scrambled-station reports. Defaults to "
            "<experiment-root>/nvchvc_station_scramble_eval."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size override. Default: model registry batch_size.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to evaluate. Default: discovered model folders.",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=None,
        help="Backward-compatible alias for --models.",
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
        help="Save confusion matrix image per evaluated model/fold.",
    )
    parser.add_argument(
        "--scramble-stations",
        dest="scramble_stations",
        action="store_true",
        help="Randomly permute station order per sample before evaluation.",
    )
    parser.add_argument(
        "--no-scramble-stations",
        dest="scramble_stations",
        action="store_false",
        help="Keep the original station order during evaluation.",
    )
    parser.add_argument(
        "--station-scramble-seed",
        type=int,
        default=42,
        help="Base seed used to build deterministic per-sample station permutations.",
    )
    parser.set_defaults(scramble_stations=DEFAULT_SCRAMBLE_STATIONS)
    return parser.parse_args()


def write_aggregate_reports(out_dir: Path) -> None:
    fold_metrics_path = out_dir / "station_scramble_fold_metrics.csv"
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

    grouped = fold_df.groupby(["model_key"], sort=True)
    for model_key, grp in grouped:
        mean_f1_summary = compute_summary(grp["test_mean_f1"].astype(float).tolist())
        mean_iou_summary = compute_summary(grp["test_mean_iou"].astype(float).tolist())
        iou_all_values = pd.to_numeric(
            grp["test_mean_iou_all"], errors="coerce"
        ).dropna()
        mean_iou_all_summary = (
            compute_summary(iou_all_values.astype(float).tolist())
            if len(iou_all_values) > 0
            else None
        )
        loss_summary = compute_summary(grp["test_loss"].astype(float).tolist())

        display_names = grp["model_display_name"].dropna().unique().tolist()
        display_name = (
            str(display_names[0]) if len(display_names) > 0 else str(model_key)
        )
        summary_row = {
            "model_key": str(model_key),
            "model_display_name": display_name,
            "n_folds": int(len(grp)),
            "test_mean_f1_mean": float(mean_f1_summary["mean"]),
            "test_mean_f1_std": float(mean_f1_summary["std"]),
            "test_mean_iou_mean": float(mean_iou_summary["mean"]),
            "test_mean_iou_std": float(mean_iou_summary["std"]),
            "test_loss_mean": float(loss_summary["mean"]),
            "test_loss_std": float(loss_summary["std"]),
        }
        if mean_iou_all_summary is not None:
            summary_row["test_mean_iou_all_mean"] = float(mean_iou_all_summary["mean"])
            summary_row["test_mean_iou_all_std"] = float(mean_iou_all_summary["std"])

        per_class_f1_row = {
            "model_key": str(model_key),
            "model_display_name": display_name,
        }
        per_class_iou_row = {
            "model_key": str(model_key),
            "model_display_name": display_name,
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
        by=["test_mean_f1_mean"],
        ascending=[False],
    )
    summary_df.to_csv(
        out_dir / "station_scramble_summary_by_model.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_f1_rows).sort_values(by=["model_key"]).to_csv(
        out_dir / "station_scramble_per_class_f1.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_iou_rows).sort_values(by=["model_key"]).to_csv(
        out_dir / "station_scramble_per_class_iou.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    comparisons_dir = out_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    summary_df.sort_values(by="test_mean_f1_mean", ascending=False).to_csv(
        comparisons_dir / "rank_by_mean_f1.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    summary_df.sort_values(by="test_mean_iou_mean", ascending=False).to_csv(
        comparisons_dir / "rank_by_mean_iou.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )


def main() -> None:
    args = parse_args()

    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    data_root = resolve_project_path(args.data_root, PROJECT_ROOT)
    out_dir = (
        resolve_project_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir is not None
        else (experiment_root / DEFAULT_OUTPUT_NAME)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations_root = experiment_root / "ablations"
    if not ablations_root.exists():
        raise FileNotFoundError(f"Ablations root not found: {ablations_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    discovered_model_dirs = sorted(
        [p.name for p in ablations_root.iterdir() if p.is_dir()]
    )
    if args.models is not None and args.ablations is not None:
        raise ValueError("Use only one of --models or --ablations.")

    available_model_keys = [
        model_key for model_key in discovered_model_dirs if model_key in MODEL_SPECS
    ]
    unknown_dirs = sorted(set(discovered_model_dirs) - set(available_model_keys))
    if len(unknown_dirs) > 0:
        print(
            "[WARN] Skipping model folders not present in registry: " f"{unknown_dirs}"
        )

    raw_selection = args.models if args.models is not None else args.ablations
    if raw_selection is not None:
        selected_models = parse_csv_selection(
            raw_selection,
            available_model_keys,
            "model keys",
        )
    else:
        selected_models = list(available_model_keys)

    if len(selected_models) == 0:
        raise RuntimeError("No models selected for evaluation.")

    for fold_id in FOLDS:
        test_npz_path = data_root / f"fold_{fold_id:02d}" / "test.npz"
        if not test_npz_path.exists():
            raise FileNotFoundError(f"Missing test artifact: {test_npz_path}")

    init_features, depth, dice_weight, ce_weight = load_unet_shape_and_loss(
        experiment_root
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "experiment_root": str(experiment_root),
        "data_root": str(data_root),
        "output_dir": str(out_dir),
        "device": str(device),
        "batch_size_override": (
            None if args.batch_size is None else int(args.batch_size)
        ),
        "selected_models": selected_models,
        "scramble_stations": bool(args.scramble_stations),
        "station_scramble_seed": int(args.station_scramble_seed),
        "allow_missing_models": bool(args.allow_missing_models),
        "allow_missing_folds": bool(args.allow_missing_folds),
        "save_confusion_matrices": bool(args.save_confusion_matrices),
        "folds": [int(f) for f in FOLDS],
        "model_specs": {
            key: {
                "display_name": MODEL_SPECS[key]["display_name"],
                "batch_size": int(MODEL_SPECS[key]["batch_size"]),
                "model_kwargs": MODEL_SPECS[key]["model_kwargs"],
            }
            for key in selected_models
        },
        "unet_shape": {
            "init_features": int(init_features),
            "depth": int(depth),
        },
        "unet_loss": {
            "dice_weight": float(dice_weight),
            "ce_weight": float(ce_weight),
        },
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 80)
    print("NVCHVC fold-test evaluation with scrambled station order")
    print(f"Experiment root: {experiment_root}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {out_dir}")
    print(f"Models ({len(selected_models)}): {selected_models}")
    print(f"Scramble stations: {bool(args.scramble_stations)}")
    print(f"Device: {device}")
    print("=" * 80)

    fieldnames = [
        "model_key",
        "model_display_name",
        "fold",
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

    completed_keys = load_completed_keys(
        out_dir / "station_scramble_fold_metrics.csv",
        id_columns=["model_key", "fold"],
    )

    newly_completed = 0
    for model_key in selected_models:
        if model_key not in MODEL_SPECS:
            print(f"[WARN] Skipping unknown model key not in registry: {model_key}")
            continue

        model_spec = MODEL_SPECS[model_key]
        trainer_kind = str(model_spec["trainer_kind"])
        model_kwargs = dict(model_spec["model_kwargs"])

        print(f"\n[MODEL] {model_key} ({model_spec['display_name']})")
        model_root = ablations_root / model_key
        if not model_root.exists():
            msg = f"Missing model folder: {model_root}"
            if args.allow_missing_models:
                print(f"[WARN] {msg}")
                continue
            raise FileNotFoundError(msg)

        for fold_id in FOLDS:
            eval_key = (model_key, int(fold_id))
            if eval_key in completed_keys:
                print(f"  [SKIP] fold={fold_id:02d} already completed")
                continue

            ckpt_path = checkpoint_path_for_fold(model_root, fold_id)
            if not ckpt_path.exists():
                msg = f"Missing checkpoint: {ckpt_path}"
                if args.allow_missing_folds:
                    print(f"[WARN] {msg}")
                    continue
                raise FileNotFoundError(msg)

            test_npz_path = data_root / f"fold_{fold_id:02d}" / "test.npz"
            batch_size = (
                int(args.batch_size)
                if args.batch_size is not None
                else int(model_spec.get("batch_size", 16))
            )
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

            if trainer_kind == "1d":
                model = model_spec["model_cls"](**model_kwargs).to(device)
                load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=ckpt_path,
                    device=device,
                    trainer_kind=trainer_kind,
                )
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
                ) = evaluate_multistation_checkpoint_on_test_fold(
                    model=model,
                    test_npz_path=test_npz_path,
                    batch_size=batch_size,
                    device=device,
                    scramble_stations=bool(args.scramble_stations),
                    station_scramble_seed=int(args.station_scramble_seed),
                )
            elif trainer_kind == "2d":
                model = model_spec["model_cls"](
                    in_channels=1,
                    out_channels=6,
                    init_features=int(init_features),
                    depth=int(depth),
                    **{
                        k: v
                        for k, v in model_kwargs.items()
                        if k
                        not in {"in_channels", "out_channels", "init_features", "depth"}
                    },
                ).to(device)
                load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=ckpt_path,
                    device=device,
                    trainer_kind=trainer_kind,
                )
                (
                    f1_per_class,
                    mean_f1,
                    iou_per_class,
                    mean_iou,
                    eval_loss,
                    cm,
                    n_samples,
                    active_event_ids,
                ) = evaluate_unet_checkpoint_on_test_fold(
                    model=model,
                    test_npz_path=test_npz_path,
                    batch_size=batch_size,
                    device=device,
                    dice_weight=float(dice_weight),
                    ce_weight=float(ce_weight),
                    scramble_stations=bool(args.scramble_stations),
                    station_scramble_seed=int(args.station_scramble_seed),
                    class_names=CLASS_NAMES,
                    len_window=LEN_WINDOW,
                    im_size=IM_SIZE,
                )
                iou_all_classes = [float("nan")] * len(ALL_CLASS_NAMES)
                mean_iou_all = float("nan")
            else:
                raise ValueError(
                    f"Unsupported trainer kind '{trainer_kind}' for key '{model_key}'"
                )

            row = {
                "model_key": model_key,
                "model_display_name": str(model_spec["display_name"]),
                "fold": int(fold_id),
                "n_test": int(n_samples),
                "checkpoint": str(ckpt_path),
                "train_best_epoch": train_best_epoch,
                "train_best_val_mean_f1": train_best_val_mean_f1,
                "test_loss": float(eval_loss),
                "test_mean_f1": float(mean_f1),
                "test_mean_iou": float(mean_iou),
                "test_mean_iou_all": float(mean_iou_all),
                "n_active_classes": int(len(active_event_ids)),
                "active_classes": ",".join(
                    [CLASS_NAMES[x - 1] for x in active_event_ids]
                ),
            }
            for idx, class_name in enumerate(CLASS_NAMES):
                row[f"test_f1_{class_name}"] = float(f1_per_class[idx])
                row[f"test_iou_{class_name}"] = float(iou_per_class[idx])
            for idx, class_name in enumerate(ALL_CLASS_NAMES):
                row[f"test_iou_all_{class_name}"] = float(iou_all_classes[idx])

            append_row_csv(
                out_dir / "station_scramble_fold_metrics.csv",
                row=row,
                fieldnames=fieldnames,
            )
            append_row_csv(
                out_dir / "by_fold" / model_key / f"fold_{fold_id:02d}.csv",
                row=row,
                fieldnames=fieldnames,
            )
            completed_keys.add(eval_key)
            newly_completed += 1

            if args.save_confusion_matrices:
                cm_dir = (
                    out_dir / "confusion_matrices" / model_key / f"fold_{fold_id:02d}"
                )
                cm_dir.mkdir(parents=True, exist_ok=True)
                save_confusion_matrix_image(
                    cm=cm,
                    labels=CLASS_NAMES,
                    out_path=cm_dir / "test_confusion_matrix.png",
                    title=f"Scrambled station order | {model_key} | fold {fold_id:02d}",
                )

            print(f"  fold={fold_id:02d} mean_f1={mean_f1:.4f} mean_iou={mean_iou:.4f}")

            del cm
            del model
            cleanup_gpu_cache()
            gc.collect()

    if newly_completed == 0 and len(completed_keys) == 0:
        raise RuntimeError("No evaluations were executed. Check models/folds.")

    write_aggregate_reports(out_dir=out_dir)

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / DEFAULT_POINTER_NAME).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(out_dir),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("NVCHVC scrambled-station evaluation complete")
    print(f"Fold metrics: {out_dir / 'station_scramble_fold_metrics.csv'}")
    print(f"Summary: {out_dir / 'station_scramble_summary_by_model.csv'}")
    print(f"Comparisons: {out_dir / 'comparisons'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
