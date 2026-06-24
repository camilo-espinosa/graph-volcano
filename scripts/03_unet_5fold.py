"""
5-fold UNet runner (2D patch input) for NVCHVC.

This script mirrors the ablation experiment structure but trains a plain UNet
using the legacy 2D input/output treatment from EXAMPLE_TRAIN.py:
- Dataset: UNetPatchDataset (patch-stacked 2D inputs)
- Loss: combined Dice + CrossEntropy
- Event-level evaluation: cm_eval + f1/iou from confusion matrix

Fold data is read from:
    data/prepared_data/NVCHVC/cv_5fold/fold_XX/{train_aug,val,test}.npz

Results are written under:
    results/experiments/EXP_<timestamp>_NVCHVC_UNet_5fold/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.UNet import UNet
from models.UNet_bottleneck_attention import UNetBottleneckAttention
from utils.metrics_report_utils import compute_iou_from_cm
from utils.script_common import resolve_project_path
from utils.train_utils import (
    BalancedBatchSampler,
    UNetPatchDataset,
    cleanup_gpu_cache,
    cm_eval,
    combined_dice_ce_loss_2d,
    compute_summary,
    ensure_fold_data_exists,
    f1_score_from_confusion_matrix,
    save_confusion_matrix_image,
)


CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
FOLDS = range(1, 6)

# Keep core training knobs aligned with scripts/02_ablation_tests.py
CONFIG = {
    "volcano": "NVCHVC",
    "arch": "UNet",
    "batch_size": 40,
    "epochs": 100,
    "early_stop_patience": 20,
    "lr": 1e-4,
    "lr_final": 1e-6,
    "dice_weight": 0.7,
    "ce_weight": 0.3,
    "save_confusion_matrix_each_epoch": True,
    "seed": 42,
    "init_features": 16,
    "depth": 5,
    "len_window": 8192,
    "im_size": 256,
}


DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / CONFIG["volcano"] / "cv_5fold"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"EXP_{TIMESTAMP}_{CONFIG['volcano']}_UNet_5fold"
EXPERIMENT_ROOT = EXPERIMENTS_ROOT / EXPERIMENT_NAME
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
        description="Run 5-fold CV training/evaluation for 2D UNet."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help=(
            "Experiment root directory (relative paths are resolved from project root). "
            "Defaults to a new timestamped folder under results/experiments/."
        ),
    )
    return parser.parse_args()


def evaluate_unet(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    len_window: int,
    im_size: int,
    config: dict,
) -> tuple[list[float], float, list[float], float, float, np.ndarray]:
    model.eval()

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for xb, y_onehot, _y_idx in dataloader:
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)
            out = model(xb)
            loss, _, _ = combined_dice_ce_loss_2d(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=config["dice_weight"],
                ce_weight=config["ce_weight"],
            )
            total_loss += float(loss.item())
            n_batches += 1
            del xb, y_onehot, out, loss

    mean_loss = float(total_loss / n_batches) if n_batches > 0 else 0.0

    cm = cm_eval(
        model=model,
        dataloader=dataloader,
        device=device,
        len_window=len_window,
        im_size=im_size,
        clases_list={1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"},
        t_bg=0,
        t_cl=0,
    )
    f1_scores, _, _ = f1_score_from_confusion_matrix(cm)
    f1_scores = [float(x) for x in f1_scores]
    mean_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
    iou_per_class, mean_iou = compute_iou_from_cm(cm)

    return f1_scores, mean_f1, iou_per_class, mean_iou, mean_loss, cm


def train_one_unet_fold(
    model_key: str,
    model_spec: dict,
    fold_id: int,
    fold_data_dir: Path,
    fold_out_dir: Path,
    device: torch.device,
    config: dict,
) -> dict:
    checkpoints_dir = fold_out_dir / "checkpoints"
    reports_dir = fold_out_dir / "reports"
    cm_dir = fold_out_dir / "confusion_matrices"
    for p in (checkpoints_dir, reports_dir, cm_dir):
        p.mkdir(parents=True, exist_ok=True)

    train_ds = UNetPatchDataset(fold_data_dir / "train_aug.npz")
    val_ds = UNetPatchDataset(fold_data_dir / "val.npz")
    test_ds = UNetPatchDataset(fold_data_dir / "test.npz")

    balanced_batch_sampler = BalancedBatchSampler(
        train_ds.label_ids,
        batch_size=config["batch_size"],
    )
    train_loader = DataLoader(train_ds, batch_sampler=balanced_batch_sampler)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    model = model_spec["model_cls"](
        in_channels=1,
        out_channels=6,
        init_features=config["init_features"],
        depth=config["depth"],
        **model_spec["model_kwargs"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"] / 4)),
        eta_min=config["lr_final"],
    )

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_mean_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    metrics_rows = []
    fold_start = time.time()

    print("=" * 80)
    print(
        f"Training {model_spec['display_name']} | fold={fold_id:02d} | "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    print(f"Output folder: {fold_out_dir}")
    print("=" * 80)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for batch_idx, (xb, y_onehot, _y_idx) in enumerate(train_loader):
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss, dice_component, ce_component = combined_dice_ce_loss_2d(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=config["dice_weight"],
                ce_weight=config["ce_weight"],
            )
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch:03d} batch {batch_idx:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} dice={dice_component.item():.4f} ce={ce_component.item():.4f}"
                )

            del xb, y_onehot, out, loss, dice_component, ce_component

        scheduler.step()

        (
            val_f1_per_class,
            val_mean_f1,
            val_iou_per_class,
            val_mean_iou,
            val_loss,
            val_cm,
        ) = evaluate_unet(
            model=model,
            dataloader=val_loader,
            device=device,
            len_window=config["len_window"],
            im_size=config["im_size"],
            config=config,
        )

        is_best_mean_f1_epoch = float(val_mean_f1) > float(best_mean_f1)
        if is_best_mean_f1_epoch:
            best_mean_f1 = float(val_mean_f1)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(val_mean_f1),
                },
                checkpoints_dir / "best_f1.pt",
            )
        else:
            epochs_without_improvement += 1

        if float(train_loss) < float(best_train_loss):
            best_train_loss = float(train_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(val_mean_f1),
                },
                checkpoints_dir / "best_train_loss.pt",
            )

        if float(val_loss) < float(best_val_loss):
            best_val_loss = float(val_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(val_mean_f1),
                },
                checkpoints_dir / "best_val_loss.pt",
            )

        if config["save_confusion_matrix_each_epoch"]:
            save_confusion_matrix_image(
                cm=val_cm,
                labels=CLASS_NAMES,
                out_path=cm_dir / f"confusion_matrix_epoch_{epoch:03d}.png",
                title=(
                    f"Confusion Matrix - {model_spec['display_name']} "
                    f"fold {fold_id:02d} - epoch {epoch:03d}"
                ),
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics_rows.append(
            [
                current_lr,
                int(epoch),
                float(train_loss),
                float(val_loss),
                float(val_f1_per_class[0]),
                float(val_f1_per_class[1]),
                float(val_f1_per_class[2]),
                float(val_f1_per_class[3]),
                float(val_f1_per_class[4]),
                float(val_mean_f1),
                float(val_iou_per_class[0]),
                float(val_iou_per_class[1]),
                float(val_iou_per_class[2]),
                float(val_iou_per_class[3]),
                float(val_iou_per_class[4]),
                float(val_mean_iou),
            ]
        )
        metrics_df = pd.DataFrame(
            metrics_rows,
            columns=[
                "lr",
                "epoch",
                "train_loss",
                "val_loss",
                "VT_f1",
                "LP_f1",
                "TR_f1",
                "AV_f1",
                "IC_f1",
                "mean_f1",
                "VT_iou",
                "LP_iou",
                "TR_iou",
                "AV_iou",
                "IC_iou",
                "mean_iou",
            ],
        )
        metrics_df.to_csv(
            reports_dir / "training_metrics.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        print(
            f"EPOCH {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"mean_f1={val_mean_f1:.4f} mean_iou={val_mean_iou:.4f} "
            f"best_epoch={best_epoch if best_epoch >= 0 else 'NA'} "
            f"no_improve={epochs_without_improvement}/{config['early_stop_patience']}"
        )

        del val_cm
        cleanup_gpu_cache()

        if epochs_without_improvement >= int(config["early_stop_patience"]):
            print(
                f"Early stopping at epoch {epoch:03d}: no mean_f1 improvement for "
                f"{config['early_stop_patience']} consecutive epochs."
            )
            break

    best_f1_ckpt = checkpoints_dir / "best_f1.pt"
    if not best_f1_ckpt.exists():
        raise RuntimeError(f"best_f1 checkpoint not found for fold output: {best_f1_ckpt}")

    ckpt = torch.load(best_f1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    (
        test_f1_per_class,
        test_mean_f1,
        test_iou_per_class,
        test_mean_iou,
        test_loss,
        test_cm,
    ) = evaluate_unet(
        model=model,
        dataloader=test_loader,
        device=device,
        len_window=config["len_window"],
        im_size=config["im_size"],
        config=config,
    )

    save_confusion_matrix_image(
        cm=test_cm,
        labels=CLASS_NAMES,
        out_path=cm_dir / "confusion_matrix_test_best_f1.png",
        title=(
            f"Test Confusion Matrix - {model_spec['display_name']} "
            f"fold {fold_id:02d} - best_f1"
        ),
    )

    fold_elapsed_sec = float(time.time() - fold_start)
    fold_summary = {
        "model": model_spec["display_name"],
        "model_key": model_key,
        "fold": int(fold_id),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_f1": float(best_mean_f1),
        "test_loss": float(test_loss),
        "test_mean_f1": float(test_mean_f1),
        "test_mean_iou": float(test_mean_iou),
        "test_f1_per_class": [float(x) for x in test_f1_per_class],
        "test_iou_per_class": [float(x) for x in test_iou_per_class],
        "fold_elapsed_seconds": fold_elapsed_sec,
    }

    with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    del optimizer, scheduler, model, ckpt
    cleanup_gpu_cache()

    return fold_summary


def write_aggregate(experiment_root: Path, fold_summaries: list[dict]) -> None:
    aggregate_dir = experiment_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

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
    best_epoch_values = [int(x["best_epoch"]) for x in fold_summaries]

    test_f1_per_class_values = [
        [float(v) for v in x["test_f1_per_class"]] for x in fold_summaries
    ]
    test_iou_per_class_values = [
        [float(v) for v in x["test_iou_per_class"]] for x in fold_summaries
    ]

    test_f1_per_class_summary = {}
    test_iou_per_class_summary = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        test_f1_per_class_summary[class_name] = compute_summary(
            [row[class_idx] for row in test_f1_per_class_values]
        )
        test_iou_per_class_summary[class_name] = compute_summary(
            [row[class_idx] for row in test_iou_per_class_values]
        )

    summary = {
        "model": str(fold_summaries[0]["model"]),
        "n_folds": len(fold_summaries),
        "best_epoch": compute_summary(best_epoch_values),
        "val_mean_f1": compute_summary(val_f1_values),
        "test_mean_f1": compute_summary(test_f1_values),
        "test_mean_iou": compute_summary(test_iou_values),
        "test_f1_per_class": test_f1_per_class_summary,
        "test_iou_per_class": test_iou_per_class_summary,
    }

    with (aggregate_dir / "cv5fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(
        aggregate_dir / "cv5fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(
        [
            {
                "model": str(summary["model"]),
                "best_epoch_mean": float(summary["best_epoch"]["mean"]),
                "best_epoch_std": float(summary["best_epoch"]["std"]),
                "best_epoch_min": float(summary["best_epoch"]["min"]),
                "best_epoch_max": float(summary["best_epoch"]["max"]),
            }
        ]
    ).to_csv(
        aggregate_dir / "best_epoch_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    per_class_f1_row = {"model": str(summary["model"])}
    per_class_iou_row = {"model": str(summary["model"])}
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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(CONFIG["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(CONFIG["seed"]))

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    experiment_root = resolve_project_path(args.experiment_root or EXPERIMENT_ROOT, PROJECT_ROOT)
    experiment_root.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "experiment_name": experiment_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "results_root": str(experiment_root),
        "device": str(device),
        "config": CONFIG,
        "models": {
            key: {
                "display_name": spec["display_name"],
                "model_kwargs": spec["model_kwargs"],
            }
            for key, spec in MODEL_SPECS.items()
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

    for model_key, model_spec in MODEL_SPECS.items():
        model_root = experiment_root / model_key
        model_root.mkdir(parents=True, exist_ok=True)

        fold_summaries = []
        for fold_id in FOLDS:
            fold_data_dir = DATA_ROOT / f"fold_{fold_id:02d}"
            ensure_fold_data_exists(fold_data_dir)

            fold_out_dir = model_root / f"fold_{fold_id:02d}"
            fold_summary = train_one_unet_fold(
                model_key=model_key,
                model_spec=model_spec,
                fold_id=fold_id,
                fold_data_dir=fold_data_dir,
                fold_out_dir=fold_out_dir,
                device=device,
                config=CONFIG,
            )
            fold_summaries.append(fold_summary)

        write_aggregate(experiment_root=model_root, fold_summaries=fold_summaries)

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "pointer_unet.json").open("w", encoding="utf-8") as f:
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
    print("UNet model-family 5-fold run complete")
    print(f"Experiment folder: {experiment_root}")
    print(f"UNet summary: {experiment_root / 'unet' / 'aggregate' / 'cv5fold_summary.json'}")
    print(
        "Attention UNet summary: "
        f"{experiment_root / 'unet_bottleneck_attention' / 'aggregate' / 'cv5fold_summary.json'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
