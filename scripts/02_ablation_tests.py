"""
5x2 ablation runner for UNet_GraphSAGE.

This script trains and evaluates selected ablations across all 5x2 folds:
- Repeats: 1..5
- Splits per repeat: 1, 2

Fold data is read from:
    data/prepared_data/NVCHVC/cv_5x2/repeat_XX/split_Y/{train_aug,val,test}.npz

Results are written under:
    results/experiments/EXP_<timestamp>_NVCHVC_5x2/
"""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import UNet_GraphSAGE
from utils.train_utils import (
    BalancedBatchSampler,
    GraphSAGEDataset,
    combined_dice_ce_loss,
    compute_event_f1_iou_graphsage,
    save_confusion_matrix_image,
    save_event_plot_payloads,
)

# ----------------------- ABLATION CONFIG (CUSTOMIZE HERE) -----------------------
GRAPH_LEVELS = [2, 3, 4]

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
    "batch_size": 16,
    "epochs": 100,
    "early_stop_patience": 20,
    "lr": 5e-4,
    "lr_final": 5e-6,
    "dice_weight": 0.7,
    "ce_weight": 0.3,
    "val_plot_events": 100,
    "save_confusion_matrix_each_epoch": True,
    "seed": 42,
}


# ------------------------------ PATHS AND OUTPUTS -------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / CONFIG["volcano"] / "cv_5x2"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"EXP_{TIMESTAMP}_{CONFIG['volcano']}_5x2"
EXPERIMENT_ROOT = EXPERIMENTS_ROOT / EXPERIMENT_NAME

REPEATS = range(1, 6)
SPLITS = (1, 2)


def cleanup_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def ensure_fold_data_exists(fold_data_dir: Path) -> None:
    needed = [
        fold_data_dir / "train_aug.npz",
        fold_data_dir / "val.npz",
        fold_data_dir / "test.npz",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing fold manifest files:\n" + "\n".join(missing))


def train_one_fold(
    ablation_name: str,
    model_kwargs: dict,
    repeat: int,
    split_id: int,
    fold_data_dir: Path,
    fold_out_dir: Path,
    device: torch.device,
) -> dict:
    checkpoints_dir = fold_out_dir / "checkpoints"
    reports_dir = fold_out_dir / "reports"
    cm_dir = fold_out_dir / "confusion_matrices"
    val_plot_dir = fold_out_dir / "validation_event_plots"

    for p in (checkpoints_dir, reports_dir, cm_dir, val_plot_dir):
        p.mkdir(parents=True, exist_ok=True)

    train_ds = GraphSAGEDataset(fold_data_dir / "train_aug.npz")
    val_ds = GraphSAGEDataset(fold_data_dir / "val.npz")
    test_ds = GraphSAGEDataset(fold_data_dir / "test.npz")

    balanced_batch_sampler = BalancedBatchSampler(
        train_ds.label_ids, batch_size=CONFIG["batch_size"]
    )
    train_loader = DataLoader(train_ds, batch_sampler=balanced_batch_sampler)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    model = UNet_GraphSAGE(
        in_channels=1,
        out_channels=6,
        init_features=16,
        depth=5,
        **model_kwargs,
    ).to(device)

    model_name = f"{CONFIG['arch']}_{ablation_name}_{CONFIG['volcano']}_r{repeat:02d}_s{split_id}"

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(CONFIG["epochs"] / 2)),
        eta_min=CONFIG["lr_final"],
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
        f"Training {ablation_name} | repeat={repeat:02d} split={split_id} | "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    print(f"Output folder: {fold_out_dir}")
    print("=" * 80)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0

        for batch_idx, (xb, y_onehot, _y_label) in enumerate(train_loader):
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss, dice_component, ce_component = combined_dice_ce_loss(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=CONFIG["dice_weight"],
                ce_weight=CONFIG["ce_weight"],
            )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

            if batch_idx % 50 == 0:
                print(
                    f"  Epoch {epoch:03d} batch {batch_idx:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} dice={dice_component.item():.4f} ce={ce_component.item():.4f}"
                )

            del xb, y_onehot, out, loss, dice_component, ce_component

        scheduler.step()

        (
            f1_per_class,
            mean_f1,
            iou_per_class,
            mean_iou,
            iou_all_classes,
            mean_iou_all,
            val_loss,
            event_plot_payloads,
            cm,
        ) = compute_event_f1_iou_graphsage(
            model,
            val_loader,
            device,
            return_cm=True,
            return_val_loss=True,
            return_event_plot_payloads=True,
            save_event_plots=False,
            event_plots_dir=val_plot_dir,
            max_event_plots=CONFIG["val_plot_events"],
            epoch=epoch,
        )

        is_best_mean_f1_epoch = float(mean_f1) > float(best_mean_f1)
        if is_best_mean_f1_epoch:
            saved_plot_count = save_event_plot_payloads(
                event_plot_payloads,
                val_plot_dir,
                epoch=epoch,
            )
            best_mean_f1 = float(mean_f1)
            best_epoch = int(epoch)
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(mean_f1),
                },
                checkpoints_dir / "best_f1.pt",
            )
        else:
            saved_plot_count = 0
            epochs_without_improvement += 1

        if float(train_loss) < float(best_train_loss):
            best_train_loss = float(train_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(mean_f1),
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
                    "f1score": float(mean_f1),
                },
                checkpoints_dir / "best_val_loss.pt",
            )

        if CONFIG["save_confusion_matrix_each_epoch"]:
            cm_labels = ["VT", "LP", "TR", "AV", "IC"]
            cm_path = cm_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
            save_confusion_matrix_image(
                cm=cm,
                labels=cm_labels,
                out_path=cm_path,
                title=f"Confusion Matrix - {model_name} - Epoch {epoch}",
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics_rows.append(
            [
                current_lr,
                epoch,
                float(train_loss),
                float(val_loss),
                float(f1_per_class[0]),
                float(f1_per_class[1]),
                float(f1_per_class[2]),
                float(f1_per_class[3]),
                float(f1_per_class[4]),
                float(mean_f1),
                float(iou_per_class[0]),
                float(iou_per_class[1]),
                float(iou_per_class[2]),
                float(iou_per_class[3]),
                float(iou_per_class[4]),
                float(mean_iou),
                float(iou_all_classes[0]),
                float(iou_all_classes[1]),
                float(iou_all_classes[2]),
                float(iou_all_classes[3]),
                float(iou_all_classes[4]),
                float(iou_all_classes[5]),
                float(mean_iou_all),
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
                "BG_iou_all",
                "VT_iou_all",
                "LP_iou_all",
                "TR_iou_all",
                "AV_iou_all",
                "IC_iou_all",
                "mean_iou_all",
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
            f"mean_f1={mean_f1:.4f} mean_iou={mean_iou:.4f} "
            f"best_epoch={best_epoch if best_epoch >= 0 else 'NA'} "
            f"no_improve={epochs_without_improvement}/{CONFIG['early_stop_patience']} "
            f"saved_best_plots={saved_plot_count}"
        )

        del event_plot_payloads, cm
        cleanup_gpu_cache()

        if epochs_without_improvement >= int(CONFIG["early_stop_patience"]):
            print(
                f"Early stopping at epoch {epoch:03d}: no mean_f1 improvement for "
                f"{CONFIG['early_stop_patience']} consecutive epochs."
            )
            break

    # Evaluate on test set using best_f1 checkpoint.
    best_f1_ckpt = checkpoints_dir / "best_f1.pt"
    if not best_f1_ckpt.exists():
        raise RuntimeError(
            f"best_f1 checkpoint not found for fold output: {best_f1_ckpt}"
        )

    ckpt = torch.load(best_f1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    (
        test_f1_per_class,
        test_mean_f1,
        test_iou_per_class,
        test_mean_iou,
        test_iou_all_classes,
        test_mean_iou_all,
        test_loss,
        test_cm,
    ) = compute_event_f1_iou_graphsage(
        model,
        test_loader,
        device,
        return_cm=True,
        return_val_loss=True,
        return_event_plot_payloads=False,
        save_event_plots=False,
        max_event_plots=0,
        epoch=None,
    )

    test_cm_path = cm_dir / "confusion_matrix_test_best_f1.png"
    save_confusion_matrix_image(
        cm=test_cm,
        labels=["VT", "LP", "TR", "AV", "IC"],
        out_path=test_cm_path,
        title=f"Test Confusion Matrix - {model_name} - best_f1",
    )

    fold_elapsed_sec = float(time.time() - fold_start)

    fold_summary = {
        "ablation": ablation_name,
        "repeat": int(repeat),
        "split": int(split_id),
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
        "test_mean_iou_all": float(test_mean_iou_all),
        "test_f1_per_class": [float(x) for x in test_f1_per_class],
        "test_iou_per_class": [float(x) for x in test_iou_per_class],
        "test_iou_all_classes": [float(x) for x in test_iou_all_classes],
        "fold_elapsed_seconds": fold_elapsed_sec,
    }

    with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    del optimizer, scheduler, model, ckpt
    cleanup_gpu_cache()

    return fold_summary


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
        "folds": [{"repeat": int(r), "split": int(s)} for r in REPEATS for s in SPLITS],
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

        for repeat in REPEATS:
            for split_id in SPLITS:
                fold_data_dir = DATA_ROOT / f"repeat_{repeat:02d}" / f"split_{split_id}"
                ensure_fold_data_exists(fold_data_dir)

                fold_out_dir = (
                    ablation_root / f"repeat_{repeat:02d}" / f"split_{split_id}"
                )
                fold_summary = train_one_fold(
                    ablation_name=ablation_name,
                    model_kwargs=ABLATION_MODEL_KWARGS[ablation_name],
                    repeat=repeat,
                    split_id=split_id,
                    fold_data_dir=fold_data_dir,
                    fold_out_dir=fold_out_dir,
                    device=device,
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

        with (aggregate_dir / "cv5x2_summary.json").open("w", encoding="utf-8") as f:
            json.dump(ablation_summary, f, indent=2)

        pd.DataFrame([ablation_summary]).to_csv(
            aggregate_dir / "cv5x2_summary.csv",
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
    print("Ablation 5x2 run complete")
    print(f"Experiment folder: {EXPERIMENT_ROOT}")
    print(f"Leaderboard: {comparisons_dir / 'ablation_leaderboard.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
