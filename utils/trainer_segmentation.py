"""
Unified trainer for 2D and 1D segmentation models (UNet, PhaseNet, MuSSeg).

This module provides a single training entry point that handles both 2D and 1D
segmentation models using conditional logic based on trainer_kind.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from utils.train_utils import (
    combined_dice_ce_loss,
    combined_dice_ce_loss_2d,
    save_confusion_matrix_image,
    compute_event_f1_iou_graphsage,
    save_event_plot_payloads,
    evaluate_unet_model,
    collect_unet_misclassified_event_plots,
    cleanup_gpu_cache,
    MultiStation1DDataset,
    UNetPatchDataset,
    BalancedBatchSampler,
)
from utils.model_registry import get_model_spec


def train_one_segmentation_fold(
    trainer_kind: str,
    model_key_or_kwargs: str | dict,
    fold_id: int,
    fold_data_dir: Path,
    fold_out_dir: Path,
    device: torch.device,
    config: dict,
) -> dict:
    """
    Train a segmentation model (2D or 1D) for one fold.

    Args:
        trainer_kind: "2d" for UNet, "1d" for PhaseNet/MuSSeg
        model_key_or_kwargs: Model registry key (str) or model_kwargs dict
        fold_id: Fold index
        fold_data_dir: Path to fold data (contains train_aug.npz, val.npz, test.npz)
        fold_out_dir: Output directory for checkpoints, reports, plots
        device: torch.device
        config: Training config dict with keys:
            - batch_size, lr, lr_final, epochs, dice_weight, ce_weight
            - early_stop_patience, val_plot_events, save_confusion_matrix_each_epoch

    Returns:
        fold_summary: dict with results (best epoch, metrics, elapsed time)
    """

    checkpoints_dir = fold_out_dir / "checkpoints"
    reports_dir = fold_out_dir / "reports"
    cm_dir = fold_out_dir / "confusion_matrices"
    val_plot_dir = fold_out_dir / "validation_event_plots"

    for p in (checkpoints_dir, reports_dir, cm_dir, val_plot_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Load datasets and dataloaders based on trainer_kind
    if trainer_kind == "2d":
        train_ds = UNetPatchDataset(fold_data_dir / "train_aug.npz")
        val_ds = UNetPatchDataset(fold_data_dir / "val.npz")
        test_ds = UNetPatchDataset(fold_data_dir / "test.npz")

        balanced_batch_sampler = BalancedBatchSampler(
            train_ds.label_ids, batch_size=config["batch_size"]
        )
        train_loader = DataLoader(train_ds, batch_sampler=balanced_batch_sampler)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(
            test_ds, batch_size=config["batch_size"], shuffle=False
        )

        len_window = int(config.get("len_window", 8192))
        im_size = int(config.get("im_size", 256))

    elif trainer_kind == "1d":
        train_ds = MultiStation1DDataset(fold_data_dir / "train_aug.npz")
        val_ds = MultiStation1DDataset(fold_data_dir / "val.npz")
        test_ds = MultiStation1DDataset(fold_data_dir / "test.npz")

        balanced_batch_sampler = BalancedBatchSampler(
            train_ds.label_ids, batch_size=config["batch_size"]
        )
        train_loader = DataLoader(train_ds, batch_sampler=balanced_batch_sampler)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(
            test_ds, batch_size=config["batch_size"], shuffle=False
        )

        len_window = None
        im_size = None

    else:
        raise ValueError(f"Unknown trainer_kind: {trainer_kind}")

    # Load model
    if isinstance(model_key_or_kwargs, str):
        spec = get_model_spec(model_key_or_kwargs)
        model = spec["model_cls"](**spec["model_kwargs"]).to(device)
        display_name = spec.get("display_name", model_key_or_kwargs)
    else:
        # model_kwargs dict provided
        model_kwargs_copy = model_key_or_kwargs.copy()
        model_class = model_kwargs_copy.pop("_model_cls", None)
        if model_class is None:
            from utils.data_utils import UNet_GraphSAGE, UNet_MPNN

            model_class_name = model_kwargs_copy.pop("_model_class", "UNet_GraphSAGE")
            model_class = (
                UNet_MPNN if model_class_name == "UNet_MPNN" else UNet_GraphSAGE
            )
        else:
            model_class_name = getattr(model_class, "__name__", str(model_class))

        if model_class_name.startswith("PhaseNet"):
            model_kwargs_copy.setdefault("in_channels", 8)
            model_kwargs_copy.setdefault("classes", 6)
        else:
            model_kwargs_copy.setdefault("in_channels", 1)
            model_kwargs_copy.setdefault("out_channels", 6)

        model = model_class(**model_kwargs_copy).to(device)
        display_name = model_class_name

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"] / 2)),
        eta_min=config["lr_final"],
    )

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_val_mean_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    metrics_rows = []
    fold_start = time.time()

    print("=" * 80)
    print(
        f"Training {display_name} (segmentation) | fold={fold_id:02d} | "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    print(f"Output folder: {fold_out_dir}")
    print("=" * 80)

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        train_loss_dice = 0.0
        train_loss_ce = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if trainer_kind == "2d":
                xb, y_onehot, _y_idx = batch
            else:  # 1d
                xb, y_onehot = batch[0], batch[1]

            xb = xb.to(device)
            y_onehot = y_onehot.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)

            if trainer_kind == "2d":
                loss, dice_component, ce_component = combined_dice_ce_loss_2d(
                    out,
                    y_onehot,
                    class_weights=None,
                    dice_weight=config["dice_weight"],
                    ce_weight=config["ce_weight"],
                )
            else:  # 1d
                loss, dice_component, ce_component = combined_dice_ce_loss(
                    out,
                    y_onehot,
                    class_weights=None,
                    dice_weight=config["dice_weight"],
                    ce_weight=config["ce_weight"],
                )

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_loss_dice += float(dice_component.item())
            train_loss_ce += float(ce_component.item())

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch:03d} batch {batch_idx:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} [dice={dice_component.item():.4f} ce={ce_component.item():.4f}]"
                )

            del xb, y_onehot, out, loss, dice_component, ce_component

        scheduler.step()

        # Validation
        if trainer_kind == "2d":
            (
                val_f1_per_class,
                val_mean_f1,
                val_iou_per_class,
                val_mean_iou,
                val_loss,
                val_cm,
            ) = evaluate_unet_model(
                model=model,
                dataloader=val_loader,
                device=device,
                len_window=len_window,
                im_size=im_size,
                config=config,
            )
            val_loss_dice = None
            val_loss_ce = None
        else:  # 1d
            (
                val_f1_per_class,
                val_mean_f1,
                val_iou_per_class,
                val_mean_iou,
                val_iou_all_classes,
                val_mean_iou_all,
                val_loss,
                event_plot_payloads,
                val_cm,
            ) = compute_event_f1_iou_graphsage(
                model,
                val_loader,
                device,
                return_cm=True,
                return_val_loss=True,
                return_event_plot_payloads=True,
                save_event_plots=False,
                event_plots_dir=val_plot_dir,
                max_event_plots=config.get("val_plot_events", 15),
                epoch=epoch,
            )
            val_loss_dice = None
            val_loss_ce = None

        is_best_val_mean_f1_epoch = float(val_mean_f1) > float(best_val_mean_f1)

        if is_best_val_mean_f1_epoch:
            if trainer_kind == "1d":
                saved_plot_count = save_event_plot_payloads(
                    event_plot_payloads,
                    val_plot_dir,
                    epoch=epoch,
                )
            else:  # 2d
                saved_plot_count = save_event_plot_payloads(
                    collect_unet_misclassified_event_plots(
                        model=model,
                        npz_path=fold_data_dir / "val.npz",
                        device=device,
                        max_per_class=int(config.get("val_plot_events", 0)),
                        class_names=["BG", "VT", "LP", "TR", "AV", "IC"],
                    ),
                    val_plot_dir,
                    epoch=epoch,
                )

            save_confusion_matrix_image(
                cm=val_cm,
                labels=["VT", "LP", "TR", "AV", "IC"],
                out_path=cm_dir / "confusion_matrix_val_best_f1.png",
                title=f"Validation Confusion Matrix - {display_name} - best_f1",
            )
            best_val_mean_f1 = float(val_mean_f1)
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

        if config.get("save_confusion_matrix_each_epoch", False):
            cm_labels = ["VT", "LP", "TR", "AV", "IC"]
            cm_path = cm_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
            save_confusion_matrix_image(
                cm=val_cm,
                labels=cm_labels,
                out_path=cm_path,
                title=f"Confusion Matrix - {display_name} - Epoch {epoch}",
            )

        # Save metrics row
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
            f"no_improve={epochs_without_improvement}/{config['early_stop_patience']} "
            f"saved_best_plots={saved_plot_count}"
        )

        if trainer_kind == "1d":
            del event_plot_payloads
        del val_cm
        cleanup_gpu_cache()

        if epochs_without_improvement >= int(config["early_stop_patience"]):
            print(
                f"Early stopping at epoch {epoch:03d}: no mean_f1 improvement for "
                f"{config['early_stop_patience']} consecutive epochs."
            )
            break

    # Test evaluation with best model
    best_f1_ckpt = checkpoints_dir / "best_f1.pt"
    if not best_f1_ckpt.exists():
        raise RuntimeError(
            f"best_f1 checkpoint not found for fold output: {best_f1_ckpt}"
        )

    ckpt = torch.load(best_f1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if trainer_kind == "2d":
        (
            test_f1_per_class,
            test_mean_f1,
            test_iou_per_class,
            test_mean_iou,
            test_loss,
            test_cm,
        ) = evaluate_unet_model(
            model=model,
            dataloader=test_loader,
            device=device,
            len_window=len_window,
            im_size=im_size,
            config=config,
        )
    else:  # 1d
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
        title=f"Test Confusion Matrix - {display_name} - best_f1",
    )

    fold_elapsed_sec = float(time.time() - fold_start)

    fold_summary = {
        "trainer_kind": trainer_kind,
        "fold": int(fold_id),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_f1": float(best_val_mean_f1),
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
