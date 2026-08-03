"""
Trainer for event detection models (MuSSED).

This module provides training for event detector models that output temporal
event predictions rather than segmentation masks. Currently supports MuSSED.
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
    cleanup_gpu_cache,
    save_confusion_matrix_image,
    MultiStation1DDataset,
    BalancedBatchSampler,
)
from utils.model_registry import get_model_spec
from utils.detr_event_loss import DETREventLoss
from utils.event_detection_metrics import EventDetectionMetrics
from utils.event_targets import batch_segmentation_to_events


def train_one_event_detection_fold(
    model_key_or_kwargs: str | dict,
    fold_id: int,
    fold_data_dir: Path,
    fold_out_dir: Path,
    device: torch.device,
    config: dict,
) -> dict:
    """
    Train an event detection model (MuSSED) for one fold.

    Args:
        model_key_or_kwargs: Model registry key (str) or model_kwargs dict
        fold_id: Fold index
        fold_data_dir: Path to fold data (contains train_aug.npz, val.npz, test.npz)
        fold_out_dir: Output directory for checkpoints, reports, plots
        device: torch.device
        config: Training config dict with keys:
            - batch_size, lr, lr_final, epochs
            - early_stop_patience, val_plot_events
            - detection-specific: loss weights for class, bbox, conf, unmatched

    Returns:
        fold_summary: dict with results (best epoch, metrics, elapsed time)
    """

    checkpoints_dir = fold_out_dir / "checkpoints"
    reports_dir = fold_out_dir / "reports"
    val_plot_dir = fold_out_dir / "validation_event_plots"

    for p in (checkpoints_dir, reports_dir, val_plot_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = MultiStation1DDataset(fold_data_dir / "train_aug.npz")
    val_ds = MultiStation1DDataset(fold_data_dir / "val.npz")
    test_ds = MultiStation1DDataset(fold_data_dir / "test.npz")

    balanced_batch_sampler = BalancedBatchSampler(
        train_ds.label_ids, batch_size=config["batch_size"]
    )
    train_loader = DataLoader(train_ds, batch_sampler=balanced_batch_sampler)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Load model
    if isinstance(model_key_or_kwargs, str):
        spec = get_model_spec(model_key_or_kwargs)
        model = spec["model_cls"](**spec["model_kwargs"]).to(device)
        model_key = model_key_or_kwargs
    else:
        # model_kwargs dict provided
        raise NotImplementedError(
            "Direct model_kwargs dict for MuSSED not yet implemented. "
            "Please use model registry key."
        )

    # Initialize loss function and metrics
    loss_fn = DETREventLoss(num_classes=6)
    metrics_fn = EventDetectionMetrics(num_classes=6)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"] / 2)),
        eta_min=config["lr_final"],
    )

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_mean_f1 = float("-inf")  # F1@0.5 for events
    best_epoch = -1
    epochs_without_improvement = 0

    metrics_rows = []
    detection_metrics_rows = []
    fold_start = time.time()

    print("=" * 80)
    print(
        f"Training {model_key} (event detection) | fold={fold_id:02d} | "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    print(f"Output folder: {fold_out_dir}")
    print("=" * 80)

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        train_loss_class = 0.0
        train_loss_bbox = 0.0
        train_loss_conf = 0.0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            xb = batch[0].to(device)
            y_onehot = batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass through MuSSED model
            predictions = model(xb)

            # Convert segmentation targets to event targets
            # batch_segmentation_to_events expects (B, C, T) and returns list of event dicts per sample
            targets = batch_segmentation_to_events(y_onehot, normalize=True)

            # Compute detection loss
            loss_dict = loss_fn(predictions, targets)
            loss = loss_dict["loss_total"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            train_loss_class += loss_dict["loss_class"].item()
            train_loss_bbox += loss_dict["loss_bbox"].item()
            train_loss_conf += loss_dict["loss_conf"].item()
            num_train_batches += 1

            # Print progress every few batches
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0 or batch_idx == 0:
                current_loss = train_loss / (batch_idx + 1)
                print(
                    f"  Epoch {epoch + 1:3d} batch {batch_idx + 1:4d}/{len(train_loader):4d} | "
                    f"loss={current_loss:.4f} "
                    f"[class={train_loss_class / (batch_idx + 1):.4f} "
                    f"bbox={train_loss_bbox / (batch_idx + 1):.4f} "
                    f"conf={train_loss_conf / (batch_idx + 1):.4f}]"
                )

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_loss_class = 0.0
        val_loss_bbox = 0.0
        val_loss_conf = 0.0
        all_predictions = {
            "class_logits": [],
            "center": [],
            "start": [],
            "end": [],
            "confidence": [],
        }
        all_targets = []
        num_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                xb = batch[0].to(device)
                y_onehot = batch[1].to(device)

                # Forward pass
                predictions = model(xb)

                # Convert targets
                targets = batch_segmentation_to_events(y_onehot, normalize=True)
                all_targets.extend(targets)

                # Compute loss
                loss_dict = loss_fn(predictions, targets)
                val_loss += loss_dict["loss_total"].item()
                val_loss_class += loss_dict["loss_class"].item()
                val_loss_bbox += loss_dict["loss_bbox"].item()
                val_loss_conf += loss_dict["loss_conf"].item()
                num_val_batches += 1

                # Collect predictions for metrics computation
                for key in all_predictions:
                    all_predictions[key].append(predictions[key].detach().cpu().numpy())

        # Average losses
        avg_train_loss = (
            train_loss / num_train_batches if num_train_batches > 0 else float("inf")
        )
        avg_train_loss_class = (
            train_loss_class / num_train_batches if num_train_batches > 0 else 0.0
        )
        avg_train_loss_bbox = (
            train_loss_bbox / num_train_batches if num_train_batches > 0 else 0.0
        )
        avg_train_loss_conf = (
            train_loss_conf / num_train_batches if num_train_batches > 0 else 0.0
        )

        avg_val_loss = (
            val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        )
        avg_val_loss_class = (
            val_loss_class / num_val_batches if num_val_batches > 0 else 0.0
        )
        avg_val_loss_bbox = (
            val_loss_bbox / num_val_batches if num_val_batches > 0 else 0.0
        )
        avg_val_loss_conf = (
            val_loss_conf / num_val_batches if num_val_batches > 0 else 0.0
        )

        # Concatenate predictions for metrics computation
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

        # Compute event detection metrics (mAP, F1@0.5, per-class AP)
        detection_metrics = metrics_fn.evaluate_batch(
            all_predictions, all_targets, confidence_threshold=0.5
        )

        # Compute F1@0.5 and mean IoU
        mean_f1 = metrics_fn.compute_f1(
            all_predictions, all_targets, iou_threshold=0.5, confidence_threshold=0.5
        )
        mean_iou = np.mean(
            [
                detection_metrics.get(f"mAP@{t:.1f}", 0.0)
                for t in [0.1, 0.3, 0.5, 0.7, 0.9]
            ]
        )

        is_best_mean_f1_epoch = mean_f1 > best_mean_f1
        if is_best_mean_f1_epoch:
            best_mean_f1 = mean_f1
            best_epoch = int(epoch)
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "f1_score": mean_f1,
                },
                checkpoints_dir / "best_f1.pt",
            )
        else:
            epochs_without_improvement += 1

        # Save metrics rows for CSV export
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Main CSV (same format as segmentation models for compatibility)
        metrics_row = {
            "epoch": int(epoch + 1),
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "train_loss_class": float(avg_train_loss_class),
            "train_loss_bbox": float(avg_train_loss_bbox),
            "train_loss_conf": float(avg_train_loss_conf),
            "val_loss_class": float(avg_val_loss_class),
            "val_loss_bbox": float(avg_val_loss_bbox),
            "val_loss_conf": float(avg_val_loss_conf),
            "mean_f1": float(mean_f1),
            "mean_iou": float(mean_iou),
            "lr": current_lr,
        }
        metrics_rows.append(metrics_row)

        # Detection metrics CSV (additional metrics specific to event detection)
        detection_row = {"epoch": int(epoch + 1)}
        detection_row.update(detection_metrics)
        detection_row["F1@0.5"] = float(mean_f1)
        detection_metrics_rows.append(detection_row)

        # Print epoch summary
        print(
            f"==============================================================================\n"
            f"EPOCH {epoch + 1:3d} SUMMARY\n"
            f"==============================================================================\n"
            f"Train Loss:  {avg_train_loss:.4f}  "
            f"[class={avg_train_loss_class:.4f} bbox={avg_train_loss_bbox:.4f} "
            f"conf={avg_train_loss_conf:.4f}]\n"
            f"Val Loss:    {avg_val_loss:.4f}  "
            f"[class={avg_val_loss_class:.4f} bbox={avg_val_loss_bbox:.4f} "
            f"conf={avg_val_loss_conf:.4f}]\n"
            f"Metrics:     mean_f1={mean_f1:.4f} mean_iou={mean_iou:.4f} "
            f"best_epoch={best_epoch + 1} no_improve={epochs_without_improvement}/{config['early_stop_patience']}\n"
            f"mAP:         {detection_metrics.get('mAP', 0.0):.4f} "
            f"(@ 0.1: {detection_metrics.get('mAP@0.1', 0.0):.4f}, "
            f"0.5: {detection_metrics.get('mAP@0.5', 0.0):.4f}, "
            f"0.9: {detection_metrics.get('mAP@0.9', 0.0):.4f})\n"
            f"======================================================================"
        )

        # Early stopping
        if epochs_without_improvement >= config["early_stop_patience"]:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs "
                f"without improvement (patience={config['early_stop_patience']})"
            )
            break

    # Save metrics to CSV
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = reports_dir / "epoch_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved epoch metrics to {metrics_csv}")

    if detection_metrics_rows:
        detection_df = pd.DataFrame(detection_metrics_rows)
        detection_csv = reports_dir / "detection_metrics.csv"
        detection_df.to_csv(detection_csv, index=False)
        print(f"Saved detection metrics to {detection_csv}")

    # Load best model and evaluate on test set
    best_model_path = checkpoints_dir / "best_f1.pt"
    if best_model_path.exists():
        checkpoint = torch.load(
            best_model_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Test set evaluation
    model.eval()
    test_loss = 0.0
    all_test_predictions = {
        "class_logits": [],
        "center": [],
        "start": [],
        "end": [],
        "confidence": [],
    }
    all_test_targets = []
    num_test_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            xb = batch[0].to(device)
            y_onehot = batch[1].to(device)

            predictions = model(xb)
            targets = batch_segmentation_to_events(y_onehot, normalize=True)
            all_test_targets.extend(targets)

            loss_dict = loss_fn(predictions, targets)
            test_loss += loss_dict["loss_total"].item()
            num_test_batches += 1

            for key in all_test_predictions:
                all_test_predictions[key].append(
                    predictions[key].detach().cpu().numpy()
                )

    avg_test_loss = (
        test_loss / num_test_batches if num_test_batches > 0 else float("inf")
    )

    for key in all_test_predictions:
        all_test_predictions[key] = np.concatenate(all_test_predictions[key], axis=0)

    test_metrics = metrics_fn.evaluate_batch(
        all_test_predictions, all_test_targets, confidence_threshold=0.5
    )
    test_f1 = metrics_fn.compute_f1(
        all_test_predictions,
        all_test_targets,
        iou_threshold=0.5,
        confidence_threshold=0.5,
    )

    fold_elapsed = time.time() - fold_start

    # Prepare fold summary
    fold_summary = {
        "fold": fold_id,
        "best_epoch": best_epoch + 1,
        "best_val_loss": float(best_val_loss),
        "best_mean_f1": float(best_mean_f1),
        "test_loss": float(avg_test_loss),
        "test_mean_f1": float(test_f1),
        "test_mAP": float(test_metrics.get("mAP", 0.0)),
        "elapsed_seconds": float(fold_elapsed),
    }
    fold_summary.update(test_metrics)
    fold_summary["test_F1@0.5"] = float(test_f1)

    # Save fold summary
    summary_path = reports_dir / "fold_summary.json"
    with summary_path.open("w") as f:
        json.dump(fold_summary, f, indent=2)

    print(f"Fold summary saved to {summary_path}")
    print(
        f"Fold {fold_id} complete | Best Epoch: {best_epoch + 1} | "
        f"Test F1: {test_f1:.4f} | Test mAP: {test_metrics.get('mAP', 0.0):.4f} | "
        f"Elapsed: {fold_elapsed / 60:.1f} min"
    )

    cleanup_gpu_cache()

    return fold_summary

    fold_elapsed_sec = float(time.time() - fold_start)

    fold_summary = {
        "trainer_kind": "detr",
        "fold": int(fold_id),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_f1": float(best_mean_f1),
        "test_loss": 0.0,  # TODO
        "test_mean_f1": 0.0,  # TODO
        "test_mAP": 0.0,  # TODO
        "fold_elapsed_seconds": fold_elapsed_sec,
    }

    with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    del optimizer, scheduler
    cleanup_gpu_cache()

    return fold_summary
