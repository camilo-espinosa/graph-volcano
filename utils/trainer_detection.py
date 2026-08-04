"""
Trainer for event detection models (MuSSED).

This module provides training for event detector models that output temporal
event predictions rather than segmentation masks. Currently supports MuSSED.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy.special import softmax

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
from utils.validation_plots import plot_event_validation


def compute_event_confusion_matrix(
    all_predictions: dict,
    all_targets: list,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    num_classes: int = 6,
) -> np.ndarray:
    """
    Compute confusion matrix from predicted and target events.

    Matches predicted events to target events using IoU, then builds confusion matrix
    from the (true_class, pred_class) pairs.

    Args:
        all_predictions: Dict with keys "class_logits", "center", "start", "end", "confidence"
                        Each is [N_pred] or [N_pred, C] array
        all_targets: List of event lists (each sample has list of EventInterval objects)
        confidence_threshold: Minimum confidence to consider prediction
        iou_threshold: Minimum IoU to match predictions to targets
        num_classes: Number of classes (including background)

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    print(f"[compute_event_confusion_matrix] Starting computation...")
    t0_cm_compute = time.time()
    class_logits = np.asarray(all_predictions["class_logits"])  # [B, Nq, C]
    pred_starts = np.asarray(all_predictions["start"])  # [B, Nq, 1]
    pred_ends = np.asarray(all_predictions["end"])  # [B, Nq, 1]
    pred_conf_logits = np.asarray(all_predictions["confidence"])  # [B, Nq, 1]

    if class_logits.ndim != 3:
        raise ValueError(
            f"Expected class_logits to be [B, Nq, C], got shape {class_logits.shape}."
        )
    if pred_starts.shape[:2] != class_logits.shape[:2]:
        raise ValueError(
            f"Start shape mismatch: starts {pred_starts.shape} vs class_logits {class_logits.shape}."
        )
    if pred_ends.shape[:2] != class_logits.shape[:2]:
        raise ValueError(
            f"End shape mismatch: ends {pred_ends.shape} vs class_logits {class_logits.shape}."
        )
    if pred_conf_logits.shape[:2] != class_logits.shape[:2]:
        raise ValueError(
            f"Confidence shape mismatch: confidence {pred_conf_logits.shape} vs class_logits {class_logits.shape}."
        )

    batch_size, n_queries, _ = class_logits.shape
    if len(all_targets) != batch_size:
        raise ValueError(
            f"Target length mismatch: targets={len(all_targets)} vs predictions batch={batch_size}."
        )

    print(
        f"[compute_event_confusion_matrix] Predictions: {batch_size} samples x {n_queries} queries"
    )
    print(
        f"[compute_event_confusion_matrix] Targets: {len(all_targets)} samples with {sum(len(t) for t in all_targets)} total events"
    )

    def temporal_iou(
        pred_start: float, pred_end: float, target_start: float, target_end: float
    ) -> float:
        ps = float(np.clip(pred_start, 0.0, 1.0))
        pe = float(np.clip(pred_end, 0.0, 1.0))
        ts = float(np.clip(target_start, 0.0, 1.0))
        te = float(np.clip(target_end, 0.0, 1.0))

        if ps > pe:
            ps, pe = pe, ps
        if ts > te:
            ts, te = te, ts

        inter = max(0.0, min(pe, te) - max(ps, ts))
        union = (pe - ps) + (te - ts) - inter
        if union <= 1e-8:
            return 0.0
        return inter / union

    true_classes = []
    pred_classes = []

    for b in range(batch_size):
        probs = softmax(class_logits[b], axis=-1)  # [Nq, C]
        conf_sigmoid = 1.0 / (1.0 + np.exp(-pred_conf_logits[b, :, 0]))  # [Nq]

        sample_preds = []
        for q in range(n_queries):
            pred_class = int(np.argmax(probs[q, 1:]) + 1)  # exclude background
            pred_score = float(probs[q, pred_class] * conf_sigmoid[q])
            if pred_score < confidence_threshold:
                continue

            sample_preds.append(
                {
                    "class_id": pred_class,
                    "start": float(np.clip(pred_starts[b, q, 0], 0.0, 1.0)),
                    "end": float(np.clip(pred_ends[b, q, 0], 0.0, 1.0)),
                    "score": pred_score,
                }
            )

        sample_preds.sort(key=lambda item: item["score"], reverse=True)
        matched_pred = np.zeros(len(sample_preds), dtype=bool)
        matched_target = np.zeros(len(all_targets[b]), dtype=bool)

        for t_idx, target in enumerate(all_targets[b]):
            if target.class_id == 0:
                continue

            best_pred_idx = -1
            best_iou = 0.0

            for p_idx, pred in enumerate(sample_preds):
                if matched_pred[p_idx]:
                    continue

                iou = temporal_iou(
                    pred["start"],
                    pred["end"],
                    target.start_norm,
                    target.end_norm,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = p_idx

            if best_pred_idx >= 0 and best_iou >= iou_threshold:
                matched_pred[best_pred_idx] = True
                matched_target[t_idx] = True
                true_classes.append(int(target.class_id))
                pred_classes.append(int(sample_preds[best_pred_idx]["class_id"]))
            else:
                # Undetected event (false negative): pred_class=0 means "No Detection"
                matched_target[t_idx] = True
                true_classes.append(int(target.class_id))
                pred_classes.append(0)

    # Build confusion matrix with classes 0-5 (0=No Detection, 1-5=VT,LP,TR,AV,IC)
    cm = confusion_matrix(
        true_classes,
        pred_classes,
        labels=list(range(num_classes)),  # Include 0 for "No Detection"
    )

    print(
        f"[compute_event_confusion_matrix] Complete: {time.time() - t0_cm_compute:.3f}s - Matched {len(true_classes)} events"
    )
    return cm


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
    cm_dir = fold_out_dir / "confusion_matrices"

    for p in (checkpoints_dir, reports_dir, val_plot_dir, cm_dir):
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

    # Build param groups with differential learning rates
    def build_param_groups(model, base_lr):
        """
        Assign different learning rates to different parts of MuSSED model.

        Strategy:
        - Encoder (backbone): 0.1 * base_lr (preserve feature extraction)
        - Attention modules: 0.3 * base_lr (intermediate)
        - Detection head: 1.0 * base_lr (fast adaptation)
        """
        param_groups = [
            {"params": [], "lr": base_lr * 0.1, "name": "encoder"},
            {"params": [], "lr": base_lr * 0.3, "name": "attention"},
            {"params": [], "lr": base_lr, "name": "detection_head"},
        ]

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Route parameters to appropriate group
            if "encoder" in name.lower() and "attention" not in name.lower():
                param_groups[0]["params"].append(param)
            elif "attention" in name.lower():
                param_groups[1]["params"].append(param)
            else:  # detection head, transformer, etc.
                param_groups[2]["params"].append(param)

        return param_groups

    param_groups = build_param_groups(model, config["lr"])

    # Set eta_min per param_group to maintain proportional scaling during annealing
    for group in param_groups:
        group["initial_lr"] = group["lr"]
        group["eta_min"] = group["lr"] * (config["lr_final"] / config["lr"])

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"] / 4)),
    )

    # Set eta_min for scheduler (will be overridden per param_group)
    for param_group in optimizer.param_groups:
        param_group["eta_min"] = param_group.get("eta_min", config["lr_final"])

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

        with torch.inference_mode():
            for batch_idx, batch in enumerate(val_loader):
                xb = batch[0].to(device)
                y_onehot = batch[1]

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

                # Release batch tensors before next iteration.
                del xb, y_onehot, predictions, targets, loss_dict

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

        # Compute event detection metrics (mAP at various IoU thresholds)
        detection_metrics = metrics_fn.evaluate_batch(
            all_predictions, all_targets, confidence_threshold=0.5
        )

        # Unified event summary from one matching pass: confusion matrix + per-class metrics.
        detection_summary = metrics_fn.compute_detection_summary(
            all_predictions,
            all_targets,
            iou_threshold=0.5,
            confidence_threshold=0.5,
        )
        per_class_f1_dict = detection_summary["per_class_f1"]
        per_class_iou_dict = detection_summary["per_class_iou"]
        per_class_stats = detection_summary["per_class"]

        # Extract per-class F1 and IoU scores: class_id 1-5 maps to VT, LP, TR, AV, IC
        class_f1_scores = [
            per_class_f1_dict.get(class_id, 0.0) for class_id in range(1, 6)
        ]
        class_iou_scores = [
            per_class_iou_dict.get(class_id, 0.0) for class_id in range(1, 6)
        ]

        # mean_f1 = average of per-class F1 scores
        mean_f1 = float(np.mean(class_f1_scores)) if class_f1_scores else 0.0
        # mean_iou = average of per-class IoU scores (actual IoU metric)
        mean_iou = float(np.mean(class_iou_scores)) if class_iou_scores else 0.0

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

            # Generate validation event plots on best epoch
            saved_plot_count = plot_event_validation(
                model=model,
                dataloader=val_loader,
                device=device,
                output_dir=val_plot_dir,
                epoch=epoch,
                max_samples=config.get("val_plot_events", 15),
                extract_attention=True,
            )

            # Confusion matrix from the same matching pass used for F1/IoU.
            val_cm = detection_summary["confusion_matrix"]

            # Save rolling best (gets replaced each new best)
            save_confusion_matrix_image(
                cm=val_cm,
                labels=["No Detection", "VT", "LP", "TR", "AV", "IC"],
                out_path=cm_dir / "confusion_matrix_val_best_f1.png",
                title=f"Event Detection Confusion Matrix - Fold {fold_id} - Best F1",
            )
            # Save per-epoch record (historical)
            save_confusion_matrix_image(
                cm=val_cm,
                labels=["No Detection", "VT", "LP", "TR", "AV", "IC"],
                out_path=cm_dir / f"confusion_matrix_epoch_{epoch:03d}.png",
                title=f"Event Detection Confusion Matrix - Fold {fold_id} - Epoch {epoch + 1}",
            )
        else:
            epochs_without_improvement += 1
            saved_plot_count = 0

        # Save metrics rows for CSV export
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Main CSV: ordered list format identical to segmentation models for comparison
        # [lr, epoch, train_loss, val_loss, VT_f1, LP_f1, TR_f1, AV_f1, IC_f1, mean_f1, VT_iou, LP_iou, TR_iou, AV_iou, IC_iou, mean_iou]
        # For event detection: use per-class F1@0.5 scores and per-class IoU scores
        vt_f1, lp_f1, tr_f1, av_f1, ic_f1 = class_f1_scores
        vt_iou, lp_iou, tr_iou, av_iou, ic_iou = class_iou_scores

        metrics_row = [
            current_lr,
            int(epoch + 1),
            float(avg_train_loss),
            float(avg_val_loss),
            vt_f1,
            lp_f1,
            tr_f1,
            av_f1,
            ic_f1,
            float(mean_f1),
            vt_iou,
            lp_iou,
            tr_iou,
            av_iou,
            ic_iou,
            float(mean_iou),
        ]
        metrics_rows.append(metrics_row)

        # Detection metrics CSV (additional metrics specific to event detection): ordered list format
        # [epoch, mAP@0.1, mAP@0.3, mAP@0.5, mAP@0.7, mAP@0.9, mAP, mean_f1, F1_VT@0.5, F1_LP@0.5, F1_TR@0.5, F1_AV@0.5, F1_IC@0.5]
        detection_row = [
            int(epoch + 1),
            float(detection_metrics.get("mAP@0.1", 0.0)),
            float(detection_metrics.get("mAP@0.3", 0.0)),
            float(detection_metrics.get("mAP@0.5", 0.0)),
            float(detection_metrics.get("mAP@0.7", 0.0)),
            float(detection_metrics.get("mAP@0.9", 0.0)),
            float(detection_metrics.get("mAP", 0.0)),
            float(mean_f1),
            vt_f1,
            lp_f1,
            tr_f1,
            av_f1,
            ic_f1,
        ]
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
            f"Plots:       {saved_plot_count} validation event plots saved\n"
            f"======================================================================"
        )

        # Save metrics incrementally after each epoch (so data is preserved if training crashes)
        from utils.metrics_reporter import save_training_history_detection

        save_training_history_detection(
            metrics_rows=metrics_rows,
            detection_metrics_rows=detection_metrics_rows,
            output_dir=reports_dir,
            main_filename="training_metrics.csv",
            detection_filename="detection_metrics.csv",
        )

        # Early stopping
        if epochs_without_improvement >= config["early_stop_patience"]:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs "
                f"without improvement (patience={config['early_stop_patience']})"
            )
            break

    # Metrics have been saved after each epoch; print final summary
    if metrics_rows:
        print(
            f"Saved training metrics to {reports_dir / 'training_metrics.csv'} "
            f"({len(metrics_rows)} epochs)"
        )
    if detection_metrics_rows:
        print(
            f"Saved detection metrics to {reports_dir / 'detection_metrics.csv'} "
            f"({len(detection_metrics_rows)} epochs)"
        )

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

    with torch.inference_mode():
        for batch in test_loader:
            xb = batch[0].to(device)
            y_onehot = batch[1]

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

            # Release batch tensors before next iteration.
            del xb, y_onehot, predictions, targets, loss_dict

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
