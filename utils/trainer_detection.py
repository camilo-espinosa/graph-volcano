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
from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.model_registry import DETR_EVAL_DEFAULTS, get_model_spec
from utils.detr_event_loss import DETREventLoss
from utils.event_detection_metrics import EventDetectionMetrics, is_interval_match
from utils.event_targets import batch_segmentation_to_events
from utils.validation_plots import plot_event_validation

DEFAULT_DETR_LOSS_WEIGHTS = {
    "class_loss": 4.0,
    "bbox_loss": 2.0,
    "giou_loss": 2.0,
    "unmatched_query": 2.0,
}


def _normalize_detr_loss_weights(raw_weights: dict | None) -> dict[str, float]:
    """Normalize alias keys to DETREventLoss schema and cast to float."""
    if not raw_weights:
        return {}

    normalized: dict[str, float] = {}
    key_aliases = {
        "class": "class_loss",
        "bbox": "bbox_loss",
        "giou": "giou_loss",
        "confidence": "unmatched_query",
        "class_loss": "class_loss",
        "bbox_loss": "bbox_loss",
        "giou_loss": "giou_loss",
        "unmatched_query": "unmatched_query",
    }
    for key, value in raw_weights.items():
        mapped_key = key_aliases.get(str(key))
        if mapped_key is None:
            continue
        normalized[mapped_key] = float(value)
    return normalized


def _resolve_detr_loss_weights(model_spec: dict, config: dict) -> dict[str, float]:
    """Resolve final DETR loss weights with precedence config > model spec > default."""
    resolved = dict(DEFAULT_DETR_LOSS_WEIGHTS)

    spec_weights = _normalize_detr_loss_weights(model_spec.get("loss_weights"))
    config_weights = _normalize_detr_loss_weights(config.get("loss_weights"))

    resolved.update(spec_weights)
    resolved.update(config_weights)
    return resolved


def _resolve_detr_eval_matching(model_spec: dict, config: dict) -> dict[str, float | str]:
    """Resolve DETR evaluation matching settings with config > model spec > defaults."""
    resolved: dict[str, float | str] = dict(DETR_EVAL_DEFAULTS)

    spec_eval = model_spec.get("eval_matching")
    if isinstance(spec_eval, dict):
        resolved.update(spec_eval)

    config_eval: dict[str, float | str] = {}
    if "match_iou_threshold" in config:
        config_eval["match_iou_threshold"] = float(config["match_iou_threshold"])
    if "matching_strategy" in config:
        config_eval["matching_strategy"] = str(config["matching_strategy"])
    if "overlap_recall_threshold" in config:
        config_eval["overlap_recall_threshold"] = float(
            config["overlap_recall_threshold"]
        )
    resolved.update(config_eval)

    strategy = str(resolved.get("matching_strategy", "iou")).strip().lower()
    if strategy not in {"iou", "overlap_recall", "dual"}:
        raise ValueError(
            "matching_strategy must be one of {'iou', 'overlap_recall', 'dual'}, "
            f"got {strategy!r}."
        )

    iou_threshold = float(resolved.get("match_iou_threshold", 0.3))
    overlap_threshold = float(resolved.get("overlap_recall_threshold", 0.8))

    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(
            f"match_iou_threshold must be in [0, 1], got {iou_threshold}."
        )
    if not (0.0 <= overlap_threshold <= 1.0):
        raise ValueError(
            "overlap_recall_threshold must be in [0, 1], got "
            f"{overlap_threshold}."
        )

    return {
        "match_iou_threshold": iou_threshold,
        "matching_strategy": strategy,
        "overlap_recall_threshold": overlap_threshold,
    }


def _checkpoint_state_dict_for_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return model state dict for checkpointing."""
    return dict(model.state_dict())


def compute_event_confusion_matrix(
    all_predictions: dict,
    all_targets: list,
    iou_threshold: float = 0.3,
    num_classes: int = 6,
) -> np.ndarray:
    """
    Compute confusion matrix from predicted and target events.

    Matches predicted events to target events using IoU, then builds confusion matrix
    from the (true_class, pred_class) pairs.

    Args:
        all_predictions: Dict with keys "class_logits", "center", "start", "end"
                        Each is [N_pred] or [N_pred, C] array
        all_targets: List of event lists (each sample has list of EventInterval objects)
        iou_threshold: Minimum IoU to match predictions to targets
        num_classes: Number of classes (including background)

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    print(f"[compute_event_confusion_matrix] Starting computation...")
    t0_cm_compute = time.time()
    all_predictions = normalize_prediction_intervals(all_predictions)
    class_logits = np.asarray(all_predictions["class_logits"])  # [B, Nq, C]
    pred_starts = np.asarray(all_predictions["start"])  # [B, Nq, 1]
    pred_ends = np.asarray(all_predictions["end"])  # [B, Nq, 1]

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

        sample_preds = []
        for q in range(n_queries):
            pred_class = int(np.argmax(probs[q]))
            if pred_class == 0:
                continue

            pred_score = float(1.0 - probs[q, 0])

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
                # Undetected event (false negative): pred_class=0 means background.
                matched_target[t_idx] = True
                true_classes.append(int(target.class_id))
                pred_classes.append(0)

    # Build confusion matrix with classes 0-5 (0=Background, 1-5=VT,LP,TR,AV,IC)
    cm = confusion_matrix(
        true_classes,
        pred_classes,
        labels=list(range(num_classes)),
    )

    print(
        f"[compute_event_confusion_matrix] Complete: {time.time() - t0_cm_compute:.3f}s - Matched {len(true_classes)} events"
    )
    return cm


def build_validation_event_predictions_dataframe(
    all_predictions: dict,
    all_targets: list,
    iou_threshold: float = 0.3,
    matching_strategy: str = "iou",
    overlap_recall_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Build per-event validation prediction records for DETR evaluation.

    The returned dataframe has one row per:
      - matched target event (correct class or class mismatch),
      - missed target event (false negative),
      - unmatched predicted event (false positive).

    This uses the same greedy IoU matching strategy used for confusion/F1 logic,
    and reuses already-collected validation predictions/targets.
    """
    normalized_predictions = normalize_prediction_intervals(all_predictions)
    class_logits = np.asarray(normalized_predictions["class_logits"])  # [B, Nq, C]
    pred_starts = np.asarray(normalized_predictions["start"])  # [B, Nq, 1]
    pred_ends = np.asarray(normalized_predictions["end"])  # [B, Nq, 1]
    pred_centers = np.asarray(normalized_predictions["center"])  # [B, Nq, 1]

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
    if pred_centers.shape[:2] != class_logits.shape[:2]:
        raise ValueError(
            f"Center shape mismatch: centers {pred_centers.shape} vs class_logits {class_logits.shape}."
        )

    batch_size, n_queries, _ = class_logits.shape
    if len(all_targets) != batch_size:
        raise ValueError(
            f"Target length mismatch: targets={len(all_targets)} vs predictions batch={batch_size}."
        )

    def temporal_iou(
        pred_start: float,
        pred_end: float,
        target_start: float,
        target_end: float,
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
        return float(inter / union)

    rows: list[dict] = []

    for sample_idx in range(batch_size):
        probs = softmax(class_logits[sample_idx], axis=-1)  # [Nq, C]

        sample_preds: list[dict] = []
        for query_idx in range(n_queries):
            pred_class = int(np.argmax(probs[query_idx]))
            if pred_class == 0:
                continue

            pred_bg_prob = float(probs[query_idx, 0])
            pred_non_bg_conf = float(1.0 - pred_bg_prob)
            pred_start = float(np.clip(pred_starts[sample_idx, query_idx, 0], 0.0, 1.0))
            pred_end = float(np.clip(pred_ends[sample_idx, query_idx, 0], 0.0, 1.0))
            pred_center = float(
                np.clip(pred_centers[sample_idx, query_idx, 0], 0.0, 1.0)
            )
            if pred_start > pred_end:
                pred_start, pred_end = pred_end, pred_start

            sample_preds.append(
                {
                    "query_idx": int(query_idx),
                    "class_id": pred_class,
                    "start": pred_start,
                    "end": pred_end,
                    "center": pred_center,
                    "duration": float(max(0.0, pred_end - pred_start)),
                    "background_probability": pred_bg_prob,
                    "pred_confidence": pred_non_bg_conf,
                }
            )

        sample_preds.sort(key=lambda item: item["pred_confidence"], reverse=True)
        matched_pred = np.zeros(len(sample_preds), dtype=bool)

        for target_idx, target in enumerate(all_targets[sample_idx]):
            true_class = int(target.class_id)
            if true_class == 0:
                continue

            true_start = float(np.clip(target.start_norm, 0.0, 1.0))
            true_end = float(np.clip(target.end_norm, 0.0, 1.0))
            if true_start > true_end:
                true_start, true_end = true_end, true_start
            true_center = float(0.5 * (true_start + true_end))
            true_duration = float(max(0.0, true_end - true_start))

            best_pred_idx = -1
            best_iou = 0.0
            for pred_idx, pred in enumerate(sample_preds):
                if matched_pred[pred_idx]:
                    continue
                iou = temporal_iou(
                    pred["start"],
                    pred["end"],
                    true_start,
                    true_end,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            is_match = False
            if best_pred_idx >= 0:
                pred = sample_preds[best_pred_idx]
                is_match, best_iou, _ = is_interval_match(
                    pred["start"],
                    pred["end"],
                    true_start,
                    true_end,
                    matching_strategy=matching_strategy,
                    match_iou_threshold=float(iou_threshold),
                    overlap_recall_threshold=float(overlap_recall_threshold),
                )

            if best_pred_idx >= 0 and is_match:
                matched_pred[best_pred_idx] = True
                pred = sample_preds[best_pred_idx]
                predicted_class = int(pred["class_id"])
                rows.append(
                    {
                        "sample_idx": int(sample_idx),
                        "target_idx": int(target_idx),
                        "pred_rank_idx": int(best_pred_idx),
                        "pred_query_idx": int(pred["query_idx"]),
                        "match_type": (
                            "matched_correct_class"
                            if predicted_class == true_class
                            else "matched_wrong_class"
                        ),
                        "true_class": int(true_class),
                        "predicted_class": int(predicted_class),
                        "temporal_iou": float(best_iou),
                        "true_start": true_start,
                        "true_end": true_end,
                        "true_duration": true_duration,
                        "true_center": true_center,
                        "pred_start": float(pred["start"]),
                        "pred_end": float(pred["end"]),
                        "pred_duration": float(pred["duration"]),
                        "pred_center": float(pred["center"]),
                        "pred_confidence": float(pred["pred_confidence"]),
                        "pred_background_probability": float(
                            pred["background_probability"]
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "sample_idx": int(sample_idx),
                        "target_idx": int(target_idx),
                        "pred_rank_idx": -1,
                        "pred_query_idx": -1,
                        "match_type": "missed_detection",
                        "true_class": int(true_class),
                        "predicted_class": 0,
                        "temporal_iou": 0.0,
                        "true_start": true_start,
                        "true_end": true_end,
                        "true_duration": true_duration,
                        "true_center": true_center,
                        "pred_start": np.nan,
                        "pred_end": np.nan,
                        "pred_duration": np.nan,
                        "pred_center": np.nan,
                        "pred_confidence": np.nan,
                        "pred_background_probability": np.nan,
                    }
                )

        for pred_idx, pred in enumerate(sample_preds):
            if matched_pred[pred_idx]:
                continue

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "target_idx": -1,
                    "pred_rank_idx": int(pred_idx),
                    "pred_query_idx": int(pred["query_idx"]),
                    "match_type": "unmatched_prediction",
                    "true_class": 0,
                    "predicted_class": int(pred["class_id"]),
                    "temporal_iou": 0.0,
                    "true_start": np.nan,
                    "true_end": np.nan,
                    "true_duration": np.nan,
                    "true_center": np.nan,
                    "pred_start": float(pred["start"]),
                    "pred_end": float(pred["end"]),
                    "pred_duration": float(pred["duration"]),
                    "pred_center": float(pred["center"]),
                    "pred_confidence": float(pred["pred_confidence"]),
                    "pred_background_probability": float(pred["background_probability"]),
                }
            )

    columns = [
        "sample_idx",
        "target_idx",
        "pred_rank_idx",
        "pred_query_idx",
        "match_type",
        "true_class",
        "predicted_class",
        "temporal_iou",
        "true_start",
        "true_end",
        "true_duration",
        "true_center",
        "pred_start",
        "pred_end",
        "pred_duration",
        "pred_center",
        "pred_confidence",
        "pred_background_probability",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


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
            - val_plot_misdetected_events_per_class
            - detection-specific: loss weights for class, bbox, giou, unmatched

    Returns:
        fold_summary: dict with results (best epoch, metrics, elapsed time)
    """

    checkpoints_dir = fold_out_dir / "checkpoints"
    reports_dir = fold_out_dir / "reports"
    val_predictions_dir = reports_dir / "validation_predictions"
    val_plot_dir = fold_out_dir / "validation_event_plots"
    cm_dir = fold_out_dir / "confusion_matrices"

    for p in (checkpoints_dir, reports_dir, val_predictions_dir, val_plot_dir, cm_dir):
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
    loss_weights = _resolve_detr_loss_weights(spec, config)
    eval_matching = _resolve_detr_eval_matching(spec, config)
    match_iou_threshold = float(eval_matching["match_iou_threshold"])
    matching_strategy = str(eval_matching["matching_strategy"])
    overlap_recall_threshold = float(eval_matching["overlap_recall_threshold"])

    print(f"Resolved DETR loss weights for {model_key}: {loss_weights}")
    print(
        "Resolved DETR eval matching for "
        f"{model_key}: strategy={matching_strategy} "
        f"iou_threshold={match_iou_threshold:.3f} "
        f"overlap_recall_threshold={overlap_recall_threshold:.3f}"
    )
    loss_fn = DETREventLoss(num_classes=6, loss_weights=loss_weights)
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
            {"params": [], "lr": base_lr, "name": "encoder"},
            {"params": [], "lr": base_lr, "name": "attention"},
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
    best_val_mean_f1 = float("-inf")
    best_epoch = -1
    best_val_loss_epoch = -1
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
        train_loss_giou = 0.0
        train_loss_unmatched_query = 0.0
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
            train_loss_giou += loss_dict["loss_giou"].item()
            train_loss_unmatched_query += loss_dict["loss_unmatched_query"].item()
            num_train_batches += 1

            # Print progress every few batches
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0 or batch_idx == 0:
                current_loss = train_loss / (batch_idx + 1)
                print(
                    f"  Epoch {epoch + 1:3d} batch {batch_idx + 1:4d}/{len(train_loader):4d} | "
                    f"loss={current_loss:.4f} "
                    f"[class={train_loss_class / (batch_idx + 1):.4f} "
                    f"bbox={train_loss_bbox / (batch_idx + 1):.4f} "
                    f"giou={train_loss_giou / (batch_idx + 1):.4f} "
                    f"unmatched={train_loss_unmatched_query / (batch_idx + 1):.4f}]"
                )

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_loss_class = 0.0
        val_loss_bbox = 0.0
        val_loss_giou = 0.0
        val_loss_unmatched_query = 0.0
        all_predictions = {
            "class_logits": [],
            "center": [],
            "start": [],
            "end": [],
        }
        all_targets = []
        num_val_batches = 0

        with torch.inference_mode():
            for batch_idx, batch in enumerate(val_loader):
                xb = batch[0].to(device)
                y_onehot = batch[1]

                # Forward pass
                predictions = normalize_prediction_intervals(model(xb))

                # Convert targets
                targets = batch_segmentation_to_events(y_onehot, normalize=True)
                all_targets.extend(targets)

                # Compute loss
                loss_dict = loss_fn(predictions, targets)
                val_loss += loss_dict["loss_total"].item()
                val_loss_class += loss_dict["loss_class"].item()
                val_loss_bbox += loss_dict["loss_bbox"].item()
                val_loss_giou += loss_dict["loss_giou"].item()
                val_loss_unmatched_query += loss_dict["loss_unmatched_query"].item()
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
        avg_train_loss_giou = (
            train_loss_giou / num_train_batches if num_train_batches > 0 else 0.0
        )
        avg_train_loss_unmatched_query = (
            train_loss_unmatched_query / num_train_batches
            if num_train_batches > 0
            else 0.0
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
        avg_val_loss_giou = (
            val_loss_giou / num_val_batches if num_val_batches > 0 else 0.0
        )
        avg_val_loss_unmatched_query = (
            val_loss_unmatched_query / num_val_batches if num_val_batches > 0 else 0.0
        )

        # Concatenate predictions for metrics computation
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

        # Compute event detection metrics (mAP at various IoU thresholds)
        detection_metrics = metrics_fn.evaluate_batch(all_predictions, all_targets)

        # Unified event summary from one matching pass: confusion matrix + per-class metrics.
        detection_summary = metrics_fn.compute_detection_summary(
            all_predictions,
            all_targets,
            iou_threshold=match_iou_threshold,
            matching_strategy=matching_strategy,
            overlap_recall_threshold=overlap_recall_threshold,
        )

        # Build per-event validation prediction rows once per epoch from already
        # collected predictions/targets (no additional model forward pass).
        val_predictions_df = build_validation_event_predictions_dataframe(
            all_predictions,
            all_targets,
            iou_threshold=match_iou_threshold,
            matching_strategy=matching_strategy,
            overlap_recall_threshold=overlap_recall_threshold,
        )
        val_predictions_latest_path = val_predictions_dir / "val_predictions_latest.csv"
        val_predictions_df.to_csv(
            val_predictions_latest_path,
            index=False,
            sep=";",
            decimal=",",
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

        active_event_class_ids = [
            class_id
            for class_id in range(1, 6)
            if int(per_class_stats[class_id]["target_count"]) > 0
        ]
        if len(active_event_class_ids) == 0:
            raise RuntimeError(
                "Validation set contains no active event classes (target_count=0 for all VT/LP/TR/AV/IC)."
            )

        # Canonical study metric: macro-F1 over active (present) non-background classes.
        mean_f1 = float(
            np.mean(
                [
                    float(per_class_f1_dict.get(class_id, 0.0))
                    for class_id in active_event_class_ids
                ]
            )
        )
        # Keep IoU aggregation aligned with active event classes.
        mean_iou = float(
            np.mean(
                [
                    float(per_class_iou_dict.get(class_id, 0.0))
                    for class_id in active_event_class_ids
                ]
            )
        )
        class_precision_scores = [
            float(per_class_stats[class_id]["precision"]) for class_id in range(1, 6)
        ]
        class_recall_scores = [
            float(per_class_stats[class_id]["recall"]) for class_id in range(1, 6)
        ]
        mean_precision = (
            float(np.mean(class_precision_scores)) if class_precision_scores else 0.0
        )
        mean_recall = (
            float(np.mean(class_recall_scores)) if class_recall_scores else 0.0
        )

        if float(avg_train_loss) < float(best_train_loss):
            best_train_loss = float(avg_train_loss)

        is_best_val_loss_epoch = float(avg_val_loss) < float(best_val_loss)
        is_best_val_mean_f1_epoch = mean_f1 > best_val_mean_f1
        val_cm = detection_summary["confusion_matrix"]

        saved_plot_count = 0
        max_misdetected_events_per_class = int(
            config.get(
                "val_plot_misdetected_events_per_class",
                config.get("val_plot_samples_per_class", 2),
            )
        )
        should_reset_early_stop = False

        if is_best_val_mean_f1_epoch:
            best_val_mean_f1 = mean_f1
            best_epoch = int(epoch)
            should_reset_early_stop = True

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _checkpoint_state_dict_for_model(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "f1_score": mean_f1,
                },
                checkpoints_dir / "best_f1.pt",
            )

            # Confusion matrix from the same matching pass used for F1/IoU.
            val_cm = detection_summary["confusion_matrix"]

            # Save rolling best (gets replaced each new best)
            save_confusion_matrix_image(
                cm=val_cm,
                labels=["Background", "VT", "LP", "TR", "AV", "IC"],
                out_path=cm_dir / "confusion_matrix_val_best_f1.png",
                title=f"Event Detection Confusion Matrix - Fold {fold_id} - Best F1",
            )

        if is_best_val_loss_epoch:
            best_val_loss = float(avg_val_loss)
            best_val_loss_epoch = int(epoch)
            should_reset_early_stop = True

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _checkpoint_state_dict_for_model(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "f1_score": mean_f1,
                },
                checkpoints_dir / "best_val_loss.pt",
            )

            save_confusion_matrix_image(
                cm=val_cm,
                labels=["Background", "VT", "LP", "TR", "AV", "IC"],
                out_path=cm_dir / "confusion_matrix_val_best_val_loss.png",
                title=(
                    f"Event Detection Confusion Matrix - Fold {fold_id} - Best Val Loss"
                ),
            )

        if is_best_val_mean_f1_epoch or is_best_val_loss_epoch:
            if is_best_val_mean_f1_epoch and is_best_val_loss_epoch:
                improvement_tag = "best_f1_best_val_loss"
            elif is_best_val_mean_f1_epoch:
                improvement_tag = "best_f1"
            else:
                improvement_tag = "best_val_loss"

            saved_plot_count += plot_event_validation(
                model=model,
                dataloader=val_loader,
                device=device,
                output_dir=val_plot_dir / f"epoch_{epoch + 1:03d}_{improvement_tag}",
                epoch=int(epoch),
                samples_per_class=config.get("val_plot_samples_per_class", 2),
                extract_attention=False,
                attention_mode="none",
                forward_batch_size=min(2, config.get("val_plot_forward_batch_size", 2)),
                misdetected_only=True,
                max_misdetected_events_per_class=max_misdetected_events_per_class,
                match_iou_threshold=match_iou_threshold,
                matching_strategy=matching_strategy,
                overlap_recall_threshold=overlap_recall_threshold,
            )

            val_predictions_df.to_csv(
                val_predictions_dir
                / f"val_predictions_epoch_{epoch + 1:03d}_{improvement_tag}.csv",
                index=False,
                sep=";",
                decimal=",",
            )

        if should_reset_early_stop:
            epochs_without_improvement = 0
            # Save per-epoch record (historical) whenever either best metric improves.
            save_confusion_matrix_image(
                cm=val_cm,
                labels=["Background", "VT", "LP", "TR", "AV", "IC"],
                out_path=cm_dir / f"confusion_matrix_epoch_{epoch:03d}.png",
                title=f"Event Detection Confusion Matrix - Fold {fold_id} - Epoch {epoch + 1}",
            )
        else:
            epochs_without_improvement += 1

        # Save metrics rows for CSV export
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Main CSV: ordered list format identical to segmentation models for comparison
        # [lr, epoch, train_loss, val_loss, VT_f1, LP_f1, TR_f1, AV_f1, IC_f1, mean_f1, VT_iou, LP_iou, TR_iou, AV_iou, IC_iou, mean_iou]
        # For event detection: use per-class F1 and per-class IoU from the
        # configured matching criterion.
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

        # Detection metrics CSV (additional metrics specific to event detection):
        # [epoch, mAP@0.1, mAP@0.3, mAP@0.5, mAP@0.7, mAP@0.9, mAP,
        #  F1_match, VT_f1_match, LP_f1_match, TR_f1_match, AV_f1_match, IC_f1_match]
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
            f"[class={avg_train_loss_class:.4f} bbox={avg_train_loss_bbox:.4f} giou={avg_train_loss_giou:.4f} unmatched={avg_train_loss_unmatched_query:.4f}]\n"
            f"Val Loss:    {avg_val_loss:.4f}  "
            f"[class={avg_val_loss_class:.4f} bbox={avg_val_loss_bbox:.4f} giou={avg_val_loss_giou:.4f} unmatched={avg_val_loss_unmatched_query:.4f}]\n"
            f"Metrics:     mean_f1={mean_f1:.4f} mean_precision={mean_precision:.4f} "
            f"mean_recall={mean_recall:.4f} mean_iou={mean_iou:.4f} "
            f"best_epoch_f1={best_epoch + 1} best_epoch_val_loss={best_val_loss_epoch + 1} "
            f"no_improve={epochs_without_improvement}/{config['early_stop_patience']}\n"
            f"mAP:         {detection_metrics.get('mAP', 0.0):.4f} "
            f"(@ 0.1: {detection_metrics.get('mAP@0.1', 0.0):.4f}, "
            f"0.5: {detection_metrics.get('mAP@0.5', 0.0):.4f}, "
            f"0.9: {detection_metrics.get('mAP@0.9', 0.0):.4f})\n"
            f"Plots:       {saved_plot_count} validation event plots saved\n"
            f"Val rows:    {len(val_predictions_df)} rows saved to {val_predictions_latest_path.name}\n"
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

        # Full attention plotting once at end-of-training on best checkpoint.
        final_attention_plot_count = plot_event_validation(
            model=model,
            dataloader=val_loader,
            device=device,
            output_dir=val_plot_dir / "best_checkpoint_full_attention",
            epoch=int(checkpoint["epoch"]),
            samples_per_class=config.get("val_plot_samples_per_class", 2),
            extract_attention=True,
            attention_mode=config.get("final_attention_mode", "full"),
            forward_batch_size=min(2, config.get("val_plot_forward_batch_size", 2)),
            misdetected_only=True,
            max_misdetected_events_per_class=int(
                config.get(
                    "val_plot_misdetected_events_per_class",
                    config.get("val_plot_samples_per_class", 2),
                )
            ),
            match_iou_threshold=match_iou_threshold,
            matching_strategy=matching_strategy,
            overlap_recall_threshold=overlap_recall_threshold,
        )
        print(
            "Final best-checkpoint attention plots saved: "
            f"{final_attention_plot_count}"
        )

    # Test set evaluation
    model.eval()
    test_loss = 0.0
    all_test_predictions = {
        "class_logits": [],
        "center": [],
        "start": [],
        "end": [],
    }
    all_test_targets = []
    num_test_batches = 0

    with torch.inference_mode():
        for batch in test_loader:
            xb = batch[0].to(device)
            y_onehot = batch[1]

            predictions = normalize_prediction_intervals(model(xb))
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

    test_metrics = metrics_fn.evaluate_batch(all_test_predictions, all_test_targets)
    test_detection_summary = metrics_fn.compute_detection_summary(
        all_test_predictions,
        all_test_targets,
        iou_threshold=match_iou_threshold,
        matching_strategy=matching_strategy,
        overlap_recall_threshold=overlap_recall_threshold,
    )
    test_f1_per_class = [
        float(test_detection_summary["per_class_f1"].get(class_id, 0.0))
        for class_id in range(1, 6)
    ]
    test_iou_per_class = [
        float(test_detection_summary["per_class_iou"].get(class_id, 0.0))
        for class_id in range(1, 6)
    ]
    test_per_class_stats = test_detection_summary["per_class"]
    active_test_class_ids = [
        class_id
        for class_id in range(1, 6)
        if int(test_per_class_stats[class_id]["target_count"]) > 0
    ]
    if len(active_test_class_ids) == 0:
        raise RuntimeError(
            "Test set contains no active event classes (target_count=0 for all VT/LP/TR/AV/IC)."
        )
    test_macro_f1_active = float(
        np.mean(
            [
                float(test_detection_summary["per_class_f1"].get(class_id, 0.0))
                for class_id in active_test_class_ids
            ]
        )
    )
    test_mean_iou = float(
        np.mean(
            [
                float(test_detection_summary["per_class_iou"].get(class_id, 0.0))
                for class_id in active_test_class_ids
            ]
        )
    )
    test_f1 = metrics_fn.compute_f1(
        all_test_predictions,
        all_test_targets,
        iou_threshold=match_iou_threshold,
        matching_strategy=matching_strategy,
        overlap_recall_threshold=overlap_recall_threshold,
    )

    fold_elapsed_sec = float(time.time() - fold_start)

    # Prepare fold summary
    fold_summary = {
        "trainer_kind": "detr",
        "fold": int(fold_id),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "best_epoch": int(best_epoch + 1),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_f1": float(best_val_mean_f1),
        "test_loss": float(avg_test_loss),
        "test_mean_f1": float(test_macro_f1_active),
        "test_mean_iou": float(test_mean_iou),
        "test_f1_per_class": [float(x) for x in test_f1_per_class],
        "test_iou_per_class": [float(x) for x in test_iou_per_class],
        "test_mAP": float(test_metrics.get("mAP", 0.0)),
        "matching_strategy": matching_strategy,
        "match_iou_threshold": float(match_iou_threshold),
        "overlap_recall_threshold": float(overlap_recall_threshold),
        "fold_elapsed_seconds": fold_elapsed_sec,
    }
    fold_summary.update(test_metrics)
    fold_summary[f"test_F1@{match_iou_threshold:.2f}"] = float(test_f1)

    # Save fold summary
    summary_path = reports_dir / "fold_summary.json"
    with summary_path.open("w") as f:
        json.dump(fold_summary, f, indent=2)

    print(f"Fold summary saved to {summary_path}")
    print(
        f"Fold {fold_id} complete | Best Epoch: {best_epoch + 1} | "
        f"Test Macro-F1(active): {test_macro_f1_active:.4f} | "
        f"Test F1(match criterion): {test_f1:.4f} | "
        f"Test mAP: {test_metrics.get('mAP', 0.0):.4f} | "
        f"Elapsed: {fold_elapsed_sec / 60:.1f} min"
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
        "best_val_mean_f1": float(best_val_mean_f1),
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
