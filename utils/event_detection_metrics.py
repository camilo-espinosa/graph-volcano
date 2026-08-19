"""
Event detection metrics: mAP@tIoU, F1@tIoU, per-class metrics.

Evaluates predictions against ground truth using temporal IoU thresholds.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.special import softmax

from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.event_targets import EventInterval


def temporal_iou(
    pred_start: float,
    pred_end: float,
    target_start: float,
    target_end: float,
) -> float:
    """Compute temporal IoU between predicted and target intervals."""
    pred_start = np.clip(pred_start, 0, 1)
    pred_end = np.clip(pred_end, 0, 1)
    target_start = np.clip(target_start, 0, 1)
    target_end = np.clip(target_end, 0, 1)

    if pred_start > pred_end:
        pred_start, pred_end = pred_end, pred_start
    if target_start > target_end:
        target_start, target_end = target_end, target_start

    inter_start = max(pred_start, target_start)
    inter_end = min(pred_end, target_end)
    inter_len = max(0.0, inter_end - inter_start)
    union_len = (pred_end - pred_start) + (target_end - target_start) - inter_len

    if union_len < 1e-7:
        return 0.0

    return float(inter_len / union_len)


def temporal_overlap_recall(
    pred_start: float,
    pred_end: float,
    target_start: float,
    target_end: float,
) -> float:
    """Compute target coverage = intersection / target_duration."""
    pred_start = np.clip(pred_start, 0, 1)
    pred_end = np.clip(pred_end, 0, 1)
    target_start = np.clip(target_start, 0, 1)
    target_end = np.clip(target_end, 0, 1)

    if pred_start > pred_end:
        pred_start, pred_end = pred_end, pred_start
    if target_start > target_end:
        target_start, target_end = target_end, target_start

    inter_start = max(pred_start, target_start)
    inter_end = min(pred_end, target_end)
    inter_len = max(0.0, inter_end - inter_start)
    target_len = max(0.0, target_end - target_start)

    if target_len < 1e-7:
        return 0.0

    return float(inter_len / target_len)


def is_interval_match(
    pred_start: float,
    pred_end: float,
    target_start: float,
    target_end: float,
    *,
    matching_strategy: str,
    match_iou_threshold: float,
    overlap_recall_threshold: float,
) -> tuple[bool, float, float]:
    """Return (is_match, iou, overlap_recall) using the configured strategy."""
    iou = temporal_iou(pred_start, pred_end, target_start, target_end)
    overlap = temporal_overlap_recall(pred_start, pred_end, target_start, target_end)

    if matching_strategy == "iou":
        return bool(iou >= float(match_iou_threshold)), float(iou), float(overlap)
    if matching_strategy == "overlap_recall":
        return bool(overlap >= float(overlap_recall_threshold)), float(iou), float(overlap)
    if matching_strategy == "dual":
        return bool(
            iou >= float(match_iou_threshold)
            or overlap >= float(overlap_recall_threshold)
        ), float(iou), float(overlap)
    raise ValueError(
        "matching_strategy must be one of {'iou', 'overlap_recall', 'dual'}, "
        f"got {matching_strategy!r}."
    )


class EventDetectionMetrics:
    """
    Compute event detection metrics at various temporal IoU thresholds.
    """

    def __init__(self, iou_thresholds: list[float] | None = None, num_classes: int = 6):
        """
        Args:
            iou_thresholds: Temporal IoU thresholds for mAP computation
            num_classes: Number of event classes
        """
        if iou_thresholds is None:
            # Standard IoU thresholds: 0.1, 0.2, ..., 0.9
            iou_thresholds = [0.1 * i for i in range(1, 10)]

        self.iou_thresholds = sorted(iou_thresholds)
        self.num_classes = int(num_classes)

    def evaluate_batch(
        self,
        predictions: Dict[str, np.ndarray],
        targets: list[list[EventInterval]],
    ) -> Dict[str, float]:
        """
        Evaluate batch of predictions.

        Args:
            predictions: Dict with keys:
                - class_logits: [B, Nq, num_classes] logits
                - center: [B, Nq, 1]
                - start: [B, Nq, 1]
                - end: [B, Nq, 1]
            targets: List of lists of EventInterval per sample

        Returns:
            Dict with metrics:
                - mAP_*: Mean average precision at each IoU threshold
                - mAP: Average over all thresholds
                - F1_*: F1 scores at each threshold
                - per_class_AP_*: Per-class AP
        """
        predictions = normalize_prediction_intervals(predictions)
        batch_size = predictions["class_logits"].shape[0]

        # Convert to numpy
        class_logits = (
            predictions["class_logits"]
            if isinstance(predictions["class_logits"], np.ndarray)
            else predictions["class_logits"].cpu().numpy()
        )
        center = (
            predictions["center"]
            if isinstance(predictions["center"], np.ndarray)
            else predictions["center"].cpu().numpy()
        )
        start = (
            predictions["start"]
            if isinstance(predictions["start"], np.ndarray)
            else predictions["start"].cpu().numpy()
        )
        end = (
            predictions["end"]
            if isinstance(predictions["end"], np.ndarray)
            else predictions["end"].cpu().numpy()
        )

        metrics = {}

        # Compute metrics per IoU threshold
        for iou_threshold in self.iou_thresholds:
            ap, per_class_ap = self._compute_ap_at_threshold(
                class_logits,
                center,
                start,
                end,
                targets,
                iou_threshold,
            )
            metrics[f"mAP@{iou_threshold:.1f}"] = float(ap)

            for class_id in range(1, self.num_classes):  # Skip background
                metrics[f"AP_class_{class_id}@{iou_threshold:.1f}"] = float(
                    per_class_ap.get(class_id, 0.0)
                )

        # Compute mean AP
        map_values = [metrics[f"mAP@{t:.1f}"] for t in self.iou_thresholds]
        metrics["mAP"] = float(np.mean(map_values))

        return metrics

    def _compute_ap_at_threshold(
        self,
        class_logits: np.ndarray,  # [B, Nq, num_classes]
        center: np.ndarray,  # [B, Nq, 1]
        start: np.ndarray,  # [B, Nq, 1]
        end: np.ndarray,  # [B, Nq, 1]
        targets: list[list[EventInterval]],
        iou_threshold: float,
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute AP at a specific IoU threshold.

        Returns:
            (mAP, per_class_AP)
        """
        batch_size = class_logits.shape[0]

        # Collect all predictions and targets across batch
        all_predictions = (
            []
        )  # (class_id, confidence, start, end, sample_idx, query_idx)
        all_targets_by_class = (
            {}
        )  # class_id -> list of (start, end, sample_idx, target_idx)

        for b in range(batch_size):
            target_list = targets[b]

            # Add targets
            for t_idx, target in enumerate(target_list):
                class_id = target.class_id
                if class_id not in all_targets_by_class:
                    all_targets_by_class[class_id] = []
                all_targets_by_class[class_id].append(
                    (target.start_norm, target.end_norm, b, t_idx)
                )

            # Extract predictions
            class_probs = softmax(class_logits[b], axis=-1)  # [Nq, num_classes]

            for q in range(class_logits.shape[1]):
                # DETR-style decode: argmax over all classes, discard background.
                pred_class_id = int(np.argmax(class_probs[q]))
                if pred_class_id == 0:
                    continue

                pred_conf = float(1.0 - class_probs[q, 0])
                all_predictions.append(
                    (
                        pred_class_id,
                        pred_conf,
                        np.clip(start[b, q, 0], 0, 1),
                        np.clip(end[b, q, 0], 0, 1),
                        b,
                        q,
                    )
                )

        # Compute AP per class
        per_class_ap = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for class_id in range(1, self.num_classes):
            # Get targets for this class
            class_targets = all_targets_by_class.get(class_id, [])

            if len(class_targets) == 0:
                # No targets for this class
                per_class_ap[class_id] = (
                    1.0 if not any(p[0] == class_id for p in all_predictions) else 0.0
                )
                total_fn += len(class_targets)
                continue

            # Get predictions for this class
            class_predictions = [p for p in all_predictions if p[0] == class_id]

            if len(class_predictions) == 0:
                # No predictions for this class but there are targets
                per_class_ap[class_id] = 0.0
                total_fn += len(class_targets)
                continue

            # Sort predictions by confidence (descending)
            class_predictions.sort(key=lambda x: x[1], reverse=True)

            # Match predictions to targets
            matched = np.zeros(len(class_targets), dtype=bool)
            tp = 0
            fp = 0

            for pred in class_predictions:
                pred_class_id, pred_conf, pred_start, pred_end, pred_b, pred_q = pred

                best_iou = 0.0
                best_target_idx = -1

                for t_idx, target in enumerate(class_targets):
                    if matched[t_idx]:
                        continue

                    t_start, t_end, t_b, _ = target

                    # Predictions can only match targets from the same sample.
                    if pred_b != t_b:
                        continue

                    # Compute temporal IoU
                    iou = self._temporal_iou(pred_start, pred_end, t_start, t_end)

                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = t_idx

                if best_iou >= iou_threshold and best_target_idx >= 0:
                    tp += 1
                    matched[best_target_idx] = True
                else:
                    fp += 1

            fn = len(class_targets) - tp

            # Compute AP for this class (simple: TP / (TP + FP + FN))
            ap_class = tp / max(1, tp + fp + fn)
            per_class_ap[class_id] = ap_class

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Mean AP across all classes
        if len(per_class_ap) > 0:
            mAP = float(np.mean(list(per_class_ap.values())))
        else:
            mAP = 0.0

        return mAP, per_class_ap

    @staticmethod
    def _temporal_iou(
        pred_start: float, pred_end: float, target_start: float, target_end: float
    ) -> float:
        """Compute temporal IoU."""
        return temporal_iou(pred_start, pred_end, target_start, target_end)

    def compute_detection_summary(
        self,
        predictions: Dict[str, np.ndarray],
        targets: list[list[EventInterval]],
        iou_threshold: float = 0.3,
        matching_strategy: str = "iou",
        overlap_recall_threshold: float = 0.8,
    ) -> Dict[str, object]:
        """
        Compute confusion matrix and per-class metrics from one shared matching pass.

        Confusion matrix convention:
        - Rows: true class (0 = Background)
        - Cols: predicted class (0 = Background)
        - Unmatched GT event => (true=class_id, pred=0)
        - Unmatched predicted event => (true=0, pred=class_id)
        """
        predictions = normalize_prediction_intervals(predictions)
        class_logits = np.asarray(predictions["class_logits"])
        pred_starts = np.asarray(predictions["start"])
        pred_ends = np.asarray(predictions["end"])

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
        if len(targets) != batch_size:
            raise ValueError(
                f"Target length mismatch: targets={len(targets)} vs predictions batch={batch_size}."
            )

        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        per_class_target_count = {c: 0 for c in range(1, self.num_classes)}
        per_class_iou_sum = {c: 0.0 for c in range(1, self.num_classes)}

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

            for target in targets[b]:
                if target.class_id == 0:
                    continue

                true_class = int(target.class_id)
                per_class_target_count[true_class] += 1

                best_pred_idx = -1
                best_iou = 0.0

                for p_idx, pred in enumerate(sample_preds):
                    if matched_pred[p_idx]:
                        continue

                    iou = self._temporal_iou(
                        pred["start"],
                        pred["end"],
                        target.start_norm,
                        target.end_norm,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = p_idx

                is_match = False
                if best_pred_idx >= 0:
                    pred = sample_preds[best_pred_idx]
                    is_match, best_iou, _ = is_interval_match(
                        pred["start"],
                        pred["end"],
                        target.start_norm,
                        target.end_norm,
                        matching_strategy=matching_strategy,
                        match_iou_threshold=float(iou_threshold),
                        overlap_recall_threshold=float(overlap_recall_threshold),
                    )

                if best_pred_idx >= 0 and is_match:
                    matched_pred[best_pred_idx] = True
                    pred_class = int(sample_preds[best_pred_idx]["class_id"])
                    cm[true_class, pred_class] += 1
                    if pred_class == true_class:
                        per_class_iou_sum[true_class] += float(best_iou)
                else:
                    # Missed detection for this true event.
                    cm[true_class, 0] += 1

            # Unmatched predictions become false positives against true background.
            for p_idx, pred in enumerate(sample_preds):
                if matched_pred[p_idx]:
                    continue
                cm[0, int(pred["class_id"])] += 1

        per_class_stats: Dict[int, Dict[str, float]] = {}
        per_class_f1: Dict[int, float] = {}
        per_class_iou: Dict[int, float] = {}

        for class_id in range(1, self.num_classes):
            tp = int(cm[class_id, class_id])
            fp = int(np.sum(cm[:, class_id]) - tp)
            fn = int(np.sum(cm[class_id, :]) - tp)

            precision = float(tp / max(1, tp + fp))
            recall = float(tp / max(1, tp + fn))
            f1 = float(2 * precision * recall / max(1e-7, precision + recall))

            iou_avg = (
                float(per_class_iou_sum[class_id] / per_class_target_count[class_id])
                if per_class_target_count[class_id] > 0
                else 0.0
            )

            per_class_stats[class_id] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou_avg,
                "target_count": int(per_class_target_count[class_id]),
            }
            per_class_f1[class_id] = f1
            per_class_iou[class_id] = iou_avg

        return {
            "confusion_matrix": cm,
            "per_class": per_class_stats,
            "per_class_f1": per_class_f1,
            "per_class_iou": per_class_iou,
        }

    def compute_f1(
        self,
        predictions: Dict[str, np.ndarray],
        targets: list[list[EventInterval]],
        iou_threshold: float = 0.3,
        matching_strategy: str = "iou",
        overlap_recall_threshold: float = 0.8,
    ) -> float:
        """
        Compute F1 score at a specific IoU threshold.
        """
        predictions = normalize_prediction_intervals(predictions)
        batch_size = predictions["class_logits"].shape[0]

        # Collect all predictions and targets with sample indices.
        all_predictions: list[tuple[int, EventInterval, float]] = []
        all_targets: list[tuple[int, EventInterval]] = []

        for b in range(batch_size):
            target_list = targets[b]

            # Add targets
            all_targets.extend((b, target) for target in target_list)

            # Extract predictions
            class_logits = predictions["class_logits"]
            if isinstance(class_logits, np.ndarray):
                class_probs = softmax(class_logits[b], axis=-1)
            else:
                class_probs = softmax(class_logits[b].cpu().numpy(), axis=-1)

            start = predictions["start"]
            end = predictions["end"]
            if not isinstance(start, np.ndarray):
                start = start.cpu().numpy()
            if not isinstance(end, np.ndarray):
                end = end.cpu().numpy()

            for q in range(class_logits.shape[1]):
                pred_class_id = int(np.argmax(class_probs[q]))
                if pred_class_id == 0:
                    continue

                pred_conf = float(1.0 - class_probs[q, 0])
                event = EventInterval(
                    class_id=pred_class_id,
                    start_norm=np.clip(start[b, q, 0], 0, 1),
                    end_norm=np.clip(end[b, q, 0], 0, 1),
                    center_norm=((start[b, q, 0] + end[b, q, 0]) / 2.0),
                    start_frame=int(start[b, q, 0]),
                    end_frame=int(end[b, q, 0]),
                )
                all_predictions.append((b, event, pred_conf))

        if len(all_predictions) == 0 and len(all_targets) == 0:
            return 1.0  # Perfect score if both empty

        if len(all_predictions) == 0 or len(all_targets) == 0:
            return 0.0  # No predictions or no targets

        # Match predictions to targets (same class and same sample only)
        matched = np.zeros(len(all_targets), dtype=bool)
        tp = 0

        all_predictions = sorted(all_predictions, key=lambda x: x[2], reverse=True)

        for pred_sample_idx, pred, _ in all_predictions:
            best_iou = 0.0
            best_target_idx = -1

            for t_idx, (target_sample_idx, target) in enumerate(all_targets):
                if matched[t_idx] or pred.class_id != target.class_id:
                    continue
                if pred_sample_idx != target_sample_idx:
                    continue

                iou = self._temporal_iou(
                    pred.start_norm,
                    pred.end_norm,
                    target.start_norm,
                    target.end_norm,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = t_idx

            is_match = False
            if best_target_idx >= 0:
                _, target = all_targets[best_target_idx]
                is_match, best_iou, _ = is_interval_match(
                    pred.start_norm,
                    pred.end_norm,
                    target.start_norm,
                    target.end_norm,
                    matching_strategy=matching_strategy,
                    match_iou_threshold=float(iou_threshold),
                    overlap_recall_threshold=float(overlap_recall_threshold),
                )

            if is_match and best_target_idx >= 0:
                tp += 1
                matched[best_target_idx] = True

        fp = len(all_predictions) - tp
        fn = len(all_targets) - tp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-7, precision + recall)

        return float(f1)

    def compute_per_class_f1(
        self,
        predictions: Dict[str, np.ndarray],
        targets: list[list[EventInterval]],
        iou_threshold: float = 0.3,
        matching_strategy: str = "iou",
        overlap_recall_threshold: float = 0.8,
    ) -> Dict[int, float]:
        """
        Compute per-class F1 scores at a specific IoU threshold.

        Returns:
            Dict[class_id] -> F1 score for each class (1-5, skipping background=0)
        """
        summary = self.compute_detection_summary(
            predictions=predictions,
            targets=targets,
            iou_threshold=iou_threshold,
            matching_strategy=matching_strategy,
            overlap_recall_threshold=overlap_recall_threshold,
        )

        return dict(summary["per_class_f1"])

    def compute_per_class_iou(
        self,
        predictions: Dict[str, np.ndarray],
        targets: list[list[EventInterval]],
        iou_threshold: float = 0.3,
        matching_strategy: str = "iou",
        overlap_recall_threshold: float = 0.8,
    ) -> Dict[int, float]:
        """
        Compute per-class IoU scores (average IoU of matched predictions per class).

        Returns:
            Dict[class_id] -> average IoU for matched predictions in each class
        """
        summary = self.compute_detection_summary(
            predictions=predictions,
            targets=targets,
            iou_threshold=iou_threshold,
            matching_strategy=matching_strategy,
            overlap_recall_threshold=overlap_recall_threshold,
        )
        return dict(summary["per_class_iou"])
