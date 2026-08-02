"""
Event detection metrics: mAP@tIoU, F1@tIoU, per-class metrics.

Evaluates predictions against ground truth using temporal IoU thresholds.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.special import softmax

from utils.event_targets import EventInterval


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
        confidence_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Evaluate batch of predictions.

        Args:
            predictions: Dict with keys:
                - class_logits: [B, Nq, num_classes] logits
                - center: [B, Nq, 1]
                - start: [B, Nq, 1]
                - end: [B, Nq, 1]
                - confidence: [B, Nq, 1]
            targets: List of lists of EventInterval per sample
            confidence_threshold: Confidence threshold for keeping predictions

        Returns:
            Dict with metrics:
                - mAP_*: Mean average precision at each IoU threshold
                - mAP: Average over all thresholds
                - F1_*: F1 scores at each threshold
                - per_class_AP_*: Per-class AP
        """
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
        confidence = (
            predictions["confidence"]
            if isinstance(predictions["confidence"], np.ndarray)
            else predictions["confidence"].cpu().numpy()
        )

        metrics = {}

        # Compute metrics per IoU threshold
        for iou_threshold in self.iou_thresholds:
            ap, per_class_ap = self._compute_ap_at_threshold(
                class_logits,
                center,
                start,
                end,
                confidence,
                targets,
                iou_threshold,
                confidence_threshold,
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
        confidence: np.ndarray,  # [B, Nq, 1]
        targets: list[list[EventInterval]],
        iou_threshold: float,
        confidence_threshold: float = 0.5,
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
            conf = 1.0 / (1.0 + np.exp(-confidence[b, :, 0]))  # Sigmoid

            for q in range(class_logits.shape[1]):
                # Predict class (argmax, excluding background=0)
                pred_class_id = np.argmax(class_probs[q, 1:]) + 1
                pred_conf = class_probs[q, pred_class_id] * conf[q]

                if pred_conf >= confidence_threshold:
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
        # Clamp to [0, 1]
        pred_start = np.clip(pred_start, 0, 1)
        pred_end = np.clip(pred_end, 0, 1)
        target_start = np.clip(target_start, 0, 1)
        target_end = np.clip(target_end, 0, 1)

        # Ensure start < end
        if pred_start > pred_end:
            pred_start, pred_end = pred_end, pred_start
        if target_start > target_end:
            target_start, target_end = target_end, target_start

        # Intersection
        inter_start = max(pred_start, target_start)
        inter_end = min(pred_end, target_end)
        inter_len = max(0, inter_end - inter_start)

        # Union
        union_len = (pred_end - pred_start) + (target_end - target_start) - inter_len

        # IoU
        if union_len < 1e-7:
            return 0.0

        return float(inter_len / union_len)

    def compute_f1(
        self,
        predictions: Dict[str, np.ndarray],
        targets: list[list[EventInterval]],
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> float:
        """
        Compute F1 score at a specific IoU threshold.
        """
        batch_size = predictions["class_logits"].shape[0]

        # Collect all predictions and targets
        all_predictions = []
        all_targets = []

        for b in range(batch_size):
            target_list = targets[b]

            # Add targets
            all_targets.extend(target_list)

            # Extract predictions
            class_logits = predictions["class_logits"]
            if isinstance(class_logits, np.ndarray):
                class_probs = softmax(class_logits[b], axis=-1)
            else:
                class_probs = softmax(class_logits[b].cpu().numpy(), axis=-1)

            conf = predictions["confidence"]
            if not isinstance(conf, np.ndarray):
                conf = conf.cpu().numpy()
            conf = 1.0 / (1.0 + np.exp(-conf[b, :, 0]))

            start = predictions["start"]
            end = predictions["end"]
            if not isinstance(start, np.ndarray):
                start = start.cpu().numpy()
            if not isinstance(end, np.ndarray):
                end = end.cpu().numpy()

            for q in range(class_logits.shape[1]):
                pred_class_id = np.argmax(class_probs[q, 1:]) + 1
                pred_conf = class_probs[q, pred_class_id] * conf[q]

                if pred_conf >= confidence_threshold:
                    event = EventInterval(
                        class_id=pred_class_id,
                        start_norm=np.clip(start[b, q, 0], 0, 1),
                        end_norm=np.clip(end[b, q, 0], 0, 1),
                        center_norm=((start[b, q, 0] + end[b, q, 0]) / 2.0),
                        start_frame=int(start[b, q, 0]),
                        end_frame=int(end[b, q, 0]),
                    )
                    all_predictions.append(event)

        if len(all_predictions) == 0 and len(all_targets) == 0:
            return 1.0  # Perfect score if both empty

        if len(all_predictions) == 0 or len(all_targets) == 0:
            return 0.0  # No predictions or no targets

        # Match predictions to targets
        matched = np.zeros(len(all_targets), dtype=bool)
        tp = 0

        for pred in all_predictions:
            best_iou = 0.0
            best_target_idx = -1

            for t_idx, target in enumerate(all_targets):
                if matched[t_idx] or pred.class_id != target.class_id:
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

            if best_iou >= iou_threshold and best_target_idx >= 0:
                tp += 1
                matched[best_target_idx] = True

        fp = len(all_predictions) - tp
        fn = len(all_targets) - tp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-7, precision + recall)

        return float(f1)
