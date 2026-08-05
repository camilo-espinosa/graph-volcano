"""
DETR-style loss for temporal event detection.

Combines:
- Focal loss for class imbalance
- L1 regression loss for temporal center and duration
- Temporal GIoU loss for interval overlap quality
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict

from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.hungarian_matcher_1d import HungarianMatcher, MatchResult
from utils.event_targets import EventInterval


class DETREventLoss(torch.nn.Module):
    """
    Combined DETR loss for event detection.
    """

    def __init__(
        self,
        num_classes: int = 6,
        loss_weights: Dict[str, float] | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        duration_smooth_l1_beta: float = 0.1,
        matcher_cost_class: float | None = None,
        matcher_cost_bbox: float | None = None,
        matcher_cost_giou: float | None = None,
    ):
        """
        Args:
            num_classes: Number of event classes (including background)
            loss_weights: Dict with keys:
                - class_loss: weight for classification loss
                - bbox_loss: weight for L1 center+duration regression loss
                - giou_loss: weight for temporal GIoU loss
                - unmatched_query: weight for unmatched query penalty
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter (focusing parameter)
            duration_smooth_l1_beta: Beta parameter for SmoothL1 on duration
            matcher_cost_*: Hungarian matcher cost weights
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.duration_smooth_l1_beta = float(duration_smooth_l1_beta)

        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                "class_loss": 2.0,
                "bbox_loss": 3.0,
                "giou_loss": 2.0,
                "unmatched_query": 0.1,
            }
        self.loss_weights = loss_weights

        if matcher_cost_class is None:
            matcher_cost_class = float(self.loss_weights["class_loss"])
        if matcher_cost_bbox is None:
            matcher_cost_bbox = float(self.loss_weights["bbox_loss"])
        if matcher_cost_giou is None:
            matcher_cost_giou = float(self.loss_weights["giou_loss"])

        # Matcher
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: list[list[EventInterval]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DETR loss.

        Args:
            predictions: Dict with keys:
                - class_logits: [B, Nq, num_classes]
                - center: [B, Nq, 1]
                - start: [B, Nq, 1]
                - end: [B, Nq, 1]
                - duration: [B, Nq, 1] (materialized by normalization)
            targets: List of lists of EventInterval per sample [B][N_gt]

        Returns:
            Dict with:
                - loss_total: scalar loss
                - loss_class: classification loss
                - loss_bbox: L1 regression loss
                - loss_giou: temporal GIoU loss
                - metrics: dict with auxiliary metrics
        """
        predictions = normalize_prediction_intervals(predictions)
        batch_size = predictions["class_logits"].shape[0]
        num_queries = predictions["class_logits"].shape[1]

        # Hungarian matching
        matches = self.matcher(predictions, targets, num_classes=self.num_classes)

        # Compute losses per sample
        class_losses = []
        bbox_losses = []
        giou_losses = []

        device = predictions["class_logits"].device

        for b in range(batch_size):
            match = matches[b]
            target_list = targets[b]

            # Extract predictions for this sample
            class_logits_b = predictions["class_logits"][b]  # [Nq, num_classes]
            center_b = predictions["center"][b]  # [Nq, 1]
            start_b = predictions["start"][b]  # [Nq, 1]
            end_b = predictions["end"][b]  # [Nq, 1]

            # Classification loss for matched queries
            if len(match.pred_indices) > 0:
                matched_class_logits = class_logits_b[
                    match.pred_indices
                ]  # [M, num_classes]
                target_classes = torch.tensor(
                    [target_list[i].class_id for i in match.target_indices],
                    dtype=torch.long,
                    device=device,
                )
                loss_class_matched = self._focal_loss(
                    matched_class_logits, target_classes
                )
            else:
                loss_class_matched = torch.tensor(0.0, device=device)

            # Classification loss for unmatched queries (predict background=0)
            if len(match.unmatched_pred) > 0:
                unmatched_class_logits = class_logits_b[
                    match.unmatched_pred
                ]  # [U, num_classes]
                unmatched_target_classes = torch.zeros(
                    len(match.unmatched_pred),
                    dtype=torch.long,
                    device=device,
                )
                loss_class_unmatched = self._focal_loss(
                    unmatched_class_logits,
                    unmatched_target_classes,
                    weight_unmatched=self.loss_weights.get("unmatched_query", 0.1),
                )
            else:
                loss_class_unmatched = torch.tensor(0.0, device=device)

            loss_class_b = loss_class_matched + loss_class_unmatched

            # Regression (bbox) loss for matched queries
            if len(match.pred_indices) > 0:
                pred_centers = center_b[match.pred_indices, 0]  # [M]
                pred_starts = start_b[match.pred_indices, 0]  # [M]
                pred_ends = end_b[match.pred_indices, 0]  # [M]
                pred_durations = (pred_ends - pred_starts).clamp(0.0, 1.0)  # [M]

                target_centers = torch.tensor(
                    [target_list[i].center_norm for i in match.target_indices],
                    dtype=torch.float32,
                    device=device,
                )
                target_starts = torch.tensor(
                    [target_list[i].start_norm for i in match.target_indices],
                    dtype=torch.float32,
                    device=device,
                )
                target_ends = torch.tensor(
                    [target_list[i].end_norm for i in match.target_indices],
                    dtype=torch.float32,
                    device=device,
                )
                target_durations = (target_ends - target_starts).clamp(0.0, 1.0)

                # L1 on center + SmoothL1 on duration.
                loss_bbox_b = (
                    F.l1_loss(pred_centers, target_centers, reduction="mean")
                    + F.smooth_l1_loss(
                        pred_durations,
                        target_durations,
                        beta=self.duration_smooth_l1_beta,
                        reduction="mean",
                    )
                ) / 2.0

                giou = self._temporal_giou_1d(
                    pred_starts, pred_ends, target_starts, target_ends
                )
                loss_giou_b = (1.0 - giou).mean()
            else:
                loss_bbox_b = torch.tensor(0.0, device=device)
                loss_giou_b = torch.tensor(0.0, device=device)

            class_losses.append(loss_class_b)
            bbox_losses.append(loss_bbox_b)
            giou_losses.append(loss_giou_b)

        # Average over batch
        loss_class = torch.stack(class_losses).mean()
        loss_bbox = torch.stack(bbox_losses).mean()
        loss_giou = torch.stack(giou_losses).mean()

        # Weighted sum
        loss_total = (
            self.loss_weights["class_loss"] * loss_class
            + self.loss_weights["bbox_loss"] * loss_bbox
            + self.loss_weights["giou_loss"] * loss_giou
        )

        return {
            "loss_total": loss_total,
            "loss_class": loss_class.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            "metrics": {
                "num_matched": sum(len(m.pred_indices) for m in matches),
                "num_targets": sum(len(t) for t in targets),
                "num_predictions": batch_size * num_queries,
            },
        }

    @staticmethod
    def _temporal_giou_1d(
        pred_starts: torch.Tensor,
        pred_ends: torch.Tensor,
        target_starts: torch.Tensor,
        target_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal GIoU for matched 1D interval pairs."""
        pred_starts = pred_starts.clamp(0.0, 1.0)
        pred_ends = pred_ends.clamp(0.0, 1.0)
        target_starts = target_starts.clamp(0.0, 1.0)
        target_ends = target_ends.clamp(0.0, 1.0)

        pred_start_fixed = torch.minimum(pred_starts, pred_ends)
        pred_end_fixed = torch.maximum(pred_starts, pred_ends)
        target_start_fixed = torch.minimum(target_starts, target_ends)
        target_end_fixed = torch.maximum(target_starts, target_ends)

        inter_start = torch.maximum(pred_start_fixed, target_start_fixed)
        inter_end = torch.minimum(pred_end_fixed, target_end_fixed)
        inter_len = (inter_end - inter_start).clamp(min=0.0)

        pred_len = (pred_end_fixed - pred_start_fixed).clamp(min=0.0)
        target_len = (target_end_fixed - target_start_fixed).clamp(min=0.0)
        union_len = pred_len + target_len - inter_len

        iou = inter_len / (union_len + 1e-7)

        enclose_start = torch.minimum(pred_start_fixed, target_start_fixed)
        enclose_end = torch.maximum(pred_end_fixed, target_end_fixed)
        enclose_len = (enclose_end - enclose_start).clamp(min=1e-7)

        giou = iou - (enclose_len - union_len) / enclose_len
        return giou

    def _focal_loss(
        self,
        predictions: torch.Tensor,  # [N, num_classes]
        targets: torch.Tensor,  # [N]
        weight_unmatched: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: [N, num_classes] logits
            targets: [N] class indices
            weight_unmatched: Weight for background class (0)

        Returns:
            Scalar focal loss
        """
        log_probs = F.log_softmax(predictions, dim=-1)
        log_probs_per_sample = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        probs = torch.exp(log_probs_per_sample)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.focal_gamma

        # Alpha weight - apply element-wise based on whether target is 0 (background)
        alpha = torch.where(targets == 0, 1.0 - self.focal_alpha, self.focal_alpha)

        focal_loss = -alpha * focal_weight * log_probs_per_sample

        # Apply unmatched weight for background predictions
        focal_loss[targets == 0] *= weight_unmatched

        return focal_loss.mean()
