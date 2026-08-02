"""
DETR-style loss for temporal event detection.

Combines:
- Focal loss for class imbalance
- L1 regression loss for temporal coordinates
- Confidence (objectness) loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict

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
        matcher_cost_class: float = 1.0,
        matcher_cost_bbox: float = 1.0,
        matcher_cost_giou: float = 1.0,
    ):
        """
        Args:
            num_classes: Number of event classes (including background)
            loss_weights: Dict with keys:
                - class_loss: weight for classification loss
                - bbox_loss: weight for regression loss
                - conf_loss: weight for confidence loss
                - unmatched_query: weight for unmatched query penalty
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter (focusing parameter)
            matcher_cost_*: Hungarian matcher cost weights
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                "class_loss": 2.0,
                "bbox_loss": 5.0,
                "conf_loss": 2.0,
                "unmatched_query": 0.1,
            }
        self.loss_weights = loss_weights

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
                - confidence: [B, Nq, 1]
            targets: List of lists of EventInterval per sample [B][N_gt]

        Returns:
            Dict with:
                - loss_total: scalar loss
                - loss_class: classification loss
                - loss_bbox: regression loss
                - loss_conf: confidence loss
                - metrics: dict with auxiliary metrics
        """
        batch_size = predictions["class_logits"].shape[0]
        num_queries = predictions["class_logits"].shape[1]

        # Hungarian matching
        matches = self.matcher(predictions, targets, num_classes=self.num_classes)

        # Compute losses per sample
        class_losses = []
        bbox_losses = []
        conf_losses = []

        device = predictions["class_logits"].device

        for b in range(batch_size):
            match = matches[b]
            target_list = targets[b]

            # Extract predictions for this sample
            class_logits_b = predictions["class_logits"][b]  # [Nq, num_classes]
            center_b = predictions["center"][b]  # [Nq, 1]
            start_b = predictions["start"][b]  # [Nq, 1]
            end_b = predictions["end"][b]  # [Nq, 1]
            confidence_b = predictions["confidence"][b]  # [Nq, 1]

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

                # L1 loss on temporal coordinates
                loss_bbox_b = (
                    F.l1_loss(pred_centers, target_centers, reduction="mean")
                    + F.l1_loss(pred_starts, target_starts, reduction="mean")
                    + F.l1_loss(pred_ends, target_ends, reduction="mean")
                ) / 3.0
            else:
                loss_bbox_b = torch.tensor(0.0, device=device)

            # Confidence loss
            # Matched queries should have high confidence, unmatched should have low confidence
            target_confidence = torch.zeros(num_queries, device=device)
            if len(match.pred_indices) > 0:
                target_confidence[match.pred_indices] = 1.0

            # BCE loss with logits
            loss_conf_b = F.binary_cross_entropy_with_logits(
                confidence_b.squeeze(-1), target_confidence, reduction="mean"
            )

            class_losses.append(loss_class_b)
            bbox_losses.append(loss_bbox_b)
            conf_losses.append(loss_conf_b)

        # Average over batch
        loss_class = torch.stack(class_losses).mean()
        loss_bbox = torch.stack(bbox_losses).mean()
        loss_conf = torch.stack(conf_losses).mean()

        # Weighted sum
        loss_total = (
            self.loss_weights["class_loss"] * loss_class
            + self.loss_weights["bbox_loss"] * loss_bbox
            + self.loss_weights["conf_loss"] * loss_conf
        )

        return {
            "loss_total": loss_total,
            "loss_class": loss_class.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_conf": loss_conf.detach(),
            "metrics": {
                "num_matched": sum(len(m.pred_indices) for m in matches),
                "num_targets": sum(len(t) for t in targets),
                "num_predictions": batch_size * num_queries,
            },
        }

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

        # Alpha weight
        alpha = self.focal_alpha if targets != 0 else (1 - self.focal_alpha)

        focal_loss = -alpha * focal_weight * log_probs_per_sample

        # Apply unmatched weight for background predictions
        focal_loss[targets == 0] *= weight_unmatched

        return focal_loss.mean()
