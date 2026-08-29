"""
Losses for query-based temporal event detection with dense localization supervision.

Combines:
- Focal classification loss (matched + unmatched/background queries)
- Confidence/objectness BCE loss
- Interval regression (start/end L1)
- Temporal GIoU loss
- Segmentation supervision per matched query (BCE + Dice)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.event_targets import EventInterval
from utils.hungarian_matcher_1d import HungarianMatcher


class EventDetectionLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        loss_weights: Dict[str, float] | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        matcher_cost_class: float | None = None,
        matcher_cost_bbox: float | None = None,
        matcher_cost_giou: float | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

        if loss_weights is None:
            loss_weights = {
                "class_loss": 2.0,
                "confidence_loss": 0.1,
                "bbox_loss": 2.0,
                "giou_loss": 2.0,
                "mask_bce_loss": 1.0,
                "mask_dice_loss": 2.0,
                "unmatched_query": 0.0,
            }
        self.loss_weights = {k: float(v) for k, v in loss_weights.items()}
        self.loss_weights.setdefault("class_loss", 2.0)
        self.loss_weights.setdefault("confidence_loss", 0.1)
        self.loss_weights.setdefault("bbox_loss", 2.0)
        self.loss_weights.setdefault("giou_loss", 2.0)
        self.loss_weights.setdefault("mask_bce_loss", 1.0)
        self.loss_weights.setdefault("mask_dice_loss", 2.0)
        self.loss_weights.setdefault("unmatched_query", 0.0)

        if matcher_cost_class is None:
            matcher_cost_class = float(self.loss_weights["class_loss"])
        if matcher_cost_bbox is None:
            matcher_cost_bbox = float(self.loss_weights["bbox_loss"])
        if matcher_cost_giou is None:
            matcher_cost_giou = float(self.loss_weights["giou_loss"])

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
        predictions = normalize_prediction_intervals(predictions)
        required_keys = {
            "class_logits",
            "confidence_logits",
            "mask_logits",
            "start",
            "end",
        }
        missing = [key for key in required_keys if key not in predictions]
        if missing:
            raise KeyError(f"Predictions missing required keys: {missing}.")

        batch_size = predictions["class_logits"].shape[0]
        num_queries = predictions["class_logits"].shape[1]
        t_len = predictions["mask_logits"].shape[-1]
        device = predictions["class_logits"].device

        matches = self.matcher(predictions, targets, num_classes=self.num_classes)

        class_losses = []
        confidence_losses = []
        bbox_losses = []
        giou_losses = []
        mask_bce_losses = []
        mask_dice_losses = []
        unmatched_query_losses = []
        matched_mask_iou_values: list[torch.Tensor] = []
        matched_interval_iou_values: list[torch.Tensor] = []

        for b in range(batch_size):
            match = matches[b]
            target_list = targets[b]
            has_targets = len(target_list) > 0

            class_logits_b = predictions["class_logits"][b]
            confidence_logits_b = predictions["confidence_logits"][b, :, 0]
            start_b = predictions["start"][b, :, 0]
            end_b = predictions["end"][b, :, 0]
            mask_logits_b = predictions["mask_logits"][b]

            target_obj = torch.zeros(num_queries, device=device, dtype=torch.float32)
            if len(match.pred_indices) > 0:
                target_obj[match.pred_indices] = 1.0

            loss_conf_b = F.binary_cross_entropy_with_logits(
                confidence_logits_b,
                target_obj,
                reduction="mean",
            )

            if len(match.pred_indices) > 0:
                target_classes = torch.tensor(
                    [target_list[i].class_id for i in match.target_indices],
                    dtype=torch.long,
                    device=device,
                )
                loss_class_matched = self._focal_loss(
                    class_logits_b[match.pred_indices],
                    target_classes,
                )
            else:
                loss_class_matched = torch.tensor(0.0, device=device)

            # For BG-only windows (no targets), keep class-loss neutral and let
            # confidence/objectness loss carry the background supervision.
            if len(match.unmatched_pred) > 0 and has_targets:
                unmatched_target_classes = torch.zeros(
                    len(match.unmatched_pred),
                    dtype=torch.long,
                    device=device,
                )
                loss_class_unmatched = self._focal_loss(
                    class_logits_b[match.unmatched_pred],
                    unmatched_target_classes,
                    weight_unmatched=self.loss_weights.get("unmatched_query", 1.0),
                )
            else:
                loss_class_unmatched = torch.tensor(0.0, device=device)

            loss_class_b = loss_class_matched + loss_class_unmatched

            if len(match.pred_indices) > 0:
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

                pred_starts = start_b[match.pred_indices]
                pred_ends = end_b[match.pred_indices]

                loss_bbox_b = 0.5 * (
                    F.l1_loss(pred_starts, target_starts, reduction="mean")
                    + F.l1_loss(pred_ends, target_ends, reduction="mean")
                )

                giou = self._temporal_giou_1d(
                    pred_starts,
                    pred_ends,
                    target_starts,
                    target_ends,
                )
                loss_giou_b = (1.0 - giou).mean()
                matched_interval_iou_values.append(
                    self._temporal_iou_1d(
                        pred_starts,
                        pred_ends,
                        target_starts,
                        target_ends,
                    ).mean()
                )

                target_masks = []
                for target_idx in match.target_indices:
                    event = target_list[target_idx]
                    mask_vec = self._build_mask_target_over_time(
                        event.start_norm,
                        event.end_norm,
                        t_len=t_len,
                        device=device,
                    )
                    target_masks.append(mask_vec)

                target_masks_t = torch.stack(target_masks, dim=0)

                pred_mask_logits = mask_logits_b[match.pred_indices]

                loss_mask_bce_b = F.binary_cross_entropy_with_logits(
                    pred_mask_logits,
                    target_masks_t,
                    reduction="mean",
                )
                pred_mask_probs = torch.sigmoid(pred_mask_logits)
                loss_mask_dice_b = self._soft_dice_loss(pred_mask_probs, target_masks_t)
                matched_mask_iou_values.append(
                    self._binary_mask_iou_mean(pred_mask_probs, target_masks_t)
                )
            else:
                loss_bbox_b = torch.tensor(0.0, device=device)
                loss_giou_b = torch.tensor(0.0, device=device)
                loss_mask_bce_b = torch.tensor(0.0, device=device)
                loss_mask_dice_b = torch.tensor(0.0, device=device)

            class_losses.append(loss_class_b)
            confidence_losses.append(loss_conf_b)
            bbox_losses.append(loss_bbox_b)
            giou_losses.append(loss_giou_b)
            mask_bce_losses.append(loss_mask_bce_b)
            mask_dice_losses.append(loss_mask_dice_b)
            unmatched_query_losses.append(loss_class_unmatched)

        loss_class = torch.stack(class_losses).mean()
        loss_confidence = torch.stack(confidence_losses).mean()
        loss_bbox = torch.stack(bbox_losses).mean()
        loss_giou = torch.stack(giou_losses).mean()
        loss_mask_bce = torch.stack(mask_bce_losses).mean()
        loss_mask_dice = torch.stack(mask_dice_losses).mean()
        loss_unmatched_query = torch.stack(unmatched_query_losses).mean()
        if matched_mask_iou_values:
            metric_mask_iou = torch.stack(matched_mask_iou_values).mean()
        else:
            metric_mask_iou = torch.tensor(0.0, device=device)
        if matched_interval_iou_values:
            metric_interval_iou = torch.stack(matched_interval_iou_values).mean()
        else:
            metric_interval_iou = torch.tensor(0.0, device=device)

        loss_total = (
            self.loss_weights["class_loss"] * loss_class
            + self.loss_weights["confidence_loss"] * loss_confidence
            + self.loss_weights["bbox_loss"] * loss_bbox
            + self.loss_weights["giou_loss"] * loss_giou
            + self.loss_weights["mask_bce_loss"] * loss_mask_bce
            + self.loss_weights["mask_dice_loss"] * loss_mask_dice
        )

        return {
            "loss_total": loss_total,
            "loss_class": loss_class.detach(),
            "loss_confidence": loss_confidence.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            "loss_mask_bce": loss_mask_bce.detach(),
            "loss_mask_dice": loss_mask_dice.detach(),
            "loss_unmatched_query": loss_unmatched_query.detach(),
            "metric_mask_iou": metric_mask_iou.detach(),
            "metric_interval_iou": metric_interval_iou.detach(),
            "metrics": {
                "num_matched": sum(len(m.pred_indices) for m in matches),
                "num_targets": sum(len(t) for t in targets),
                "num_predictions": batch_size * num_queries,
            },
        }

    @staticmethod
    def _soft_dice_loss(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        intersection = (probabilities * targets).sum(dim=-1)
        denominator = probabilities.sum(dim=-1) + targets.sum(dim=-1)
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1.0 - dice.mean()

    @staticmethod
    def _binary_mask_iou_mean(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        pred_bin = (probabilities >= float(threshold)).to(torch.float32)
        target_bin = (targets >= 0.5).to(torch.float32)
        inter = (pred_bin * target_bin).sum(dim=-1)
        union = pred_bin.sum(dim=-1) + target_bin.sum(dim=-1) - inter
        iou = inter / union.clamp_min(float(eps))
        return iou.mean()

    def _build_mask_target_over_time(
        self,
        start_norm: float,
        end_norm: float,
        *,
        t_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        start = float(max(0.0, min(1.0, start_norm)))
        end = float(max(0.0, min(1.0, end_norm)))
        start, end = (start, end) if start <= end else (end, start)

        time = torch.linspace(0.0, 1.0, steps=t_len, device=device)
        mask = ((time >= start) & (time <= end)).to(torch.float32)
        return mask

    @staticmethod
    def _temporal_iou_1d(
        pred_starts: torch.Tensor,
        pred_ends: torch.Tensor,
        target_starts: torch.Tensor,
        target_ends: torch.Tensor,
    ) -> torch.Tensor:
        pred_start_fixed = torch.minimum(pred_starts, pred_ends).clamp(0.0, 1.0)
        pred_end_fixed = torch.maximum(pred_starts, pred_ends).clamp(0.0, 1.0)
        target_start_fixed = torch.minimum(target_starts, target_ends).clamp(0.0, 1.0)
        target_end_fixed = torch.maximum(target_starts, target_ends).clamp(0.0, 1.0)

        inter_start = torch.maximum(pred_start_fixed, target_start_fixed)
        inter_end = torch.minimum(pred_end_fixed, target_end_fixed)
        inter_len = (inter_end - inter_start).clamp(min=0.0)

        pred_len = (pred_end_fixed - pred_start_fixed).clamp(min=0.0)
        target_len = (target_end_fixed - target_start_fixed).clamp(min=0.0)
        union_len = pred_len + target_len - inter_len
        return inter_len / union_len.clamp_min(1e-7)

    @staticmethod
    def _temporal_giou_1d(
        pred_starts: torch.Tensor,
        pred_ends: torch.Tensor,
        target_starts: torch.Tensor,
        target_ends: torch.Tensor,
    ) -> torch.Tensor:
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
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weight_unmatched: float = 1.0,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(predictions, dim=-1)
        log_probs_per_sample = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        probs = torch.exp(log_probs_per_sample)
        focal_weight = (1.0 - probs) ** self.focal_gamma
        alpha = torch.where(targets == 0, 1.0 - self.focal_alpha, self.focal_alpha)

        focal_loss = -alpha * focal_weight * log_probs_per_sample
        focal_loss[targets == 0] *= weight_unmatched
        return focal_loss.mean()
