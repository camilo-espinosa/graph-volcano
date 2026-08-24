"""
Losses for query-based temporal event detection with dense localization supervision.

Combines:
- Focal classification loss (matched + unmatched/background queries)
- Confidence/objectness BCE loss
- Interval regression (start/end L1)
- Temporal GIoU loss
- Segmentation supervision per matched query (BCE + Dice)
- Start/end heatmap supervision (distributional NLL over time)
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
        peak_sigma: float = 0.015,
        matcher_cost_class: float | None = None,
        matcher_cost_bbox: float | None = None,
        matcher_cost_giou: float | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.peak_sigma = float(peak_sigma)

        if loss_weights is None:
            loss_weights = {
                "class_loss": 2.0,
                "confidence_loss": 0.1,
                "bbox_loss": 2.0,
                "giou_loss": 2.0,
                "mask_bce_loss": 1.0,
                "mask_dice_loss": 2.0,
                "start_heatmap_loss": 0.0,
                "end_heatmap_loss": 0.0,
                "unmatched_query": 1.0,
            }
        self.loss_weights = {k: float(v) for k, v in loss_weights.items()}

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
            "start_heatmap_logits",
            "end_heatmap_logits",
            "start",
            "end",
            "center",
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
        start_hm_losses = []
        end_hm_losses = []
        unmatched_query_losses = []

        for b in range(batch_size):
            match = matches[b]
            target_list = targets[b]

            class_logits_b = predictions["class_logits"][b]
            confidence_logits_b = predictions["confidence_logits"][b, :, 0]
            start_b = predictions["start"][b, :, 0]
            end_b = predictions["end"][b, :, 0]
            mask_logits_b = predictions["mask_logits"][b]
            start_hm_logits_b = predictions["start_heatmap_logits"][b]
            end_hm_logits_b = predictions["end_heatmap_logits"][b]

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

            if len(match.unmatched_pred) > 0:
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

                target_masks = []
                target_start_hm = []
                target_end_hm = []
                for target_idx in match.target_indices:
                    event = target_list[target_idx]
                    mask_vec, start_peak, end_peak = self._build_targets_over_time(
                        event.start_norm,
                        event.end_norm,
                        t_len=t_len,
                        device=device,
                    )
                    target_masks.append(mask_vec)
                    target_start_hm.append(start_peak)
                    target_end_hm.append(end_peak)

                target_masks_t = torch.stack(target_masks, dim=0)
                target_start_hm_t = torch.stack(target_start_hm, dim=0)
                target_end_hm_t = torch.stack(target_end_hm, dim=0)

                pred_mask_logits = mask_logits_b[match.pred_indices]
                pred_start_hm_logits = start_hm_logits_b[match.pred_indices]
                pred_end_hm_logits = end_hm_logits_b[match.pred_indices]

                loss_mask_bce_b = F.binary_cross_entropy_with_logits(
                    pred_mask_logits,
                    target_masks_t,
                    reduction="mean",
                )
                pred_mask_probs = torch.sigmoid(pred_mask_logits)
                loss_mask_dice_b = self._soft_dice_loss(pred_mask_probs, target_masks_t)

                if self.loss_weights["start_heatmap_loss"] > 0.0:
                    loss_start_hm_b = self._distribution_nll(
                        pred_start_hm_logits,
                        target_start_hm_t,
                    )
                else:
                    loss_start_hm_b = torch.tensor(0.0, device=device)

                if self.loss_weights["end_heatmap_loss"] > 0.0:
                    loss_end_hm_b = self._distribution_nll(
                        pred_end_hm_logits,
                        target_end_hm_t,
                    )
                else:
                    loss_end_hm_b = torch.tensor(0.0, device=device)
            else:
                loss_bbox_b = torch.tensor(0.0, device=device)
                loss_giou_b = torch.tensor(0.0, device=device)
                loss_mask_bce_b = torch.tensor(0.0, device=device)
                loss_mask_dice_b = torch.tensor(0.0, device=device)
                loss_start_hm_b = torch.tensor(0.0, device=device)
                loss_end_hm_b = torch.tensor(0.0, device=device)

            class_losses.append(loss_class_b)
            confidence_losses.append(loss_conf_b)
            bbox_losses.append(loss_bbox_b)
            giou_losses.append(loss_giou_b)
            mask_bce_losses.append(loss_mask_bce_b)
            mask_dice_losses.append(loss_mask_dice_b)
            start_hm_losses.append(loss_start_hm_b)
            end_hm_losses.append(loss_end_hm_b)
            unmatched_query_losses.append(loss_class_unmatched)

        loss_class = torch.stack(class_losses).mean()
        loss_confidence = torch.stack(confidence_losses).mean()
        loss_bbox = torch.stack(bbox_losses).mean()
        loss_giou = torch.stack(giou_losses).mean()
        loss_mask_bce = torch.stack(mask_bce_losses).mean()
        loss_mask_dice = torch.stack(mask_dice_losses).mean()
        loss_start_heatmap = torch.stack(start_hm_losses).mean()
        loss_end_heatmap = torch.stack(end_hm_losses).mean()
        loss_unmatched_query = torch.stack(unmatched_query_losses).mean()

        loss_total = (
            self.loss_weights["class_loss"] * loss_class
            + self.loss_weights["confidence_loss"] * loss_confidence
            + self.loss_weights["bbox_loss"] * loss_bbox
            + self.loss_weights["giou_loss"] * loss_giou
            + self.loss_weights["mask_bce_loss"] * loss_mask_bce
            + self.loss_weights["mask_dice_loss"] * loss_mask_dice
            + self.loss_weights["start_heatmap_loss"] * loss_start_heatmap
            + self.loss_weights["end_heatmap_loss"] * loss_end_heatmap
        )

        return {
            "loss_total": loss_total,
            "loss_class": loss_class.detach(),
            "loss_confidence": loss_confidence.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
            "loss_mask_bce": loss_mask_bce.detach(),
            "loss_mask_dice": loss_mask_dice.detach(),
            "loss_start_heatmap": loss_start_heatmap.detach(),
            "loss_end_heatmap": loss_end_heatmap.detach(),
            "loss_unmatched_query": loss_unmatched_query.detach(),
            "metrics": {
                "num_matched": sum(len(m.pred_indices) for m in matches),
                "num_targets": sum(len(t) for t in targets),
                "num_predictions": batch_size * num_queries,
            },
        }

    def _distribution_nll(
        self,
        logits: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return (-(target_distribution * log_probs).sum(dim=-1)).mean()

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

    def _build_targets_over_time(
        self,
        start_norm: float,
        end_norm: float,
        *,
        t_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = float(max(0.0, min(1.0, start_norm)))
        end = float(max(0.0, min(1.0, end_norm)))
        start, end = (start, end) if start <= end else (end, start)

        time = torch.linspace(0.0, 1.0, steps=t_len, device=device)
        mask = ((time >= start) & (time <= end)).to(torch.float32)

        sigma = max(self.peak_sigma, 1e-4)
        start_peak = torch.exp(-0.5 * ((time - start) / sigma) ** 2)
        end_peak = torch.exp(-0.5 * ((time - end) / sigma) ** 2)
        start_peak = start_peak / start_peak.sum().clamp_min(1e-8)
        end_peak = end_peak / end_peak.sum().clamp_min(1e-8)

        return mask, start_peak, end_peak

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
