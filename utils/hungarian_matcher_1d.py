"""
Hungarian matcher for matching predicted queries to ground-truth events.

Uses linear_sum_assignment (Hungarian algorithm) to find optimal matching
between predicted events (queries) and ground-truth events based on cost matrix.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from utils.event_targets import EventInterval


class MatchResult(NamedTuple):
    """Result of Hungarian matching."""

    pred_indices: np.ndarray  # [num_matches] which queries matched
    target_indices: np.ndarray  # [num_matches] which targets matched
    unmatched_pred: np.ndarray  # [num_unmatched] unmatched query indices
    unmatched_target: np.ndarray  # [num_unmatched] unmatched target indices
    total_cost: float  # sum of matched costs


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for 1D temporal event detection.

    Matches predicted event queries to ground-truth events by minimizing
    a cost matrix combining classification, temporal, and confidence costs.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
    ):
        """
        Args:
            cost_class: Relative weight of classification cost
            cost_bbox: Relative weight of bbox (temporal) regression cost
            cost_giou: Relative weight of temporal GIoU cost
        """
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)

        if self.cost_class + self.cost_bbox + self.cost_giou == 0:
            raise ValueError("All costs cannot be zero")

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: list[list[EventInterval]],
        num_classes: int = 6,
    ) -> list[MatchResult]:
        """
        Perform matching for a batch.

        Args:
            predictions: Dict with keys:
                - class_logits: [B, Nq, num_classes]
                - center: [B, Nq, 1]
                - start: [B, Nq, 1]
                - end: [B, Nq, 1]
                - confidence: [B, Nq, 1]
            targets: List of lists of EventInterval per sample [B][N_gt]
            num_classes: Number of classes (for focal loss weighting)

        Returns:
            List of MatchResult per sample.
        """
        batch_size = predictions["class_logits"].shape[0]

        class_logits = predictions["class_logits"]  # [B, Nq, num_classes]
        center_pred = predictions["center"]  # [B, Nq, 1]
        start_pred = predictions["start"]  # [B, Nq, 1]
        end_pred = predictions["end"]  # [B, Nq, 1]
        confidence = predictions["confidence"]  # [B, Nq, 1]

        # Softmax over classes
        class_probs = torch.softmax(class_logits, dim=-1)  # [B, Nq, num_classes]

        results = []
        for b in range(batch_size):
            target_list = targets[b]

            if len(target_list) == 0:
                # No targets in this sample
                num_queries = class_logits.shape[1]
                result = MatchResult(
                    pred_indices=np.array([], dtype=np.int64),
                    target_indices=np.array([], dtype=np.int64),
                    unmatched_pred=np.arange(num_queries, dtype=np.int64),
                    unmatched_target=np.array([], dtype=np.int64),
                    total_cost=0.0,
                )
                results.append(result)
                continue

            # Compute cost matrix [Nq, N_gt]
            cost_matrix = self._compute_cost_matrix(
                class_probs[b],
                center_pred[b],
                start_pred[b],
                end_pred[b],
                confidence[b],
                target_list,
                num_classes=num_classes,
            )

            # Hungarian matching
            match_result = self._hungarian_match(cost_matrix, target_list)
            results.append(match_result)

        return results

    def _compute_cost_matrix(
        self,
        class_probs: torch.Tensor,  # [Nq, num_classes]
        center_pred: torch.Tensor,  # [Nq, 1]
        start_pred: torch.Tensor,  # [Nq, 1]
        end_pred: torch.Tensor,  # [Nq, 1]
        confidence: torch.Tensor,  # [Nq, 1]
        target_list: list[EventInterval],
        num_classes: int = 6,
    ) -> np.ndarray:
        """
        Compute cost matrix between predictions and targets.

        Returns:
            [Nq, N_gt] cost matrix (numpy)
        """
        num_queries = class_probs.shape[0]
        num_targets = len(target_list)

        if num_targets == 0:
            return np.zeros((num_queries, 0), dtype=np.float32)

        # Detach and move to numpy
        class_probs_np = class_probs.detach().cpu().numpy()  # [Nq, num_classes]
        center_pred_np = center_pred.detach().cpu().numpy().squeeze(-1)  # [Nq]
        start_pred_np = start_pred.detach().cpu().numpy().squeeze(-1)  # [Nq]
        end_pred_np = end_pred.detach().cpu().numpy().squeeze(-1)  # [Nq]

        # Target class IDs
        target_classes = np.array(
            [t.class_id for t in target_list], dtype=np.int64
        )  # [N_gt]
        target_centers = np.array(
            [t.center_norm for t in target_list], dtype=np.float32
        )  # [N_gt]
        target_starts = np.array(
            [t.start_norm for t in target_list], dtype=np.float32
        )  # [N_gt]
        target_ends = np.array(
            [t.end_norm for t in target_list], dtype=np.float32
        )  # [N_gt]

        # Cost 1: Classification cost (negative log probability of target class)
        # [Nq, N_gt]
        cost_class = -class_probs_np[:, target_classes]

        # Cost 2: L1 distance for bbox (center, start, end)
        # [Nq] -> [Nq, 1], [N_gt] -> [1, N_gt]
        # Broadcast to [Nq, N_gt]
        cost_bbox = (
            np.abs(center_pred_np[:, None] - target_centers[None, :])
            + np.abs(start_pred_np[:, None] - target_starts[None, :])
            + np.abs(end_pred_np[:, None] - target_ends[None, :])
        )

        # Cost 3: Temporal GIoU (1 - GIoU)
        giou_matrix = self._temporal_giou_matrix(
            center_pred_np,
            start_pred_np,
            end_pred_np,
            target_centers,
            target_starts,
            target_ends,
        )  # [Nq, N_gt]
        cost_giou = 1.0 - giou_matrix

        # Combine costs
        cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        return cost.astype(np.float32)

    @staticmethod
    def _temporal_giou_matrix(
        center_pred: np.ndarray,  # [Nq]
        start_pred: np.ndarray,  # [Nq]
        end_pred: np.ndarray,  # [Nq]
        center_target: np.ndarray,  # [N_gt]
        start_target: np.ndarray,  # [N_gt]
        end_target: np.ndarray,  # [N_gt]
    ) -> np.ndarray:
        """
        Compute temporal GIoU matrix.

        Args:
            pred: [Nq] predictions
            target: [N_gt] targets

        Returns:
            [Nq, N_gt] GIoU matrix
        """
        # Clamp predictions to [0, 1]
        start_pred = np.clip(start_pred, 0, 1)
        end_pred = np.clip(end_pred, 0, 1)
        start_target = np.clip(start_target, 0, 1)
        end_target = np.clip(end_target, 0, 1)

        # Ensure start < end
        start_pred = np.minimum(start_pred, end_pred)
        end_pred = np.maximum(start_pred, end_pred)
        start_target = np.minimum(start_target, end_target)
        end_target = np.maximum(start_target, end_target)

        # Intersection: [Nq, N_gt]
        inter_start = np.maximum(start_pred[:, None], start_target[None, :])
        inter_end = np.minimum(end_pred[:, None], end_target[None, :])
        inter_len = np.maximum(0, inter_end - inter_start)

        # Union
        union_len = (
            (end_pred[:, None] - start_pred[:, None])
            + (end_target[None, :] - start_target[None, :])
            - inter_len
        )

        # IoU
        iou = inter_len / (union_len + 1e-7)

        # Enclosing interval
        enclose_start = np.minimum(start_pred[:, None], start_target[None, :])
        enclose_end = np.maximum(end_pred[:, None], end_target[None, :])
        enclose_len = np.maximum(enclose_end - enclose_start, 1e-7)

        # GIoU
        giou = iou - (enclose_len - union_len) / enclose_len

        return giou.astype(np.float32)

    @staticmethod
    def _hungarian_match(
        cost_matrix: np.ndarray,  # [Nq, N_gt]
        target_list: list[EventInterval],
    ) -> MatchResult:
        """
        Solve Hungarian algorithm for matching.

        Args:
            cost_matrix: [Nq, N_gt]
            target_list: For reference only

        Returns:
            MatchResult with matched and unmatched indices.
        """
        if cost_matrix.shape[1] == 0:
            # No targets
            num_queries = cost_matrix.shape[0]
            return MatchResult(
                pred_indices=np.array([], dtype=np.int64),
                target_indices=np.array([], dtype=np.int64),
                unmatched_pred=np.arange(num_queries, dtype=np.int64),
                unmatched_target=np.array([], dtype=np.int64),
                total_cost=0.0,
            )

        pred_idx, target_idx = linear_sum_assignment(cost_matrix)

        num_queries = cost_matrix.shape[0]
        num_targets = cost_matrix.shape[1]

        unmatched_pred = np.setdiff1d(np.arange(num_queries, dtype=np.int64), pred_idx)
        unmatched_target = np.setdiff1d(
            np.arange(num_targets, dtype=np.int64), target_idx
        )

        total_cost = float(cost_matrix[pred_idx, target_idx].sum())

        return MatchResult(
            pred_indices=pred_idx.astype(np.int64),
            target_indices=target_idx.astype(np.int64),
            unmatched_pred=unmatched_pred,
            unmatched_target=unmatched_target,
            total_cost=total_cost,
        )
