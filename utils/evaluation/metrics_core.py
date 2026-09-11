from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

CANONICAL_METRIC_COLUMNS = {
    "macro_f1_6c": "macro_f1_6c",
    "event_f1_agnostic": "event_f1_agnostic",
    "event_iou_active_only": "event_iou_active_only",
}


def summarize_scalar_values(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def per_class_f1_from_confusion_matrix(confusion_matrix: np.ndarray) -> list[float]:
    if (
        confusion_matrix.ndim != 2
        or confusion_matrix.shape[0] != confusion_matrix.shape[1]
    ):
        raise ValueError(
            "confusion_matrix must be square [C,C], got " f"{confusion_matrix.shape}."
        )

    f1_scores: list[float] = []
    for i in range(confusion_matrix.shape[0]):
        tp = float(confusion_matrix[i, i])
        fp = float(np.sum(confusion_matrix[:, i]) - tp)
        fn = float(np.sum(confusion_matrix[i, :]) - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
        f1_scores.append(float(f1))

    return f1_scores


def macro_f1_6c_from_confusion_matrix(confusion_matrix: np.ndarray) -> float:
    if confusion_matrix.shape[0] != 6 or confusion_matrix.shape[1] != 6:
        raise ValueError(
            "macro_f1_6c requires a 6x6 confusion matrix ordered as "
            "[BG, VT, LP, TR, AV, IC]."
        )

    f1_scores = per_class_f1_from_confusion_matrix(confusion_matrix)
    return float(np.mean(np.asarray(f1_scores, dtype=np.float64)))


def event_f1_agnostic_from_confusion_matrix(confusion_matrix: np.ndarray) -> float:
    if (
        confusion_matrix.ndim != 2
        or confusion_matrix.shape[0] != confusion_matrix.shape[1]
    ):
        raise ValueError(
            "confusion_matrix must be square [C,C], got " f"{confusion_matrix.shape}."
        )
    if confusion_matrix.shape[0] < 2:
        raise ValueError(
            "confusion_matrix must include BG + at least one event class, got "
            f"{confusion_matrix.shape[0]} class(es)."
        )

    tp_evt = int(np.sum(confusion_matrix[1:, 1:]))
    fp_evt = int(np.sum(confusion_matrix[0, 1:]))
    fn_evt = int(np.sum(confusion_matrix[1:, 0]))

    precision = float(tp_evt / (tp_evt + fp_evt)) if (tp_evt + fp_evt) > 0 else 0.0
    recall = float(tp_evt / (tp_evt + fn_evt)) if (tp_evt + fn_evt) > 0 else 0.0
    if (precision + recall) <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def event_iou_active_only_from_class_indices(
    pred_class_idx: np.ndarray,
    true_class_idx: np.ndarray,
) -> float:
    pred = np.asarray(pred_class_idx)
    true = np.asarray(true_class_idx)
    if pred.shape != true.shape:
        raise ValueError(
            "pred_class_idx and true_class_idx must have the same shape, got "
            f"{pred.shape} vs {true.shape}."
        )
    if pred.ndim < 2:
        raise ValueError(
            "event_iou_active_only expects at least 2 dimensions [N, ...], got "
            f"shape={pred.shape}."
        )

    pred_flat = pred.reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)

    pred_event = pred_flat != 0
    true_event = true_flat != 0

    union = np.logical_or(pred_event, true_event).sum(axis=1)
    active_mask = union > 0
    if not np.any(active_mask):
        return 0.0

    inter = np.logical_and(pred_event, true_event).sum(axis=1)
    iou_per_window = np.zeros_like(union, dtype=np.float64)
    iou_per_window[active_mask] = inter[active_mask] / union[active_mask]
    return float(np.mean(iou_per_window[active_mask]))


def detection_temporal_iou_active_only_from_rows(
    predictions_df: pd.DataFrame,
) -> tuple[float, float]:
    """Return (temporal_iou_mean, mask_iou_mean) over active matched detections."""
    if predictions_df.empty:
        return 0.0, 0.0

    matched = predictions_df[
        predictions_df["match_type"].isin(
            ["matched_correct_class", "matched_wrong_class"]
        )
    ]
    if matched.empty:
        return 0.0, 0.0

    temporal_mean = float(matched["temporal_iou"].mean(skipna=True))
    mask_mean = float(matched["mask_iou"].mean(skipna=True))
    if not np.isfinite(temporal_mean):
        temporal_mean = 0.0
    if not np.isfinite(mask_mean):
        mask_mean = 0.0
    return temporal_mean, mask_mean
