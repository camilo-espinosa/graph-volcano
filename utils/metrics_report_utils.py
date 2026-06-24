from __future__ import annotations

import numpy as np

from utils.train_utils import compute_summary


def compute_iou_from_cm(cm: np.ndarray) -> tuple[list[float], float]:
    iou_per_class = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        denom = tp + fp + fn
        iou_per_class.append(float(tp / denom) if denom > 0 else 0.0)
    mean_iou = float(np.mean(iou_per_class)) if len(iou_per_class) > 0 else 0.0
    return iou_per_class, mean_iou


def compute_per_class_summary(
    per_fold_values: list[list[float]],
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    per_class_summary: dict[str, dict[str, float]] = {}
    for class_idx, class_name in enumerate(class_names):
        values = [float(v[class_idx]) for v in per_fold_values]
        per_class_summary[class_name] = compute_summary(values)
    return per_class_summary
