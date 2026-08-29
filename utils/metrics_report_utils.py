from __future__ import annotations

from utils.train_utils import compute_summary


def compute_per_class_summary(
    per_fold_values: list[list[float]],
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    per_class_summary: dict[str, dict[str, float]] = {}
    for class_idx, class_name in enumerate(class_names):
        values = [float(v[class_idx]) for v in per_fold_values]
        per_class_summary[class_name] = compute_summary(values)
    return per_class_summary
