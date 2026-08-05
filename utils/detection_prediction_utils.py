from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _clip_unit_interval(values: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.clamp(0.0, 1.0)
    return np.clip(values, 0.0, 1.0)


def normalize_prediction_intervals(
    predictions: dict[str, Any],
) -> dict[str, Any]:
    """Materialize center/start/end/duration for either MuSSED output format."""
    if "center" not in predictions:
        raise KeyError("Predictions must include 'center'.")

    center = predictions["center"]

    if "start" in predictions and "end" in predictions:
        start = _clip_unit_interval(predictions["start"])
        end = _clip_unit_interval(predictions["end"])
        duration = _clip_unit_interval(end - start)
    elif "duration" in predictions:
        duration = _clip_unit_interval(predictions["duration"])
        half_duration = 0.5 * duration
        start = _clip_unit_interval(center - half_duration)
        end = _clip_unit_interval(center + half_duration)
    else:
        raise KeyError(
            "Predictions must include either ('start', 'end') or 'duration'."
        )

    normalized = dict(predictions)
    normalized["center"] = center
    normalized["start"] = start
    normalized["end"] = end
    normalized["duration"] = duration
    return normalized
