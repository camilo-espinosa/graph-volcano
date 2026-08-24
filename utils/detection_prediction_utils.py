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
    center = None
    if "center" in predictions:
        center = _clip_unit_interval(predictions["center"])

    if "start" in predictions and "end" in predictions:
        start_raw = _clip_unit_interval(predictions["start"])
        end_raw = _clip_unit_interval(predictions["end"])

        # Canonicalize interval ordering for all downstream consumers.
        if isinstance(start_raw, torch.Tensor):
            start = torch.minimum(start_raw, end_raw)
            end = torch.maximum(start_raw, end_raw)
        else:
            start = np.minimum(start_raw, end_raw)
            end = np.maximum(start_raw, end_raw)

        duration = _clip_unit_interval(end - start)
        center = _clip_unit_interval(0.5 * (start + end))
    elif "duration" in predictions:
        if center is None:
            raise KeyError(
                "Predictions that include 'duration' must also include 'center'."
            )
        duration = _clip_unit_interval(predictions["duration"])
        half_duration = 0.5 * duration
        start = _clip_unit_interval(center - half_duration)
        end = _clip_unit_interval(center + half_duration)

        # Recompute canonical center from the realized interval endpoints.
        center = _clip_unit_interval(0.5 * (start + end))
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
