"""
Event target converter: segmentation labels -> event intervals.

Converts one-hot segmentation [C, T] to a list of event intervals (class, start, end, center).
Background class (index 0) is ignored. Consecutive frames of the same class are merged.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class EventInterval(NamedTuple):
    """Represents a single seismic event."""

    class_id: int  # 1-5 (0 is background)
    start_norm: float  # Normalized start time [0, 1]
    end_norm: float  # Normalized end time [0, 1]
    center_norm: float  # Normalized center time [0, 1]
    start_frame: int  # Absolute start frame
    end_frame: int  # Absolute end frame


def segmentation_to_events(
    y_onehot: np.ndarray | torch.Tensor,
    normalize: bool = True,
) -> list[EventInterval]:
    """
    Convert one-hot segmentation to event intervals.

    Args:
        y_onehot: [C, T] one-hot segmentation
                  C=6 (classes 0-5), where 0 is background
                  T: time samples
        normalize: If True, return normalized times [0, 1]. If False, return frame indices.

    Returns:
        List of EventInterval objects, sorted by start_norm/start_frame.
    """
    if isinstance(y_onehot, torch.Tensor):
        y_onehot = y_onehot.cpu().numpy()

    y_onehot = np.asarray(y_onehot, dtype=np.float32)

    if y_onehot.ndim != 2:
        raise ValueError(f"Expected [C, T], got shape {y_onehot.shape}")

    n_classes, t_len = y_onehot.shape

    # Find dominant class at each timestep (argmax), excluding background
    class_per_frame = np.argmax(y_onehot, axis=0)  # [T]

    events: list[EventInterval] = []

    # Scan for class boundaries
    current_class = None
    start_frame = None

    for t in range(t_len):
        c = int(class_per_frame[t])

        if c == 0:  # Background class
            if current_class is not None:
                # End current event
                end_frame = t - 1
                if end_frame >= start_frame:
                    if normalize:
                        start_norm = start_frame / max(1, t_len - 1)
                        end_norm = end_frame / max(1, t_len - 1)
                        center_norm = (start_frame + end_frame) / (
                            2 * max(1, t_len - 1)
                        )
                    else:
                        start_norm = float(start_frame)
                        end_norm = float(end_frame)
                        center_norm = (start_frame + end_frame) / 2.0

                    events.append(
                        EventInterval(
                            class_id=current_class,
                            start_norm=start_norm,
                            end_norm=end_norm,
                            center_norm=center_norm,
                            start_frame=start_frame,
                            end_frame=end_frame,
                        )
                    )
                current_class = None
                start_frame = None

        else:  # Non-background class
            if current_class is None:
                # Start new event
                current_class = c
                start_frame = t
            elif current_class != c:
                # Switch to different class (treat as end-start)
                end_frame = t - 1
                if end_frame >= start_frame:
                    if normalize:
                        start_norm = start_frame / max(1, t_len - 1)
                        end_norm = end_frame / max(1, t_len - 1)
                        center_norm = (start_frame + end_frame) / (
                            2 * max(1, t_len - 1)
                        )
                    else:
                        start_norm = float(start_frame)
                        end_norm = float(end_frame)
                        center_norm = (start_frame + end_frame) / 2.0

                    events.append(
                        EventInterval(
                            class_id=current_class,
                            start_norm=start_norm,
                            end_norm=end_norm,
                            center_norm=center_norm,
                            start_frame=start_frame,
                            end_frame=end_frame,
                        )
                    )

                # Start new event
                current_class = c
                start_frame = t

    # Handle final event
    if current_class is not None:
        end_frame = t_len - 1
        if end_frame >= start_frame:
            if normalize:
                start_norm = start_frame / max(1, t_len - 1)
                end_norm = end_frame / max(1, t_len - 1)
                center_norm = (start_frame + end_frame) / (2 * max(1, t_len - 1))
            else:
                start_norm = float(start_frame)
                end_norm = float(end_frame)
                center_norm = (start_frame + end_frame) / 2.0

            events.append(
                EventInterval(
                    class_id=current_class,
                    start_norm=start_norm,
                    end_norm=end_norm,
                    center_norm=center_norm,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

    return events


def batch_segmentation_to_events(
    y_batch: np.ndarray | torch.Tensor,
    normalize: bool = True,
) -> list[list[EventInterval]]:
    """
    Convert batch of one-hot segmentations to events.

    Args:
        y_batch: [B, C, T] batch of one-hot segmentations
        normalize: If True, return normalized times [0, 1]

    Returns:
        List of lists of EventInterval objects, one list per sample.
    """
    if isinstance(y_batch, torch.Tensor):
        y_batch = y_batch.cpu().numpy()

    y_batch = np.asarray(y_batch, dtype=np.float32)

    if y_batch.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got shape {y_batch.shape}")

    batch_size = y_batch.shape[0]
    return [
        segmentation_to_events(y_batch[b], normalize=normalize)
        for b in range(batch_size)
    ]


def events_to_binary_targets(
    events: list[EventInterval],
    num_frames: int,
) -> np.ndarray:
    """
    Convert event intervals back to binary segmentation for validation.

    Args:
        events: List of EventInterval objects
        num_frames: Number of time frames (T)

    Returns:
        [C, T] binary segmentation (not one-hot, just shows which class at each frame)
    """
    # Use 6 classes (background + 5 event classes)
    n_classes = 6
    binary_seg = np.zeros((n_classes, num_frames), dtype=np.float32)

    for event in events:
        # Convert normalized times to frame indices
        start_frame = int(np.round(event.start_frame))
        end_frame = int(np.round(event.end_frame))

        # Clip to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        # Fill frames
        binary_seg[event.class_id, start_frame : end_frame + 1] = 1.0

    return binary_seg
