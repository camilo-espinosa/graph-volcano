"""
Validation plot generation for segmentation and event detection models.

Handles:
- Segmentation validation plots (refactored from existing train_utils functions)
- Event detection validation plots (for MuSSED)
- Attention weight extraction (station and temporal)
"""

import gc
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import data_utils
from utils.detection_prediction_utils import normalize_prediction_intervals

# Keep delivered attention artifacts compact for plotting diagnostics.
MAX_TEMPORAL_ATTN_POINTS = 16


def plot_segmentation_validation(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    samples_per_class: int = 2,
    class_names: list = None,
) -> int:
    """
    Generate validation plots for segmentation models (2D or 1D).

    Args:
        model: Trained segmentation model
        dataloader: Validation dataloader
        device: torch.device
        output_dir: Directory to save plots
        epoch: Epoch number (for naming)
        samples_per_class: Number of examples to save per non-background class
        class_names: List of class names (default: ["BG", "VT", "LP", "TR", "AV", "IC"])

    Returns:
        Number of plots saved
    """
    if class_names is None:
        class_names = ["BG", "VT", "LP", "TR", "AV", "IC"]

    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    plot_count = 0
    class_counts: dict = {}  # {class_id: n_saved}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:
                xb, y_onehot = batch
                _y_idx = None
            else:
                xb, y_onehot, _y_idx = batch

            xb = xb.to(device)
            outputs = model(xb)

            xb_np = xb.cpu().numpy()
            y_np = y_onehot.cpu().numpy()

            if outputs.ndim == 4:  # 2D output [B, C, H, W]
                pred_np = torch.softmax(outputs, dim=1).cpu().numpy()
            else:  # 1D output [B, C, T] or [B, S, C, T]
                pred_np = (
                    torch.softmax(outputs, dim=1 if outputs.ndim == 3 else 2)
                    .cpu()
                    .numpy()
                )

            for sample_idx in range(len(xb_np)):
                y_sample = y_np[sample_idx]  # [C, T] or [C, H, W]
                sample_classes = [
                    c for c in range(1, y_sample.shape[0]) if y_sample[c].any()
                ]
                if not sample_classes:
                    continue
                if not any(
                    class_counts.get(c, 0) < samples_per_class for c in sample_classes
                ):
                    continue

                plot_segmentation_sample(
                    x=xb_np[sample_idx],
                    y_true=y_sample,
                    y_pred=pred_np[sample_idx],
                    output_dir=output_dir,
                    epoch=epoch,
                    sample_id=batch_idx * len(xb_np) + sample_idx,
                    class_names=class_names,
                )
                plot_count += 1
                for c in sample_classes:
                    class_counts[c] = class_counts.get(c, 0) + 1

            del xb, y_onehot, outputs

    return plot_count


def plot_segmentation_sample(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    epoch: int,
    sample_id: int,
    class_names: list,
):
    """
    Plot a single segmentation sample with ground truth and predictions.

    Args:
        x: Input waveforms [S, T] or [H, W]
        y_true: Ground truth labels [C, T] or [C, H, W]
        y_pred: Predicted probabilities [C, T] or [C, H, W]
        output_dir: Output directory
        epoch: Epoch number
        sample_id: Sample index
        class_names: List of class names
    """
    # TODO: Implement segmentation plot visualization
    # Should show:
    # - Input waveforms (8 stations for 1D, or image for 2D)
    # - Ground truth labels
    # - Raw predictions (softmax activations)
    # - Post-processed predictions
    pass  # Placeholder: Segmentation visualization to be implemented separately


def plot_event_validation(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    samples_per_class: int = 2,
    class_names: list = None,
    extract_attention: bool = True,
    attention_mode: str = "full",
    forward_batch_size: int = 5,
) -> int:
    """
    Generate validation plots for event detection models (MuSSED).

    Args:
        model: Trained event detection model
        dataloader: Validation dataloader
        device: torch.device
        output_dir: Directory to save plots
        epoch: Epoch number
        samples_per_class: Number of examples to save per non-background class
        class_names: List of class names
        extract_attention: Backward-compat flag. If False, attention_mode is forced to "none"
        attention_mode: One of {"none", "station", "full"}
        forward_batch_size: Max plotting forward micro-batch size

    Returns:
        Number of plots saved

    Plot structure (per sample):
        - 8 station waveforms (stacked vertically)
        - Ground-truth events (temporal intervals)
        - Predicted events (temporal intervals with confidence)
        - Station attention heatmap (encoded as panel opacity)
        - Temporal attention heatmap (encoded as color saturation)
    """
    if class_names is None:
        class_names = ["BG", "VT", "LP", "TR", "AV", "IC"]

    if not extract_attention:
        attention_mode = "none"
    allowed_attention_modes = {"none", "station", "full"}
    if attention_mode not in allowed_attention_modes:
        raise ValueError(
            "attention_mode must be one of "
            f"{sorted(allowed_attention_modes)}, got {attention_mode!r}."
        )
    if forward_batch_size < 1:
        raise ValueError(f"forward_batch_size must be >= 1, got {forward_batch_size}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    plot_count = 0
    class_counts: dict = {}  # {class_id: n_saved}
    num_non_bg_classes: int | None = None

    need_station_attention = attention_mode in {"station", "full"}
    need_temporal_attention = attention_mode == "full"

    station_hook = (
        try_attach_station_attention_hook(model) if need_station_attention else None
    )
    temporal_hook = (
        try_attach_temporal_attention_hook(model) if need_temporal_attention else None
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            y_onehot = batch[1]
            if num_non_bg_classes is None:
                num_non_bg_classes = int(y_onehot.shape[1] - 1)

            remaining_classes = [
                c
                for c in range(1, y_onehot.shape[1])
                if class_counts.get(c, 0) < samples_per_class
            ]
            if not remaining_classes:
                break

            # Forward only samples that can still contribute to the remaining quotas.
            selected_indices: list[int] = []
            y_onehot_np = y_onehot.numpy()
            remaining_set = set(remaining_classes)
            for sample_idx in range(y_onehot_np.shape[0]):
                sample_classes = {
                    c
                    for c in range(1, y_onehot_np.shape[1])
                    if y_onehot_np[sample_idx, c].any()
                }
                if sample_classes & remaining_set:
                    selected_indices.append(sample_idx)

            if not selected_indices:
                continue

            for chunk_start in range(0, len(selected_indices), forward_batch_size):
                chunk_indices = selected_indices[
                    chunk_start : chunk_start + forward_batch_size
                ]
                xb = batch[0][chunk_indices].to(device)
                y_onehot_chunk = y_onehot[chunk_indices]

                # Forward pass through MuSSED
                outputs = model(xb)

                station_attn_batch = None
                temporal_attn_batch = None
                if need_station_attention:
                    station_attn_batch = station_hook.get_attention()  # [B, S]
                    if station_attn_batch is None:
                        raise RuntimeError(
                            "Station attention weights were not captured. "
                            "Check station-attention extraction."
                        )
                    if station_attn_batch.shape != (xb.shape[0], xb.shape[1]):
                        raise RuntimeError(
                            "Station attention shape mismatch: expected "
                            f"{(xb.shape[0], xb.shape[1])}, got {station_attn_batch.shape}."
                        )

                if need_temporal_attention:
                    temporal_attn_batch = temporal_hook.get_attention(
                        batch_size=xb.shape[0],
                        n_stations=xb.shape[1],
                        collapse_stations=False,
                    )  # [B, S, T]
                    if temporal_attn_batch is None:
                        raise RuntimeError(
                            "Temporal attention weights were not captured. "
                            "Check temporal-attention extraction."
                        )
                    if temporal_attn_batch.shape[:2] != (xb.shape[0], xb.shape[1]):
                        raise RuntimeError(
                            "Temporal attention shape mismatch: expected [B, S, T] "
                            f"with B={xb.shape[0]}, S={xb.shape[1]}, got {temporal_attn_batch.shape}."
                        )

                # normalized_outputs computed once per chunk
                normalized_outputs = normalize_prediction_intervals(outputs)

                # Plot each sample that contributes to an under-represented class
                for sample_idx in range(len(xb)):
                    y_sample = y_onehot_chunk[sample_idx].numpy()  # [C, T]
                    sample_classes = [
                        c for c in range(1, y_sample.shape[0]) if y_sample[c].any()
                    ]
                    if not sample_classes:
                        continue
                    if not any(
                        class_counts.get(c, 0) < samples_per_class
                        for c in sample_classes
                    ):
                        continue

                    station_attn = (
                        station_attn_batch[sample_idx]
                        if station_attn_batch is not None
                        else None
                    )
                    temporal_attn = (
                        temporal_attn_batch[sample_idx]
                        if temporal_attn_batch is not None
                        else None
                    )

                    outputs_sample = {}
                    key_mapping = {
                        "class_logits": "class_logits",
                        "centers": "center",
                        "starts": "start",
                        "ends": "end",
                        "confidence": "confidence",
                    }
                    for plot_key, model_key in key_mapping.items():
                        if model_key in normalized_outputs:
                            outputs_sample[plot_key] = (
                                normalized_outputs[model_key][sample_idx].cpu().numpy()
                            )

                    plot_event_sample(
                        x=xb[sample_idx].cpu().numpy(),
                        y_true=y_sample,
                        outputs=outputs_sample,
                        output_dir=output_dir,
                        epoch=epoch,
                        sample_id=batch_idx * y_onehot_np.shape[0]
                        + chunk_indices[sample_idx],
                        class_names=class_names,
                        station_attn=station_attn,
                        temporal_attn=temporal_attn,
                    )
                    plot_count += 1
                    for c in sample_classes:
                        class_counts[c] = class_counts.get(c, 0) + 1

                    if num_non_bg_classes is not None and all(
                        class_counts.get(c, 0) >= samples_per_class
                        for c in range(1, num_non_bg_classes + 1)
                    ):
                        break

                del xb, y_onehot_chunk, outputs, station_attn_batch, temporal_attn_batch

                if num_non_bg_classes is not None and all(
                    class_counts.get(c, 0) >= samples_per_class
                    for c in range(1, num_non_bg_classes + 1)
                ):
                    break

            if num_non_bg_classes is not None and all(
                class_counts.get(c, 0) >= samples_per_class
                for c in range(1, num_non_bg_classes + 1)
            ):
                break

    # Clean up hooks
    if station_hook is not None:
        station_hook.detach()
    if temporal_hook is not None:
        temporal_hook.detach()

    # Explicitly release cached CUDA blocks used during attention plotting.
    if device.type == "cuda" and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return plot_count


def plot_event_sample(
    x: np.ndarray,
    y_true: np.ndarray,
    outputs: dict,
    output_dir: Path,
    epoch: int,
    sample_id: int,
    class_names: list,
    station_attn: Optional[np.ndarray] = None,
    temporal_attn: Optional[np.ndarray] = None,
):
    """
    Plot a single event detection sample.

    Args:
        x: Input waveforms [S, T] where S=8 stations
        y_true: Ground truth segmentation [C, T] one-hot
        outputs: Dict with predicted events (class_logits, centers, starts, ends, confidence)
        output_dir: Output directory
        epoch: Epoch number
        sample_id: Sample index
        class_names: List of class names
        station_attn: Station importance weights [S], normalized to [0, 1]
        temporal_attn: Temporal importance weights [T] or [S, T], normalized to [0, 1]

    Plot layout:
        - Panels 0-7: Station waveforms with strong station-attention visual encoding
        - Temporal attention: Blue background heatmap varying by time
        - Ground-truth and predicted event intervals (top/bottom tracks)
        - Final panel: Ground-truth segmentation strip (class per timestep)
    """
    if class_names is None:
        class_names = ["BG", "VT", "LP", "TR", "AV", "IC"]

    # Class colors (skip background)
    # Using hex palette: VT=#df8d5e, LP=#2ca02c, TR=#d62728, AV=#9467bd, IC=#8c564b
    class_colors = {
        0: (0.5, 0.5, 0.5),  # BG: gray
        1: (0.875, 0.553, 0.369),  # VT: #df8d5e (tan/burnt orange)
        2: (0.173, 0.627, 0.173),  # LP: #2ca02c (green)
        3: (0.839, 0.153, 0.157),  # TR: #d62728 (red)
        4: (0.580, 0.404, 0.741),  # AV: #9467bd (purple)
        5: (0.549, 0.337, 0.294),  # IC: #8c564b (brown)
    }

    # Normalize attention weights if provided
    if station_attn is not None:
        station_attn = np.clip(station_attn.astype(float), 0, 1)
    if temporal_attn is not None:
        temporal_attn = np.clip(temporal_attn.astype(float), 0, 1)

    # Extract ground truth events
    from utils.event_targets import segmentation_to_events

    events_gt = segmentation_to_events(torch.from_numpy(y_true), normalize=True)

    # Get time dimension for interpolation
    T = y_true.shape[1]

    # Interpolate temporal attention to match waveform time dimension.
    # Supports a shared [T] trace or station-wise [S, T] traces.
    if temporal_attn is not None:
        if temporal_attn.ndim == 1:
            if len(temporal_attn) != T:
                temporal_attn = np.interp(
                    np.linspace(0, 1, T),
                    np.linspace(0, 1, len(temporal_attn)),
                    temporal_attn,
                )
        elif temporal_attn.ndim == 2:
            if temporal_attn.shape[1] != T:
                temporal_interp = np.zeros(
                    (temporal_attn.shape[0], T), dtype=temporal_attn.dtype
                )
                for s_idx in range(temporal_attn.shape[0]):
                    temporal_interp[s_idx] = np.interp(
                        np.linspace(0, 1, T),
                        np.linspace(0, 1, temporal_attn.shape[1]),
                        temporal_attn[s_idx],
                    )
                temporal_attn = temporal_interp
        else:
            raise ValueError(
                f"temporal_attn must be 1D or 2D, got shape {temporal_attn.shape}."
            )

    # Extract predicted events (if available)
    events_pred = []
    if outputs:
        required_pred_keys = {
            "centers",
            "starts",
            "ends",
            "confidence",
            "class_logits",
        }
        missing_keys = required_pred_keys - set(outputs.keys())
        if missing_keys:
            raise ValueError(
                f"Missing required prediction keys for event plotting: {sorted(missing_keys)}"
            )

        centers = outputs["centers"]  # [Nq]
        starts = outputs["starts"]  # [Nq]
        ends = outputs["ends"]  # [Nq]
        confidence = outputs["confidence"]  # [Nq]
        class_logits = outputs["class_logits"]  # [Nq, C]

        for q_idx in range(len(centers)):
            conf = float(confidence[q_idx])
            if conf > 0.0:
                class_id = (
                    int(np.argmax(class_logits[q_idx]))
                    if class_logits is not None
                    else 1
                )
                if class_id == 0:
                    continue
                events_pred.append(
                    {
                        "class_id": class_id,
                        "start": float(starts[q_idx]),
                        "end": float(ends[q_idx]),
                        "center": float(centers[q_idx]),
                        "confidence": conf,
                    }
                )

    # Create figure
    n_stations = min(x.shape[0], 8)  # Max 8 stations
    fig_height = max(11, 1.4 * n_stations)
    fig, axes = plt.subplots(
        n_stations,
        1,
        figsize=(14, fig_height),
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    station_axes = axes

    # Time axis in seconds (sampling rate = 100 Hz)
    SAMPLING_RATE_HZ = 100.0
    duration_seconds = T / SAMPLING_RATE_HZ
    time_axis = np.linspace(0, duration_seconds, T)

    # Build high-contrast station attention for visibility.
    if station_attn is not None:
        station_vis = station_attn[:n_stations]
        s_min = float(np.min(station_vis))
        s_max = float(np.max(station_vis))
        station_vis = (station_vis - s_min) / (s_max - s_min + 1e-8)
    else:
        station_vis = np.ones(n_stations)

    # Build high-contrast temporal attention map(s).
    temporal_img = None
    temporal_img_by_station = None
    if temporal_attn is not None:
        if temporal_attn.ndim == 1:
            t_min = float(np.min(temporal_attn))
            t_max = float(np.max(temporal_attn))
            temporal_vis = (temporal_attn - t_min) / (t_max - t_min + 1e-8)
            temporal_img = temporal_vis[np.newaxis, :]
        else:
            temporal_img_by_station = []
            for s_idx in range(min(n_stations, temporal_attn.shape[0])):
                s_trace = temporal_attn[s_idx]
                s_min = float(np.min(s_trace))
                s_max = float(np.max(s_trace))
                s_vis = (s_trace - s_min) / (s_max - s_min + 1e-8)
                temporal_img_by_station.append(s_vis[np.newaxis, :])

    # Plot waveforms for each station
    for s_idx in range(n_stations):
        ax = station_axes[s_idx]
        waveform = x[s_idx]
        waveform_sum_abs = float(np.sum(np.abs(waveform)))

        # Use original waveform values (no normalization)
        waveform_norm = waveform.copy()

        # Fixed y-limits: [-1.1, 1.1] for all plots
        # This allows comparing actual amplitudes across stations
        y_lim_min, y_lim_max = -1.1, 1.1

        # Temporal attention as one rasterized heatmap (fast).
        # Station attention modulates heatmap opacity
        if temporal_img is not None or temporal_img_by_station is not None:
            heatmap_alpha = float(
                0.15 + 0.50 * station_vis[s_idx]
            )  # Station attention controls heatmap opacity
            img = (
                temporal_img_by_station[s_idx]
                if temporal_img_by_station is not None
                and s_idx < len(temporal_img_by_station)
                else temporal_img
            )
            ax.imshow(
                img,
                aspect="auto",
                extent=[0, duration_seconds, y_lim_min, y_lim_max],
                cmap="Blues",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                alpha=heatmap_alpha,
                origin="lower",
                zorder=0,
            )

        # Store waveform for drawing on top later (always 100% opacity)
        waveform_to_plot = {
            "time_axis": time_axis,
            "waveform_norm": waveform_norm,
            "lw_station": float(1.0),  # Constant linewidth
        }

        # Compute event box positions relative to y-limits
        # GT boxes: below y=0 center
        # Predicted boxes: above y=0 center
        y_range = y_lim_max - y_lim_min
        y_center = (y_lim_max + y_lim_min) / 2.0
        box_height = y_range * 0.25  # Quarter of visible range

        gt_y_top = y_center  # y=0 equivalent
        gt_y_bottom = gt_y_top - box_height

        pred_y_bottom = y_center  # y=0 equivalent
        pred_y_top = pred_y_bottom + box_height

        # Ground-truth event track (hollow rectangles with border only)
        for event in events_gt:
            if event.class_id > 0:  # Skip background
                color = class_colors.get(event.class_id, (0, 0, 0))
                # Convert normalized coordinates to seconds
                start_sec = event.start_norm * duration_seconds
                end_sec = event.end_norm * duration_seconds
                # Draw hollow rectangle from gt_y_bottom to gt_y_top
                rect = Rectangle(
                    (start_sec, gt_y_bottom),
                    end_sec - start_sec,
                    box_height,
                    linewidth=3.0,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.9,
                    zorder=10,
                )
                ax.add_patch(rect)

        # Predicted event track (hollow rectangles with border only)
        if events_pred:
            for event_pred in events_pred:
                class_id = event_pred["class_id"]
                color = class_colors.get(class_id, (0, 0, 0))
                conf = event_pred["confidence"]
                pred_alpha = float(min(1.0, 0.2 + 0.8 * conf))
                # Convert normalized coordinates to seconds
                start_sec = event_pred["start"] * duration_seconds
                end_sec = event_pred["end"] * duration_seconds
                center_sec = event_pred["center"] * duration_seconds
                # Draw hollow rectangle from pred_y_bottom to pred_y_top
                rect = Rectangle(
                    (start_sec, pred_y_bottom),
                    end_sec - start_sec,
                    box_height,
                    linewidth=3.0,
                    edgecolor=color,
                    facecolor="none",
                    linestyle="--",
                    alpha=pred_alpha,
                    zorder=10,
                )
                ax.add_patch(rect)
                # Center marker at middle of box
                marker_y = (pred_y_bottom + pred_y_top) / 2.0
                ax.plot(
                    center_sec,
                    marker_y,
                    marker="o",
                    color=color,
                    markersize=6,
                    alpha=pred_alpha,
                    zorder=15,
                )

        # Draw waveform on top of everything (after all events/heatmaps)
        # Always 100% opacity (alpha=1.0)
        ax.plot(
            waveform_to_plot["time_axis"],
            waveform_to_plot["waveform_norm"],
            color="black",
            linewidth=waveform_to_plot["lw_station"],
            alpha=1.0,
            zorder=200,
        )

        # Labels and formatting
        ax.set_ylim(y_lim_min, y_lim_max)
        if station_attn is not None:
            ax.set_ylabel(
                f"S{s_idx} a={station_attn[s_idx]:.2f}\nabs={waveform_sum_abs:.3e}",
                fontsize=11,
                fontweight="bold",
                rotation=0,
            )
        else:
            ax.set_ylabel(
                f"S{s_idx}\nabs={waveform_sum_abs:.3e}",
                fontsize=11,
                fontweight="bold",
                rotation=0,
            )
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.grid(True, alpha=0.2, zorder=0)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(0, duration_seconds)
        ax.set_xlabel("Time (seconds)", fontsize=10)

    # Title with metadata
    title_parts = [f"Epoch {epoch} | Fold Sample {sample_id}"]

    # Extract ground truth class names
    if events_gt:
        gt_classes = sorted(
            set(event.class_id for event in events_gt if event.class_id > 0)
        )
        gt_class_names = [
            class_names[c_id] for c_id in gt_classes if c_id < len(class_names)
        ]
        title_parts.append(f"GT Classes: {', '.join(gt_class_names)}")

    # Extract predicted class names
    if events_pred:
        pred_classes = sorted(set(event["class_id"] for event in events_pred))
        pred_class_names = [
            class_names[c_id] for c_id in pred_classes if c_id < len(class_names)
        ]
        title_parts.append(f"Pred Classes: {', '.join(pred_class_names)}")

    if station_attn is not None:
        title_parts.append(
            f"Station attn spread: {float(np.max(station_attn) - np.min(station_attn)):.3f}"
        )
    plt.suptitle(" | ".join(title_parts), fontsize=11, fontweight="bold")

    # X-axis label on last subplot
    axes[-1].set_xlabel("Normalized Time [0-1]", fontsize=10)

    plt.tight_layout()

    # Save
    plot_filename = f"epoch_{epoch:03d}_sample_{sample_id:04d}_events.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=80, bbox_inches="tight")
    plt.close()


def extract_station_attention_weights(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    sample_idx: int = 0,
) -> np.ndarray:
    """
    Extract station attention weights from model.

    Extracts from StationAttentionBlock in MuSSED encoder.

    Args:
        model: MuSSED event detection model
        dataloader: Validation dataloader
        device: torch.device
        sample_idx: Index of sample in first batch to extract for

    Returns:
        Station attention weights [S] normalized to [0, 1]
    """
    hook = try_attach_station_attention_hook(model)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            xb = batch[0].to(device)
            _ = model(xb)
            attn = hook.get_attention()
            hook.detach()
            if attn is not None and len(attn) > sample_idx:
                return attn[sample_idx]
    raise RuntimeError("Station attention weights could not be extracted.")


def extract_temporal_attention_weights(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    sample_idx: int = 0,
) -> np.ndarray:
    """
    Extract temporal attention weights from model.

    Extracts from TemporalBottleneckAttention in MuSSED encoder.

    Args:
        model: MuSSED event detection model
        dataloader: Validation dataloader
        device: torch.device
        sample_idx: Index of sample in first batch to extract for

    Returns:
        Temporal attention weights [T] normalized to [0, 1]
    """
    hook = try_attach_temporal_attention_hook(model)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            xb = batch[0].to(device)
            _ = model(xb)
            attn = hook.get_attention(batch_size=xb.shape[0], n_stations=xb.shape[1])
            hook.detach()
            if attn is not None and len(attn) > sample_idx:
                return attn[sample_idx]
    raise RuntimeError("Temporal attention weights could not be extracted.")


def try_attach_station_attention_hook(
    model: torch.nn.Module,
) -> "StationAttentionHook":
    """
    Attach a hook to the station attention module in MuSSED encoder.
    Fails fast if module not found.

    Returns:
        StationAttentionHook

    Raises:
        RuntimeError: If station attention module cannot be found
    """
    # Try common naming patterns for station attention module
    for module_name in [
        "encoder.station_attention",
        "encoder.station_attention_block",
        "encoder.attention_station",
        "station_attention",
    ]:
        try:
            module = dict(model.named_modules())[module_name]
            return StationAttentionHook(module)
        except KeyError:
            continue
    raise RuntimeError(
        "Station attention module not found in model. "
        "Tried: encoder.station_attention, encoder.station_attention_block, "
        "encoder.attention_station, station_attention"
    )


def try_attach_temporal_attention_hook(
    model: torch.nn.Module,
) -> "TemporalAttentionHook":
    """
    Attach a hook to the temporal attention module in MuSSED encoder.
    Fails fast if module not found.

    Returns:
        TemporalAttentionHook

    Raises:
        RuntimeError: If temporal attention module cannot be found
    """
    # Try common naming patterns for temporal attention module
    for module_name in [
        "encoder.temporal_attention",
        "encoder.temporal_bottleneck_attention",
        "encoder.bottleneck_attention",
        "temporal_attention",
    ]:
        try:
            module = dict(model.named_modules())[module_name]
            return TemporalAttentionHook(module)
        except KeyError:
            continue
    raise RuntimeError(
        "Temporal attention module not found in model. "
        "Tried: encoder.temporal_attention, encoder.temporal_bottleneck_attention, "
        "encoder.bottleneck_attention, temporal_attention"
    )


class AttentionHook:
    """Base helper class to capture attention weights during forward pass."""

    def __init__(self, module):
        self.module = module
        self.attention_weights = None
        if hasattr(module, "capture_attention_weights"):
            module.capture_attention_weights = True
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Capture attention weights from module output. Override in subclasses."""
        pass

    def get_attention(self) -> Optional[np.ndarray]:
        """Return captured attention weights, normalized to [0, 1]."""
        if self.attention_weights is None:
            return None
        return self.attention_weights

    def detach(self):
        """Remove the hook."""
        self.hook.remove()
        if hasattr(self.module, "capture_attention_weights"):
            self.module.capture_attention_weights = False


class StationAttentionHook(AttentionHook):
    """Hook to capture station-level attention weights [B, S]."""

    def _hook_fn(self, module, input, output):
        """
        Capture station attention from attention module.
        Uses StationAttentionBlock.last_attn_weights with shape [B, H, S, S].
        """
        attn = getattr(module, "last_attn_weights", None)
        if attn is None:
            self.attention_weights = None
            return

        if not isinstance(attn, torch.Tensor) or attn.ndim != 4:
            raise RuntimeError(
                "Station attention capture expected tensor [B, H, S, S]."
            )

        attn = attn.detach().cpu().mean(dim=1).mean(dim=1)  # [B, S]
        attn_np = attn.numpy()
        batch_min = attn_np.min(axis=-1, keepdims=True)
        batch_max = attn_np.max(axis=-1, keepdims=True)
        attn_np = (attn_np - batch_min) / (batch_max - batch_min + 1e-8)
        self.attention_weights = np.clip(attn_np, 0.0, 1.0).astype(np.float16)
        module.last_attn_weights = None


class TemporalAttentionHook(AttentionHook):
    """Hook to capture temporal attention weights [B, T]."""

    def _hook_fn(self, module, input, output):
        """
        Capture temporal attention from attention module.
        Uses TemporalBottleneckAttention.last_attn_weights with shape [N, H, T, T],
        where N = B * S.
        """
        attn = getattr(module, "last_attn_weights", None)
        if attn is None:
            self.attention_weights = None
            return

        if not isinstance(attn, torch.Tensor) or attn.ndim not in (3, 4):
            raise RuntimeError(
                "Temporal attention capture expected tensor [N, H, T, T] or [N, T, T]."
            )

        if attn.ndim == 4:
            attn = attn.detach().cpu().mean(dim=1).mean(dim=1)  # [N, T]
        else:
            attn = attn.detach().cpu().mean(dim=1)  # [N, T]
        attn_np = attn.numpy()
        row_min = attn_np.min(axis=-1, keepdims=True)
        row_max = attn_np.max(axis=-1, keepdims=True)
        attn_np = (attn_np - row_min) / (row_max - row_min + 1e-8)
        attn_np = np.clip(attn_np, 0.0, 1.0)

        # Downsample temporal attention traces for compact delivery to plotting.
        if (
            isinstance(MAX_TEMPORAL_ATTN_POINTS, int)
            and MAX_TEMPORAL_ATTN_POINTS > 0
            and attn_np.shape[1] > MAX_TEMPORAL_ATTN_POINTS
        ):
            idx = np.linspace(
                0,
                attn_np.shape[1] - 1,
                num=MAX_TEMPORAL_ATTN_POINTS,
                dtype=np.int64,
            )
            attn_np = attn_np[:, idx]

        self.attention_weights = attn_np.astype(np.float16)
        module.last_attn_weights = None

    def get_attention(
        self,
        batch_size: Optional[int] = None,
        n_stations: Optional[int] = None,
        collapse_stations: bool = True,
    ) -> Optional[np.ndarray]:
        attn = super().get_attention()
        if attn is None:
            return None

        if batch_size is None or n_stations is None:
            return attn

        expected_rows = batch_size * n_stations
        if attn.shape[0] != expected_rows:
            raise RuntimeError(
                f"Temporal attention row mismatch: expected {expected_rows}, got {attn.shape[0]}."
            )

        attn_by_station = attn.reshape(batch_size, n_stations, attn.shape[1])
        if collapse_stations:
            return attn_by_station.mean(axis=1)
        return attn_by_station
