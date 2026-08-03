"""
Validation plot generation for segmentation and event detection models.

Handles:
- Segmentation validation plots (refactored from existing train_utils functions)
- Event detection validation plots (for MuSSED)
- Attention weight extraction (station and temporal)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import data_utils


def plot_segmentation_validation(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_samples: int = 15,
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
        max_samples: Maximum number of random samples to plot
        class_names: List of class names (default: ["BG", "VT", "LP", "TR", "AV", "IC"])

    Returns:
        Number of plots saved
    """
    if class_names is None:
        class_names = ["BG", "VT", "LP", "TR", "AV", "IC"]

    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    plot_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:
                xb, y_onehot = batch
                _y_idx = None
            else:
                xb, y_onehot, _y_idx = batch

            xb = xb.to(device)
            y_onehot = y_onehot.to(device)

            outputs = model(xb)

            # Convert to CPU for plotting
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

            # Plot each sample in batch
            for sample_idx in range(min(len(xb_np), max_samples - plot_count)):
                plot_segmentation_sample(
                    x=xb_np[sample_idx],
                    y_true=y_np[sample_idx],
                    y_pred=pred_np[sample_idx],
                    output_dir=output_dir,
                    epoch=epoch,
                    sample_id=batch_idx * len(xb_np) + sample_idx,
                    class_names=class_names,
                )
                plot_count += 1

                if plot_count >= max_samples:
                    break

            if plot_count >= max_samples:
                break

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
    pass


def plot_event_validation(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_samples: int = 15,
    class_names: list = None,
    extract_attention: bool = True,
) -> int:
    """
    Generate validation plots for event detection models (MuSSED).

    Args:
        model: Trained event detection model
        dataloader: Validation dataloader
        device: torch.device
        output_dir: Directory to save plots
        epoch: Epoch number
        max_samples: Maximum number of samples to plot
        class_names: List of class names
        extract_attention: Whether to extract and visualize attention weights

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

    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    plot_count = 0

    # TODO: Hook attention modules for extraction
    # station_attention_hook = attach_station_attention_hook(model)
    # temporal_attention_hook = attach_temporal_attention_hook(model)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            xb, y_onehot = batch[0], batch[1]
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)

            # TODO: Forward pass through MuSSED
            # outputs = model(xb)
            # Extract:
            # - class_logits: [B, Nq, C]
            # - centers, starts, ends: [B, Nq]
            # - confidence: [B, Nq]

            # TODO: Extract attention weights if hooks attached
            # station_attn = station_attention_hook.get_attention()  # [B, S] normalized
            # temporal_attn = temporal_attention_hook.get_attention()  # [B, T] normalized

            # TODO: Convert segmentation to events
            # events_gt = segmentation_to_events(y_onehot, normalize=True)

            # Plot each sample
            for sample_idx in range(min(len(xb), max_samples - plot_count)):
                plot_event_sample(
                    x=xb[sample_idx].cpu().numpy(),
                    y_true=y_onehot[sample_idx].cpu().numpy(),
                    outputs=None,  # TODO: Extract from batch
                    output_dir=output_dir,
                    epoch=epoch,
                    sample_id=batch_idx * len(xb) + sample_idx,
                    class_names=class_names,
                    station_attn=None,  # TODO
                    temporal_attn=None,  # TODO
                )
                plot_count += 1

                if plot_count >= max_samples:
                    break

            if plot_count >= max_samples:
                break

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
        x: Input waveforms [S, T]
        y_true: Ground truth segmentation [C, T]
        outputs: Dict with predicted events (class_logits, centers, starts, ends, confidence)
        output_dir: Output directory
        epoch: Epoch number
        sample_id: Sample index
        class_names: List of class names
        station_attn: Station importance weights [S], normalized to [0, 1]
        temporal_attn: Temporal importance weights [T], normalized to [0, 1]

    Plot layout:
        - Panels 0-7: Station waveforms with opacity encoding station attention
        - Panel 8: Ground-truth events (red dashed lines)
        - Panel 9: Predicted events (colored intervals with confidence)
        - Background: Temporal attention encoded as blue saturation
    """
    # TODO: Implement event plot visualization
    # Should show:
    # - 8 waveform panels (stations)
    # - Ground truth events (horizontal bars with class colors)
    # - Predicted events (intervals with confidence alpha)
    # - Station attention: opacity variation per station
    # - Temporal attention: blue background saturation varying by time
    pass


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
    # TODO: Implement attention hook attachment
    # Hook into model.encoder.station_attention_block
    # Extract [B, num_heads, S, S], average over B and num_heads
    # Normalize to [0, 1]
    raise NotImplementedError(
        "Station attention extraction requires MuSSED model structure definition."
    )


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
    # TODO: Implement attention hook attachment
    # Hook into model.encoder.temporal_attention_block
    # Extract [B, num_heads, T, T], average over B and num_heads
    # Normalize to [0, 1]
    raise NotImplementedError(
        "Temporal attention extraction requires MuSSED model structure definition."
    )


class AttentionHook:
    """Helper class to capture attention weights during forward pass."""

    def __init__(self, module):
        self.module = module
        self.attention_weights = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Capture attention weights from module output."""
        # TODO: Implement based on actual attention module structure
        pass

    def get_attention(self) -> Optional[np.ndarray]:
        """Return captured attention weights, normalized to [0, 1]."""
        if self.attention_weights is None:
            return None
        return self.attention_weights

    def detach(self):
        """Remove the hook."""
        self.hook.remove()


def segmentation_to_events(
    y_onehot: torch.Tensor,
    normalize: bool = True,
) -> list:
    """
    Convert segmentation labels to event list.

    Args:
        y_onehot: One-hot segmentation [C, T] or [B, C, T]
        normalize: If True, normalize event times to [0, 1]

    Returns:
        List of events, each with format:
            {"class": int, "start": float, "end": float, "confidence": 1.0}
    """
    # TODO: Implement segmentation-to-events conversion
    # Use existing segmentation_to_events from event_targets.py if available
    # Or implement here
    raise NotImplementedError("segmentation_to_events requires event target utilities.")
