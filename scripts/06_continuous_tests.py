"""Continuous 10-hour inference benchmark for all trained ablations.

This script runs sliding-window inference on the NVCHVC 10-hour continuous trace
for every ablation present in an experiment folder and every available fold
checkpoint. It supports:

- 1D segmentation models (PhaseNet and MuSSeg variants)
- 2D segmentation models (UNet variants)
- Event-detection models (MuSSED variants)

Outputs include per-fold raw/clean detections, event-level evaluation files,
and stride/model aggregate summaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.active_eval_utils import load_checkpoint_into_model
from utils.data_utils import activation_unstacking, patch_stacking_X
from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.fold_io_utils import checkpoint_path_for_fold
from utils.model_registry import MODEL_SPECS, build_model_from_spec
from utils.script_common import parse_csv_selection, resolve_project_path
from utils.train_utils import cleanup_gpu_cache

CLASS_ID_TO_NAME = {
    1: "VT",
    2: "LP",
    3: "TR",
    4: "AV",
    5: "IC",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}
EVENT_CLASSES = ["VT", "LP", "TR", "AV", "IC"]

RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "complete_experiment"
DEFAULT_CONTINUOUS_NPY = (
    PROJECT_ROOT
    / "data"
    / "NVCh_10h_continuous_trace"
    / "NVCh_10h_continuous_trace.npy"
)
DEFAULT_REFERENCE_CSV = (
    PROJECT_ROOT
    / "data"
    / "NVCh_10h_continuous_trace"
    / "NVCh_10h_continuous_trace_reference.csv"
)
DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_DET_MIN_DURATION_SEC = "VT:5,LP:7,TR:40,AV:7,IC:2"
DEFAULT_DET_MAX_DURATION_SEC = "VT:45,LP:80,TR:360,AV:110,IC:25"
DEFAULT_DET_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_WINDOW_RSAM_THRESHOLD = 5.0
DEFAULT_EVENT_RSAM_THRESHOLD = 14.10

def log_stage(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run continuous 10-hour sliding-window inference for all ablations in "
            "an experiment folder, across multiple stride values."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Experiment root containing the 'ablations' folder.",
    )
    parser.add_argument(
        "--continuous-npy",
        type=Path,
        default=DEFAULT_CONTINUOUS_NPY,
        help="Path to NVCh_10h continuous trace .npy file.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=DEFAULT_REFERENCE_CSV,
        help="Reference event CSV with columns event_type, idx_start, idx_end.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <experiment-root>/continuous_tests. "
            "Results are grouped by stride/model/fold."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to run. Default: all ablations in folder.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated folds to run (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Sliding window size in samples (default: 8192).",
    )
    parser.add_argument(
        "--strides",
        type=str,
        default="2048",
        help="Comma-separated stride values in samples, e.g. 1024,2048,4096.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help=(
            "Batch size override. When omitted, each model uses its registry batch_size."
        ),
    )

    # Segmentation decoding defaults (100 Hz data).
    parser.add_argument(
        "--seg-min-event-len-samples",
        type=int,
        default=100,
        help="Minimum decoded segmentation event duration in samples.",
    )
    parser.add_argument(
        "--seg-max-bg-hole-samples",
        type=int,
        default=25,
        help="Fill background holes up to this size if same class on both sides.",
    )

    # Event-detection decode / cleaning.
    parser.add_argument(
        "--enable-window-rsam-filter",
        action="store_true",
        help=(
            "Enable RSAM-based window filtering before model inference. Disabled by "
            "default."
        ),
    )
    parser.add_argument(
        "--window-rsam-threshold",
        type=float,
        default=DEFAULT_WINDOW_RSAM_THRESHOLD,
        help=(
            "RSAM threshold for window-level filtering (used only when "
            "--enable-window-rsam-filter is set)."
        ),
    )
    parser.add_argument(
        "--event-rsam-threshold",
        type=float,
        default=DEFAULT_EVENT_RSAM_THRESHOLD,
        help=(
            "RSAM threshold for event-level filtering. Events below this threshold "
            "are discarded before merge/post-processing."
        ),
    )
    parser.add_argument(
        "--cross-class-overlap-threshold",
        type=float,
        default=0.9,
        help=(
            "If two different-class detections overlap with IoU >= threshold and both "
            "have confidence, keep only the highest-confidence one."
        ),
    )
    parser.add_argument(
        "--cross-class-confidence-margin",
        type=float,
        default=0.02,
        help=(
            "Minimum confidence gap required to suppress one class when cross-class "
            "events overlap. Larger values make suppression less aggressive."
        ),
    )
    parser.add_argument(
        "--cross-class-max-duration-ratio",
        type=float,
        default=2.5,
        help=(
            "Do not suppress overlapping cross-class events when duration ratio "
            "(longer/shorter) exceeds this value."
        ),
    )
    parser.add_argument(
        "--det-class-min-duration-sec",
        type=str,
        default=DEFAULT_DET_MIN_DURATION_SEC,
        help=(
            "Minimum duration (seconds) per class for event-detection decode. "
            "Format: VT:5,LP:7,TR:40,AV:7,IC:2"
        ),
    )
    parser.add_argument(
        "--det-class-max-duration-sec",
        type=str,
        default=DEFAULT_DET_MAX_DURATION_SEC,
        help=(
            "Maximum duration (seconds) per class for event-detection decode. "
            "Format: VT:45,LP:80,TR:360,AV:110,IC:25"
        ),
    )

    # Evaluation matching.
    parser.add_argument(
        "--matching-strategy",
        type=str,
        default="dual",
        choices=["iou", "overlap_recall", "dual"],
        help="Event matching strategy for evaluation.",
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold used by matching strategy.",
    )
    parser.add_argument(
        "--overlap-recall-threshold",
        type=float,
        default=0.9,
        help="Target coverage threshold used by overlap_recall/dual matching.",
    )
    parser.add_argument(
        "--debug-plot-windows",
        action="store_true",
        help=(
            "Save one debug plot per window (normalized traces + GT/pred overlays) to inspect model inputs. "
            "This is intended for temporary debugging and can be very heavy."
        ),
    )
    return parser.parse_args()


def parse_int_csv(raw_value: str, *, name: str) -> list[int]:
    values = [x.strip() for x in raw_value.split(",") if x.strip()]
    if len(values) == 0:
        raise ValueError(f"No values parsed for {name}.")
    parsed: list[int] = []
    for value in values:
        try:
            parsed_value = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {name}: {value}") from exc
        if parsed_value <= 0:
            raise ValueError(f"{name} values must be > 0, got {parsed_value}.")
        if parsed_value not in parsed:
            parsed.append(parsed_value)
    return parsed


def parse_class_float_map(raw_value: str, *, name: str) -> dict[str, float]:
    pairs = [x.strip() for x in raw_value.split(",") if x.strip()]
    if len(pairs) == 0:
        raise ValueError(f"No class:value pairs parsed for {name}.")

    out: dict[str, float] = {}
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(
                f"Invalid token in {name}: {pair!r}. Expected CLASS:VALUE format."
            )
        key_raw, value_raw = pair.split(":", maxsplit=1)
        key = key_raw.strip().upper()
        if key not in CLASS_NAME_TO_ID:
            raise ValueError(
                f"Unknown class {key!r} in {name}. Expected one of {EVENT_CLASSES}."
            )
        try:
            value = float(value_raw.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid float in {name}: {pair!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} values must be > 0, got {pair!r}.")
        out[key] = float(value)

    missing = [c for c in EVENT_CLASSES if c not in out]
    if missing:
        raise ValueError(
            f"Missing classes in {name}: {missing}. Required classes: {EVENT_CLASSES}."
        )
    return out


def discover_ablation_keys(experiment_root: Path) -> list[str]:
    ablations_root = experiment_root / "ablations"
    if not ablations_root.exists():
        raise FileNotFoundError(f"Ablations folder not found: {ablations_root}")

    discovered = sorted([p.name for p in ablations_root.iterdir() if p.is_dir()])
    if len(discovered) == 0:
        raise RuntimeError(f"No ablation folders found under: {ablations_root}")

    unknown = sorted([k for k in discovered if k not in MODEL_SPECS])
    if len(unknown) > 0:
        raise RuntimeError(
            "Ablation folders found that are missing in model registry: "
            f"{unknown}. Add them to utils/model_registry.py first."
        )
    return discovered


def parse_folds(raw_folds: str) -> list[int]:
    folds = parse_int_csv(raw_folds, name="--folds")
    valid = {1, 2, 3, 4, 5}
    if any(f not in valid for f in folds):
        raise ValueError(f"Folds must be in [1..5], got {folds}.")
    return folds


def ensure_detection_df_schema(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["class", "idx_start", "idx_end", "confidence"]
    if df.empty:
        return pd.DataFrame(columns=expected)
    out = df.copy()
    for col in expected:
        if col not in out.columns:
            out[col] = np.nan
    return out[expected]


def build_window_starts(total_len: int, window_size: int, stride: int) -> np.ndarray:
    if total_len < window_size:
        raise ValueError(
            f"Continuous trace length {total_len} is shorter than window size {window_size}."
        )
    starts = list(range(0, total_len - window_size + 1, stride))
    last_start = total_len - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return np.asarray(starts, dtype=np.int64)


def normalize_windows_global_max_abs(batch_x: np.ndarray) -> np.ndarray:
    """Normalize each window by its max absolute value across all stations/time."""
    if batch_x.ndim != 3:
        raise ValueError(f"Expected batch_x with shape [B, S, T], got {batch_x.shape}.")
    denom = np.max(np.abs(batch_x), axis=(1, 2), keepdims=True)
    # Keep all-zero windows unchanged while avoiding division by zero.
    denom = np.where(denom > 0.0, denom, 1.0)
    return batch_x / denom


def bandpass_windows_butterworth(
    batch_x: np.ndarray,
    *,
    low_hz: float = 1.0,
    high_hz: float = 15.0,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass to each station trace in each window."""
    if batch_x.ndim != 3:
        raise ValueError(f"Expected batch_x with shape [B, S, T], got {batch_x.shape}.")
    nyquist = 0.5 * float(sample_rate_hz)
    if not (0.0 < float(low_hz) < float(high_hz) < nyquist):
        raise ValueError(
            "Invalid bandpass limits: "
            f"low_hz={low_hz}, high_hz={high_hz}, nyquist={nyquist}."
        )

    sos = butter(
        int(order),
        [float(low_hz), float(high_hz)],
        btype="bandpass",
        fs=float(sample_rate_hz),
        output="sos",
    )
    filtered = sosfiltfilt(sos, batch_x, axis=-1)
    return filtered.astype(np.float32, copy=False)


def save_window_debug_plot(
    *,
    normalized_window: np.ndarray,
    window_start: int,
    output_dir: Path,
    gt_events_local: list[dict[str, float | int | str]] | None = None,
    pred_events_local: list[dict[str, float | int | str]] | None = None,
) -> None:
    """Save a per-window debug plot with one normalized row per station."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:  # pragma: no cover - debug-only path
        raise RuntimeError(
            "matplotlib is required for --debug-plot-windows. "
            "Install it with: pip install matplotlib"
        ) from exc

    station_count, time_len = normalized_window.shape
    fig, axes = plt.subplots(station_count, 1, figsize=(14, max(6, 1.8 * station_count)), sharex=True)
    if station_count == 1:
        axes = [axes]
    time_axis = np.arange(time_len, dtype=np.int32)

    for station_idx in range(station_count):
        axes[station_idx].plot(
            time_axis,
            normalized_window[station_idx],
            linewidth=1.0,
            alpha=1.0,
            color="black",
            zorder=200,
        )
        axes[station_idx].set_ylabel(f"S{station_idx + 1}")
        axes[station_idx].grid(True, alpha=0.25)

    gt_events_local = gt_events_local or []
    pred_events_local = pred_events_local or []

    # Match validation_plots.py class colors and event-box semantics.
    class_colors = {
        0: (0.5, 0.5, 0.5),
        1: (0.875, 0.553, 0.369),
        2: (0.173, 0.627, 0.173),
        3: (0.839, 0.153, 0.157),
        4: (0.580, 0.404, 0.741),
        5: (0.549, 0.337, 0.294),
    }

    y_lim_min, y_lim_max = -1.1, 1.1
    y_range = y_lim_max - y_lim_min
    y_center = (y_lim_max + y_lim_min) / 2.0
    box_height = y_range * 0.25
    gt_y_top = y_center
    gt_y_bottom = gt_y_top - box_height
    pred_y_bottom = y_center
    pred_y_top = pred_y_bottom + box_height

    for ax in axes:
        # Ground-truth event track: hollow rectangles with solid border.
        for event in gt_events_local:
            cls = str(event.get("class", ""))
            class_id = int(CLASS_NAME_TO_ID.get(cls, 0))
            color = class_colors.get(class_id, (0.0, 0.0, 0.0))
            s = int(event["idx_start"])
            e = int(event["idx_end"])
            rect = Rectangle(
                (s, gt_y_bottom),
                max(1, e - s + 1),
                box_height,
                linewidth=3.0,
                edgecolor=color,
                facecolor="none",
                alpha=0.9,
                zorder=10,
            )
            ax.add_patch(rect)

        # Predicted event track: hollow dashed rectangles + center marker.
        for event in pred_events_local:
            cls = str(event.get("class", ""))
            class_id = int(CLASS_NAME_TO_ID.get(cls, 0))
            color = class_colors.get(class_id, (0.0, 0.0, 0.0))
            conf = event.get("confidence", np.nan)
            conf_val = float(conf) if pd.notna(conf) else 0.5
            pred_alpha = float(min(1.0, 0.2 + 0.8 * conf_val))
            s = int(event["idx_start"])
            e = int(event["idx_end"])
            rect = Rectangle(
                (s, pred_y_bottom),
                max(1, e - s + 1),
                box_height,
                linewidth=3.0,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                alpha=pred_alpha,
                zorder=10,
            )
            ax.add_patch(rect)

            center_x = 0.5 * (s + e)
            marker_y = 0.5 * (pred_y_bottom + pred_y_top)
            ax.plot(
                center_x,
                marker_y,
                marker="o",
                color=color,
                markersize=5,
                alpha=pred_alpha,
                zorder=15,
            )

        ax.set_ylim(y_lim_min, y_lim_max)

    norm_max_abs = (
        float(np.max(np.abs(normalized_window))) if normalized_window.size else 0.0
    )

    axes[0].set_title(
        f"Normalized window start={window_start} (global max_abs={norm_max_abs:.6g})"
    )
    axes[0].text(
        0.01,
        0.98,
        f"GT={len(gt_events_local)} | Pred={len(pred_events_local)}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    axes[-1].set_xlabel("Sample index in window")

    output_path = output_dir / f"window_{window_start:07d}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _fill_small_background_holes(
    class_idx: np.ndarray,
    max_bg_hole_samples: int,
) -> np.ndarray:
    out = class_idx.copy()
    t = 0
    n = len(out)
    while t < n:
        if out[t] != 0:
            t += 1
            continue

        hole_start = t
        while t < n and out[t] == 0:
            t += 1
        hole_end = t - 1
        hole_len = hole_end - hole_start + 1

        if hole_len > max_bg_hole_samples:
            continue
        if hole_start == 0 or hole_end == n - 1:
            continue

        left_class = int(out[hole_start - 1])
        right_class = int(out[hole_end + 1])
        if left_class > 0 and left_class == right_class:
            out[hole_start : hole_end + 1] = left_class

    return out


def decode_segmentation_events(
    logits: torch.Tensor,
    *,
    min_event_len_samples: int,
    max_bg_hole_samples: int,
) -> list[list[dict[str, float | int | str]]]:
    probs = torch.softmax(logits, dim=1)
    # Event-only decode for segmentation baselines: pick among VT/LP/TR/AV/IC
    # and rely on duration + RSAM postprocessing for no-event rejection.
    pred_idx = (torch.argmax(probs[:, 1:, :], dim=1) + 1).detach().cpu().numpy()  # [B, T]

    batch_events: list[list[dict[str, float | int | str]]] = []
    for sample_idx in range(pred_idx.shape[0]):
        class_idx = pred_idx[sample_idx]

        events: list[dict[str, float | int | str]] = []
        t = 0
        n = int(class_idx.shape[0])
        while t < n:
            cls = int(class_idx[t])
            if cls == 0:
                t += 1
                continue

            start = t
            while t < n and int(class_idx[t]) == cls:
                t += 1
            end = t - 1

            duration = end - start + 1
            if duration < int(min_event_len_samples):
                continue

            if cls in CLASS_ID_TO_NAME:
                events.append(
                    {
                        "class": CLASS_ID_TO_NAME[cls],
                        "idx_start": int(start),
                        "idx_end": int(end),
                        "confidence": np.nan,
                    }
                )

        batch_events.append(events)
    return batch_events


def decode_event_detection_events(
    predictions: dict[str, torch.Tensor],
    *,
    window_size: int,
    confidence_threshold: float,
    min_duration_samples_by_class: dict[str, int],
    max_duration_samples_by_class: dict[str, int],
) -> list[list[dict[str, float | int | str]]]:
    pred = normalize_prediction_intervals(predictions)

    class_logits = pred["class_logits"]  # [B, Nq, C]
    confidence_logits = pred.get("confidence_logits")
    starts = pred["start"]  # [B, Nq, 1]
    ends = pred["end"]  # [B, Nq, 1]

    probs = torch.softmax(class_logits, dim=-1)
    batch_size, num_queries, _ = probs.shape

    out: list[list[dict[str, float | int | str]]] = []
    for b in range(batch_size):
        sample_events: list[dict[str, float | int | str]] = []
        for q in range(num_queries):
            if confidence_logits is not None:
                conf_logit = torch.clamp(confidence_logits[b, q, 0], -60.0, 60.0)
                conf_prob = float(torch.sigmoid(conf_logit).item())
            else:
                conf_prob = float(1.0 - probs[b, q, 0].item())

            # Low confidence => treat query as background/no-event.
            if conf_prob < float(confidence_threshold):
                continue

            # Event class is selected among non-background classes only.
            cls = int(torch.argmax(probs[b, q, 1:]).item()) + 1

            if cls not in CLASS_ID_TO_NAME:
                continue

            class_name = CLASS_ID_TO_NAME[cls]
            conf = float(conf_prob * float(probs[b, q, cls].item()))
            start_norm = float(torch.clamp(starts[b, q, 0], 0.0, 1.0).item())
            end_norm = float(torch.clamp(ends[b, q, 0], 0.0, 1.0).item())
            if start_norm > end_norm:
                start_norm, end_norm = end_norm, start_norm

            start_idx = int(round(start_norm * max(1, window_size - 1)))
            end_idx = int(round(end_norm * max(1, window_size - 1)))
            start_idx = max(0, min(start_idx, window_size - 1))
            end_idx = max(0, min(end_idx, window_size - 1))
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            duration_samples = int(end_idx - start_idx + 1)
            min_len = int(min_duration_samples_by_class[class_name])
            max_len = int(max_duration_samples_by_class[class_name])
            if duration_samples < min_len or duration_samples > max_len:
                continue

            sample_events.append(
                {
                    "class": class_name,
                    "idx_start": int(start_idx),
                    "idx_end": int(end_idx),
                    "confidence": conf,
                }
            )
        out.append(sample_events)
    return out


def merge_same_class_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return ensure_detection_df_schema(df)

    merged_rows: list[dict[str, float | int | str]] = []
    for class_name, group in df.groupby("class", sort=True):
        group_sorted = group.sort_values(by=["idx_start", "idx_end"]).reset_index(drop=True)
        cur_start = int(group_sorted.loc[0, "idx_start"])
        cur_end = int(group_sorted.loc[0, "idx_end"])
        cur_conf = group_sorted.loc[0, "confidence"]

        for i in range(1, len(group_sorted)):
            row = group_sorted.loc[i]
            s = int(row["idx_start"])
            e = int(row["idx_end"])
            c = row["confidence"]

            if s <= cur_end:
                cur_end = max(cur_end, e)
                if pd.notna(c):
                    if pd.isna(cur_conf):
                        cur_conf = float(c)
                    else:
                        cur_conf = float(max(float(cur_conf), float(c)))
            else:
                merged_rows.append(
                    {
                        "class": class_name,
                        "idx_start": int(cur_start),
                        "idx_end": int(cur_end),
                        "confidence": cur_conf,
                    }
                )
                cur_start, cur_end, cur_conf = s, e, c

        merged_rows.append(
            {
                "class": class_name,
                "idx_start": int(cur_start),
                "idx_end": int(cur_end),
                "confidence": cur_conf,
            }
        )

    return ensure_detection_df_schema(pd.DataFrame(merged_rows))


def _event_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _event_overlap_recall(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    b_len = b_end - b_start + 1
    if b_len <= 0:
        return 0.0
    return float(inter / b_len)


def resolve_cross_class_conflicts(
    df: pd.DataFrame,
    *,
    overlap_threshold: float,
    confidence_margin: float,
    max_duration_ratio: float,
) -> pd.DataFrame:
    if df.empty:
        return ensure_detection_df_schema(df)

    # Only apply when confidence is available (detection models).
    if not df["confidence"].notna().any():
        return ensure_detection_df_schema(df)

    events = df.sort_values(by=["idx_start", "idx_end"]).reset_index(drop=True)
    drop = np.zeros(len(events), dtype=bool)

    for i in range(len(events)):
        if drop[i]:
            continue
        ci = str(events.loc[i, "class"])
        si = int(events.loc[i, "idx_start"])
        ei = int(events.loc[i, "idx_end"])
        confi = events.loc[i, "confidence"]
        if pd.isna(confi):
            continue

        for j in range(i + 1, len(events)):
            if drop[j]:
                continue
            sj = int(events.loc[j, "idx_start"])
            if sj > ei:
                break

            cj = str(events.loc[j, "class"])
            if cj == ci:
                continue

            confj = events.loc[j, "confidence"]
            if pd.isna(confj):
                continue

            ej = int(events.loc[j, "idx_end"])
            iou = _event_iou(si, ei, sj, ej)
            if iou < float(overlap_threshold):
                continue

            dur_i = max(1, ei - si + 1)
            dur_j = max(1, ej - sj + 1)
            duration_ratio = max(dur_i, dur_j) / max(1, min(dur_i, dur_j))
            if duration_ratio > float(max_duration_ratio):
                continue

            if abs(float(confi) - float(confj)) < float(confidence_margin):
                continue

            if float(confi) >= float(confj):
                drop[j] = True
            else:
                drop[i] = True
                break

    out = events.loc[~drop].copy()
    return ensure_detection_df_schema(out)


def postprocess_detections(
    raw_df: pd.DataFrame,
    *,
    cross_class_overlap_threshold: float,
    cross_class_confidence_margin: float,
    cross_class_max_duration_ratio: float,
) -> pd.DataFrame:
    merged = merge_same_class_overlaps(raw_df)
    cleaned = resolve_cross_class_conflicts(
        merged,
        overlap_threshold=cross_class_overlap_threshold,
        confidence_margin=cross_class_confidence_margin,
        max_duration_ratio=cross_class_max_duration_ratio,
    )
    if cleaned.empty:
        return cleaned
    return ensure_detection_df_schema(
        cleaned.sort_values(by=["idx_start", "idx_end", "class"]).reset_index(drop=True)
    )


def _is_match(
    pred_start: int,
    pred_end: int,
    gt_start: int,
    gt_end: int,
    *,
    strategy: str,
    match_iou_threshold: float,
    overlap_recall_threshold: float,
) -> tuple[bool, float, float]:
    iou = _event_iou(int(pred_start), int(pred_end), int(gt_start), int(gt_end))
    overlap = _event_overlap_recall(
        int(pred_start), int(pred_end), int(gt_start), int(gt_end)
    )

    if strategy == "iou":
        ok = iou >= float(match_iou_threshold)
    elif strategy == "overlap_recall":
        ok = overlap >= float(overlap_recall_threshold)
    elif strategy == "dual":
        ok = (iou >= float(match_iou_threshold)) or (
            overlap >= float(overlap_recall_threshold)
        )
    else:
        raise ValueError(f"Unknown matching strategy: {strategy}")

    return bool(ok), float(iou), float(overlap)


def _confidence_sort_value(value: float) -> float:
    if pd.isna(value):
        return -1.0
    return float(value)


def evaluate_event_detections(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    matching_strategy: str,
    match_iou_threshold: float,
    overlap_recall_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    pred_df = ensure_detection_df_schema(pred_df)
    gt_df = ensure_detection_df_schema(gt_df)

    pair_rows: list[dict[str, float | int | str]] = []
    per_class_metrics: dict[str, dict[str, float]] = {}

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_matched_ious: list[float] = []

    for class_name in EVENT_CLASSES:
        pred_class = pred_df[pred_df["class"] == class_name].copy()
        gt_class = gt_df[gt_df["class"] == class_name].copy()

        pred_class = pred_class.sort_values(
            by=["confidence", "idx_start"],
            ascending=[False, True],
            key=lambda col: col.map(_confidence_sort_value)
            if col.name == "confidence"
            else col,
        ).reset_index(drop=True)
        gt_class = gt_class.sort_values(by=["idx_start", "idx_end"]).reset_index(drop=True)

        gt_used = np.zeros(len(gt_class), dtype=bool)
        tp = 0
        fp = 0
        matched_ious: list[float] = []

        for p_idx in range(len(pred_class)):
            p_row = pred_class.loc[p_idx]
            ps = int(p_row["idx_start"])
            pe = int(p_row["idx_end"])
            pconf = p_row["confidence"]

            best_g = -1
            best_iou = -1.0
            best_overlap = 0.0
            for g_idx in range(len(gt_class)):
                if gt_used[g_idx]:
                    continue
                g_row = gt_class.loc[g_idx]
                ok, iou_val, overlap_val = _is_match(
                    ps,
                    pe,
                    int(g_row["idx_start"]),
                    int(g_row["idx_end"]),
                    strategy=matching_strategy,
                    match_iou_threshold=match_iou_threshold,
                    overlap_recall_threshold=overlap_recall_threshold,
                )
                if not ok:
                    continue
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_overlap = overlap_val
                    best_g = g_idx

            if best_g >= 0:
                gt_used[best_g] = True
                tp += 1
                matched_ious.append(float(best_iou))
                g_row = gt_class.loc[best_g]
                gs = int(g_row["idx_start"])
                ge = int(g_row["idx_end"])
                pair_rows.append(
                    {
                        "status": "TP",
                        "class": class_name,
                        "pred_idx_start": ps,
                        "pred_idx_end": pe,
                        "pred_duration": int(pe - ps + 1),
                        "pred_confidence": pconf,
                        "gt_idx_start": gs,
                        "gt_idx_end": ge,
                        "gt_duration": int(ge - gs + 1),
                        "temporal_iou": float(best_iou),
                        "overlap_recall": float(best_overlap),
                        "start_error": int(ps - gs),
                        "end_error": int(pe - ge),
                        "duration_error": int((pe - ps) - (ge - gs)),
                    }
                )
            else:
                fp += 1
                pair_rows.append(
                    {
                        "status": "FP",
                        "class": class_name,
                        "pred_idx_start": ps,
                        "pred_idx_end": pe,
                        "pred_duration": int(pe - ps + 1),
                        "pred_confidence": pconf,
                        "gt_idx_start": np.nan,
                        "gt_idx_end": np.nan,
                        "gt_duration": np.nan,
                        "temporal_iou": 0.0,
                        "overlap_recall": 0.0,
                        "start_error": np.nan,
                        "end_error": np.nan,
                        "duration_error": np.nan,
                    }
                )

        fn = int((~gt_used).sum())
        for g_idx in np.where(~gt_used)[0]:
            g_row = gt_class.loc[int(g_idx)]
            gs = int(g_row["idx_start"])
            ge = int(g_row["idx_end"])
            pair_rows.append(
                {
                    "status": "FN",
                    "class": class_name,
                    "pred_idx_start": np.nan,
                    "pred_idx_end": np.nan,
                    "pred_duration": np.nan,
                    "pred_confidence": np.nan,
                    "gt_idx_start": gs,
                    "gt_idx_end": ge,
                    "gt_duration": int(ge - gs + 1),
                    "temporal_iou": 0.0,
                    "overlap_recall": 0.0,
                    "start_error": np.nan,
                    "end_error": np.nan,
                    "duration_error": np.nan,
                }
            )

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0

        per_class_metrics[class_name] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": mean_iou,
        }

        global_tp += int(tp)
        global_fp += int(fp)
        global_fn += int(fn)
        global_matched_ious.extend(matched_ious)

    global_precision = (
        float(global_tp / (global_tp + global_fp)) if (global_tp + global_fp) > 0 else 0.0
    )
    global_recall = (
        float(global_tp / (global_tp + global_fn)) if (global_tp + global_fn) > 0 else 0.0
    )
    global_f1 = (
        float(2.0 * global_precision * global_recall / (global_precision + global_recall))
        if (global_precision + global_recall) > 0
        else 0.0
    )
    global_iou = (
        float(np.mean(global_matched_ious)) if len(global_matched_ious) > 0 else 0.0
    )

    macro_f1 = float(np.mean([per_class_metrics[c]["f1"] for c in EVENT_CLASSES]))
    macro_iou = float(np.mean([per_class_metrics[c]["iou"] for c in EVENT_CLASSES]))
    macro_precision = float(
        np.mean([per_class_metrics[c]["precision"] for c in EVENT_CLASSES])
    )
    macro_recall = float(np.mean([per_class_metrics[c]["recall"] for c in EVENT_CLASSES]))

    metrics: dict[str, float] = {
        "tp": float(global_tp),
        "fp": float(global_fp),
        "fn": float(global_fn),
        "precision": global_precision,
        "recall": global_recall,
        "f1": global_f1,
        "iou": global_iou,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
        "n_predictions": float(len(pred_df)),
        "n_reference": float(len(gt_df)),
    }

    for class_name in EVENT_CLASSES:
        stats = per_class_metrics[class_name]
        metrics[f"precision_{class_name}"] = float(stats["precision"])
        metrics[f"recall_{class_name}"] = float(stats["recall"])
        metrics[f"f1_{class_name}"] = float(stats["f1"])
        metrics[f"iou_{class_name}"] = float(stats["iou"])

    pairs_df = pd.DataFrame(pair_rows)
    return metrics, pairs_df


def load_reference_events(reference_csv: Path) -> pd.DataFrame:
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")

    ref = pd.read_csv(reference_csv)
    required = {"event_type", "idx_start", "idx_end"}
    if not required.issubset(ref.columns):
        raise ValueError(
            f"Reference CSV must contain columns {sorted(required)}, "
            f"got {list(ref.columns)}"
        )

    rows = []
    for _, row in ref.iterrows():
        class_name = str(row["event_type"]).strip()
        if class_name not in CLASS_NAME_TO_ID:
            continue
        rows.append(
            {
                "class": class_name,
                "idx_start": int(row["idx_start"]),
                "idx_end": int(row["idx_end"]),
                "confidence": np.nan,
            }
        )
    return ensure_detection_df_schema(pd.DataFrame(rows))


def build_window_event_table(
    *,
    starts: np.ndarray,
    window_size: int,
    events_df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Map absolute events to overlapping windows with local window coordinates."""
    rows: list[dict[str, float | int | str]] = []
    events = ensure_detection_df_schema(events_df)

    for window_idx, window_start in enumerate(starts.tolist()):
        w_start = int(window_start)
        w_end = int(w_start + window_size - 1)

        if events.empty:
            continue

        overlaps = events[
            (events["idx_end"].astype(int) >= w_start)
            & (events["idx_start"].astype(int) <= w_end)
        ]

        for _, event in overlaps.iterrows():
            e_start = int(event["idx_start"])
            e_end = int(event["idx_end"])
            clipped_start = max(e_start, w_start)
            clipped_end = min(e_end, w_end)
            rows.append(
                {
                    "source": source,
                    "window_idx": int(window_idx),
                    "window_start": int(w_start),
                    "window_end": int(w_end),
                    "event_class": str(event["class"]),
                    "event_idx_start": int(e_start),
                    "event_idx_end": int(e_end),
                    "event_idx_start_in_window": int(clipped_start - w_start),
                    "event_idx_end_in_window": int(clipped_end - w_start),
                    "confidence": event["confidence"],
                }
            )

    columns = [
        "source",
        "window_idx",
        "window_start",
        "window_end",
        "event_class",
        "event_idx_start",
        "event_idx_end",
        "event_idx_start_in_window",
        "event_idx_end_in_window",
        "confidence",
    ]
    if len(rows) == 0:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_window_count_table(
    *,
    starts: np.ndarray,
    window_size: int,
    gt_window_df: pd.DataFrame,
    pred_raw_window_df: pd.DataFrame,
    pred_clean_window_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per window with GT/raw/clean event counts."""
    window_df = pd.DataFrame(
        {
            "window_idx": np.arange(len(starts), dtype=np.int64),
            "window_start": starts.astype(np.int64),
            "window_end": (starts + int(window_size) - 1).astype(np.int64),
        }
    )

    gt_counts = (
        gt_window_df.groupby("window_idx", as_index=False)
        .size()
        .rename(columns={"size": "n_gt_events"})
        if not gt_window_df.empty
        else pd.DataFrame(columns=["window_idx", "n_gt_events"])
    )
    raw_counts = (
        pred_raw_window_df.groupby("window_idx", as_index=False)
        .size()
        .rename(columns={"size": "n_pred_raw_events"})
        if not pred_raw_window_df.empty
        else pd.DataFrame(columns=["window_idx", "n_pred_raw_events"])
    )
    clean_counts = (
        pred_clean_window_df.groupby("window_idx", as_index=False)
        .size()
        .rename(columns={"size": "n_pred_clean_events"})
        if not pred_clean_window_df.empty
        else pd.DataFrame(columns=["window_idx", "n_pred_clean_events"])
    )

    out = window_df.merge(gt_counts, on="window_idx", how="left")
    out = out.merge(raw_counts, on="window_idx", how="left")
    out = out.merge(clean_counts, on="window_idx", how="left")
    for col in ["n_gt_events", "n_pred_raw_events", "n_pred_clean_events"]:
        out[col] = out[col].fillna(0).astype(int)
    return out

def compute_rsam_mask(batch_x_filtered, threshold):
    """
    RSAM amplitude gate. Call on bandpass-filtered, NOT-YET-normalized windows.

    batch_x_filtered : np.ndarray, shape (B, S, W)
    threshold        : float, tau from compute_rsam_threshold_from_training

    Returns
    -------
    keep_mask : np.ndarray of bool, (B,) -- True = keep the window
    rsam_net  : np.ndarray of float, (B,) -- for logging/diagnostics
    """
    zero_channel = ~np.any(batch_x_filtered, axis=-1)  # (B, S), drop no-data stations

    rsam_k = np.mean(np.abs(batch_x_filtered.astype(np.float64)), axis=-1)  # (B, S)
    rsam_k_masked = np.where(zero_channel, np.nan, rsam_k)

    with np.errstate(all="ignore"):
        rsam_net = np.nanmedian(rsam_k_masked, axis=-1)  # (B,)

    # if every station was zero-filled, fail open rather than silently dropping
    keep_mask = np.where(np.isnan(rsam_net), True, rsam_net >= threshold)
    return keep_mask, rsam_net


def filter_events_by_rsam(
    events_df: pd.DataFrame,
    *,
    x_stations_bandpassed: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Discard events whose RSAM is below threshold (fail-open for all-zero spans)."""
    events_df = ensure_detection_df_schema(events_df)
    if events_df.empty:
        return events_df

    if x_stations_bandpassed.ndim != 2:
        raise ValueError(
            "x_stations_bandpassed must have shape [S, T], got "
            f"{x_stations_bandpassed.shape}."
        )
    if threshold <= 0:
        return events_df.copy()

    n_stations, total_len = x_stations_bandpassed.shape
    abs_x = np.abs(x_stations_bandpassed.astype(np.float64, copy=False))
    nonzero_x = (x_stations_bandpassed != 0.0).astype(np.int32, copy=False)

    # Prefix sums for fast interval RSAM queries.
    abs_prefix = np.pad(np.cumsum(abs_x, axis=1), ((0, 0), (1, 0)), mode="constant")
    nz_prefix = np.pad(
        np.cumsum(nonzero_x, axis=1),
        ((0, 0), (1, 0)),
        mode="constant",
    )

    keep_flags = np.zeros(len(events_df), dtype=bool)
    for i, row in events_df.reset_index(drop=True).iterrows():
        start_raw = int(row["idx_start"])
        end_raw = int(row["idx_end"])
        start_idx = max(0, min(start_raw, total_len - 1))
        end_idx = max(0, min(end_raw, total_len - 1))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        seg_len = max(1, end_idx - start_idx + 1)
        sum_abs = abs_prefix[:, end_idx + 1] - abs_prefix[:, start_idx]
        nz_count = nz_prefix[:, end_idx + 1] - nz_prefix[:, start_idx]

        rsam_station = sum_abs / float(seg_len)
        rsam_station = np.where(nz_count > 0, rsam_station, np.nan)
        rsam_event = float(np.nanmedian(rsam_station))

        # If all stations are zero/no-data for this span, keep the event (fail-open).
        keep_flags[i] = bool(np.isnan(rsam_event) or (rsam_event >= float(threshold)))

    filtered = events_df.reset_index(drop=True).loc[keep_flags].copy()
    return ensure_detection_df_schema(filtered)

def infer_one_model_fold(
    *,
    model_key: str,
    model_spec: dict,
    checkpoint_path: Path,
    x_stations: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    seg_min_event_len_samples: int,
    seg_max_bg_hole_samples: int,
    det_confidence_threshold: float,
    enable_window_rsam_filter: bool,
    window_rsam_threshold: float,
    det_min_duration_samples_by_class: dict[str, int],
    det_max_duration_samples_by_class: dict[str, int],
    gt_df: pd.DataFrame | None = None,
    run_label: str | None = None,
    debug_plot_windows: bool = False,
    debug_plot_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    label = run_label or model_key
    model = build_model_from_spec(model_key=model_key, n_classes=6)
    model = model.to(device)
    model.eval()

    load_checkpoint_into_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        trainer_kind=str(model_spec["trainer_kind"]),
    )

    starts = build_window_starts(
        total_len=int(x_stations.shape[1]),
        window_size=window_size,
        stride=stride,
    )
    num_windows = int(len(starts))
    num_batches = int(np.ceil(num_windows / batch_size))
    log_every_batches = max(1, num_batches // 10)

    detections: list[dict[str, float | int | str]] = []
    skipped_all_missing_windows = 0
    skipped_rsam_windows = 0
    debug_plotted_windows = 0

    if debug_plot_windows:
        if debug_plot_dir is None:
            raise ValueError("debug_plot_dir must be provided when debug_plot_windows=True")
        debug_plot_dir.mkdir(parents=True, exist_ok=True)
        log_stage(f"{label}: debug window plots enabled -> {debug_plot_dir}")

    log_stage(
        f"{label}: model ready, starting inference "
        f"({num_windows} windows, {num_batches} batches, batch_size={batch_size})."
    )
    log_stage(
        f"{label}: applying Butterworth bandpass (1-15 Hz, fs={DEFAULT_SAMPLE_RATE_HZ:.1f} Hz)."
    )
    if enable_window_rsam_filter:
        log_stage(
            f"{label}: window RSAM filtering enabled "
            f"(threshold={window_rsam_threshold:.4f})."
        )
    else:
        log_stage(f"{label}: window RSAM filtering disabled.")

    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, b_start in enumerate(range(0, num_windows, batch_size), start=1):
            b_end = min(b_start + batch_size, num_windows)
            batch_starts = starts[b_start:b_end]
            batch_x_raw = np.stack(
                [
                    x_stations[:, int(win_start) : int(win_start) + window_size]
                    for win_start in batch_starts
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            batch_x_filtered = bandpass_windows_butterworth(batch_x_raw)
            if enable_window_rsam_filter:
                keep_mask, _ = compute_rsam_mask(
                    batch_x_filtered,
                    threshold=float(window_rsam_threshold),
                )
                skipped_rsam_windows += int((~keep_mask).sum())
                if not keep_mask.any():
                    continue
                batch_x_filtered = batch_x_filtered[keep_mask]
                batch_starts = batch_starts[keep_mask]

            batch_x = normalize_windows_global_max_abs(batch_x_filtered)

            xb = torch.from_numpy(batch_x).to(device)
            trainer_kind = str(model_spec["trainer_kind"])

            if trainer_kind == "1d":
                logits = model(xb)  # [B, C, T]
                batch_events = decode_segmentation_events(
                    logits,
                    min_event_len_samples=seg_min_event_len_samples,
                    max_bg_hole_samples=seg_max_bg_hole_samples,
                )
            elif trainer_kind == "2d":
                x2d = patch_stacking_X(xb, N=256)
                logits_2d = model(x2d)
                logits = activation_unstacking(
                    logits_2d,
                    len_window=window_size,
                    N=256,
                    n_classes=6,
                    n_stations=8,
                )
                batch_events = decode_segmentation_events(
                    logits,
                    min_event_len_samples=seg_min_event_len_samples,
                    max_bg_hole_samples=seg_max_bg_hole_samples,
                )
                del x2d, logits_2d
            elif trainer_kind == "event_detection":
                missing_threshold = float(
                    getattr(
                        getattr(model, "encoder", None),
                        "station_mask_abs_sum_threshold",
                        10.0,
                    )
                )
                station_abs_sum = xb.abs().sum(dim=-1)
                valid_window_mask = (station_abs_sum > missing_threshold).any(dim=1)

                if not bool(valid_window_mask.any().item()):
                    skipped_all_missing_windows += int(valid_window_mask.numel())
                    batch_events = [[] for _ in range(len(batch_starts))]
                elif bool(valid_window_mask.all().item()):
                    predictions = model(xb)
                    batch_events = decode_event_detection_events(
                        predictions,
                        window_size=window_size,
                        confidence_threshold=det_confidence_threshold,
                        min_duration_samples_by_class=det_min_duration_samples_by_class,
                        max_duration_samples_by_class=det_max_duration_samples_by_class,
                    )
                    del predictions
                else:
                    valid_window_mask_np = valid_window_mask.detach().cpu().numpy().astype(bool)
                    skipped_all_missing_windows += int((~valid_window_mask_np).sum())
                    xb_valid = xb[valid_window_mask]

                    predictions = model(xb_valid)
                    valid_events = decode_event_detection_events(
                        predictions,
                        window_size=window_size,
                        confidence_threshold=det_confidence_threshold,
                        min_duration_samples_by_class=det_min_duration_samples_by_class,
                        max_duration_samples_by_class=det_max_duration_samples_by_class,
                    )
                    del predictions

                    batch_events = [[] for _ in range(len(batch_starts))]
                    valid_indices = np.flatnonzero(valid_window_mask_np)
                    for out_i, local_i in enumerate(valid_indices.tolist()):
                        batch_events[int(local_i)] = valid_events[int(out_i)]

                    del xb_valid, valid_events
            else:
                raise ValueError(
                    f"Unsupported trainer_kind {trainer_kind} for model {model_key}."
                )

            for local_i, win_start in enumerate(batch_starts.tolist()):
                for event in batch_events[local_i]:
                    abs_start = int(win_start) + int(event["idx_start"])
                    abs_end = int(win_start) + int(event["idx_end"])
                    detections.append(
                        {
                            "class": str(event["class"]),
                            "idx_start": int(abs_start),
                            "idx_end": int(abs_end),
                            "confidence": event["confidence"],
                        }
                    )

            if debug_plot_windows:
                for local_i, win_start in enumerate(batch_starts.tolist()):
                    pred_events_local = batch_events[local_i]
                    gt_events_local: list[dict[str, float | int | str]] = []
                    if gt_df is not None and not gt_df.empty:
                        w_start = int(win_start)
                        w_end = int(win_start) + int(window_size) - 1
                        overlaps = gt_df[
                            (gt_df["idx_end"].astype(int) >= w_start)
                            & (gt_df["idx_start"].astype(int) <= w_end)
                        ]
                        for _, gt_row in overlaps.iterrows():
                            clipped_start = max(int(gt_row["idx_start"]), w_start)
                            clipped_end = min(int(gt_row["idx_end"]), w_end)
                            gt_events_local.append(
                                {
                                    "class": str(gt_row["class"]),
                                    "idx_start": int(clipped_start - w_start),
                                    "idx_end": int(clipped_end - w_start),
                                    "confidence": gt_row["confidence"],
                                }
                            )

                    save_window_debug_plot(
                        normalized_window=batch_x[local_i],
                        window_start=int(win_start),
                        output_dir=debug_plot_dir,
                        gt_events_local=gt_events_local,
                        pred_events_local=pred_events_local,
                    )
                    debug_plotted_windows += 1

            if (
                batch_idx == 1
                or batch_idx % log_every_batches == 0
                or batch_idx == num_batches
            ):
                log_stage(
                    f"{label}: batch {batch_idx}/{num_batches} "
                    f"({b_end}/{num_windows} windows processed, "
                    f"rsam_skipped={skipped_rsam_windows}, "
                    f"skipped={skipped_all_missing_windows}, "
                    f"detected_events={len(detections)})."
                )

            del xb, batch_x, batch_x_filtered, batch_x_raw, batch_events

    elapsed_s = float(time.perf_counter() - t0)

    raw_df = ensure_detection_df_schema(pd.DataFrame(detections))
    if len(raw_df) > 0:
        raw_df = raw_df.sort_values(by=["idx_start", "idx_end", "class"]).reset_index(drop=True)

    timing = {
        "total_time_10h_s": elapsed_s,
        "total_windows": float(num_windows),
        "processed_windows": float(
            num_windows - skipped_all_missing_windows - skipped_rsam_windows
        ),
        "skipped_rsam_windows": float(skipped_rsam_windows),
        "skipped_all_missing_windows": float(skipped_all_missing_windows),
        "total_batches": float(num_batches),
        "mean_time_per_window_ms": float(1000.0 * elapsed_s / max(1, num_windows)),
        "mean_time_per_batch_ms": float(1000.0 * elapsed_s / max(1, num_batches)),
        "windows_per_second": float(num_windows / max(1e-9, elapsed_s)),
    }

    log_stage(
        f"{label}: inference finished in {elapsed_s:.2f}s "
        f"(detections={len(raw_df)}, skipped_rsam_windows={skipped_rsam_windows}, "
        f"skipped_all_missing_windows={skipped_all_missing_windows}, "
        f"debug_plots={debug_plotted_windows})."
    )

    del model
    cleanup_gpu_cache()
    gc.collect()
    return raw_df, timing


def summarize_group(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols)

    rows = []
    grouped = df.groupby(group_cols, sort=True)
    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row["n_folds"] = int(len(grp))
        for metric in metric_cols:
            values = pd.to_numeric(grp[metric], errors="coerce").dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) > 0 else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    log_stage("Starting continuous 10-hour tests.")

    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    continuous_npy = resolve_project_path(args.continuous_npy, PROJECT_ROOT)
    reference_csv = resolve_project_path(args.reference_csv, PROJECT_ROOT)
    output_root = (
        resolve_project_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir is not None
        else experiment_root / "continuous_tests"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_stage(f"Output directory: {output_root}")

    if not continuous_npy.exists():
        raise FileNotFoundError(f"Continuous NPY not found: {continuous_npy}")

    discovered_models = discover_ablation_keys(experiment_root)
    selected_models = parse_csv_selection(args.models, discovered_models, "models")
    selected_folds = parse_folds(args.folds)
    strides = parse_int_csv(args.strides, name="--strides")

    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if args.seg_min_event_len_samples <= 0:
        raise ValueError("--seg-min-event-len-samples must be > 0")
    if args.seg_max_bg_hole_samples < 0:
        raise ValueError("--seg-max-bg-hole-samples must be >= 0")
    if args.cross_class_confidence_margin < 0:
        raise ValueError("--cross-class-confidence-margin must be >= 0")
    if args.cross_class_max_duration_ratio <= 0:
        raise ValueError("--cross-class-max-duration-ratio must be > 0")
    if args.window_rsam_threshold <= 0:
        raise ValueError("--window-rsam-threshold must be > 0")
    if args.event_rsam_threshold <= 0:
        raise ValueError("--event-rsam-threshold must be > 0")

    min_duration_sec_by_class = parse_class_float_map(
        args.det_class_min_duration_sec,
        name="--det-class-min-duration-sec",
    )
    max_duration_sec_by_class = parse_class_float_map(
        args.det_class_max_duration_sec,
        name="--det-class-max-duration-sec",
    )
    for cls in EVENT_CLASSES:
        if min_duration_sec_by_class[cls] > max_duration_sec_by_class[cls]:
            raise ValueError(
                f"Duration gate invalid for class {cls}: "
                f"min {min_duration_sec_by_class[cls]} > max {max_duration_sec_by_class[cls]}."
            )

    det_min_duration_samples_by_class = {
        cls: max(1, int(round(min_duration_sec_by_class[cls] * DEFAULT_SAMPLE_RATE_HZ)))
        for cls in EVENT_CLASSES
    }
    det_max_duration_samples_by_class = {
        cls: max(1, int(round(max_duration_sec_by_class[cls] * DEFAULT_SAMPLE_RATE_HZ)))
        for cls in EVENT_CLASSES
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_stage(f"Using device: {device}")

    log_stage(f"Loading continuous trace: {continuous_npy}")
    cont = np.load(continuous_npy, mmap_mode="r")
    if cont.ndim != 2 or cont.shape[0] < 9:
        raise ValueError(
            "Continuous array must be [rows, T] with at least 9 rows "
            "(timestamp + 8 stations)."
        )
    x_stations = np.asarray(cont[1:9, :], dtype=np.float32)
    log_stage(
        f"Prepared station matrix with shape={x_stations.shape} "
        "(rows 1..8 from continuous trace)."
    )
    log_stage("Precomputing bandpassed continuous trace for event RSAM filtering.")
    x_stations_bandpassed = bandpass_windows_butterworth(
        x_stations[np.newaxis, :, :]
    )[0]
    log_stage(
        "Bandpassed continuous trace ready "
        f"(shape={x_stations_bandpassed.shape})."
    )
    log_stage(f"Loading reference events: {reference_csv}")
    gt_df = load_reference_events(reference_csv)
    log_stage(f"Loaded {len(gt_df)} reference events.")

    run_manifest = {
        "experiment_root": str(experiment_root),
        "continuous_npy": str(continuous_npy),
        "reference_csv": str(reference_csv),
        "output_root": str(output_root),
        "device": str(device),
        "window_size": int(args.window_size),
        "strides": [int(s) for s in strides],
        "models": selected_models,
        "folds": selected_folds,
        "seg_min_event_len_samples": int(args.seg_min_event_len_samples),
        "seg_max_bg_hole_samples": int(args.seg_max_bg_hole_samples),
        "det_confidence_threshold": float(DEFAULT_DET_CONFIDENCE_THRESHOLD),
        "enable_window_rsam_filter": bool(args.enable_window_rsam_filter),
        "window_rsam_threshold": float(args.window_rsam_threshold),
        "event_rsam_threshold": float(args.event_rsam_threshold),
        "cross_class_overlap_threshold": float(args.cross_class_overlap_threshold),
        "cross_class_confidence_margin": float(args.cross_class_confidence_margin),
        "cross_class_max_duration_ratio": float(args.cross_class_max_duration_ratio),
        "det_class_min_duration_sec": {
            cls: float(min_duration_sec_by_class[cls]) for cls in EVENT_CLASSES
        },
        "det_class_max_duration_sec": {
            cls: float(max_duration_sec_by_class[cls]) for cls in EVENT_CLASSES
        },
        "matching_strategy": str(args.matching_strategy),
        "match_iou_threshold": float(args.match_iou_threshold),
        "overlap_recall_threshold": float(args.overlap_recall_threshold),
        "debug_plot_windows": bool(args.debug_plot_windows),
    }
    with (output_root / "run_manifest_continuous.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    fold_summary_rows: list[dict[str, float | int | str]] = []

    for stride in strides:
        log_stage(f"Starting stride={int(stride)}.")
        stride_root = output_root / f"stride_{int(stride)}"
        stride_root.mkdir(parents=True, exist_ok=True)

        for model_key in selected_models:
            model_spec = MODEL_SPECS[model_key]
            model_batch_size = int(args.batch_size or int(model_spec["batch_size"]))
            model_root = stride_root / model_key
            log_stage(
                f"Starting model={model_key} (trainer_kind={model_spec['trainer_kind']}, "
                f"batch_size={model_batch_size})."
            )

            for fold in selected_folds:
                ckpt = checkpoint_path_for_fold(
                    root=experiment_root / "ablations" / model_key,
                    fold_id=int(fold),
                    checkpoint_name="best_f1.pt",
                )
                if not ckpt.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint for model={model_key} fold={fold}: {ckpt}"
                    )

                fold_root = model_root / f"fold_{int(fold):02d}"
                fold_root.mkdir(parents=True, exist_ok=True)
                run_label = f"stride={int(stride)} model={model_key} fold={int(fold):02d}"
                log_stage(f"{run_label}: loading checkpoint {ckpt.name}.")
                debug_plot_dir = (
                    fold_root / "debug_window_plots"
                    if args.debug_plot_windows
                    else None
                )

                raw_df, timing = infer_one_model_fold(
                    model_key=model_key,
                    model_spec=model_spec,
                    checkpoint_path=ckpt,
                    x_stations=x_stations,
                    window_size=int(args.window_size),
                    stride=int(stride),
                    batch_size=int(model_batch_size),
                    device=device,
                    seg_min_event_len_samples=int(args.seg_min_event_len_samples),
                    seg_max_bg_hole_samples=int(args.seg_max_bg_hole_samples),
                    det_confidence_threshold=float(DEFAULT_DET_CONFIDENCE_THRESHOLD),
                    enable_window_rsam_filter=bool(args.enable_window_rsam_filter),
                    window_rsam_threshold=float(args.window_rsam_threshold),
                    det_min_duration_samples_by_class=det_min_duration_samples_by_class,
                    det_max_duration_samples_by_class=det_max_duration_samples_by_class,
                    gt_df=gt_df,
                    run_label=run_label,
                    debug_plot_windows=bool(args.debug_plot_windows),
                    debug_plot_dir=debug_plot_dir,
                )

                raw_count_before_event_rsam = int(len(raw_df))
                raw_df = filter_events_by_rsam(
                    raw_df,
                    x_stations_bandpassed=x_stations_bandpassed,
                    threshold=float(args.event_rsam_threshold),
                )
                dropped_by_event_rsam = raw_count_before_event_rsam - int(len(raw_df))
                if dropped_by_event_rsam > 0:
                    log_stage(
                        f"{run_label}: event RSAM filter dropped "
                        f"{dropped_by_event_rsam}/{raw_count_before_event_rsam} "
                        f"events (threshold={float(args.event_rsam_threshold):.4f})."
                    )
                else:
                    log_stage(
                        f"{run_label}: event RSAM filter dropped 0 events "
                        f"(threshold={float(args.event_rsam_threshold):.4f})."
                    )

                clean_df = postprocess_detections(
                    raw_df,
                    cross_class_overlap_threshold=float(args.cross_class_overlap_threshold),
                    cross_class_confidence_margin=float(args.cross_class_confidence_margin),
                    cross_class_max_duration_ratio=float(args.cross_class_max_duration_ratio),
                )

                window_starts = build_window_starts(
                    total_len=int(x_stations.shape[1]),
                    window_size=int(args.window_size),
                    stride=int(stride),
                )
                gt_window_df = build_window_event_table(
                    starts=window_starts,
                    window_size=int(args.window_size),
                    events_df=gt_df,
                    source="gt",
                )
                pred_raw_window_df = build_window_event_table(
                    starts=window_starts,
                    window_size=int(args.window_size),
                    events_df=raw_df,
                    source="pred_raw",
                )
                pred_clean_window_df = build_window_event_table(
                    starts=window_starts,
                    window_size=int(args.window_size),
                    events_df=clean_df,
                    source="pred_clean",
                )
                window_events_df = pd.concat(
                    [gt_window_df, pred_raw_window_df, pred_clean_window_df],
                    axis=0,
                    ignore_index=True,
                )
                window_counts_df = build_window_count_table(
                    starts=window_starts,
                    window_size=int(args.window_size),
                    gt_window_df=gt_window_df,
                    pred_raw_window_df=pred_raw_window_df,
                    pred_clean_window_df=pred_clean_window_df,
                )

                raw_metrics, raw_pairs = evaluate_event_detections(
                    raw_df,
                    gt_df,
                    matching_strategy=str(args.matching_strategy),
                    match_iou_threshold=float(args.match_iou_threshold),
                    overlap_recall_threshold=float(args.overlap_recall_threshold),
                )
                clean_metrics, clean_pairs = evaluate_event_detections(
                    clean_df,
                    gt_df,
                    matching_strategy=str(args.matching_strategy),
                    match_iou_threshold=float(args.match_iou_threshold),
                    overlap_recall_threshold=float(args.overlap_recall_threshold),
                )

                raw_df.to_csv(
                    fold_root / "raw_detections.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )
                clean_df.to_csv(
                    fold_root / "cleaned_detections.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )
                raw_pairs.to_csv(
                    fold_root / "event_pairs_raw.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )
                clean_pairs.to_csv(
                    fold_root / "event_pairs_cleaned.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )
                window_events_df.to_csv(
                    fold_root / "window_events_detailed.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )
                window_counts_df.to_csv(
                    fold_root / "window_events_summary.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )

                metrics_payload = {
                    "raw": raw_metrics,
                    "cleaned": clean_metrics,
                    "timing": timing,
                    "stride": int(stride),
                    "model_key": model_key,
                    "fold": int(fold),
                }
                with (fold_root / "metrics_and_timing.json").open("w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, indent=2)

                summary_row: dict[str, float | int | str] = {
                    "stride": int(stride),
                    "model_key": model_key,
                    "fold": int(fold),
                    "trainer_kind": str(model_spec["trainer_kind"]),
                }
                for key, value in raw_metrics.items():
                    summary_row[f"raw_{key}"] = float(value)
                for key, value in clean_metrics.items():
                    summary_row[f"clean_{key}"] = float(value)
                for key, value in timing.items():
                    summary_row[key] = float(value)
                fold_summary_rows.append(summary_row)
                log_stage(
                    f"{run_label}: metrics saved "
                    f"(raw_f1={summary_row.get('raw_f1', np.nan):.4f}, "
                    f"clean_f1={summary_row.get('clean_f1', np.nan):.4f}, "
                    f"raw_detected_events={len(raw_df)}, "
                    f"clean_detected_events={len(clean_df)}, "
                    f"window_event_rows={len(window_events_df)})."
                )

                cleanup_gpu_cache()
                gc.collect()

    fold_summary_df = pd.DataFrame(fold_summary_rows)
    if fold_summary_df.empty:
        raise RuntimeError("No fold results were generated.")

    fold_summary_df.to_csv(
        output_root / "continuous_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    metric_cols = [
        "raw_precision",
        "raw_recall",
        "raw_f1",
        "raw_iou",
        "clean_precision",
        "clean_recall",
        "clean_f1",
        "clean_iou",
        "total_time_10h_s",
        "mean_time_per_window_ms",
        "mean_time_per_batch_ms",
        "windows_per_second",
    ]
    aggregate_df = summarize_group(
        fold_summary_df,
        group_cols=["stride", "model_key"],
        metric_cols=metric_cols,
    )
    aggregate_df = aggregate_df.sort_values(
        by=["stride", "clean_f1_mean"],
        ascending=[True, False],
    )
    aggregate_df.to_csv(
        output_root / "continuous_summary_by_model_stride.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )


if __name__ == "__main__":
    main()
