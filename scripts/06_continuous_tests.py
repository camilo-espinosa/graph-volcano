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
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.active_eval_utils import load_checkpoint_into_model
from utils.data_utils import activation_unstacking, patch_stacking_X
from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.event_detection_metrics import temporal_iou, temporal_overlap_recall
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
        default=None,
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
        "--det-bg-prob-threshold",
        type=float,
        default=0.5,
        help="Skip event-detection query if background probability is >= threshold.",
    )
    parser.add_argument(
        "--cross-class-overlap-threshold",
        type=float,
        default=0.8,
        help=(
            "If two different-class detections overlap with IoU >= threshold and both "
            "have confidence, keep only the highest-confidence one."
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
    pred_idx = torch.argmax(probs, dim=1).detach().cpu().numpy()  # [B, T]

    batch_events: list[list[dict[str, float | int | str]]] = []
    for sample_idx in range(pred_idx.shape[0]):
        class_idx = pred_idx[sample_idx]
        class_idx = _fill_small_background_holes(
            class_idx,
            max_bg_hole_samples=max_bg_hole_samples,
        )

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
    bg_prob_threshold: float,
) -> list[list[dict[str, float | int | str]]]:
    pred = normalize_prediction_intervals(predictions)

    class_logits = pred["class_logits"]  # [B, Nq, C]
    starts = pred["start"]  # [B, Nq, 1]
    ends = pred["end"]  # [B, Nq, 1]

    probs = torch.softmax(class_logits, dim=-1)
    batch_size, num_queries, _ = probs.shape

    out: list[list[dict[str, float | int | str]]] = []
    for b in range(batch_size):
        sample_events: list[dict[str, float | int | str]] = []
        for q in range(num_queries):
            cls = int(torch.argmax(probs[b, q]).item())
            if cls == 0:
                continue

            bg_prob = float(probs[b, q, 0].item())
            if bg_prob >= float(bg_prob_threshold):
                continue

            if cls not in CLASS_ID_TO_NAME:
                continue

            conf = float(1.0 - bg_prob)
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

            sample_events.append(
                {
                    "class": CLASS_ID_TO_NAME[cls],
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


def resolve_cross_class_conflicts(
    df: pd.DataFrame,
    *,
    overlap_threshold: float,
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
) -> pd.DataFrame:
    merged = merge_same_class_overlaps(raw_df)
    cleaned = resolve_cross_class_conflicts(
        merged,
        overlap_threshold=cross_class_overlap_threshold,
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
    p_start = float(pred_start)
    p_end = float(pred_end)
    g_start = float(gt_start)
    g_end = float(gt_end)
    iou = temporal_iou(p_start, p_end, g_start, g_end)
    overlap = temporal_overlap_recall(p_start, p_end, g_start, g_end)

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
    det_bg_prob_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
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

    detections: list[dict[str, float | int | str]] = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for b_start in range(0, num_windows, batch_size):
            b_end = min(b_start + batch_size, num_windows)
            batch_starts = starts[b_start:b_end]
            batch_x = np.stack(
                [
                    x_stations[:, int(win_start) : int(win_start) + window_size]
                    for win_start in batch_starts
                ],
                axis=0,
            ).astype(np.float32, copy=False)

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
                predictions = model(xb)
                batch_events = decode_event_detection_events(
                    predictions,
                    window_size=window_size,
                    bg_prob_threshold=det_bg_prob_threshold,
                )
                del predictions
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

            del xb, batch_x, batch_events

    elapsed_s = float(time.perf_counter() - t0)

    raw_df = ensure_detection_df_schema(pd.DataFrame(detections))
    if len(raw_df) > 0:
        raw_df = raw_df.sort_values(by=["idx_start", "idx_end", "class"]).reset_index(drop=True)

    timing = {
        "total_time_10h_s": elapsed_s,
        "total_windows": float(num_windows),
        "total_batches": float(num_batches),
        "mean_time_per_window_ms": float(1000.0 * elapsed_s / max(1, num_windows)),
        "mean_time_per_batch_ms": float(1000.0 * elapsed_s / max(1, num_batches)),
        "windows_per_second": float(num_windows / max(1e-9, elapsed_s)),
    }

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

    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    continuous_npy = resolve_project_path(args.continuous_npy, PROJECT_ROOT)
    reference_csv = resolve_project_path(args.reference_csv, PROJECT_ROOT)
    output_root = (
        resolve_project_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir is not None
        else experiment_root / "continuous_tests"
    )
    output_root.mkdir(parents=True, exist_ok=True)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cont = np.load(continuous_npy, mmap_mode="r")
    if cont.ndim != 2 or cont.shape[0] < 9:
        raise ValueError(
            "Continuous array must be [rows, T] with at least 9 rows "
            "(timestamp + 8 stations)."
        )
    x_stations = np.asarray(cont[1:9, :], dtype=np.float32)
    gt_df = load_reference_events(reference_csv)

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
        "det_bg_prob_threshold": float(args.det_bg_prob_threshold),
        "cross_class_overlap_threshold": float(args.cross_class_overlap_threshold),
        "matching_strategy": str(args.matching_strategy),
        "match_iou_threshold": float(args.match_iou_threshold),
        "overlap_recall_threshold": float(args.overlap_recall_threshold),
    }
    with (output_root / "run_manifest_continuous.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    fold_summary_rows: list[dict[str, float | int | str]] = []

    for stride in strides:
        stride_root = output_root / f"stride_{int(stride)}"
        stride_root.mkdir(parents=True, exist_ok=True)

        for model_key in selected_models:
            model_spec = MODEL_SPECS[model_key]
            model_batch_size = int(args.batch_size or int(model_spec["batch_size"]))
            model_root = stride_root / model_key

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
                    det_bg_prob_threshold=float(args.det_bg_prob_threshold),
                )

                clean_df = postprocess_detections(
                    raw_df,
                    cross_class_overlap_threshold=float(args.cross_class_overlap_threshold),
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
