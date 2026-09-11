"""Select per-model fused alpha and threshold using F1/precision/recall.

This script computes event-level relative RSAM for predicted intervals using
surrounding signal only (no GT masking), and then
sweeps:
1) alpha in score_fused = alpha*z(confidence) + (1-alpha)*z(log(relative_rsam))
2) threshold quantiles on score_fused

Selection objective is best F1 per model. Precision and recall at that operating
point are reported explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

EVENT_CLASSES = ["VT", "LP", "TR", "AV", "IC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select fused alpha + threshold per model using F1."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/experiments/complete_experiment/continuous_tests_v3_conf09"
        ),
        help="Root folder with stride_* model outputs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7000,
        help="Stride to analyze (reads output_root/stride_<stride>).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="cleaned",
        choices=["raw", "cleaned"],
        help="Use raw or cleaned event-pairs files.",
    )
    parser.add_argument(
        "--continuous-npy",
        type=Path,
        default=Path("data/NVCh_10h_continuous_trace/NVCh_10h_continuous_trace.npy"),
        help="Continuous trace NPY [rows, T] with timestamp row + 8 stations.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Sample rate in Hz.",
    )
    parser.add_argument(
        "--bandpass-low-hz",
        type=float,
        default=1.0,
        help="Bandpass low cutoff (Hz).",
    )
    parser.add_argument(
        "--bandpass-high-hz",
        type=float,
        default=15.0,
        help="Bandpass high cutoff (Hz).",
    )
    parser.add_argument(
        "--bandpass-order",
        type=int,
        default=4,
        help="Butterworth bandpass order.",
    )
    parser.add_argument(
        "--context-multiplier",
        type=float,
        default=1.0,
        help=(
            "Each side local-background context length = context_multiplier * interval_length."
        ),
    )
    parser.add_argument(
        "--fusion-alpha-step",
        type=float,
        default=0.05,
        help="Grid step for alpha in fused z-score: alpha*z_conf + (1-alpha)*z_log_rsam.",
    )
    parser.add_argument(
        "--threshold-quantile-step",
        type=float,
        default=0.01,
        help=(
            "Quantile step in [0,1] for threshold sweep over fused score. "
            "At quantile q, threshold = quantile(score_fused, q)."
        ),
    )
    return parser.parse_args()


def read_csv_local(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",")


def bandpass_trace(
    x_stations: np.ndarray,
    *,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    if x_stations.ndim != 2:
        raise ValueError(f"Expected [S,T], got {x_stations.shape}")
    nyquist = 0.5 * float(sample_rate_hz)
    if not (0.0 < float(low_hz) < float(high_hz) < nyquist):
        raise ValueError(
            "Invalid bandpass limits: "
            f"low={low_hz}, high={high_hz}, nyquist={nyquist}."
        )
    sos = butter(
        int(order),
        [float(low_hz), float(high_hz)],
        btype="bandpass",
        fs=float(sample_rate_hz),
        output="sos",
    )
    filtered = sosfiltfilt(sos, x_stations[np.newaxis, :, :], axis=-1)[0]
    return filtered.astype(np.float32, copy=False)


def local_background_segment(
    x_band: np.ndarray,
    *,
    idx_start: int,
    idx_end: int,
    context_multiplier: float,
) -> np.ndarray:
    if x_band.ndim != 2:
        raise ValueError(f"Expected x_band [S,T], got {x_band.shape}")
    if idx_start < 0 or idx_end >= x_band.shape[1] or idx_start > idx_end:
        raise ValueError(
            f"Invalid interval [{idx_start}, {idx_end}] for length {x_band.shape[1]}."
        )
    if context_multiplier <= 0.0:
        raise ValueError("context_multiplier must be > 0.")

    interval_len = idx_end - idx_start + 1
    context_len = max(1, int(round(float(context_multiplier) * float(interval_len))))
    total_len = int(x_band.shape[1])

    left_s = max(0, idx_start - context_len)
    left_e = idx_start - 1
    right_s = idx_end + 1
    right_e = min(total_len - 1, idx_end + context_len)

    parts: list[np.ndarray] = []
    if left_s <= left_e:
        parts.append(x_band[:, left_s : left_e + 1])

    if right_s <= right_e:
        parts.append(x_band[:, right_s : right_e + 1])

    if len(parts) == 0:
        return np.empty((x_band.shape[0], 0), dtype=x_band.dtype)

    return np.concatenate(parts, axis=1)


def rsam_from_segment(seg: np.ndarray) -> float:
    if seg.ndim != 2:
        raise ValueError(f"Expected segment [S,T], got {seg.shape}")
    if seg.shape[1] <= 0:
        return np.nan

    abs_seg = np.abs(seg.astype(np.float64, copy=False))
    rsam_station = np.mean(abs_seg, axis=1)
    zero_station = ~np.any(seg != 0.0, axis=1)
    rsam_station = np.where(zero_station, np.nan, rsam_station)
    if np.all(np.isnan(rsam_station)):
        return np.nan
    return float(np.nanmedian(rsam_station))


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0.0 or not np.isfinite(std):
        raise ValueError("Cannot z-score constant or non-finite vector.")
    return (values - mean) / std


def build_alpha_grid(step: float) -> np.ndarray:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"fusion-alpha-step must be in (0,1], got {step}")
    n = int(round(1.0 / step))
    grid = np.linspace(0.0, 1.0, n + 1)
    return np.unique(np.clip(grid, 0.0, 1.0))


def build_quantile_grid(step: float) -> np.ndarray:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"threshold-quantile-step must be in (0,1], got {step}")
    n = int(round(1.0 / step))
    grid = np.linspace(0.0, 1.0, n + 1)
    return np.unique(np.clip(grid, 0.0, 1.0))


def ensure_pair_columns(df: pd.DataFrame, *, file_name: str) -> pd.DataFrame:
    required = {
        "status",
        "class",
        "pred_idx_start",
        "pred_idx_end",
        "pred_confidence",
    }
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing columns in {file_name}: {missing}")
    return df.copy()


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def main() -> None:
    args = parse_args()
    stage = str(args.stage)

    stride_root = args.output_root / f"stride_{int(args.stride)}"
    if not stride_root.exists():
        raise FileNotFoundError(f"Stride folder not found: {stride_root}")

    if not args.continuous_npy.exists():
        raise FileNotFoundError(f"Continuous NPY not found: {args.continuous_npy}")

    cont = np.load(args.continuous_npy, mmap_mode="r")
    if cont.ndim != 2 or cont.shape[0] < 9:
        raise ValueError(
            "Continuous array must be [rows,T] with at least 9 rows "
            "(timestamp + 8 stations)."
        )

    x_stations = np.asarray(cont[1:9, :], dtype=np.float32)
    total_len = int(x_stations.shape[1])

    x_band = bandpass_trace(
        x_stations,
        sample_rate_hz=float(args.sample_rate_hz),
        low_hz=float(args.bandpass_low_hz),
        high_hz=float(args.bandpass_high_hz),
        order=int(args.bandpass_order),
    )

    pair_file_name = (
        "event_pairs_cleaned.csv" if stage == "cleaned" else "event_pairs_raw.csv"
    )

    rows_all: list[dict[str, float | int | str]] = []
    totals_by_model: dict[str, dict[str, int]] = {}
    class_totals_by_model: dict[str, dict[str, dict[str, int]]] = {}

    model_dirs = sorted(
        [p for p in stride_root.iterdir() if p.is_dir()], key=lambda p: p.name
    )
    for model_dir in model_dirs:
        for fold_dir in sorted(
            [p for p in model_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        ):
            if not fold_dir.name.startswith("fold_"):
                continue
            fold_id = int(fold_dir.name.split("_")[-1])
            pair_path = fold_dir / pair_file_name
            if not pair_path.exists():
                continue

            pairs_all = ensure_pair_columns(
                read_csv_local(pair_path), file_name=str(pair_path)
            )

            model_key = str(model_dir.name)
            totals = totals_by_model.setdefault(
                model_key,
                {"tp_total": 0, "fp_total": 0, "fn_total": 0},
            )
            totals["tp_total"] += int((pairs_all["status"] == "TP").sum())
            totals["fp_total"] += int((pairs_all["status"] == "FP").sum())
            totals["fn_total"] += int((pairs_all["status"] == "FN").sum())

            class_totals = class_totals_by_model.setdefault(
                model_key,
                {
                    class_name: {"tp_total": 0, "fp_total": 0, "fn_total": 0}
                    for class_name in EVENT_CLASSES
                },
            )
            pairs_all["class"] = pairs_all["class"].astype(str)
            for class_name in EVENT_CLASSES:
                class_rows = pairs_all[pairs_all["class"] == class_name]
                class_totals[class_name]["tp_total"] += int(
                    (class_rows["status"] == "TP").sum()
                )
                class_totals[class_name]["fp_total"] += int(
                    (class_rows["status"] == "FP").sum()
                )
                class_totals[class_name]["fn_total"] += int(
                    (class_rows["status"] == "FN").sum()
                )

            pairs = pairs_all[pairs_all["status"].isin(["TP", "FP"])].copy()

            for _, row in pairs.iterrows():
                start = int(round(float(row["pred_idx_start"])))
                end = int(round(float(row["pred_idx_end"])))
                start = max(0, min(start, total_len - 1))
                end = max(0, min(end, total_len - 1))
                if start > end:
                    start, end = end, start

                event_seg = x_band[:, start : end + 1]
                bg_seg = local_background_segment(
                    x_band,
                    idx_start=start,
                    idx_end=end,
                    context_multiplier=float(args.context_multiplier),
                )

                rsam_interval = rsam_from_segment(event_seg)
                rsam_local_bg = rsam_from_segment(bg_seg)
                local_bg_samples = int(bg_seg.shape[1])
                if (
                    np.isfinite(rsam_interval)
                    and np.isfinite(rsam_local_bg)
                    and rsam_local_bg > 0.0
                ):
                    relative_rsam = float(rsam_interval / rsam_local_bg)
                else:
                    relative_rsam = np.nan

                invalid_reason = ""
                if not np.isfinite(relative_rsam):
                    if local_bg_samples == 0:
                        invalid_reason = "empty_local_background"
                    elif not np.isfinite(rsam_local_bg):
                        invalid_reason = "local_background_all_zero_or_nan"
                    elif not np.isfinite(rsam_interval):
                        invalid_reason = "event_all_zero_or_nan"
                    elif rsam_local_bg <= 0.0:
                        invalid_reason = "local_background_non_positive"
                    else:
                        invalid_reason = "unknown_invalid_relative_rsam"

                rows_all.append(
                    {
                        "stage": stage,
                        "model_key": model_key,
                        "fold": int(fold_id),
                        "status": str(row["status"]),
                        "class": str(row["class"]),
                        "pred_idx_start": int(start),
                        "pred_idx_end": int(end),
                        "pred_confidence": float(row["pred_confidence"]),
                        "rsam_interval": rsam_interval,
                        "rsam_local_background": rsam_local_bg,
                        "local_bg_samples": local_bg_samples,
                        "relative_rsam": relative_rsam,
                        "invalid_relative_rsam_reason": invalid_reason,
                    }
                )

    if len(rows_all) == 0:
        raise RuntimeError("No TP/FP rows found for analysis.")

    all_df = pd.DataFrame(rows_all)
    invalid_conf = int((~np.isfinite(all_df["pred_confidence"]).to_numpy()).sum())
    invalid_rsam = int((~np.isfinite(all_df["relative_rsam"]).to_numpy()).sum())
    nonpos_rsam = int((all_df["relative_rsam"].to_numpy(dtype=np.float64) <= 0.0).sum())
    if invalid_conf > 0 or invalid_rsam > 0 or nonpos_rsam > 0:
        debug_dir = stride_root / "confidence_rsam_poc_06f"
        debug_dir.mkdir(parents=True, exist_ok=True)

        invalid_rows = all_df[
            (~np.isfinite(all_df["pred_confidence"]))
            | (~np.isfinite(all_df["relative_rsam"]))
            | (all_df["relative_rsam"].astype(float) <= 0.0)
        ].copy()
        invalid_rows.to_csv(
            debug_dir / f"invalid_scoring_rows_{stage}.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        invalid_reason_counts = (
            invalid_rows["invalid_relative_rsam_reason"]
            .value_counts(dropna=False)
            .to_dict()
            if "invalid_relative_rsam_reason" in invalid_rows.columns
            else {}
        )

        invalid_reason_counts_str = ", ".join(
            [
                f"{k}:{v}"
                for k, v in sorted(
                    invalid_reason_counts.items(), key=lambda x: str(x[0])
                )
            ]
        )

        raise ValueError(
            "Invalid TP/FP scoring rows found: "
            f"invalid_confidence={invalid_conf}, "
            f"invalid_relative_rsam={invalid_rsam}, "
            f"non_positive_relative_rsam={nonpos_rsam}. "
            f"Reason breakdown=({invalid_reason_counts_str}). "
            f"See {debug_dir / f'invalid_scoring_rows_{stage}.csv'}"
        )

    all_df["log_relative_rsam"] = np.log(all_df["relative_rsam"].astype(float))
    alphas = build_alpha_grid(float(args.fusion_alpha_step))
    quantiles = build_quantile_grid(float(args.threshold_quantile_step))

    scored_parts: list[pd.DataFrame] = []
    sweep_rows: list[dict[str, float | int | str]] = []
    best_rows: list[dict[str, float | int | str]] = []
    best_rows_macro: list[dict[str, float | int | str]] = []

    for model_key, grp in all_df.groupby("model_key", sort=True):
        totals = totals_by_model.get(model_key)
        if totals is None:
            raise KeyError(f"Missing totals for model {model_key}.")
        class_totals = class_totals_by_model.get(model_key)
        if class_totals is None:
            raise KeyError(f"Missing class totals for model {model_key}.")

        tp_total = int(totals["tp_total"])
        fp_total = int(totals["fp_total"])
        fn_total = int(totals["fn_total"])
        if tp_total <= 0:
            raise ValueError(
                f"Model {model_key} has no TP rows; cannot compute recall."
            )

        work = grp.copy().reset_index(drop=True)
        conf = work["pred_confidence"].to_numpy(dtype=np.float64)
        lrr = work["log_relative_rsam"].to_numpy(dtype=np.float64)
        z_conf = zscore(conf)
        z_lrr = zscore(lrr)

        work["z_confidence"] = z_conf
        work["z_log_relative_rsam"] = z_lrr
        scored_parts.append(work)

        best_for_model: dict[str, float | int | str] | None = None
        best_for_model_macro: dict[str, float | int | str] | None = None

        is_tp = (work["status"] == "TP").to_numpy(dtype=bool)
        class_values = work["class"].astype(str).to_numpy()

        for alpha in alphas.tolist():
            score = float(alpha) * z_conf + (1.0 - float(alpha)) * z_lrr

            for q in quantiles.tolist():
                threshold = float(np.quantile(score, q))
                keep = score >= threshold

                tp_kept = int(np.logical_and(is_tp, keep).sum())
                fp_kept = int(np.logical_and(~is_tp, keep).sum())
                fn_effective = int(fn_total + (tp_total - tp_kept))

                precision, recall, f1 = compute_prf(tp_kept, fp_kept, fn_effective)
                per_class_f1: list[float] = []
                for class_name in EVENT_CLASSES:
                    class_mask = class_values == class_name
                    tp_kept_cls = int(
                        np.logical_and(np.logical_and(is_tp, keep), class_mask).sum()
                    )
                    fp_kept_cls = int(
                        np.logical_and(np.logical_and(~is_tp, keep), class_mask).sum()
                    )
                    tp_total_cls = int(class_totals[class_name]["tp_total"])
                    fn_total_cls = int(class_totals[class_name]["fn_total"])
                    class_support = tp_total_cls + fn_total_cls
                    if class_support <= 0:
                        continue
                    fn_effective_cls = int(fn_total_cls + (tp_total_cls - tp_kept_cls))
                    _, _, f1_cls = compute_prf(
                        tp_kept_cls, fp_kept_cls, fn_effective_cls
                    )
                    per_class_f1.append(float(f1_cls))
                macro_f1 = (
                    float(np.mean(per_class_f1)) if len(per_class_f1) > 0 else 0.0
                )
                tp_retention = float(tp_kept / tp_total) if tp_total > 0 else 0.0
                fp_reduction = (
                    float(1.0 - (fp_kept / fp_total)) if fp_total > 0 else 0.0
                )

                row = {
                    "stage": stage,
                    "model_key": str(model_key),
                    "alpha_confidence": float(alpha),
                    "threshold_quantile": float(q),
                    "score_threshold": threshold,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "class_agnostic_f1": float(f1),
                    "macro_f1": float(macro_f1),
                    "tp_kept": int(tp_kept),
                    "fp_kept": int(fp_kept),
                    "fn_effective": int(fn_effective),
                    "tp_total": int(tp_total),
                    "fp_total": int(fp_total),
                    "fn_total": int(fn_total),
                    "tp_retention": float(tp_retention),
                    "fp_reduction": float(fp_reduction),
                }
                sweep_rows.append(row)

                if best_for_model is None:
                    best_for_model = row
                else:
                    old = best_for_model
                    better = False
                    if row["f1"] > old["f1"]:
                        better = True
                    elif row["f1"] == old["f1"] and row["recall"] > old["recall"]:
                        better = True
                    elif (
                        row["f1"] == old["f1"]
                        and row["recall"] == old["recall"]
                        and row["precision"] > old["precision"]
                    ):
                        better = True
                    elif (
                        row["f1"] == old["f1"]
                        and row["recall"] == old["recall"]
                        and row["precision"] == old["precision"]
                        and row["threshold_quantile"] < old["threshold_quantile"]
                    ):
                        better = True
                    if better:
                        best_for_model = row

                if best_for_model_macro is None:
                    best_for_model_macro = row
                else:
                    old_macro = best_for_model_macro
                    better_macro = False
                    if row["macro_f1"] > old_macro["macro_f1"]:
                        better_macro = True
                    elif (
                        row["macro_f1"] == old_macro["macro_f1"]
                        and row["recall"] > old_macro["recall"]
                    ):
                        better_macro = True
                    elif (
                        row["macro_f1"] == old_macro["macro_f1"]
                        and row["recall"] == old_macro["recall"]
                        and row["precision"] > old_macro["precision"]
                    ):
                        better_macro = True
                    if better_macro:
                        best_for_model_macro = row

        if best_for_model is None:
            raise RuntimeError(
                f"Failed to compute best operating point for model {model_key}."
            )
        if best_for_model_macro is None:
            raise RuntimeError(
                f"Failed to compute best macro operating point for model {model_key}."
            )
        best_rows.append(best_for_model)
        best_rows_macro.append(best_for_model_macro)

    scored_df = pd.concat(scored_parts, axis=0, ignore_index=True)
    sweep_df = pd.DataFrame(sweep_rows)
    best_df = (
        pd.DataFrame(best_rows).sort_values(by=["model_key"]).reset_index(drop=True)
    )
    best_macro_df = (
        pd.DataFrame(best_rows_macro)
        .sort_values(by=["model_key"])
        .reset_index(drop=True)
    )

    out_dir = stride_root / "confidence_rsam_poc_06f"
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(
        out_dir / f"tp_fp_with_relative_rsam_{stage}.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    sweep_df.to_csv(
        out_dir / f"fused_threshold_sweep_by_model_{stage}.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    best_df.to_csv(
        out_dir / f"fused_best_alpha_threshold_by_model_{stage}.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    best_macro_df.to_csv(
        out_dir / f"fused_best_alpha_threshold_by_model_{stage}_macro.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    best_global_idx = int(best_df["f1"].astype(float).idxmax())
    best_global_row = best_df.loc[best_global_idx].to_dict()
    best_global_macro_idx = int(best_macro_df["macro_f1"].astype(float).idxmax())
    best_global_macro_row = best_macro_df.loc[best_global_macro_idx].to_dict()

    summary = {
        "stage": stage,
        "n_models": int(best_df["model_key"].nunique()),
        "n_scored_rows": int(len(scored_df)),
        "alpha_grid_size": int(len(alphas)),
        "quantile_grid_size": int(len(quantiles)),
        "best_by_model_class_agnostic_f1": best_df.to_dict(orient="records"),
        "best_by_model_macro_f1": best_macro_df.to_dict(orient="records"),
        "best_global_model_entry_by_class_agnostic_f1": best_global_row,
        "best_global_model_entry_by_macro_f1": best_global_macro_row,
        "notes": [
            "Alpha is feature-mix weight only; threshold on fused score is selected separately.",
            "Selection is in-sample; validate out-of-fold before fixing deployment policy.",
        ],
    }
    with (out_dir / f"fused_best_summary_{stage}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved fused selection artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
