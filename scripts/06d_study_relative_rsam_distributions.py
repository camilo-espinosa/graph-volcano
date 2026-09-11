"""Study absolute and relative RSAM distributions on continuous data.

This script computes RSAM statistics for:
- reference events (grouped by class)
- background-only windows (no GT overlap)

Relative RSAM is defined as:
    relative_rsam = rsam_event / rsam_local_background

The outputs help decide RSAM thresholds with class-aware and duration-aware evidence.
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
        description=(
            "Compute class-wise absolute/relative RSAM distributions from continuous "
            "trace and background-only windows."
        )
    )
    parser.add_argument(
        "--continuous-npy",
        type=Path,
        default=Path("data/NVCh_10h_continuous_trace/NVCh_10h_continuous_trace.npy"),
        help="Continuous trace NPY [rows, T] with timestamp row + 8 stations.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path(
            "data/NVCh_10h_continuous_trace/NVCh_10h_continuous_trace_reference.csv"
        ),
        help="Reference CSV with columns event_type, idx_start, idx_end.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Window size used for background-only windows.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7000,
        help="Stride used for background-only windows.",
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
        "--duration-bins-sec",
        type=str,
        default="0,5,10,20,40,80,9999",
        help=(
            "Comma-separated bin edges in seconds for duration-aware summaries. "
            "Example: 0,5,10,20,40,80,9999"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/complete_experiment/rsam_distribution_study"),
        help="Output folder for diagnostic artifacts.",
    )
    return parser.parse_args()


def parse_duration_bins(raw: str) -> list[float]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) < 2:
        raise ValueError("--duration-bins-sec must contain at least two edges.")
    bins = [float(x) for x in parts]
    if any(not np.isfinite(x) for x in bins):
        raise ValueError("Duration bins contain non-finite values.")
    if any(bins[i + 1] <= bins[i] for i in range(len(bins) - 1)):
        raise ValueError("Duration bins must be strictly increasing.")
    return bins


def build_window_starts(total_len: int, window_size: int, stride: int) -> np.ndarray:
    if total_len < window_size:
        raise ValueError(
            f"Trace length {total_len} is shorter than window size {window_size}."
        )
    starts = list(range(0, total_len - window_size + 1, stride))
    last_start = total_len - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return np.asarray(starts, dtype=np.int64)


def load_reference_events(reference_csv: Path) -> pd.DataFrame:
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")

    ref = pd.read_csv(reference_csv)
    required = {"event_type", "idx_start", "idx_end"}
    if not required.issubset(ref.columns):
        raise ValueError(
            f"Reference CSV must contain {sorted(required)}, got {list(ref.columns)}"
        )

    rows: list[dict[str, int | str]] = []
    for _, row in ref.iterrows():
        class_name = str(row["event_type"]).strip()
        if class_name not in EVENT_CLASSES:
            continue
        start = int(row["idx_start"])
        end = int(row["idx_end"])
        if start > end:
            start, end = end, start
        rows.append(
            {
                "class": class_name,
                "idx_start": start,
                "idx_end": end,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid events loaded from reference CSV.")
    return out.sort_values(by=["idx_start", "idx_end"]).reset_index(drop=True)


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


def peak_from_segment(seg: np.ndarray) -> float:
    if seg.ndim != 2:
        raise ValueError(f"Expected segment [S,T], got {seg.shape}")
    if seg.shape[1] <= 0:
        return np.nan
    return float(np.max(np.abs(seg.astype(np.float64, copy=False))))


def local_background_segment(
    x_band: np.ndarray,
    *,
    idx_start: int,
    idx_end: int,
    occupied_mask: np.ndarray,
    context_multiplier: float,
) -> np.ndarray:
    if x_band.ndim != 2:
        raise ValueError(f"Expected x_band [S,T], got {x_band.shape}")
    if occupied_mask.ndim != 1:
        raise ValueError(f"Expected occupied_mask [T], got {occupied_mask.shape}")
    if x_band.shape[1] != occupied_mask.shape[0]:
        raise ValueError("x_band and occupied_mask length mismatch.")
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
        left_seg = x_band[:, left_s : left_e + 1]
        left_mask = ~occupied_mask[left_s : left_e + 1]
        if np.any(left_mask):
            parts.append(left_seg[:, left_mask])

    if right_s <= right_e:
        right_seg = x_band[:, right_s : right_e + 1]
        right_mask = ~occupied_mask[right_s : right_e + 1]
        if np.any(right_mask):
            parts.append(right_seg[:, right_mask])

    if len(parts) == 0:
        return np.empty((x_band.shape[0], 0), dtype=x_band.dtype)

    return np.concatenate(parts, axis=1)


def metric_summary(values: pd.Series, *, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_q01": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q50": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_q99": np.nan,
        }

    return {
        f"{prefix}_count": float(len(numeric)),
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_q01": float(numeric.quantile(0.01)),
        f"{prefix}_q05": float(numeric.quantile(0.05)),
        f"{prefix}_q10": float(numeric.quantile(0.10)),
        f"{prefix}_q25": float(numeric.quantile(0.25)),
        f"{prefix}_q50": float(numeric.quantile(0.50)),
        f"{prefix}_q75": float(numeric.quantile(0.75)),
        f"{prefix}_q90": float(numeric.quantile(0.90)),
        f"{prefix}_q95": float(numeric.quantile(0.95)),
        f"{prefix}_q99": float(numeric.quantile(0.99)),
    }


def main() -> None:
    args = parse_args()

    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    duration_bins_sec = parse_duration_bins(args.duration_bins_sec)

    if not args.continuous_npy.exists():
        raise FileNotFoundError(f"Continuous NPY not found: {args.continuous_npy}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cont = np.load(args.continuous_npy, mmap_mode="r")
    if cont.ndim != 2 or cont.shape[0] < 9:
        raise ValueError(
            "Continuous array must be [rows,T] with at least 9 rows "
            "(timestamp + 8 stations)."
        )

    x_stations = np.asarray(cont[1:9, :], dtype=np.float32)
    total_len = int(x_stations.shape[1])

    gt_df = load_reference_events(args.reference_csv)

    x_band = bandpass_trace(
        x_stations,
        sample_rate_hz=float(args.sample_rate_hz),
        low_hz=float(args.bandpass_low_hz),
        high_hz=float(args.bandpass_high_hz),
        order=int(args.bandpass_order),
    )

    occupied_mask = np.zeros(total_len, dtype=bool)
    for _, row in gt_df.iterrows():
        s = max(0, min(int(row["idx_start"]), total_len - 1))
        e = max(0, min(int(row["idx_end"]), total_len - 1))
        if s > e:
            s, e = e, s
        occupied_mask[s : e + 1] = True

    event_rows: list[dict[str, float | int | str]] = []
    for event_idx, row in gt_df.reset_index(drop=True).iterrows():
        class_name = str(row["class"])
        start = max(0, min(int(row["idx_start"]), total_len - 1))
        end = max(0, min(int(row["idx_end"]), total_len - 1))
        if start > end:
            start, end = end, start

        event_seg = x_band[:, start : end + 1]
        bg_seg = local_background_segment(
            x_band,
            idx_start=start,
            idx_end=end,
            occupied_mask=occupied_mask,
            context_multiplier=float(args.context_multiplier),
        )

        event_rsam = rsam_from_segment(event_seg)
        bg_rsam = rsam_from_segment(bg_seg)
        event_peak = peak_from_segment(event_seg)
        bg_peak = peak_from_segment(bg_seg)

        rel_rsam = (
            float(event_rsam / bg_rsam)
            if np.isfinite(event_rsam) and np.isfinite(bg_rsam) and bg_rsam > 0.0
            else np.nan
        )
        rel_peak = (
            float(event_peak / bg_peak)
            if np.isfinite(event_peak) and np.isfinite(bg_peak) and bg_peak > 0.0
            else np.nan
        )

        event_rows.append(
            {
                "source": "reference_event",
                "group": class_name,
                "event_index": int(event_idx),
                "idx_start": int(start),
                "idx_end": int(end),
                "duration_samples": int(end - start + 1),
                "duration_sec": float((end - start + 1) / float(args.sample_rate_hz)),
                "rsam_interval": event_rsam,
                "rsam_local_background": bg_rsam,
                "relative_rsam": rel_rsam,
                "peak_interval": event_peak,
                "peak_local_background": bg_peak,
                "relative_peak": rel_peak,
                "local_bg_samples": int(bg_seg.shape[1]),
            }
        )

    event_df = pd.DataFrame(event_rows)

    starts = build_window_starts(total_len, int(args.window_size), int(args.stride))
    ends = starts + int(args.window_size) - 1
    bg_window_rows: list[dict[str, float | int | str]] = []

    for i, start in enumerate(starts.tolist()):
        end = int(ends[i])
        if np.any(occupied_mask[int(start) : end + 1]):
            continue

        win_seg = x_band[:, int(start) : end + 1]
        bg_seg = local_background_segment(
            x_band,
            idx_start=int(start),
            idx_end=end,
            occupied_mask=occupied_mask,
            context_multiplier=float(args.context_multiplier),
        )

        win_rsam = rsam_from_segment(win_seg)
        bg_rsam = rsam_from_segment(bg_seg)
        win_peak = peak_from_segment(win_seg)
        bg_peak = peak_from_segment(bg_seg)

        rel_rsam = (
            float(win_rsam / bg_rsam)
            if np.isfinite(win_rsam) and np.isfinite(bg_rsam) and bg_rsam > 0.0
            else np.nan
        )
        rel_peak = (
            float(win_peak / bg_peak)
            if np.isfinite(win_peak) and np.isfinite(bg_peak) and bg_peak > 0.0
            else np.nan
        )

        bg_window_rows.append(
            {
                "source": "background_window",
                "group": "BG_ONLY_WINDOW",
                "window_index": int(i),
                "idx_start": int(start),
                "idx_end": int(end),
                "duration_samples": int(end - int(start) + 1),
                "duration_sec": float(
                    (end - int(start) + 1) / float(args.sample_rate_hz)
                ),
                "rsam_interval": win_rsam,
                "rsam_local_background": bg_rsam,
                "relative_rsam": rel_rsam,
                "peak_interval": win_peak,
                "peak_local_background": bg_peak,
                "relative_peak": rel_peak,
                "local_bg_samples": int(bg_seg.shape[1]),
            }
        )

    bg_df = pd.DataFrame(bg_window_rows)

    combined_df = pd.concat([event_df, bg_df], axis=0, ignore_index=True)
    combined_df.to_csv(
        output_dir / "rsam_relative_values_all_intervals.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    summary_rows: list[dict[str, float | int | str]] = []
    for group_name, grp in combined_df.groupby("group", sort=True):
        row: dict[str, float | int | str] = {"group": str(group_name)}
        row.update(metric_summary(grp["rsam_interval"], prefix="rsam_interval"))
        row.update(metric_summary(grp["relative_rsam"], prefix="relative_rsam"))
        row.update(metric_summary(grp["peak_interval"], prefix="peak_interval"))
        row.update(metric_summary(grp["relative_peak"], prefix="relative_peak"))
        row.update(metric_summary(grp["duration_sec"], prefix="duration_sec"))
        summary_rows.append(row)

    summary_df = (
        pd.DataFrame(summary_rows).sort_values(by=["group"]).reset_index(drop=True)
    )
    summary_df.to_csv(
        output_dir / "rsam_distribution_summary_by_group.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    duration_bin_labels = []
    for i in range(len(duration_bins_sec) - 1):
        duration_bin_labels.append(
            f"[{duration_bins_sec[i]},{duration_bins_sec[i + 1]})"
        )

    event_df = event_df.copy()
    event_df["duration_bin"] = pd.cut(
        event_df["duration_sec"],
        bins=duration_bins_sec,
        labels=duration_bin_labels,
        include_lowest=True,
        right=False,
    )

    duration_rows: list[dict[str, float | int | str]] = []
    for class_name in EVENT_CLASSES:
        cls_df = event_df[event_df["group"] == class_name]
        for bin_label in duration_bin_labels:
            bin_df = cls_df[cls_df["duration_bin"] == bin_label]
            if len(bin_df) == 0:
                continue
            row: dict[str, float | int | str] = {
                "class": class_name,
                "duration_bin": str(bin_label),
                "n_events": int(len(bin_df)),
            }
            row.update(metric_summary(bin_df["rsam_interval"], prefix="rsam_interval"))
            row.update(metric_summary(bin_df["relative_rsam"], prefix="relative_rsam"))
            duration_rows.append(row)

    duration_df = pd.DataFrame(duration_rows)
    duration_df.to_csv(
        output_dir / "rsam_duration_binned_summary_by_class.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    bg_values_abs = (
        pd.to_numeric(bg_df["rsam_interval"], errors="coerce").dropna().astype(float)
    )
    bg_values_rel = (
        pd.to_numeric(bg_df["relative_rsam"], errors="coerce").dropna().astype(float)
    )

    threshold_rows: list[dict[str, float | int | str]] = []
    retention_targets = [0.99, 0.95, 0.90, 0.85]

    for group_name in EVENT_CLASSES + ["ALL_EVENTS"]:
        if group_name == "ALL_EVENTS":
            grp = event_df
        else:
            grp = event_df[event_df["group"] == group_name]

        event_abs = (
            pd.to_numeric(grp["rsam_interval"], errors="coerce").dropna().astype(float)
        )
        event_rel = (
            pd.to_numeric(grp["relative_rsam"], errors="coerce").dropna().astype(float)
        )

        for metric_name, event_values, bg_values in [
            ("absolute_rsam", event_abs, bg_values_abs),
            ("relative_rsam", event_rel, bg_values_rel),
        ]:
            if len(event_values) == 0:
                continue
            for target in retention_targets:
                quantile_level = max(0.0, min(1.0, 1.0 - float(target)))
                threshold = float(event_values.quantile(quantile_level))
                tp_retention = float((event_values >= threshold).mean())
                bg_rejection = (
                    float((bg_values < threshold).mean())
                    if len(bg_values) > 0
                    else np.nan
                )
                threshold_rows.append(
                    {
                        "group": group_name,
                        "metric": metric_name,
                        "target_tp_retention": float(target),
                        "threshold": threshold,
                        "achieved_tp_retention": tp_retention,
                        "background_rejection": bg_rejection,
                        "n_event_intervals": int(len(event_values)),
                        "n_background_windows": int(len(bg_values)),
                    }
                )

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        output_dir / "rsam_threshold_candidates_by_group.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    manifest = {
        "continuous_npy": str(args.continuous_npy),
        "reference_csv": str(args.reference_csv),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "sample_rate_hz": float(args.sample_rate_hz),
        "bandpass_low_hz": float(args.bandpass_low_hz),
        "bandpass_high_hz": float(args.bandpass_high_hz),
        "bandpass_order": int(args.bandpass_order),
        "context_multiplier": float(args.context_multiplier),
        "duration_bins_sec": [float(x) for x in duration_bins_sec],
        "n_reference_events": int(len(event_df)),
        "n_background_windows": int(len(bg_df)),
    }
    (output_dir / "run_manifest_rsam_distribution_study.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    report_lines: list[str] = []
    report_lines.append("# RSAM Distribution Study")
    report_lines.append("")
    report_lines.append("## Scope")
    report_lines.append("")
    report_lines.append(f"- Reference events analyzed: {int(len(event_df))}")
    report_lines.append(f"- Background-only windows analyzed: {int(len(bg_df))}")
    report_lines.append(
        f"- Window size / stride: {int(args.window_size)} / {int(args.stride)}"
    )
    report_lines.append(
        f"- Relative RSAM definition: RSAM(interval) / RSAM(local background)"
    )
    report_lines.append("")

    report_lines.append("## Group Medians")
    report_lines.append("")
    report_lines.append(
        "| group | rsam_interval_q50 | relative_rsam_q50 | peak_interval_q50 | relative_peak_q50 |"
    )
    report_lines.append("|---|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        report_lines.append(
            "| "
            f"{row['group']} | "
            f"{float(row['rsam_interval_q50']):.4f} | "
            f"{float(row['relative_rsam_q50']):.4f} | "
            f"{float(row['peak_interval_q50']):.4f} | "
            f"{float(row['relative_peak_q50']):.4f} |"
        )

    report_lines.append("")
    report_lines.append("## Threshold Candidate File")
    report_lines.append("")
    report_lines.append(
        "- Use rsam_threshold_candidates_by_group.csv to choose per-class or global "
        "thresholds at fixed TP retention (0.99, 0.95, 0.90, 0.85), and inspect "
        "background rejection achieved by each candidate."
    )
    report_lines.append("")
    report_lines.append("## Generated Files")
    report_lines.append("")
    report_lines.append("- rsam_relative_values_all_intervals.csv")
    report_lines.append("- rsam_distribution_summary_by_group.csv")
    report_lines.append("- rsam_duration_binned_summary_by_class.csv")
    report_lines.append("- rsam_threshold_candidates_by_group.csv")
    report_lines.append("- run_manifest_rsam_distribution_study.json")

    (output_dir / "RSAM_DISTRIBUTION_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
