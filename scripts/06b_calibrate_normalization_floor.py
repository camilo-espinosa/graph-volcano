"""Calibrate normalization-floor candidates from continuous trace amplitudes.

This script estimates robust floor values for per-window normalization used by
scripts/06_continuous_tests.py. It compares amplitude distributions between:

- windows overlapping reference events (event windows)
- windows with no reference events (background windows)

The goal is to choose a floor that dampens low-energy background amplification
while minimally affecting event windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

CLASS_NAME_TO_ID = {"VT": 1, "LP": 2, "TR": 3, "AV": 4, "IC": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate normalization-floor candidates from true-event vs background "
            "amplitude distributions on the continuous trace."
        )
    )
    parser.add_argument(
        "--continuous-npy",
        type=Path,
        default=Path("data/NVCh_10h_continuous_trace/NVCh_10h_continuous_trace.npy"),
        help="Path to continuous trace NPY [rows, T] (timestamp + stations).",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path(
            "data/NVCh_10h_continuous_trace/NVCh_10h_continuous_trace_reference.csv"
        ),
        help="Reference CSV with event_type, idx_start, idx_end.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Window size in samples.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7000,
        help="Stride in samples.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Sample rate used for bandpass/RSAM context.",
    )
    parser.add_argument(
        "--bandpass-low-hz",
        type=float,
        default=1.0,
        help="Bandpass lower cutoff.",
    )
    parser.add_argument(
        "--bandpass-high-hz",
        type=float,
        default=15.0,
        help="Bandpass upper cutoff.",
    )
    parser.add_argument(
        "--bandpass-order",
        type=int,
        default=4,
        help="Bandpass filter order.",
    )
    parser.add_argument(
        "--event-window-max-affected-limit",
        type=float,
        default=0.05,
        help=(
            "Maximum acceptable fraction of event windows whose max abs is below "
            "the floor (default 0.05)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/experiments/complete_experiment/normalization_floor_calibration"
        ),
        help="Output directory for calibration artifacts.",
    )
    return parser.parse_args()


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
            }
        )
    return pd.DataFrame(rows, columns=["class", "idx_start", "idx_end"])


def bandpass_windows(
    batch_x: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    sample_rate_hz: float,
    order: int,
) -> np.ndarray:
    if batch_x.ndim != 3:
        raise ValueError(f"Expected [B,S,T], got {batch_x.shape}")
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
    return sosfiltfilt(sos, batch_x, axis=-1).astype(np.float32, copy=False)


def quantile_dict(values: np.ndarray) -> dict[str, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Quantile input has no finite values.")

    return {
        "q01": float(np.quantile(finite_values, 0.01)),
        "q05": float(np.quantile(finite_values, 0.05)),
        "q10": float(np.quantile(finite_values, 0.10)),
        "q25": float(np.quantile(finite_values, 0.25)),
        "q50": float(np.quantile(finite_values, 0.50)),
        "q75": float(np.quantile(finite_values, 0.75)),
        "q90": float(np.quantile(finite_values, 0.90)),
        "q95": float(np.quantile(finite_values, 0.95)),
        "q99": float(np.quantile(finite_values, 0.99)),
    }


def main() -> None:
    args = parse_args()

    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if (
        args.event_window_max_affected_limit < 0
        or args.event_window_max_affected_limit > 1
    ):
        raise ValueError("--event-window-max-affected-limit must be in [0,1]")

    if not args.continuous_npy.exists():
        raise FileNotFoundError(f"Continuous NPY not found: {args.continuous_npy}")

    cont = np.load(args.continuous_npy, mmap_mode="r")
    if cont.ndim != 2 or cont.shape[0] < 9:
        raise ValueError("Continuous array must be [rows,T] with at least 9 rows")

    x_raw = np.asarray(cont[1:9, :], dtype=np.float32)
    total_len = int(x_raw.shape[1])
    starts = build_window_starts(total_len, int(args.window_size), int(args.stride))
    ends = starts + int(args.window_size) - 1

    gt_df = load_reference_events(args.reference_csv)
    if gt_df.empty:
        raise ValueError("No valid reference events loaded from reference CSV.")

    # Build overlap mask: event window if ANY GT event intersects.
    event_mask = np.zeros(len(starts), dtype=bool)
    for _, row in gt_df.iterrows():
        es = int(row["idx_start"])
        ee = int(row["idx_end"])
        overlap = (ends >= es) & (starts <= ee)
        event_mask = event_mask | overlap

    bg_mask = ~event_mask
    if not bg_mask.any():
        raise ValueError("No background windows found for this window/stride setup.")

    # Compute bandpassed trace exactly like continuous tests.
    x_band = bandpass_windows(
        x_raw[np.newaxis, :, :],
        low_hz=float(args.bandpass_low_hz),
        high_hz=float(args.bandpass_high_hz),
        sample_rate_hz=float(args.sample_rate_hz),
        order=int(args.bandpass_order),
    )[0]

    # Window max amplitudes (normalization denominator source) and window RSAM.
    window_max_raw = np.zeros(len(starts), dtype=np.float64)
    window_max_band = np.zeros(len(starts), dtype=np.float64)
    window_rsam_band = np.zeros(len(starts), dtype=np.float64)
    for i, s in enumerate(starts.tolist()):
        e = int(s) + int(args.window_size)
        wr = x_raw[:, int(s) : e]
        wb = x_band[:, int(s) : e]
        window_max_raw[i] = float(np.max(np.abs(wr)))
        window_max_band[i] = float(np.max(np.abs(wb)))

        rsam_k = np.mean(np.abs(wb.astype(np.float64)), axis=-1)
        zero_channel = np.all(np.isclose(wb, 0.0), axis=-1)
        rsam_k = np.where(zero_channel, np.nan, rsam_k)
        if np.all(np.isnan(rsam_k)):
            window_rsam_band[i] = np.nan
        else:
            window_rsam_band[i] = float(np.nanmedian(rsam_k))

    event_window_max_band = window_max_band[event_mask]
    bg_window_max_band = window_max_band[bg_mask]

    # Event-level amplitudes from GT intervals (raw + bandpassed RSAM).
    event_peak_raw = []
    event_rsam_band = []
    abs_band = np.abs(x_band.astype(np.float64, copy=False))
    nonzero_band = (x_band != 0.0).astype(np.int32, copy=False)
    abs_prefix = np.pad(np.cumsum(abs_band, axis=1), ((0, 0), (1, 0)), mode="constant")
    nz_prefix = np.pad(
        np.cumsum(nonzero_band, axis=1), ((0, 0), (1, 0)), mode="constant"
    )

    for _, row in gt_df.iterrows():
        s = int(max(0, min(int(row["idx_start"]), total_len - 1)))
        e = int(max(0, min(int(row["idx_end"]), total_len - 1)))
        if s > e:
            s, e = e, s

        seg_raw = x_raw[:, s : e + 1]
        event_peak_raw.append(float(np.max(np.abs(seg_raw))))

        seg_len = max(1, e - s + 1)
        sum_abs = abs_prefix[:, e + 1] - abs_prefix[:, s]
        nz_count = nz_prefix[:, e + 1] - nz_prefix[:, s]
        rsam_station = sum_abs / float(seg_len)
        rsam_station = np.where(nz_count > 0, rsam_station, np.nan)
        event_rsam_band.append(float(np.nanmedian(rsam_station)))

    event_peak_raw = np.asarray(event_peak_raw, dtype=np.float64)
    event_rsam_band = np.asarray(event_rsam_band, dtype=np.float64)

    # Candidate floors from BG quantiles + event-anchored fractions.
    candidates: list[dict[str, float | str]] = []

    for q in [0.90, 0.95, 0.975, 0.99]:
        floor = float(np.quantile(bg_window_max_band, q))
        bg_affected = float(np.mean(bg_window_max_band < floor))
        evt_affected = float(np.mean(event_window_max_band < floor))
        candidates.append(
            {
                "family": "bg_quantile",
                "name": f"bg_q{int(round(1000*q))/10:g}",
                "floor": floor,
                "bg_windows_affected": bg_affected,
                "event_windows_affected": evt_affected,
                "event_to_bg_affected_ratio": (
                    evt_affected / bg_affected if bg_affected > 0 else np.nan
                ),
            }
        )

    p10_event_window_max = float(np.quantile(event_window_max_band, 0.10))
    for alpha in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]:
        floor = float(alpha * p10_event_window_max)
        bg_affected = float(np.mean(bg_window_max_band < floor))
        evt_affected = float(np.mean(event_window_max_band < floor))
        candidates.append(
            {
                "family": "event_anchored",
                "name": f"alpha_{alpha:.2f}_of_event_p10",
                "floor": floor,
                "bg_windows_affected": bg_affected,
                "event_windows_affected": evt_affected,
                "event_to_bg_affected_ratio": (
                    evt_affected / bg_affected if bg_affected > 0 else np.nan
                ),
            }
        )

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values(
        by=["event_windows_affected", "bg_windows_affected", "floor"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    acceptable = cand_df[
        cand_df["event_windows_affected"] <= float(args.event_window_max_affected_limit)
    ]
    if len(acceptable) > 0:
        recommended = acceptable.sort_values(
            by=["bg_windows_affected", "floor"],
            ascending=[False, True],
        ).iloc[0]
    else:
        recommended = cand_df.iloc[0]

    summary = {
        "inputs": {
            "continuous_npy": str(args.continuous_npy),
            "reference_csv": str(args.reference_csv),
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "sample_rate_hz": float(args.sample_rate_hz),
            "bandpass_low_hz": float(args.bandpass_low_hz),
            "bandpass_high_hz": float(args.bandpass_high_hz),
            "bandpass_order": int(args.bandpass_order),
            "event_window_max_affected_limit": float(
                args.event_window_max_affected_limit
            ),
        },
        "counts": {
            "n_windows": int(len(starts)),
            "n_event_windows": int(event_mask.sum()),
            "n_bg_windows": int(bg_mask.sum()),
            "n_events_reference": int(len(gt_df)),
        },
        "distributions": {
            "window_max_band_event": quantile_dict(event_window_max_band),
            "window_max_band_bg": quantile_dict(bg_window_max_band),
            "window_rsam_band_event": quantile_dict(window_rsam_band[event_mask]),
            "window_rsam_band_bg": quantile_dict(window_rsam_band[bg_mask]),
            "event_peak_raw": quantile_dict(event_peak_raw),
            "event_rsam_band": quantile_dict(event_rsam_band),
        },
        "recommended": {
            "family": str(recommended["family"]),
            "name": str(recommended["name"]),
            "floor": float(recommended["floor"]),
            "bg_windows_affected": float(recommended["bg_windows_affected"]),
            "event_windows_affected": float(recommended["event_windows_affected"]),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "normalization_floor_calibration_summary.json"
    candidates_path = args.output_dir / "normalization_floor_candidates.csv"
    windows_path = args.output_dir / "window_energy_table.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cand_df.to_csv(candidates_path, index=False)

    window_df = pd.DataFrame(
        {
            "window_start": starts,
            "window_end": ends,
            "is_event_window": event_mask.astype(np.int32),
            "window_max_raw": window_max_raw,
            "window_max_band": window_max_band,
            "window_rsam_band": window_rsam_band,
        }
    )
    window_df.to_csv(windows_path, index=False)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {candidates_path}")
    print(f"Wrote: {windows_path}")
    print("Recommended floor:", float(recommended["floor"]))
    print(
        "Affected windows (bg/event):",
        float(recommended["bg_windows_affected"]),
        float(recommended["event_windows_affected"]),
    )


if __name__ == "__main__":
    main()
