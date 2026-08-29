"""Extract fixed background windows from NEW_DATASET AV traces.

Behavior per input trace:
1) Load trace.
2) Keep only samples where BG channel == 1 (BG channel defaults to row 10).
3) Apply a light low-pass filter on the stitched background stream.
4) Extract two fixed windows: first 8192 and last 8192 samples.
5) Apply 1-15 Hz Butterworth bandpass (N=5, fs=100 Hz).
6) Save both windows to data/NVCHVC/BG as <original>_bg1.npy and <original>_bg2.npy.

No manifests or metadata are produced.

Run:
    python scripts/00_extract_bg_windows_from_new_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.script_common import resolve_project_path

DEFAULT_TRACES_DIR = Path(r"D:\Camilo\Volcanes_UFRO\DATOS\NEW_DATASET\traces\AV")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "NVCHVC" / "BG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract two background-only 8192 windows per NEW_DATASET trace."
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=DEFAULT_TRACES_DIR,
        help="Folder with input .npy traces.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination folder for extracted BG windows.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Output window length in samples.",
    )
    parser.add_argument(
        "--bg-row",
        type=int,
        default=9,
        help="Zero-based BG segmentation row index in the trace array.",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.5,
        help="Treat BG as present where trace[bg_row] >= threshold.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Sampling rate used for filtering.",
    )
    parser.add_argument(
        "--stitch-lowpass-hz",
        type=float,
        default=10.0,
        help="Low-pass cutoff used after BG stitching (set <=0 to disable).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, process only the first N files.",
    )
    return parser.parse_args()


def apply_lowpass_if_enabled(
    x: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
) -> np.ndarray:
    if cutoff_hz <= 0.0:
        return x

    sos = signal.butter(
        N=3,
        Wn=float(cutoff_hz),
        btype="lowpass",
        fs=float(sample_rate_hz),
        output="sos",
    )
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def apply_bandpass_1_15_hz(
    x: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    sos = signal.butter(
        N=5,
        Wn=[1.0, 15.0],
        btype="bandpass",
        fs=float(sample_rate_hz),
        output="sos",
    )
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def normalize_window_global_max_abs(x: np.ndarray) -> np.ndarray:
    denom = float(np.max(np.abs(x))) if x.size > 0 else 1.0
    if denom <= 0.0:
        denom = 1.0
    return (x / denom).astype(np.float32, copy=False)


tag_bg = [np.ones([1,8192])]
tag_bg.extend([np.zeros([1,8192]) for i in range(5)])
tag_bg = np.concat(tag_bg,axis=0)


def main() -> None:
    args = parse_args()

    traces_dir = resolve_project_path(args.traces_dir, PROJECT_ROOT)
    output_dir = resolve_project_path(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    if int(args.window_size) <= 0:
        raise ValueError("--window-size must be > 0")
    if float(args.sample_rate_hz) <= 0.0:
        raise ValueError("--sample-rate-hz must be > 0")

    trace_paths = sorted(traces_dir.glob("*.npy"))
    if int(args.max_files) > 0:
        trace_paths = trace_paths[: int(args.max_files)]

    if not trace_paths:
        raise FileNotFoundError(f"No .npy files found in {traces_dir}")

    n_processed = 0
    n_skipped_shape = 0
    n_skipped_bg = 0
    n_saved = 0

    win = int(args.window_size)
    bg_row = int(args.bg_row)
    bg_thr = float(args.bg_threshold)

    for idx, path in enumerate(trace_paths, start=1):
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            n_skipped_shape += 1
            continue

        if arr.ndim != 2 or arr.shape[0] < max(11, bg_row + 1):
            n_skipped_shape += 1
            continue

        x_stations = np.asarray(arr[1:9, :], dtype=np.float32)
        bg_mask = np.asarray(arr[bg_row, :] >= bg_thr, dtype=bool)

        if x_stations.shape[1] != bg_mask.shape[0]:
            n_skipped_shape += 1
            continue

        stitched_bg = x_stations[:, bg_mask]
        if stitched_bg.shape[1] < 2 * win:
            n_skipped_bg += 1
            continue

        try:
            stitched_bg = apply_lowpass_if_enabled(
                stitched_bg,
                sample_rate_hz=float(args.sample_rate_hz),
                cutoff_hz=float(args.stitch_lowpass_hz),
            )
        except Exception:
            # If low-pass fails for numerical reasons, continue without it.
            stitched_bg = stitched_bg.astype(np.float32, copy=False)

        w1 = stitched_bg[:, :win]
        w2 = stitched_bg[:, -win:]

        w1 = apply_bandpass_1_15_hz(w1, float(args.sample_rate_hz))
        w2 = apply_bandpass_1_15_hz(w2, float(args.sample_rate_hz))
        w1 = normalize_window_global_max_abs(w1)
        w2 = normalize_window_global_max_abs(w2)
        w1 = np.concat([w1,tag_bg],axis=0)
        w2 = np.concat([w2,tag_bg],axis=0)
        stem = path.stem
        out_1 = output_dir / f"{stem}_bg1.npy"
        out_2 = output_dir / f"{stem}_bg2.npy"

        np.save(out_1, w1.astype(np.float32, copy=False))
        np.save(out_2, w2.astype(np.float32, copy=False))

        n_processed += 1
        n_saved += 2

        if idx % 100 == 0 or idx == len(trace_paths):
            print(
                f"Progress {idx}/{len(trace_paths)} | "
                f"processed={n_processed} skipped_shape={n_skipped_shape} "
                f"skipped_short_bg={n_skipped_bg} saved={n_saved}"
            )

    print("Extraction complete.")
    print(f"Input files scanned: {len(trace_paths)}")
    print(f"Traces processed: {n_processed}")
    print(f"Skipped (shape/row issues): {n_skipped_shape}")
    print(f"Skipped (insufficient BG for 2 windows): {n_skipped_bg}")
    print(f"Saved windows: {n_saved}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
