"""Build ready-to-read section-A diagnostics from existing continuous test outputs.

This script consumes outputs produced by scripts/06_continuous_tests.py and
creates fold-level diagnostic CSVs plus a stride-level markdown report that can
be read quickly to confirm/reject core hypotheses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_CLASSES = ["VT", "LP", "TR", "AV", "IC"]

FIXED_EVENT_RSAM_THRESHOLD = 20.0
FIXED_SAME_CLASS_MERGE_GAP_SAMPLES = 150
FIXED_SEG_MAX_BG_HOLE_SAMPLES = 15
FIXED_NORMALIZATION_FLOOR = 160.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze existing continuous test outputs and generate immediate "
            "diagnostics + readable hypothesis report."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/experiments/complete_experiment/continuous_tests_v1"),
        help="Root folder containing stride_* outputs and run_manifest_continuous.json.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7000,
        help="Stride ID to analyze (reads output_root/stride_<stride>).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated model keys filter.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Optional comma-separated fold ids filter (1..5).",
    )
    parser.add_argument(
        "--top-k-windows",
        type=int,
        default=50,
        help="How many top no-GT burst windows to include per fold.",
    )
    return parser.parse_args()


def parse_csv_selection(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [v.strip() for v in raw_value.split(",") if v.strip()]
    if len(values) == 0:
        raise ValueError("Parsed empty CSV selection.")
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def parse_fold_selection(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    values = parse_csv_selection(raw_value)
    assert values is not None
    out: list[int] = []
    for value in values:
        fold = int(value)
        if fold < 1 or fold > 5:
            raise ValueError(f"Fold must be in [1..5], got {fold}.")
        if fold not in out:
            out.append(fold)
    return out


def read_csv_local(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",")


def ensure_detection_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "source",
        "window_idx",
        "window_start",
        "window_end",
        "event_class",
        "event_idx_start",
        "event_idx_end",
        "event_idx_start_in_window",
        "event_idx_end_in_window",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing columns in window_events_detailed.csv: {missing}")
    return out


def ensure_window_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "window_idx",
        "window_start",
        "window_end",
        "n_gt_events",
        "n_pred_raw_events",
        "n_pred_clean_events",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing columns in window_events_summary.csv: {missing}")
    return out


def ensure_pairs_columns(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = df.copy()
    required = [
        "status",
        "class",
        "pred_duration",
        "pred_idx_start",
        "pred_idx_end",
        "start_error",
        "end_error",
        "duration_error",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
    return out


def _describe(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {
            "count": 0.0,
            "mean": np.nan,
            "median": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q90": np.nan,
        }
    return {
        "count": float(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "q10": float(numeric.quantile(0.10)),
        "q25": float(numeric.quantile(0.25)),
        "q75": float(numeric.quantile(0.75)),
        "q90": float(numeric.quantile(0.90)),
    }


def _hist(
    values: pd.Series, *, bin_width: int, min_edge: int, max_edge: int
) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    bins = np.arange(min_edge, max_edge + bin_width, bin_width, dtype=np.int64)
    if bins[-1] < max_edge:
        bins = np.append(bins, max_edge)

    if numeric.empty:
        counts = np.zeros(len(bins) - 1, dtype=np.int64)
        edges = bins
    else:
        clipped = numeric.clip(lower=min_edge, upper=max_edge - 1e-9)
        counts, edges = np.histogram(clipped.to_numpy(), bins=bins)

    rows = []
    for i, count in enumerate(counts.tolist()):
        rows.append(
            {
                "bin_left": int(edges[i]),
                "bin_right": int(edges[i + 1]),
                "count": int(count),
            }
        )
    return pd.DataFrame(rows)


def build_a1(
    *,
    window_summary: pd.DataFrame,
    window_detailed: pd.DataFrame,
) -> pd.DataFrame:
    no_gt_windows = window_summary[window_summary["n_gt_events"] == 0][
        ["window_idx", "window_start", "window_end", "n_pred_clean_events"]
    ].copy()

    pred_clean_rows = window_detailed[window_detailed["source"] == "pred_clean"].copy()
    pred_clean_no_gt = pred_clean_rows.merge(
        no_gt_windows[["window_idx", "window_start", "window_end"]],
        on=["window_idx", "window_start", "window_end"],
        how="inner",
    )

    if pred_clean_no_gt.empty:
        out = no_gt_windows.sort_values(
            by=["n_pred_clean_events", "window_idx"],
            ascending=[False, True],
        ).copy()
        for class_name in EVENT_CLASSES:
            out[f"count_{class_name}"] = 0
        out["dominant_class"] = "NONE"
        out["dominant_class_fraction"] = 0.0
        return out.reset_index(drop=True)

    class_counts = (
        pred_clean_no_gt.groupby(["window_idx", "event_class"])
        .size()
        .unstack(fill_value=0)
    )
    class_counts = class_counts.reindex(columns=EVENT_CLASSES, fill_value=0)
    class_counts = class_counts.rename(
        columns={class_name: f"count_{class_name}" for class_name in EVENT_CLASSES}
    ).reset_index()

    out = no_gt_windows.merge(class_counts, on="window_idx", how="left")
    for class_name in EVENT_CLASSES:
        col = f"count_{class_name}"
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)

    count_cols = [f"count_{class_name}" for class_name in EVENT_CLASSES]
    count_matrix = out[count_cols].to_numpy(dtype=np.float64, copy=False)
    dominant_idx = count_matrix.argmax(axis=1)
    out["dominant_class"] = [EVENT_CLASSES[i] for i in dominant_idx.tolist()]

    total_pred = out["n_pred_clean_events"].replace(0, np.nan)
    row_idx = np.arange(count_matrix.shape[0], dtype=np.int64)
    dominant_counts = count_matrix[row_idx, dominant_idx]
    out["dominant_class_fraction"] = (
        pd.Series(dominant_counts, index=out.index) / total_pred
    ).fillna(0.0)

    out = out.sort_values(
        by=["n_pred_clean_events", "window_idx"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return out


def build_a2(
    raw_pairs: pd.DataFrame, clean_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_tr = raw_pairs[
        (raw_pairs["class"] == "TR") & (raw_pairs["status"].isin(["TP", "FP"]))
    ]
    clean_tr = clean_pairs[
        (clean_pairs["class"] == "TR") & (clean_pairs["status"].isin(["TP", "FP"]))
    ]

    summary_rows: list[dict[str, float | int | str]] = []
    for stage_name, frame in [("raw", raw_tr), ("cleaned", clean_tr)]:
        tp_count = int((frame["status"] == "TP").sum())
        fp_count = int((frame["status"] == "FP").sum())
        dur = _describe(frame["pred_duration"])
        summary_rows.append(
            {
                "stage": stage_name,
                "n_pred_events": int(len(frame)),
                "n_tp": tp_count,
                "n_fp": fp_count,
                "tp_fraction": float(tp_count / len(frame)) if len(frame) > 0 else 0.0,
                "duration_mean": dur["mean"],
                "duration_median": dur["median"],
                "duration_q10": dur["q10"],
                "duration_q25": dur["q25"],
                "duration_q75": dur["q75"],
                "duration_q90": dur["q90"],
            }
        )

    events_df = pd.concat(
        [
            raw_tr.assign(stage="raw")[
                ["stage", "status", "pred_duration", "pred_idx_start", "pred_idx_end"]
            ],
            clean_tr.assign(stage="cleaned")[
                ["stage", "status", "pred_duration", "pred_idx_start", "pred_idx_end"]
            ],
        ],
        axis=0,
        ignore_index=True,
    )

    return pd.DataFrame(summary_rows), events_df


def build_a3(clean_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tp_clean = clean_pairs[clean_pairs["status"] == "TP"].copy()

    stats_rows: list[dict[str, float | int | str]] = []
    hist_parts: list[pd.DataFrame] = []

    for class_name in EVENT_CLASSES:
        class_tp = tp_clean[tp_clean["class"] == class_name]
        for metric_name, min_edge, max_edge in [
            ("start_error", -2000, 2000),
            ("end_error", -2000, 2000),
            ("duration_error", -3000, 3000),
        ]:
            stats = _describe(class_tp[metric_name])
            stats_rows.append(
                {
                    "class": class_name,
                    "metric": metric_name,
                    "count": int(stats["count"]),
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "q10": stats["q10"],
                    "q25": stats["q25"],
                    "q75": stats["q75"],
                    "q90": stats["q90"],
                }
            )
            hist_df = _hist(
                class_tp[metric_name],
                bin_width=10,
                min_edge=min_edge,
                max_edge=max_edge,
            )
            hist_df["class"] = class_name
            hist_df["metric"] = metric_name
            hist_parts.append(hist_df)

    stats_df = pd.DataFrame(stats_rows)
    hist_df = pd.concat(hist_parts, axis=0, ignore_index=True)
    return stats_df, hist_df


def hypothesis_flags(
    *,
    a2_summary: pd.DataFrame,
    a3_stats: pd.DataFrame,
    a1_top: pd.DataFrame,
) -> dict[str, bool | float]:
    raw_row = a2_summary[a2_summary["stage"] == "raw"]
    clean_row = a2_summary[a2_summary["stage"] == "cleaned"]

    if len(raw_row) != 1 or len(clean_row) != 1:
        raise ValueError("Expected exactly one raw and one cleaned row in A2 summary.")

    raw_med = float(raw_row.iloc[0]["duration_median"])
    clean_med = float(clean_row.iloc[0]["duration_median"])
    raw_count = float(raw_row.iloc[0]["n_pred_events"])
    clean_count = float(clean_row.iloc[0]["n_pred_events"])

    tr_start = a3_stats[
        (a3_stats["class"] == "TR") & (a3_stats["metric"] == "start_error")
    ]
    tr_end = a3_stats[(a3_stats["class"] == "TR") & (a3_stats["metric"] == "end_error")]

    if len(tr_start) != 1 or len(tr_end) != 1:
        raise ValueError("Missing TR start/end error stats in A3 table.")

    tr_start_median = float(tr_start.iloc[0]["median"])
    tr_end_median = float(tr_end.iloc[0]["median"])

    under_segment_classes = ["VT", "LP", "IC"]
    under_segment_hits = 0
    for class_name in under_segment_classes:
        row = a3_stats[
            (a3_stats["class"] == class_name) & (a3_stats["metric"] == "duration_error")
        ]
        if len(row) != 1:
            raise ValueError(f"Missing duration_error stats for class {class_name}.")
        if float(row.iloc[0]["median"]) < 0:
            under_segment_hits += 1

    no_gt_windows_total = float(len(a1_top))
    no_gt_windows_with_fp = float((a1_top["n_pred_clean_events"] > 0).sum())
    fp_window_rate = (
        no_gt_windows_with_fp / no_gt_windows_total if no_gt_windows_total > 0 else 0.0
    )

    duration_ratio = (
        clean_med / raw_med if np.isfinite(raw_med) and raw_med > 0 else np.nan
    )

    return {
        "h_fp_background_windows_high": bool(fp_window_rate >= 0.50),
        "h_tr_merge_inflates_duration": bool(
            np.isfinite(duration_ratio) and duration_ratio > 1.05
        ),
        "h_tr_merge_reduces_count": bool(
            np.isfinite(raw_count)
            and np.isfinite(clean_count)
            and clean_count < raw_count
        ),
        "h_tr_boundary_early_start_late_end": bool(
            tr_start_median < 0 and tr_end_median > 0
        ),
        "h_vt_lp_ic_under_segmented": bool(under_segment_hits >= 2),
        "fp_window_rate_no_gt": float(fp_window_rate),
        "tr_duration_median_ratio_clean_vs_raw": float(duration_ratio),
        "tr_start_error_median": float(tr_start_median),
        "tr_end_error_median": float(tr_end_median),
    }


def check_fixed_operating_point(manifest_path: Path) -> dict[str, str | bool | float]:
    if not manifest_path.exists():
        return {
            "manifest_found": False,
            "compliant": False,
            "message": "run_manifest_continuous.json not found",
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    observed_event_rsam = float(manifest["event_rsam_threshold"])
    observed_merge_gap = int(manifest["same_class_merge_gap_samples"])
    observed_bg_hole = int(manifest["seg_max_bg_hole_samples"])
    observed_norm_floor = float(manifest["normalization_floor"])

    compliant = (
        abs(observed_event_rsam - FIXED_EVENT_RSAM_THRESHOLD) < 1e-9
        and observed_merge_gap == FIXED_SAME_CLASS_MERGE_GAP_SAMPLES
        and observed_bg_hole == FIXED_SEG_MAX_BG_HOLE_SAMPLES
        and abs(observed_norm_floor - FIXED_NORMALIZATION_FLOOR) < 1e-9
    )

    return {
        "manifest_found": True,
        "compliant": bool(compliant),
        "observed_event_rsam_threshold": observed_event_rsam,
        "observed_same_class_merge_gap_samples": observed_merge_gap,
        "observed_seg_max_bg_hole_samples": observed_bg_hole,
        "observed_normalization_floor": observed_norm_floor,
        "target_event_rsam_threshold": FIXED_EVENT_RSAM_THRESHOLD,
        "target_same_class_merge_gap_samples": FIXED_SAME_CLASS_MERGE_GAP_SAMPLES,
        "target_seg_max_bg_hole_samples": FIXED_SEG_MAX_BG_HOLE_SAMPLES,
        "target_normalization_floor": FIXED_NORMALIZATION_FLOOR,
    }


def find_fold_roots(
    stride_root: Path, models_filter: list[str] | None, folds_filter: list[int] | None
) -> list[Path]:
    if not stride_root.exists():
        raise FileNotFoundError(f"Stride folder not found: {stride_root}")

    model_dirs = [p for p in stride_root.iterdir() if p.is_dir()]
    model_dirs = sorted(model_dirs, key=lambda p: p.name)

    if models_filter is not None:
        model_dirs = [p for p in model_dirs if p.name in models_filter]

    fold_roots: list[Path] = []
    for model_dir in model_dirs:
        for fold_dir in sorted(
            [p for p in model_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        ):
            if not fold_dir.name.startswith("fold_"):
                continue
            fold_id = int(fold_dir.name.split("_")[-1])
            if folds_filter is not None and fold_id not in folds_filter:
                continue
            required = [
                fold_dir / "window_events_detailed.csv",
                fold_dir / "window_events_summary.csv",
                fold_dir / "event_pairs_raw.csv",
                fold_dir / "event_pairs_cleaned.csv",
            ]
            if all(path.exists() for path in required):
                fold_roots.append(fold_dir)

    if len(fold_roots) == 0:
        raise RuntimeError(
            "No fold outputs found with required files under stride root: "
            f"{stride_root}"
        )
    return fold_roots


def main() -> None:
    args = parse_args()
    if args.top_k_windows <= 0:
        raise ValueError("--top-k-windows must be > 0")

    output_root = args.output_root.resolve()
    stride_root = output_root / f"stride_{int(args.stride)}"

    models_filter = parse_csv_selection(args.models)
    folds_filter = parse_fold_selection(args.folds)
    fold_roots = find_fold_roots(stride_root, models_filter, folds_filter)

    report_root = stride_root / "immediate_diagnostics_06c"
    report_root.mkdir(parents=True, exist_ok=True)

    fixed_op = check_fixed_operating_point(output_root / "run_manifest_continuous.json")
    (report_root / "fixed_operating_point_check.json").write_text(
        json.dumps(fixed_op, indent=2),
        encoding="utf-8",
    )

    fold_report_rows: list[dict[str, float | int | str | bool]] = []

    for fold_root in fold_roots:
        model_key = fold_root.parent.name
        fold_name = fold_root.name
        fold_id = int(fold_name.split("_")[-1])

        window_summary = ensure_window_summary_columns(
            read_csv_local(fold_root / "window_events_summary.csv")
        )
        window_detailed = ensure_detection_columns(
            read_csv_local(fold_root / "window_events_detailed.csv")
        )
        raw_pairs = ensure_pairs_columns(
            read_csv_local(fold_root / "event_pairs_raw.csv"),
            name="event_pairs_raw.csv",
        )
        clean_pairs = ensure_pairs_columns(
            read_csv_local(fold_root / "event_pairs_cleaned.csv"),
            name="event_pairs_cleaned.csv",
        )

        fold_diag_root = fold_root / "immediate_diagnostics_06c"
        fold_diag_root.mkdir(parents=True, exist_ok=True)

        a1 = build_a1(window_summary=window_summary, window_detailed=window_detailed)
        a1.to_csv(
            fold_diag_root / "a1_fp_burst_structure_by_window.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )
        a1.head(int(args.top_k_windows)).to_csv(
            fold_diag_root / "a1_fp_burst_top_windows.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        a2_summary, a2_events = build_a2(raw_pairs=raw_pairs, clean_pairs=clean_pairs)
        a2_summary.to_csv(
            fold_diag_root / "a2_tr_merge_raw_vs_clean_summary.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )
        a2_events.to_csv(
            fold_diag_root / "a2_tr_pred_duration_events.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        a3_stats, a3_hist = build_a3(clean_pairs=clean_pairs)
        a3_stats.to_csv(
            fold_diag_root / "a3_boundary_error_stats_by_class.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )
        a3_hist.to_csv(
            fold_diag_root / "a3_boundary_error_histograms_by_class.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        flags = hypothesis_flags(a2_summary=a2_summary, a3_stats=a3_stats, a1_top=a1)

        top_window = a1.iloc[0]
        fold_report_rows.append(
            {
                "model_key": model_key,
                "fold": fold_id,
                "top_no_gt_window_idx": int(top_window["window_idx"]),
                "top_no_gt_window_pred_clean_events": int(
                    top_window["n_pred_clean_events"]
                ),
                "top_no_gt_window_dominant_class": str(top_window["dominant_class"]),
                "fp_window_rate_no_gt": float(flags["fp_window_rate_no_gt"]),
                "tr_duration_median_ratio_clean_vs_raw": float(
                    flags["tr_duration_median_ratio_clean_vs_raw"]
                ),
                "tr_start_error_median": float(flags["tr_start_error_median"]),
                "tr_end_error_median": float(flags["tr_end_error_median"]),
                "h_fp_background_windows_high": bool(
                    flags["h_fp_background_windows_high"]
                ),
                "h_tr_merge_inflates_duration": bool(
                    flags["h_tr_merge_inflates_duration"]
                ),
                "h_tr_merge_reduces_count": bool(flags["h_tr_merge_reduces_count"]),
                "h_tr_boundary_early_start_late_end": bool(
                    flags["h_tr_boundary_early_start_late_end"]
                ),
                "h_vt_lp_ic_under_segmented": bool(flags["h_vt_lp_ic_under_segmented"]),
            }
        )

    fold_report_df = pd.DataFrame(fold_report_rows).sort_values(
        by=["model_key", "fold"],
        ascending=[True, True],
    )
    fold_report_df.to_csv(
        report_root / "hypothesis_status_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    metrics_snapshot_path = output_root / "continuous_fold_summary.csv"
    has_metrics_snapshot = metrics_snapshot_path.exists()
    metrics_summary_df = pd.DataFrame()
    if has_metrics_snapshot:
        metrics_snapshot_df = read_csv_local(metrics_snapshot_path)
        required_cols = {
            "stride",
            "model_key",
            "fold",
            "clean_f1",
            "clean_macro_f1",
            "raw_f1",
            "raw_macro_f1",
        }
        missing_cols = sorted(required_cols.difference(metrics_snapshot_df.columns))
        if len(missing_cols) > 0:
            raise ValueError(
                "continuous_fold_summary.csv is missing required columns: "
                f"{missing_cols}"
            )

        metrics_snapshot_df = metrics_snapshot_df[
            metrics_snapshot_df["stride"].astype(int) == int(args.stride)
        ].copy()
        metrics_snapshot_df = metrics_snapshot_df[
            [
                "model_key",
                "fold",
                "raw_f1",
                "raw_macro_f1",
                "clean_f1",
                "clean_macro_f1",
            ]
        ].copy()
        metrics_snapshot_df.to_csv(
            report_root / "evaluation_f1_by_fold.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        metrics_summary_df = (
            metrics_snapshot_df.groupby("model_key", as_index=False)
            .agg(
                raw_f1_mean=("raw_f1", "mean"),
                raw_macro_f1_mean=("raw_macro_f1", "mean"),
                clean_f1_mean=("clean_f1", "mean"),
                clean_macro_f1_mean=("clean_macro_f1", "mean"),
            )
            .sort_values(
                by=["clean_macro_f1_mean", "clean_f1_mean"], ascending=[False, False]
            )
        )
        metrics_summary_df.to_csv(
            report_root / "evaluation_f1_by_model.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

    aggregate = {
        "n_model_folds": int(len(fold_report_df)),
        "mean_fp_window_rate_no_gt": float(
            fold_report_df["fp_window_rate_no_gt"].mean()
        ),
        "mean_tr_duration_median_ratio_clean_vs_raw": float(
            fold_report_df["tr_duration_median_ratio_clean_vs_raw"].mean()
        ),
        "fraction_h_fp_background_windows_high_true": float(
            fold_report_df["h_fp_background_windows_high"].mean()
        ),
        "fraction_h_tr_merge_inflates_duration_true": float(
            fold_report_df["h_tr_merge_inflates_duration"].mean()
        ),
        "fraction_h_tr_boundary_early_start_late_end_true": float(
            fold_report_df["h_tr_boundary_early_start_late_end"].mean()
        ),
        "fraction_h_vt_lp_ic_under_segmented_true": float(
            fold_report_df["h_vt_lp_ic_under_segmented"].mean()
        ),
    }
    (report_root / "hypothesis_aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )

    md_lines: list[str] = []
    md_lines.append("# Immediate Diagnostics Report (06c)")
    md_lines.append("")
    md_lines.append(f"- Output root: {output_root}")
    md_lines.append(f"- Stride: stride_{int(args.stride)}")
    md_lines.append(f"- Model-fold runs analyzed: {int(len(fold_report_df))}")
    md_lines.append("")
    md_lines.append("## Fixed Operating Point Check")
    md_lines.append("")

    md_lines.append("## F1 Snapshot (Class-Agnostic and Class-Specific)")
    md_lines.append("")
    if not has_metrics_snapshot:
        md_lines.append(
            "- continuous_fold_summary.csv was not found; skipped F1 snapshot section."
        )
    elif len(metrics_summary_df) == 0:
        md_lines.append(
            "- No rows found for selected stride in continuous_fold_summary.csv."
        )
    else:
        md_lines.append(
            "| model | clean_f1_mean | clean_macro_f1_mean | raw_f1_mean | raw_macro_f1_mean |"
        )
        md_lines.append("|---|---:|---:|---:|---:|")
        for _, row in metrics_summary_df.iterrows():
            md_lines.append(
                "| "
                f"{row['model_key']} | {float(row['clean_f1_mean']):.4f} | "
                f"{float(row['clean_macro_f1_mean']):.4f} | {float(row['raw_f1_mean']):.4f} | "
                f"{float(row['raw_macro_f1_mean']):.4f} |"
            )
    md_lines.append("")
    if not bool(fixed_op["manifest_found"]):
        md_lines.append(
            "- Manifest not found, cannot verify operating point compliance."
        )
    else:
        md_lines.append(
            f"- Compliant with section-B defaults: {bool(fixed_op['compliant'])}"
        )
        md_lines.append(
            "- Observed values: "
            f"event_rsam_threshold={fixed_op['observed_event_rsam_threshold']}, "
            f"same_class_merge_gap_samples={fixed_op['observed_same_class_merge_gap_samples']}, "
            f"seg_max_bg_hole_samples={fixed_op['observed_seg_max_bg_hole_samples']}, "
            f"normalization_floor={fixed_op['observed_normalization_floor']}"
        )
        md_lines.append(
            "- Target values: "
            f"event_rsam_threshold={fixed_op['target_event_rsam_threshold']}, "
            f"same_class_merge_gap_samples={fixed_op['target_same_class_merge_gap_samples']}, "
            f"seg_max_bg_hole_samples={fixed_op['target_seg_max_bg_hole_samples']}, "
            f"normalization_floor={fixed_op['target_normalization_floor']}"
        )
    md_lines.append("")

    md_lines.append("## A1/A2/A3 Hypothesis Status by Model/Fold")
    md_lines.append("")
    md_lines.append(
        "| model | fold | fp_rate_no_gt | tr_duration_ratio_clean_raw | TR early/late bias | VT/LP/IC under-seg |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in fold_report_df.iterrows():
        md_lines.append(
            "| "
            f"{row['model_key']} | {int(row['fold'])} | "
            f"{float(row['fp_window_rate_no_gt']):.3f} | "
            f"{float(row['tr_duration_median_ratio_clean_vs_raw']):.3f} | "
            f"{bool(row['h_tr_boundary_early_start_late_end'])} | "
            f"{bool(row['h_vt_lp_ic_under_segmented'])} |"
        )
    md_lines.append("")

    md_lines.append("## Aggregate")
    md_lines.append("")
    md_lines.append(
        f"- Mean no-GT FP window rate: {aggregate['mean_fp_window_rate_no_gt']:.3f}"
    )
    md_lines.append(
        "- Mean TR duration median ratio (clean/raw): "
        f"{aggregate['mean_tr_duration_median_ratio_clean_vs_raw']:.3f}"
    )
    md_lines.append(
        "- Fraction with strong no-GT FP hypothesis true: "
        f"{aggregate['fraction_h_fp_background_windows_high_true']:.3f}"
    )
    md_lines.append(
        "- Fraction with TR merge inflation hypothesis true: "
        f"{aggregate['fraction_h_tr_merge_inflates_duration_true']:.3f}"
    )
    md_lines.append(
        "- Fraction with TR early-start/late-end hypothesis true: "
        f"{aggregate['fraction_h_tr_boundary_early_start_late_end_true']:.3f}"
    )
    md_lines.append(
        "- Fraction with VT/LP/IC under-segmentation hypothesis true: "
        f"{aggregate['fraction_h_vt_lp_ic_under_segmented_true']:.3f}"
    )
    md_lines.append("")

    md_lines.append("## Generated Files")
    md_lines.append("")
    md_lines.append("- stride-level:")
    md_lines.append("  - immediate_diagnostics_06c/hypothesis_status_by_fold.csv")
    md_lines.append("  - immediate_diagnostics_06c/hypothesis_aggregate.json")
    md_lines.append("  - immediate_diagnostics_06c/fixed_operating_point_check.json")
    if has_metrics_snapshot:
        md_lines.append("  - immediate_diagnostics_06c/evaluation_f1_by_fold.csv")
        md_lines.append("  - immediate_diagnostics_06c/evaluation_f1_by_model.csv")
    md_lines.append("- fold-level (for each model/fold):")
    md_lines.append("  - immediate_diagnostics_06c/a1_fp_burst_structure_by_window.csv")
    md_lines.append("  - immediate_diagnostics_06c/a1_fp_burst_top_windows.csv")
    md_lines.append(
        "  - immediate_diagnostics_06c/a2_tr_merge_raw_vs_clean_summary.csv"
    )
    md_lines.append("  - immediate_diagnostics_06c/a2_tr_pred_duration_events.csv")
    md_lines.append(
        "  - immediate_diagnostics_06c/a3_boundary_error_stats_by_class.csv"
    )
    md_lines.append(
        "  - immediate_diagnostics_06c/a3_boundary_error_histograms_by_class.csv"
    )

    report_path = report_root / "IMMEDIATE_DIAGNOSTICS_REPORT.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
