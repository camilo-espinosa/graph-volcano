"""Run open confidence diagnostics on continuous-test outputs.

Implements the pending diagnostics for both raw and cleaned event pairs:
1) confidence separability (TP vs FP)
2) confidence-threshold response curves
3) TR raw-to-clean merge pathology with confidence deltas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_CLASSES = ["VT", "LP", "TR", "AV", "IC"]
PAIR_STAGES = ["raw", "cleaned"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute open confidence diagnostics from continuous test outputs."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/experiments/complete_experiment/continuous_tests_v2_confidence"
        ),
        help="Root folder with stride_* outputs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7000,
        help="Stride to analyze (reads output_root/stride_<stride>).",
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
        help="Optional comma-separated fold ids filter.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold sweep step in [0,1].",
    )
    return parser.parse_args()


def parse_csv_selection(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [v.strip() for v in raw_value.split(",") if v.strip()]
    if len(values) == 0:
        raise ValueError("Parsed empty CSV selection.")
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def parse_folds(raw_value: str | None) -> list[int] | None:
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


def ensure_pair_columns(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {
        "status",
        "class",
        "pred_confidence",
        "pred_idx_start",
        "pred_idx_end",
        "pred_duration",
    }
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
    return df.copy()


def ensure_detection_columns(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {"class", "idx_start", "idx_end", "confidence"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
    return df.copy()


def mann_whitney_auc(tp_values: np.ndarray, fp_values: np.ndarray) -> float:
    n_tp = int(len(tp_values))
    n_fp = int(len(fp_values))
    if n_tp == 0 or n_fp == 0:
        return np.nan

    combined = np.concatenate([tp_values, fp_values], axis=0)
    labels = np.concatenate(
        [np.ones(n_tp, dtype=np.int32), np.zeros(n_fp, dtype=np.int32)],
        axis=0,
    )
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_tp = float(ranks[labels == 1].sum())
    u_stat = rank_sum_tp - float(n_tp * (n_tp + 1) / 2)
    return float(u_stat / float(n_tp * n_fp))


def find_fold_roots(
    stride_root: Path,
    models_filter: list[str] | None,
    folds_filter: list[int] | None,
) -> list[Path]:
    if not stride_root.exists():
        raise FileNotFoundError(f"Stride folder not found: {stride_root}")

    model_dirs = sorted(
        [p for p in stride_root.iterdir() if p.is_dir()], key=lambda p: p.name
    )
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
                fold_dir / "event_pairs_raw.csv",
                fold_dir / "event_pairs_cleaned.csv",
                fold_dir / "raw_detections.csv",
                fold_dir / "cleaned_detections.csv",
            ]
            if all(path.exists() for path in required):
                fold_roots.append(fold_dir)

    if len(fold_roots) == 0:
        raise RuntimeError(f"No valid fold outputs found under {stride_root}")
    return fold_roots


def summarize_confidence_distribution(
    rows: pd.DataFrame,
    *,
    stage: str,
    model_key: str,
    fold: int,
    class_name: str,
) -> dict[str, float | int | str]:
    tp = rows[rows["status"] == "TP"]["pred_confidence"].to_numpy(dtype=np.float64)
    fp = rows[rows["status"] == "FP"]["pred_confidence"].to_numpy(dtype=np.float64)

    out: dict[str, float | int | str] = {
        "stage": stage,
        "model_key": model_key,
        "fold": int(fold),
        "class": class_name,
        "n_tp": int(len(tp)),
        "n_fp": int(len(fp)),
        "tp_conf_q10": float(np.quantile(tp, 0.10)) if len(tp) > 0 else np.nan,
        "tp_conf_q50": float(np.quantile(tp, 0.50)) if len(tp) > 0 else np.nan,
        "tp_conf_q90": float(np.quantile(tp, 0.90)) if len(tp) > 0 else np.nan,
        "fp_conf_q10": float(np.quantile(fp, 0.10)) if len(fp) > 0 else np.nan,
        "fp_conf_q50": float(np.quantile(fp, 0.50)) if len(fp) > 0 else np.nan,
        "fp_conf_q90": float(np.quantile(fp, 0.90)) if len(fp) > 0 else np.nan,
        "auc_tp_gt_fp": mann_whitney_auc(tp, fp),
    }
    if len(tp) > 0 and len(fp) > 0:
        out["median_gap_tp_minus_fp"] = float(np.median(tp) - np.median(fp))
    else:
        out["median_gap_tp_minus_fp"] = np.nan
    return out


def build_threshold_grid(step: float) -> np.ndarray:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"threshold-step must be in (0,1], got {step}")
    n = int(np.floor(1.0 / step))
    grid = np.linspace(0.0, 1.0, n + 1)
    if grid[-1] != 1.0:
        grid = np.append(grid, 1.0)
    return np.unique(np.clip(grid, 0.0, 1.0))


def compute_threshold_curve(
    pairs_df: pd.DataFrame,
    *,
    stage: str,
    model_key: str,
    fold: int,
    scope: str,
    class_name: str,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    work = pairs_df.copy()
    if scope == "class":
        work = work[work["class"] == class_name].copy()

    tp_total = int((work["status"] == "TP").sum())
    fp_total = int((work["status"] == "FP").sum())
    fn_total = int((work["status"] == "FN").sum())
    gt_total = int(tp_total + fn_total)

    pred_rows = work[work["status"].isin(["TP", "FP"])].copy()
    pred_rows = pred_rows[np.isfinite(pred_rows["pred_confidence"])].copy()

    rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds.tolist():
        kept = pred_rows[pred_rows["pred_confidence"] >= float(threshold)]
        tp_kept = int((kept["status"] == "TP").sum())
        fp_kept = int((kept["status"] == "FP").sum())

        precision = (
            float(tp_kept / (tp_kept + fp_kept)) if (tp_kept + fp_kept) > 0 else 0.0
        )
        recall = float(tp_kept / gt_total) if gt_total > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "stage": stage,
                "model_key": model_key,
                "fold": int(fold),
                "scope": scope,
                "class": class_name,
                "threshold": float(threshold),
                "tp_total": int(tp_total),
                "fp_total": int(fp_total),
                "fn_total": int(fn_total),
                "gt_total": int(gt_total),
                "tp_kept": int(tp_kept),
                "fp_kept": int(fp_kept),
                "tp_retention": float(tp_kept / tp_total) if tp_total > 0 else np.nan,
                "fp_reduction": (
                    float(1.0 - (fp_kept / fp_total)) if fp_total > 0 else np.nan
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def overlap_count(
    raw_df: pd.DataFrame,
    clean_start: int,
    clean_end: int,
) -> pd.DataFrame:
    return raw_df[
        (raw_df["idx_end"] >= clean_start) & (raw_df["idx_start"] <= clean_end)
    ]


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    stride_root = output_root / f"stride_{int(args.stride)}"

    models_filter = parse_csv_selection(args.models)
    folds_filter = parse_folds(args.folds)
    thresholds = build_threshold_grid(float(args.threshold_step))

    fold_roots = find_fold_roots(stride_root, models_filter, folds_filter)

    report_root = stride_root / "confidence_diagnostics_06e"
    report_root.mkdir(parents=True, exist_ok=True)

    pair_rows_all: list[pd.DataFrame] = []
    separability_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[pd.DataFrame] = []

    tr_lineage_rows: list[dict[str, float | int | str | bool]] = []

    for fold_root in fold_roots:
        model_key = fold_root.parent.name
        fold = int(fold_root.name.split("_")[-1])

        pairs_by_stage: dict[str, pd.DataFrame] = {}
        for stage_name, filename in [
            ("raw", "event_pairs_raw.csv"),
            ("cleaned", "event_pairs_cleaned.csv"),
        ]:
            pairs = ensure_pair_columns(
                read_csv_local(fold_root / filename),
                name=filename,
            )
            pairs["class"] = pairs["class"].astype(str)
            pairs["status"] = pairs["status"].astype(str)
            for col in [
                "pred_confidence",
                "pred_idx_start",
                "pred_idx_end",
                "pred_duration",
            ]:
                pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
            pairs_by_stage[stage_name] = pairs

            pred_subset = pairs[pairs["status"].isin(["TP", "FP"])].copy()
            pred_subset = pred_subset[
                np.isfinite(pred_subset["pred_confidence"])
            ].copy()
            pred_subset["stage"] = stage_name
            pred_subset["model_key"] = model_key
            pred_subset["fold"] = int(fold)
            pair_rows_all.append(pred_subset)

            classes = ["ALL"] + EVENT_CLASSES
            for class_name in classes:
                if class_name == "ALL":
                    cls_rows = pred_subset
                else:
                    cls_rows = pred_subset[pred_subset["class"] == class_name]
                separability_rows.append(
                    summarize_confidence_distribution(
                        cls_rows,
                        stage=stage_name,
                        model_key=model_key,
                        fold=int(fold),
                        class_name=class_name,
                    )
                )

            curve_rows.append(
                compute_threshold_curve(
                    pairs,
                    stage=stage_name,
                    model_key=model_key,
                    fold=int(fold),
                    scope="global",
                    class_name="ALL",
                    thresholds=thresholds,
                )
            )
            for class_name in EVENT_CLASSES:
                curve_rows.append(
                    compute_threshold_curve(
                        pairs,
                        stage=stage_name,
                        model_key=model_key,
                        fold=int(fold),
                        scope="class",
                        class_name=class_name,
                        thresholds=thresholds,
                    )
                )

        raw_det = ensure_detection_columns(
            read_csv_local(fold_root / "raw_detections.csv"),
            name="raw_detections.csv",
        )
        clean_det = ensure_detection_columns(
            read_csv_local(fold_root / "cleaned_detections.csv"),
            name="cleaned_detections.csv",
        )
        for col in ["idx_start", "idx_end", "confidence"]:
            raw_det[col] = pd.to_numeric(raw_det[col], errors="coerce")
            clean_det[col] = pd.to_numeric(clean_det[col], errors="coerce")

        raw_tr = raw_det[raw_det["class"].astype(str) == "TR"].copy()
        clean_tr = clean_det[clean_det["class"].astype(str) == "TR"].copy()

        raw_tr = raw_tr[
            np.isfinite(raw_tr["idx_start"]) & np.isfinite(raw_tr["idx_end"])
        ].copy()
        clean_tr = clean_tr[
            np.isfinite(clean_tr["idx_start"]) & np.isfinite(clean_tr["idx_end"])
        ].copy()

        raw_tr["idx_start"] = raw_tr["idx_start"].astype(np.int64)
        raw_tr["idx_end"] = raw_tr["idx_end"].astype(np.int64)
        clean_tr["idx_start"] = clean_tr["idx_start"].astype(np.int64)
        clean_tr["idx_end"] = clean_tr["idx_end"].astype(np.int64)

        clean_tr["duration"] = (clean_tr["idx_end"] - clean_tr["idx_start"] + 1).astype(
            np.int64
        )
        clean_conf_finite = pd.to_numeric(clean_tr["confidence"], errors="coerce")
        long_thr = (
            float(clean_tr["duration"].quantile(0.75)) if len(clean_tr) > 0 else np.nan
        )
        low_conf_thr = (
            float(clean_conf_finite.quantile(0.25))
            if clean_conf_finite.notna().any()
            else np.nan
        )

        for _, c_row in clean_tr.iterrows():
            c_start = int(c_row["idx_start"])
            c_end = int(c_row["idx_end"])
            c_conf = (
                float(c_row["confidence"])
                if np.isfinite(c_row["confidence"])
                else np.nan
            )
            c_duration = int(c_end - c_start + 1)

            overlaps = overlap_count(raw_tr, c_start, c_end)
            n_raw = int(len(overlaps))

            if n_raw > 0:
                raw_union_start = int(overlaps["idx_start"].min())
                raw_union_end = int(overlaps["idx_end"].max())
                raw_union_duration = int(raw_union_end - raw_union_start + 1)
                raw_conf_values = pd.to_numeric(overlaps["confidence"], errors="coerce")
                raw_conf_mean = float(raw_conf_values.mean())
                raw_conf_median = float(raw_conf_values.median())
                raw_conf_min = float(raw_conf_values.min())
                raw_conf_max = float(raw_conf_values.max())
            else:
                raw_union_start = np.nan
                raw_union_end = np.nan
                raw_union_duration = np.nan
                raw_conf_mean = np.nan
                raw_conf_median = np.nan
                raw_conf_min = np.nan
                raw_conf_max = np.nan

            is_multi = bool(n_raw > 1)
            duration_ratio = (
                float(c_duration / raw_union_duration)
                if np.isfinite(raw_union_duration) and raw_union_duration > 0
                else np.nan
            )
            long_flag = bool(np.isfinite(long_thr) and c_duration >= long_thr)
            low_conf_flag = bool(
                np.isfinite(low_conf_thr)
                and np.isfinite(c_conf)
                and c_conf <= low_conf_thr
            )
            harmful_flag = bool(is_multi and long_flag and low_conf_flag)

            tr_lineage_rows.append(
                {
                    "model_key": model_key,
                    "fold": int(fold),
                    "clean_idx_start": int(c_start),
                    "clean_idx_end": int(c_end),
                    "clean_duration": int(c_duration),
                    "clean_confidence": c_conf,
                    "n_overlapping_raw_tr": int(n_raw),
                    "is_multi_raw_merge": is_multi,
                    "raw_union_idx_start": raw_union_start,
                    "raw_union_idx_end": raw_union_end,
                    "raw_union_duration": raw_union_duration,
                    "duration_ratio_clean_over_raw_union": duration_ratio,
                    "raw_conf_mean": raw_conf_mean,
                    "raw_conf_median": raw_conf_median,
                    "raw_conf_min": raw_conf_min,
                    "raw_conf_max": raw_conf_max,
                    "long_duration_threshold_p75": long_thr,
                    "low_conf_threshold_p25": low_conf_thr,
                    "is_long_duration": long_flag,
                    "is_low_confidence": low_conf_flag,
                    "is_harmful_merge_candidate": harmful_flag,
                }
            )

    pair_rows_df = pd.concat(pair_rows_all, axis=0, ignore_index=True)
    separability_df = pd.DataFrame(separability_rows)
    curve_df = pd.concat(curve_rows, axis=0, ignore_index=True)
    tr_lineage_df = pd.DataFrame(tr_lineage_rows)

    pair_rows_df.to_csv(
        report_root / "confidence_pairs_tp_fp_raw_and_cleaned.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    separability_df.to_csv(
        report_root / "confidence_separability_by_class_stage_model_fold.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    curve_df.to_csv(
        report_root / "confidence_threshold_curves_raw_and_cleaned.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    tr_lineage_df.to_csv(
        report_root / "tr_merge_raw_to_clean_lineage.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    sep_agg = separability_df.groupby(
        ["stage", "model_key", "class"], as_index=False
    ).agg(
        n_tp=("n_tp", "sum"),
        n_fp=("n_fp", "sum"),
        auc_tp_gt_fp_mean=("auc_tp_gt_fp", "mean"),
        median_gap_tp_minus_fp_mean=("median_gap_tp_minus_fp", "mean"),
        tp_conf_q50_mean=("tp_conf_q50", "mean"),
        fp_conf_q50_mean=("fp_conf_q50", "mean"),
    )
    sep_agg.to_csv(
        report_root / "confidence_separability_aggregated_by_model.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    curve_agg = curve_df.groupby(
        ["stage", "model_key", "scope", "class", "threshold"], as_index=False
    ).agg(
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        tp_retention_mean=("tp_retention", "mean"),
        fp_reduction_mean=("fp_reduction", "mean"),
    )
    curve_agg.to_csv(
        report_root / "confidence_threshold_curves_aggregated_by_model.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    curve_macro = (
        curve_agg[curve_agg["scope"] == "class"]
        .groupby(["stage", "model_key", "threshold"], as_index=False)
        .agg(
            macro_f1_mean=("f1_mean", "mean"),
            macro_precision_mean=("precision_mean", "mean"),
            macro_recall_mean=("recall_mean", "mean"),
            macro_tp_retention_mean=("tp_retention_mean", "mean"),
            macro_fp_reduction_mean=("fp_reduction_mean", "mean"),
        )
    )
    curve_macro.to_csv(
        report_root / "confidence_threshold_macro_f1_by_model.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    curve_candidates = []
    for (stage, model_key, scope, class_name), grp in curve_agg.groupby(
        ["stage", "model_key", "scope", "class"]
    ):
        grp_sorted = grp.sort_values(
            by=["f1_mean", "precision_mean"], ascending=[False, False]
        )
        best_f1 = grp_sorted.iloc[0]
        curve_candidates.append(
            {
                "stage": stage,
                "model_key": model_key,
                "scope": scope,
                "class": class_name,
                "candidate_type": "best_f1",
                "threshold": float(best_f1["threshold"]),
                "precision_mean": float(best_f1["precision_mean"]),
                "recall_mean": float(best_f1["recall_mean"]),
                "f1_mean": float(best_f1["f1_mean"]),
                "tp_retention_mean": float(best_f1["tp_retention_mean"]),
                "fp_reduction_mean": float(best_f1["fp_reduction_mean"]),
            }
        )

        high_ret = grp[grp["tp_retention_mean"] >= 0.90].copy()
        if len(high_ret) > 0:
            chosen = high_ret.sort_values(
                by=["precision_mean", "f1_mean", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]
            curve_candidates.append(
                {
                    "stage": stage,
                    "model_key": model_key,
                    "scope": scope,
                    "class": class_name,
                    "candidate_type": "high_recall_guardrail",
                    "threshold": float(chosen["threshold"]),
                    "precision_mean": float(chosen["precision_mean"]),
                    "recall_mean": float(chosen["recall_mean"]),
                    "f1_mean": float(chosen["f1_mean"]),
                    "tp_retention_mean": float(chosen["tp_retention_mean"]),
                    "fp_reduction_mean": float(chosen["fp_reduction_mean"]),
                }
            )

    curve_candidates_df = pd.DataFrame(curve_candidates)
    curve_candidates_df.to_csv(
        report_root / "confidence_threshold_candidates.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    tr_fold_summary = tr_lineage_df.groupby(["model_key", "fold"], as_index=False).agg(
        n_clean_tr=("clean_idx_start", "count"),
        n_multi_raw_merges=("is_multi_raw_merge", "sum"),
        frac_multi_raw_merges=("is_multi_raw_merge", "mean"),
        median_duration_ratio=("duration_ratio_clean_over_raw_union", "median"),
        n_harmful_merge_candidates=("is_harmful_merge_candidate", "sum"),
        frac_harmful_merge_candidates=("is_harmful_merge_candidate", "mean"),
        clean_conf_median=("clean_confidence", "median"),
    )
    tr_fold_summary.to_csv(
        report_root / "tr_merge_pathology_summary_by_model_fold.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    tr_model_summary = tr_fold_summary.groupby("model_key", as_index=False).agg(
        n_folds=("fold", "count"),
        n_clean_tr_mean=("n_clean_tr", "mean"),
        frac_multi_raw_merges_mean=("frac_multi_raw_merges", "mean"),
        median_duration_ratio_mean=("median_duration_ratio", "mean"),
        frac_harmful_merge_candidates_mean=("frac_harmful_merge_candidates", "mean"),
        clean_conf_median_mean=("clean_conf_median", "mean"),
    )
    tr_model_summary.to_csv(
        report_root / "tr_merge_pathology_summary_by_model.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    report_lines: list[str] = []
    report_lines.append("# Confidence Open Diagnostics (06e)")
    report_lines.append("")
    report_lines.append(f"- Output root: {output_root}")
    report_lines.append(f"- Stride: stride_{int(args.stride)}")
    report_lines.append(f"- Model-fold runs analyzed: {len(fold_roots)}")
    report_lines.append(f"- Threshold step: {float(args.threshold_step):.4f}")
    report_lines.append("")

    report_lines.append("## Implemented Diagnostics")
    report_lines.append("")
    report_lines.append(
        "- Confidence separability TP vs FP on RAW and CLEANED event pairs."
    )
    report_lines.append(
        "- Confidence threshold response curves on RAW and CLEANED pairs (global + class-wise)."
    )
    report_lines.append("- TR raw-to-clean merge lineage and harmful-merge candidates.")
    report_lines.append("")

    report_lines.append("## Quick Snapshot")
    report_lines.append("")
    all_sep = (
        separability_df[(separability_df["class"] == "ALL")]
        .groupby("stage", as_index=False)
        .agg(
            auc_mean=("auc_tp_gt_fp", "mean"),
            median_gap_mean=("median_gap_tp_minus_fp", "mean"),
            n_tp=("n_tp", "sum"),
            n_fp=("n_fp", "sum"),
        )
    )
    if len(all_sep) > 0:
        report_lines.append(
            "| stage | auc_tp_gt_fp_mean | median_gap_tp_minus_fp_mean | n_tp | n_fp |"
        )
        report_lines.append("|---|---:|---:|---:|---:|")
        for _, row in all_sep.iterrows():
            report_lines.append(
                "| "
                f"{row['stage']} | {float(row['auc_mean']):.4f} | "
                f"{float(row['median_gap_mean']):.4f} | {int(row['n_tp'])} | {int(row['n_fp'])} |"
            )
        report_lines.append("")

    best_global = (
        curve_agg[(curve_agg["scope"] == "global") & (curve_agg["class"] == "ALL")]
        .sort_values(by=["f1_mean", "precision_mean"], ascending=[False, False])
        .groupby(["stage", "model_key"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_macro = (
        curve_macro.sort_values(
            by=["macro_f1_mean", "macro_precision_mean"],
            ascending=[False, False],
        )
        .groupby(["stage", "model_key"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    if len(best_global) > 0:
        report_lines.append("## Best Thresholds By Metric Family")
        report_lines.append("")
        report_lines.append(
            "| stage | model | best_global_f1 | threshold_at_global_f1 | best_macro_f1 | threshold_at_macro_f1 |"
        )
        report_lines.append("|---|---|---:|---:|---:|---:|")

        merged_best = best_global.merge(
            best_macro,
            on=["stage", "model_key"],
            how="left",
            suffixes=("_global", "_macro"),
        )
        for _, row in merged_best.iterrows():
            report_lines.append(
                "| "
                f"{row['stage']} | {row['model_key']} | {float(row['f1_mean']):.4f} | "
                f"{float(row['threshold_global']):.4f} | "
                f"{float(row['macro_f1_mean']):.4f} | {float(row['threshold_macro']):.4f} |"
            )
        report_lines.append("")

    report_lines.append("## Output Files")
    report_lines.append("")
    report_lines.append("- confidence_pairs_tp_fp_raw_and_cleaned.csv")
    report_lines.append("- confidence_separability_by_class_stage_model_fold.csv")
    report_lines.append("- confidence_separability_aggregated_by_model.csv")
    report_lines.append("- confidence_threshold_curves_raw_and_cleaned.csv")
    report_lines.append("- confidence_threshold_curves_aggregated_by_model.csv")
    report_lines.append("- confidence_threshold_macro_f1_by_model.csv")
    report_lines.append("- confidence_threshold_candidates.csv")
    report_lines.append("- tr_merge_raw_to_clean_lineage.csv")
    report_lines.append("- tr_merge_pathology_summary_by_model_fold.csv")
    report_lines.append("- tr_merge_pathology_summary_by_model.csv")

    (report_root / "CONFIDENCE_OPEN_DIAGNOSTICS_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "output_root": str(output_root),
        "stride": int(args.stride),
        "threshold_step": float(args.threshold_step),
        "n_model_fold_runs": int(len(fold_roots)),
        "models_filter": models_filter,
        "folds_filter": folds_filter,
    }
    (report_root / "run_manifest_confidence_open_diagnostics.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
