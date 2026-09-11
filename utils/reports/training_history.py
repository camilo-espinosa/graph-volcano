"""
Unified metrics reporting and output formatting for segmentation and detection models.

Handles:
- Epoch summary formatting (identical format for all model types)
- Training history CSV generation (compatible format for all models)
- Detection-specific metrics CSV (for MuSSED)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def format_epoch_summary_segmentation(
    epoch: int,
    train_loss: float,
    train_loss_dice: float,
    train_loss_ce: float,
    val_loss: float,
    val_loss_dice: Optional[float],
    val_loss_ce: Optional[float],
    mean_f1: float,
    mean_iou: float,
    best_epoch: int,
    epochs_without_improvement: int,
    early_stop_patience: int,
    saved_plot_count: int,
) -> str:
    """
    Format epoch summary for segmentation models.

    Output format:
        EPOCH 003 | train_loss=0.4567 val_loss=0.3456 mean_f1=0.8512 mean_iou=0.7643
                  best_epoch=002 no_improve=1/20 saved_best_plots=5
    """
    line = (
        f"EPOCH {epoch:03d} | "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"mean_f1={mean_f1:.4f} mean_iou={mean_iou:.4f} "
        f"best_epoch={best_epoch if best_epoch >= 0 else 'NA'} "
        f"no_improve={epochs_without_improvement}/{early_stop_patience} "
        f"saved_best_plots={saved_plot_count}"
    )
    return line


def format_epoch_summary_detection(
    epoch: int,
    train_loss: float,
    train_loss_class: float,
    train_loss_bbox: float,
    val_loss: float,
        mean_f1: float,
    mAP: float,
    best_epoch: int,
    epochs_without_improvement: int,
    early_stop_patience: int,
    saved_plot_count: int,
) -> str:
    """
    Format epoch summary for event detection models.

    Output format:
        EPOCH 003 | train_loss=0.8234 val_loss=0.7890 mean_f1=0.8512 mAP=0.7634
                  best_epoch=002 no_improve=1/20 saved_best_plots=5
    """
    line = (
        f"EPOCH {epoch:03d} | "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"mean_f1={mean_f1:.4f} mAP={mAP:.4f} "
        f"best_epoch={best_epoch if best_epoch >= 0 else 'NA'} "
        f"no_improve={epochs_without_improvement}/{early_stop_patience} "
        f"saved_best_plots={saved_plot_count}"
    )
    return line


def save_training_history_segmentation(
    metrics_rows: List[List[float]],
    output_dir: Path,
    filename: str = "training_metrics.csv",
):
    """
    Save training history for segmentation models.

    Creates CSV with columns:
        lr, epoch, train_loss, val_loss,
        VT_f1, LP_f1, TR_f1, AV_f1, IC_f1, mean_f1,
        mean_iou

    Args:
        metrics_rows: List of metric rows (one per epoch)
        output_dir: Directory to save CSV
        filename: Output filename
    """
    if not metrics_rows:
        return

    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "lr",
            "epoch",
            "train_loss",
            "val_loss",
            "VT_f1",
            "LP_f1",
            "TR_f1",
            "AV_f1",
            "IC_f1",
            "mean_f1",
            "mean_iou",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(
        output_dir / filename,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )


def save_training_history_detection(
    metrics_rows: List[List[float]],
    detection_metrics_rows: List[List[float]],
    output_dir: Path,
    main_filename: str = "training_metrics.csv",
    detection_filename: str = "detection_metrics.csv",
):
    """
    Save training history for event detection models.

    Creates two CSVs:

    1. Main CSV (compatible with segmentation format):
        lr, epoch, train_loss, val_loss,
        VT_f1, LP_f1, TR_f1, AV_f1, IC_f1, mean_f1,
        mean_iou

    2. Detection-specific CSV:
        epoch, mAP@0.1, mAP@0.3, mAP@0.5, mAP@0.7, mAP@0.9, mAP,
        macro_f1, VT_f1, LP_f1, TR_f1, AV_f1, IC_f1

    Args:
        metrics_rows: List of metric rows for main CSV
        detection_metrics_rows: List of detection metric rows
        output_dir: Directory to save CSVs
        main_filename: Output filename for main CSV
        detection_filename: Output filename for detection CSV
    """
    if not metrics_rows:
        return

    # Main CSV (compatible format)
    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "lr",
            "epoch",
            "train_loss",
            "val_loss",
            "VT_f1",
            "LP_f1",
            "TR_f1",
            "AV_f1",
            "IC_f1",
            "mean_f1",
            "mean_iou",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(
        output_dir / main_filename,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    # Detection-specific CSV
    if detection_metrics_rows:
        detection_df = pd.DataFrame(
            detection_metrics_rows,
            columns=[
                "epoch",
                "mAP@0.1",
                "mAP@0.3",
                "mAP@0.5",
                "mAP@0.7",
                "mAP@0.9",
                "mAP",
                "macro_f1",
                "VT_f1",
                "LP_f1",
                "TR_f1",
                "AV_f1",
                "IC_f1",
            ],
        )
        detection_df.to_csv(
            output_dir / detection_filename,
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )


def map_detection_metrics_to_compatible_columns(
    mAP_dict: Dict[str, float],
) -> Tuple[List[float], List[float], float, float]:
    """
    Map detection metrics to compatible segmentation columns.

    Allows detection model results to be compared with segmentation models
    using identical CSV columns.

    Args:
        mAP_dict: Detection metrics dict with keys like:
            - "mAP": overall mAP
            - "macro_f1": macro F1 across configured classes
            - "VT_f1", "LP_f1", "TR_f1", "AV_f1", "IC_f1"

    Returns:
        Tuple of:
        - per_class_f1: [VT, LP, TR, AV, IC] F1 scores
        - per_class_iou: [VT, LP, TR, AV, IC] IoU proxy scores
        - mean_f1: Overall macro F1
        - mean_iou: Overall mAP (as proxy for mean IoU)
    """
    # For detection models: keep macro/per-class F1 plus mAP.

    per_class_f1 = [
        mAP_dict.get("VT_f1", 0.0),
        mAP_dict.get("LP_f1", 0.0),
        mAP_dict.get("TR_f1", 0.0),
        mAP_dict.get("AV_f1", 0.0),
        mAP_dict.get("IC_f1", 0.0),
    ]

    per_class_iou = per_class_f1.copy()  # Same as F1 for detection

    mean_f1 = mAP_dict.get("macro_f1", 0.0)
    mean_iou = mAP_dict.get("mAP", 0.0)

    return per_class_f1, per_class_iou, mean_f1, mean_iou


def aggregate_fold_metrics(
    fold_summaries: List[Dict],
    trainer_kind: str,
) -> Dict[str, float]:
    """
    Aggregate metrics across all folds.

    Args:
        fold_summaries: List of fold summary dicts from train_one_*_fold()
        trainer_kind: "2d", "1d", or "event_detection"

    Returns:
        Aggregated metrics dict with mean and std for:
        - best_val_mean_f1
        - test_mean_f1
        - test_mean_iou
        - test_loss
        (and additional fields for detection models)
    """
    import numpy as np

    if not fold_summaries:
        return {}

    fold_results = {
        "best_val_mean_f1": [f["best_val_mean_f1"] for f in fold_summaries],
        "test_mean_f1": [f["test_mean_f1"] for f in fold_summaries],
        "test_mean_iou": [f["test_mean_iou"] for f in fold_summaries],
        "test_loss": [f["test_loss"] for f in fold_summaries],
    }

    aggregated = {}
    for metric_name, values in fold_results.items():
        aggregated[f"{metric_name}_mean"] = float(np.mean(values))
        aggregated[f"{metric_name}_std"] = float(np.std(values))
        aggregated[f"{metric_name}_min"] = float(np.min(values))
        aggregated[f"{metric_name}_max"] = float(np.max(values))

    if trainer_kind == "event_detection":
        # Add detection-specific aggregation
        test_mAP = [f.get("test_mAP", 0.0) for f in fold_summaries]
        aggregated["test_mAP_mean"] = float(np.mean(test_mAP))
        aggregated["test_mAP_std"] = float(np.std(test_mAP))

    aggregated["n_folds"] = len(fold_summaries)

    return aggregated


def format_aggregated_results(
    aggregated: Dict[str, float],
    trainer_kind: str,
) -> str:
    """
    Format aggregated results for console output.

    Args:
        aggregated: Aggregated metrics dict from aggregate_fold_metrics()
        trainer_kind: "2d", "1d", or "event_detection"

    Returns:
        Formatted string for printing
    """
    lines = [
        "=" * 80,
        f"AGGREGATED RESULTS ({aggregated['n_folds']} folds)",
        "=" * 80,
        f"Val Mean F1:     {aggregated['best_val_mean_f1_mean']:.4f} ± {aggregated['best_val_mean_f1_std']:.4f}",
        f"Test Mean F1:    {aggregated['test_mean_f1_mean']:.4f} ± {aggregated['test_mean_f1_std']:.4f}",
        f"Test Mean IoU:   {aggregated['test_mean_iou_mean']:.4f} ± {aggregated['test_mean_iou_std']:.4f}",
        f"Test Loss:       {aggregated['test_loss_mean']:.4f} ± {aggregated['test_loss_std']:.4f}",
    ]

    if trainer_kind == "event_detection":
        lines.insert(
            -1,
            f"Test mAP:        {aggregated['test_mAP_mean']:.4f} ± {aggregated['test_mAP_std']:.4f}",
        )

    lines.append("=" * 80)

    return "\n".join(lines)
