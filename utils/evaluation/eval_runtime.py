from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.evaluation.detection_prediction_utils import normalize_prediction_intervals
from utils.evaluation.metrics_core import (
    detection_temporal_iou_active_only_from_rows,
    macro_f1_6c_from_confusion_matrix,
)
from utils.training.losses_detection import EventDetectionLoss
from utils.evaluation.event_detection_metrics import EventDetectionMetrics
from utils.evaluation.event_targets import batch_segmentation_to_events
from utils.training.trainer_detection import (
    build_validation_event_predictions_dataframe,
    _resolve_event_detection_eval_matching,
    _resolve_event_detection_loss_weights,
)
from utils.training.train_utils import (
    MultiStation1DDataset,
    cleanup_gpu_cache,
    compute_event_f1_iou_multistation,
)

ACTIVE_EVENT_LABEL_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
def active_event_ids_from_label_ids(
    label_ids: np.ndarray,
) -> tuple[list[int], list[int]]:
    active_event_ids = sorted(
        [
            int(label_id)
            for label_id in np.unique(label_ids).tolist()
            if int(label_id) in ACTIVE_EVENT_LABEL_IDS
        ]
    )
    active_class_indices = [event_id - 1 for event_id in active_event_ids]
    return active_event_ids, active_class_indices


def evaluate_multistation_checkpoint(
    model: torch.nn.Module,
    test_npz_path: Path,
    batch_size: int,
    device: torch.device,
    scramble_stations: bool,
    station_scramble_seed: int,
) -> tuple[
    list[float],
    float,
    float,
    float,
    np.ndarray,
    int,
    list[int],
]:
    ds = MultiStation1DDataset(
        test_npz_path,
        scramble_stations=scramble_stations,
        station_scramble_seed=station_scramble_seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    active_event_ids, _ = active_event_ids_from_label_ids(ds.label_ids)

    model.eval()
    with torch.inference_mode():
        (
            f1_per_class,
            mean_f1,
            mean_iou,
            eval_loss,
            cm,
        ) = compute_event_f1_iou_multistation(
            model,
            loader,
            device,
            return_cm=True,
            return_val_loss=True,
            return_event_plot_payloads=False,
            save_event_plots=False,
            max_event_plots=0,
            epoch=None,
        )

    n_samples = int(len(ds))
    del ds, loader
    cleanup_gpu_cache()

    return (
        [float(x) for x in f1_per_class],
        float(mean_f1),
        float(mean_iou),
        float(eval_loss),
        cm,
        n_samples,
        active_event_ids,
    )


def evaluate_event_detection_checkpoint(
    model: torch.nn.Module,
    test_npz_path: Path,
    batch_size: int,
    device: torch.device,
    scramble_stations: bool,
    station_scramble_seed: int,
    model_spec: dict,
) -> tuple[
    list[float],
    float,
    float,
    float,
    np.ndarray,
    int,
    list[int],
    float,
]:
    ds = MultiStation1DDataset(
        test_npz_path,
        scramble_stations=scramble_stations,
        station_scramble_seed=station_scramble_seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    active_event_ids, _ = active_event_ids_from_label_ids(ds.label_ids)

    loss_weights = _resolve_event_detection_loss_weights(
        model_spec=model_spec, config={}
    )
    eval_matching = _resolve_event_detection_eval_matching(
        model_spec=model_spec, config={}
    )
    loss_fn = EventDetectionLoss(
        num_classes=6,
        loss_weights=loss_weights,
    )
    metrics_fn = EventDetectionMetrics(num_classes=6)

    model.eval()
    test_loss = 0.0
    all_predictions = {
        "class_logits": [],
        "start": [],
        "end": [],
        "mask_logits": [],
    }
    all_targets = []
    n_batches = 0

    with torch.inference_mode():
        for xb, y_onehot, _ in loader:
            xb = xb.to(device)
            predictions = normalize_prediction_intervals(model(xb))
            targets = batch_segmentation_to_events(y_onehot, normalize=True)
            all_targets.extend(targets)

            loss_dict = loss_fn(predictions, targets)
            test_loss += float(loss_dict["loss_total"].item())
            n_batches += 1

            for key in all_predictions:
                all_predictions[key].append(predictions[key].detach().cpu().numpy())

            del xb, y_onehot, predictions, targets, loss_dict

    avg_test_loss = float(test_loss / n_batches) if n_batches > 0 else float("inf")

    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

    predictions_df = build_validation_event_predictions_dataframe(
        all_predictions,
        all_targets,
        iou_threshold=float(eval_matching["match_iou_threshold"]),
        matching_strategy=str(eval_matching["matching_strategy"]),
        overlap_recall_threshold=float(eval_matching["overlap_recall_threshold"]),
    )
    temporal_iou_agnostic, _ = detection_temporal_iou_active_only_from_rows(
        predictions_df
    )

    detection_summary = metrics_fn.compute_detection_summary(
        all_predictions,
        all_targets,
        iou_threshold=float(eval_matching["match_iou_threshold"]),
        matching_strategy=str(eval_matching["matching_strategy"]),
        overlap_recall_threshold=float(eval_matching["overlap_recall_threshold"]),
    )
    f1_per_class = [
        float(detection_summary["per_class_f1"].get(class_id, 0.0))
        for class_id in range(1, 6)
    ]

    per_class_stats = detection_summary["per_class"]
    active_event_class_ids = [
        class_id
        for class_id in range(1, 6)
        if int(per_class_stats[class_id]["target_count"]) > 0
    ]
    if len(active_event_class_ids) == 0:
        raise RuntimeError(
            "No active event classes found in event-detection evaluation target set."
        )
    mean_f1 = float(
        macro_f1_6c_from_confusion_matrix(detection_summary["confusion_matrix"])
    )
    mean_iou = float(temporal_iou_agnostic)

    test_metrics = metrics_fn.evaluate_batch(all_predictions, all_targets)
    test_map = float(test_metrics.get("mAP", 0.0))
    cm = detection_summary["confusion_matrix"]

    n_samples = int(len(ds))
    del ds, loader
    cleanup_gpu_cache()

    return (
        [float(x) for x in f1_per_class],
        float(mean_f1),
        float(mean_iou),
        float(avg_test_loss),
        cm,
        n_samples,
        active_event_ids,
        test_map,
    )


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    trainer_kind: str,
    allowed_missing_keys: Sequence[str] = (),
    ignore_checkpoint_keys: Sequence[str] = (),
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_state = dict(ckpt["model_state_dict"])

    for key in ignore_checkpoint_keys:
        ckpt_state.pop(str(key), None)

    incompat = model.load_state_dict(ckpt_state, strict=False)
    missing_keys = sorted(
        set(incompat.missing_keys) - {str(key) for key in allowed_missing_keys}
    )
    unexpected_keys = sorted(set(incompat.unexpected_keys))

    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        raise RuntimeError(
            "Checkpoint/model mismatch detected while loading state_dict. "
            f"checkpoint={checkpoint_path} trainer_kind={trainer_kind} "
            f"missing_keys={missing_keys} unexpected_keys={unexpected_keys}"
        )

    del ckpt
    cleanup_gpu_cache()
