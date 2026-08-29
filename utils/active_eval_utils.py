from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.event_detection_loss import EventDetectionLoss
from utils.event_detection_metrics import EventDetectionMetrics
from utils.event_targets import batch_segmentation_to_events
from utils.trainer_detection import (
    _class_agnostic_detection_iou_from_rows,
    build_validation_event_predictions_dataframe,
    _resolve_event_detection_eval_matching,
    _resolve_event_detection_loss_weights,
)
from utils.train_utils import (
    MultiStation1DDataset,
    UNetPatchDataset,
    cleanup_gpu_cache,
    cm_eval,
    combined_dice_ce_loss_2d,
    compute_event_f1_iou_multistation,
    f1_score_from_confusion_matrix,
)

ACTIVE_EVENT_LABEL_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_EVENT_CLASS_MAP: dict[float, str] = {
    1.0: "VT",
    2.0: "LP",
    3.0: "TR",
    4.0: "AV",
    5.0: "IC",
}


def load_unet_shape_and_loss(
    experiment_root: Path,
    *,
    default_init_features: int = 16,
    default_depth: int = 5,
    default_dice_weight: float = 0.7,
    default_ce_weight: float = 0.3,
) -> tuple[int, int, float, float]:
    init_features = int(default_init_features)
    depth = int(default_depth)
    dice_weight = float(default_dice_weight)
    ce_weight = float(default_ce_weight)

    run_manifest_path = experiment_root / "run_manifest.json"
    if run_manifest_path.exists():
        with run_manifest_path.open("r", encoding="utf-8") as f:
            run_manifest = json.load(f)
        config = run_manifest.get("config", {})
        init_features = int(config.get("init_features", init_features))
        depth = int(config.get("depth", depth))
        dice_weight = float(config.get("dice_weight", dice_weight))
        ce_weight = float(config.get("ce_weight", ce_weight))

    return init_features, depth, dice_weight, ce_weight


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


def evaluate_unet_checkpoint(
    model: torch.nn.Module,
    test_npz_path: Path,
    batch_size: int,
    device: torch.device,
    dice_weight: float,
    ce_weight: float,
    scramble_stations: bool,
    station_scramble_seed: int,
    *,
    class_names: Sequence[str],
    len_window: int,
    im_size: int,
    event_class_map: dict[float, str] | None = None,
) -> tuple[list[float], float, float, float, np.ndarray, int, list[int]]:
    ds = UNetPatchDataset(
        test_npz_path,
        scramble_stations=scramble_stations,
        station_scramble_seed=station_scramble_seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    active_event_ids, _ = active_event_ids_from_label_ids(ds.label_ids)

    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.inference_mode():
        for xb, y_onehot, _ in loader:
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)
            out = model(xb)
            loss, _, _ = combined_dice_ce_loss_2d(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=dice_weight,
                ce_weight=ce_weight,
            )
            total_loss += float(loss.item())
            n_batches += 1
            del xb, y_onehot, out, loss

    mean_loss = float(total_loss / n_batches) if n_batches > 0 else 0.0

    cm = cm_eval(
        model=model,
        dataloader=loader,
        device=device,
        len_window=len_window,
        im_size=im_size,
        clases_list=event_class_map or DEFAULT_EVENT_CLASS_MAP,
        t_bg=0,
        t_cl=0,
    )
    f1_scores, _, _ = f1_score_from_confusion_matrix(cm)
    f1_scores = [float(x) for x in f1_scores]
    support = np.sum(cm, axis=1)
    active_mask = support > 0
    mean_f1 = (
        float(np.mean([f1_scores[i] for i, active in enumerate(active_mask) if active]))
        if np.any(active_mask)
        else 0.0
    )

    # Class-agnostic event-vs-background IoU over temporal masks.
    event_inter = 0
    event_union = 0
    with torch.inference_mode():
        for xb, y_onehot, _ in loader:
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)
            out = model(xb)
            pred_idx = torch.argmax(out, dim=1)
            true_idx = torch.argmax(y_onehot, dim=1)
            pred_event = pred_idx > 0
            true_event = true_idx > 0
            event_inter += int(torch.logical_and(pred_event, true_event).sum().item())
            event_union += int(torch.logical_or(pred_event, true_event).sum().item())
            del xb, y_onehot, out

    mean_iou = float(event_inter / event_union) if event_union > 0 else 0.0

    n_samples = int(len(ds))
    del ds, loader
    cleanup_gpu_cache()

    return (
        [float(x) for x in f1_scores],
        float(mean_f1),
        float(mean_iou),
        float(mean_loss),
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

    loss_weights = _resolve_event_detection_loss_weights(model_spec=model_spec, config={})
    eval_matching = _resolve_event_detection_eval_matching(model_spec=model_spec, config={})
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
    temporal_iou_agnostic, _ = _class_agnostic_detection_iou_from_rows(predictions_df)

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
        np.mean(
            [
                float(detection_summary["per_class_f1"].get(class_id, 0.0))
                for class_id in active_event_class_ids
            ]
        )
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
