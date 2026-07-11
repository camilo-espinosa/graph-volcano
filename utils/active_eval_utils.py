from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    list[float],
    float,
    list[float],
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

    active_event_ids, active_class_indices = active_event_ids_from_label_ids(
        ds.label_ids
    )

    model.eval()
    with torch.inference_mode():
        (
            f1_per_class,
            mean_f1,
            iou_per_class,
            mean_iou,
            iou_all_classes,
            mean_iou_all,
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

    if len(active_class_indices) > 0:
        mean_f1 = float(np.mean([f1_per_class[i] for i in active_class_indices]))
        mean_iou = float(np.mean([iou_per_class[i] for i in active_class_indices]))

    n_samples = int(len(ds))
    del ds, loader
    cleanup_gpu_cache()

    return (
        [float(x) for x in f1_per_class],
        float(mean_f1),
        [float(x) for x in iou_per_class],
        float(mean_iou),
        [float(x) for x in iou_all_classes],
        float(mean_iou_all),
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
) -> tuple[list[float], float, list[float], float, float, np.ndarray, int, list[int]]:
    ds = UNetPatchDataset(
        test_npz_path,
        scramble_stations=scramble_stations,
        station_scramble_seed=station_scramble_seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    active_event_ids, active_class_indices = active_event_ids_from_label_ids(
        ds.label_ids
    )

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
    mean_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0

    iou_per_class = []
    for class_idx in range(len(class_names)):
        tp = float(cm[class_idx, class_idx])
        fp = float(cm[:, class_idx].sum() - tp)
        fn = float(cm[class_idx, :].sum() - tp)
        denom = tp + fp + fn
        iou_per_class.append(float(tp / denom) if denom > 0 else 0.0)
    mean_iou = float(np.mean(iou_per_class)) if len(iou_per_class) > 0 else 0.0

    if len(active_class_indices) > 0:
        mean_f1 = float(np.mean([f1_scores[i] for i in active_class_indices]))
        mean_iou = float(np.mean([iou_per_class[i] for i in active_class_indices]))

    n_samples = int(len(ds))
    del ds, loader
    cleanup_gpu_cache()

    return (
        [float(x) for x in f1_scores],
        float(mean_f1),
        [float(x) for x in iou_per_class],
        float(mean_iou),
        float(mean_loss),
        cm,
        n_samples,
        active_event_ids,
    )


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    trainer_kind: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_state = ckpt["model_state_dict"]

    if trainer_kind == "2d":
        model.load_state_dict(ckpt_state)
    else:
        model.load_state_dict(ckpt_state, strict=False)

    del ckpt
    cleanup_gpu_cache()
