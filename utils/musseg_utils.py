from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

EVENT_LABEL_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
EVENT_LABEL_NAMES: dict[int, str] = {
    1: "VT",
    2: "LP",
    3: "TR",
    4: "AV",
    5: "IC",
}


def _station_permutation(n_stations: int, idx: int, base_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(base_seed) + int(idx))
    return rng.permutation(int(n_stations)).astype(np.int64, copy=False)


@dataclass(frozen=True)
class MuSSegBatch:
    x: torch.Tensor
    y_onehot: torch.Tensor
    y_label: torch.Tensor
    station_mask: torch.Tensor


def _infer_station_rows(total_rows: int, num_classes: int) -> int:
    station_rows = int(total_rows) - int(num_classes)
    if station_rows <= 0:
        raise ValueError(
            "Cannot infer number of station rows. "
            f"total_rows={total_rows}, num_classes={num_classes}."
        )
    return station_rows


class MuSSegWindowDataset(Dataset):
    """
    Dataset for MuSSeg manifests (.npz with filepaths/labels/label_ids).

    Each sample waveform file is expected as a 2D array [S + C, T], where:
    - first S rows are station traces
    - next C rows are one-hot target traces
    """

    def __init__(
        self,
        npz_path: Path,
        num_classes: int = 6,
        station_rows: int | None = None,
        use_zero_mask: bool = True,
        scramble_stations: bool = False,
        station_scramble_seed: int = 42,
    ):
        with np.load(npz_path) as data:
            self.filepaths = data["filepaths"].copy()
            self.labels = data["labels"].copy()
            self.label_ids = data["label_ids"].astype(np.int64, copy=True)

        self.num_classes = int(num_classes)
        self.station_rows = int(station_rows) if station_rows is not None else None
        self.use_zero_mask = bool(use_zero_mask)
        self.scramble_stations = bool(scramble_stations)
        self.station_scramble_seed = int(station_scramble_seed)

    def __len__(self) -> int:
        return int(len(self.filepaths))

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        arr = np.load(self.filepaths[idx], mmap_mode="r")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected sample array [rows,T], got shape {tuple(arr.shape)} at index {idx}."
            )

        station_rows = self.station_rows
        if station_rows is None:
            station_rows = _infer_station_rows(arr.shape[0], self.num_classes)

        x_raw = arr[:station_rows, :]
        y_raw = arr[station_rows : station_rows + self.num_classes, :]

        if y_raw.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class rows, got {y_raw.shape[0]} at index {idx}."
            )

        if self.use_zero_mask:
            station_mask_np = np.abs(x_raw).sum(axis=1) > 0.0
        else:
            station_mask_np = np.ones(x_raw.shape[0], dtype=np.bool_)

        if self.scramble_stations:
            permutation = _station_permutation(
                n_stations=x_raw.shape[0],
                idx=idx,
                base_seed=self.station_scramble_seed,
            )
            x_raw = x_raw[permutation, :]
            station_mask_np = station_mask_np[permutation]

        x = torch.from_numpy(x_raw)
        y_onehot = torch.from_numpy(y_raw)
        y_label = torch.tensor(int(self.label_ids[idx]), dtype=torch.long)
        station_mask = torch.from_numpy(station_mask_np.astype(np.bool_, copy=False))

        return x, y_onehot, y_label, station_mask


def musseg_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> MuSSegBatch:
    if len(batch) == 0:
        raise ValueError("Empty batch is not supported.")

    x_list, y_list, label_list, mask_list = zip(*batch)

    batch_size = len(x_list)
    max_stations = int(max(x.shape[0] for x in x_list))
    t_len = int(x_list[0].shape[1])
    n_classes = int(y_list[0].shape[0])

    x_batch = x_list[0].new_zeros((batch_size, max_stations, t_len))
    mask_batch = torch.zeros((batch_size, max_stations), dtype=torch.bool)

    for i, (x_i, mask_i) in enumerate(zip(x_list, mask_list)):
        n_stations_i = int(x_i.shape[0])
        if int(x_i.shape[1]) != t_len:
            raise ValueError(
                f"All samples in batch must share T. Got {x_i.shape[1]} and expected {t_len}."
            )
        x_batch[i, :n_stations_i, :] = x_i
        mask_batch[i, :n_stations_i] = mask_i

    y_batch = torch.stack(y_list, dim=0)
    label_batch = torch.stack(label_list, dim=0).long()

    if y_batch.ndim != 3 or y_batch.shape[1] != n_classes:
        raise ValueError(
            f"Unexpected y batch shape {tuple(y_batch.shape)}. Expected [B,{n_classes},T]."
        )

    return MuSSegBatch(
        x=x_batch,
        y_onehot=y_batch,
        y_label=label_batch,
        station_mask=mask_batch,
    )


def build_musseg_dataloader(
    npz_path: Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    num_classes: int = 6,
    station_rows: int | None = None,
    use_zero_mask: bool = True,
    scramble_stations: bool = False,
    station_scramble_seed: int = 42,
) -> tuple[MuSSegWindowDataset, DataLoader]:
    dataset = MuSSegWindowDataset(
        npz_path=npz_path,
        num_classes=num_classes,
        station_rows=station_rows,
        use_zero_mask=use_zero_mask,
        scramble_stations=scramble_stations,
        station_scramble_seed=station_scramble_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=musseg_collate_fn,
    )
    return dataset, loader


def _longest_event(bg_diff: np.ndarray) -> tuple[int, int]:
    start_indices = np.where(bg_diff == -1)[0]
    if len(start_indices) == 0:
        return 0, int(len(bg_diff) - 1)

    end_indices = np.where(bg_diff == 1)[0]
    if len(end_indices) == 0:
        return int(start_indices[0]), int(len(bg_diff) - 1)

    events: list[tuple[int, int, int]] = []
    for start in start_indices:
        valid_ends = end_indices[end_indices > start]
        if valid_ends.size > 0:
            end = int(valid_ends[0])
            events.append((int(start), end, int(end - start)))

    if len(events) == 0:
        return 0, int(len(bg_diff) - 1)

    best = max(events, key=lambda item: item[2])
    return int(best[0]), int(best[1])


def _predicted_event_from_output(out_np: np.ndarray) -> int:
    max_indices = np.argmax(out_np, axis=0)
    processed_out = np.eye(len(out_np), dtype=np.float32)[max_indices].T
    bg_diff = np.diff(processed_out[0])
    if np.abs(bg_diff).sum() != 0:
        start_idx, end_idx = _longest_event(bg_diff)
    else:
        start_idx, end_idx = 0, int(processed_out.shape[1] - 1)

    if end_idx <= start_idx:
        class_scores = processed_out[1:, :].sum(axis=1)
    else:
        class_scores = processed_out[1:, start_idx:end_idx].sum(axis=1)

    return int(np.argmax(class_scores) + 1)


def evaluate_musseg_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate MuSSeg classifier from event-level labels.

    Expects loader batches created by `musseg_collate_fn`.
    """
    model.eval()
    pred_labels: list[int] = []
    true_labels: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, MuSSegBatch):
                xb = batch.x.to(device)
                yb = batch.y_onehot.to(device)
                y_label = batch.y_label.to(device).long()
                station_mask = batch.station_mask.to(device)
            else:
                xb, yb, y_label, station_mask = batch
                xb = xb.to(device)
                yb = yb.to(device)
                y_label = y_label.to(device).long()
                station_mask = station_mask.to(device)

            out = model(xb, station_mask=station_mask)

            if out.ndim == 4:
                probs = torch.softmax(out, dim=2).mean(dim=1)
            elif out.ndim == 3:
                probs = torch.softmax(out, dim=1)
            else:
                raise ValueError(
                    f"Unexpected model output shape {tuple(out.shape)}; expected [B,C,T] or [B,S,C,T]."
                )

            for i in range(probs.shape[0]):
                out_np = probs[i].detach().cpu().numpy()
                pred_evt = _predicted_event_from_output(out_np)
                pred_labels.append(int(pred_evt))
                true_labels.append(int(y_label[i].item()))

            del xb, yb, y_label, station_mask, out, probs

    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(pred_labels, dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=list(EVENT_LABEL_IDS))
    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=list(EVENT_LABEL_IDS),
        average=None,
        zero_division=0,
    )
    mean_f1 = float(np.mean(f1_per_class)) if len(f1_per_class) > 0 else 0.0

    iou_per_class: list[float] = []
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        denom = tp + fp + fn
        iou_per_class.append(float(tp / denom) if denom > 0 else 0.0)
    mean_iou = float(np.mean(iou_per_class)) if len(iou_per_class) > 0 else 0.0

    return {
        "n_samples": int(len(y_true)),
        "f1_per_class": [float(x) for x in f1_per_class],
        "mean_f1": mean_f1,
        "iou_per_class": iou_per_class,
        "mean_iou": mean_iou,
        "confusion_matrix": cm,
        "event_label_ids": list(EVENT_LABEL_IDS),
        "event_label_names": [EVENT_LABEL_NAMES[k] for k in EVENT_LABEL_IDS],
    }
