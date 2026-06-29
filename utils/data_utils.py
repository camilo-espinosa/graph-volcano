from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CLASS_TO_ID = {"VT": 1, "LP": 2, "TR": 3, "AV": 4, "IC": 5}
VALID_CLASSES = tuple(CLASS_TO_ID.keys())

DEFAULT_AUGMENT_POLICIES = {
    "AV": (("shift", 0.50), ("shift_amp", 0.25), ("shift_noise", 0.25)),
    "IC": (("shift", 0.70), ("shift_amp", 0.20), ("shift_noise", 0.10)),
}


def patch_stacking_X(x: torch.Tensor, N: int = 256) -> torch.Tensor:
    """
    Convert waveform tensor to 2D patch image format expected by UNet.

    Input:
    - [S, T] or [B, S, T]

    Output:
    - [B, 1, N, N]

    This mirrors the transform used in EXAMPLE_TRAIN.CustomTrace2DDataset.
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(
            f"patch_stacking_X expected [S,T] or [B,S,T], got shape {tuple(x.shape)}"
        )

    batch_out = []
    for sample in x:
        patches = sample.unfold(1, N, N)
        patches = patches.permute(1, 0, 2)
        sample_img = patches.reshape(-1, N).unsqueeze(0)
        batch_out.append(sample_img)

    return torch.stack(batch_out, dim=0)


def patch_stacking_y(
    y: torch.Tensor,
    N: int = 256,
    n_classes: int = 6,
    n_stations: int = 8,
) -> torch.Tensor:
    """
    Convert one-hot labels to 2D patch image format expected by UNet loss/eval.

    Input:
    - [C, T] or [B, C, T]

    Output:
    - [B, C, N, N]

    This mirrors the transform used in EXAMPLE_TRAIN.CustomTrace2DDataset.
    """
    if y.ndim == 2:
        y = y.unsqueeze(0)
    if y.ndim != 3:
        raise ValueError(
            f"patch_stacking_y expected [C,T] or [B,C,T], got shape {tuple(y.shape)}"
        )

    if y.shape[1] != n_classes:
        raise ValueError(
            f"patch_stacking_y expected class dim={n_classes}, got {y.shape[1]}"
        )

    batch_out = []
    for sample in y:
        patches = sample.repeat(n_stations, 1, 1)
        patches = patches.permute(1, 0, 2)
        patches = patches.unfold(2, N, N)
        patches = patches.permute(0, 2, 1, 3)
        sample_img = patches.reshape(n_classes, -1, N)
        batch_out.append(sample_img)

    return torch.stack(batch_out, dim=0)


def activation_unstacking(
    img: torch.Tensor,
    len_window: int = 8192,
    N: int = 256,
    n_classes: int = 6,
    n_stations: int = 8,
) -> torch.Tensor:
    """
    Map patch-domain activations [B,C,N,N] back to trace-domain [B,C,T].

    This reproduces EXAMPLE_TRAIN.img_to_trace_y behavior.
    """
    if img.ndim != 4:
        raise ValueError(
            f"activation_unstacking expected [B,C,N,N], got shape {tuple(img.shape)}"
        )

    output = torch.zeros(
        [len(img), n_classes, len_window],
        dtype=img.dtype,
        device=img.device,
    )
    for idx, patches in enumerate(img):
        patches = patches.unfold(1, n_stations, n_stations)
        patches = patches.permute(0, 3, 1, 2).reshape(
            n_classes,
            n_stations,
            N * N // n_stations,
        )
        patches_y = patches.sum(dim=1)
        max_val = torch.max(patches_y)
        if max_val > 0:
            patches_y = patches_y / max_val
        output[idx] = patches_y

    return output


@dataclass(frozen=True)
class SplitSpec:
    repeat: int
    split_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def collect_volcano_samples(
    volcano_root: Path,
    class_to_id: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_to_id = class_to_id or CLASS_TO_ID

    all_paths: list[str] = []
    all_labels: list[str] = []
    all_label_ids: list[int] = []

    for class_name in sorted(class_to_id):
        class_dir = volcano_root / class_name
        if not class_dir.exists():
            continue
        for npy_path in sorted(class_dir.glob("*.npy")):
            all_paths.append(str(npy_path.as_posix()))
            all_labels.append(class_name)
            all_label_ids.append(class_to_id[class_name])

    if not all_paths:
        raise RuntimeError(f"No .npy files found under {volcano_root}")

    return (
        np.asarray(all_paths),
        np.asarray(all_labels),
        np.asarray(all_label_ids, dtype=np.int64),
    )


def save_manifest(
    output_path: Path,
    filepaths: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "filepaths": np.asarray(filepaths),
        "labels": np.asarray(labels),
        "label_ids": np.asarray(label_ids, dtype=np.int64),
    }
    if extra_fields:
        payload.update(extra_fields)
    np.savez(output_path, **payload)


def build_stratified_kfold_specs(
    labels: np.ndarray,
    n_splits: int = 5,
    random_seed: int = 42,
) -> list[SplitSpec]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    rng = np.random.default_rng(seed=random_seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]

    for class_name in sorted(np.unique(labels)):
        class_idx = np.where(labels == class_name)[0]
        rng.shuffle(class_idx)

        class_chunks = np.array_split(class_idx, n_splits)
        for fold_idx, chunk in enumerate(class_chunks):
            fold_parts[fold_idx].append(chunk.astype(np.int64, copy=False))

    split_specs: list[SplitSpec] = []
    for fold_idx in range(n_splits):
        test_idx = (
            np.concatenate(fold_parts[fold_idx])
            if fold_parts[fold_idx]
            else np.empty(0, dtype=np.int64)
        )

        train_parts: list[np.ndarray] = []
        for other_fold in range(n_splits):
            if other_fold == fold_idx:
                continue
            train_parts.extend(fold_parts[other_fold])

        train_idx = (
            np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
        )
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        split_specs.append(
            SplitSpec(
                repeat=1,
                split_id=fold_idx + 1,
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )

    return split_specs


def random_time_shift(
    x: np.ndarray,
    y: np.ndarray,
    max_shift: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(x, shift=shift, axis=1), np.roll(y, shift=shift, axis=1), shift


def amplitude_scaling(
    x: np.ndarray,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    scale = float(rng.uniform(low, high))
    return x * scale, scale


def add_noise(
    x: np.ndarray,
    std_factor: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    x_std = float(np.std(x))
    noise_std = max(std_factor * x_std, 1e-6)
    noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x + noise, noise_std


def augment_trace_for_storage(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    policy: str,
    rng: np.random.Generator,
    time_shift: int,
    amplitude_low: float,
    amplitude_high: float,
    noise_std_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_aug, y_aug, _ = random_time_shift(
        x_raw,
        y_raw,
        max_shift=time_shift,
        rng=rng,
    )

    if policy == "shift_amp":
        x_aug, _ = amplitude_scaling(
            x_aug,
            low=amplitude_low,
            high=amplitude_high,
            rng=rng,
        )
    elif policy == "shift_noise":
        x_aug, _ = add_noise(
            x_aug,
            std_factor=noise_std_factor,
            rng=rng,
        )
    elif policy != "shift":
        raise ValueError(f"Unknown augmentation policy: {policy}")

    return x_aug.astype(np.float32), y_aug.astype(np.float32)


def build_augmented_path(
    src_path: Path,
    volcano_root: Path,
    split_augmented_root: Path,
    aug_index: int,
    policy: str,
) -> Path:
    rel = src_path.relative_to(volcano_root)
    class_name = rel.parts[0]
    out_dir = split_augmented_root / class_name
    out_name = f"{src_path.stem}_aug_{aug_index:05d}_{policy}.npy"
    return out_dir / out_name


def infer_augment_target_counts(
    train_labels: np.ndarray,
    reference_classes: tuple[str, ...] = ("VT", "LP", "TR"),
    target_classes: tuple[str, ...] = ("AV", "IC"),
) -> dict[str, int]:
    ref_counts = []
    for class_name in reference_classes:
        count = int(np.sum(train_labels == class_name))
        if count > 0:
            ref_counts.append(count)

    if not ref_counts:
        raise RuntimeError(
            "Cannot infer augmentation target: no samples for reference classes VT/LP/TR in training split."
        )

    target_count = int(round(float(np.mean(ref_counts))))
    return {class_name: target_count for class_name in target_classes}


def expand_training_set_with_augmentation(
    train_paths: np.ndarray,
    train_labels: np.ndarray,
    train_label_ids: np.ndarray,
    volcano_root: Path,
    split_augmented_root: Path,
    random_seed: int,
    augment_policies: dict[str, tuple[tuple[str, float], ...]] | None = None,
    augment_time_shift: int = 1000,
    augment_amplitude_low: float = 0.85,
    augment_amplitude_high: float = 1.15,
    augment_noise_std_factor: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    augment_target_counts = infer_augment_target_counts(
        train_labels=train_labels,
    )
    augment_policies = augment_policies or DEFAULT_AUGMENT_POLICIES
    rng = np.random.default_rng(seed=random_seed)

    combined_filepaths = [str(p) for p in train_paths]
    combined_labels = [str(lbl) for lbl in train_labels]
    combined_label_ids = [int(lbl_id) for lbl_id in train_label_ids]
    is_augmented = [False for _ in range(len(train_paths))]

    augmented_paths: list[str] = []
    aug_counter = 0
    split_augmented_root.mkdir(parents=True, exist_ok=True)

    for class_name, target_count in augment_target_counts.items():
        class_idx = np.where(train_labels == class_name)[0]
        current_count = len(class_idx)
        needed = max(0, int(target_count) - int(current_count))

        print(
            f"Augmentation target for {class_name}: current={current_count}, target={target_count}, needed={needed}"
        )
        if needed == 0:
            continue
        if current_count == 0:
            raise RuntimeError(
                f"Cannot augment class {class_name}: no training samples available in this split."
            )

        class_policy = augment_policies.get(class_name)
        if not class_policy:
            raise RuntimeError(f"Missing augmentation policy for class '{class_name}'.")

        source_probs = np.array([p for _, p in class_policy], dtype=np.float32)
        source_probs = source_probs / np.sum(source_probs)
        policies = [name for name, _ in class_policy]

        for _ in range(needed):
            src_pos = int(rng.choice(class_idx))
            src_path = Path(str(train_paths[src_pos]))

            src_arr = np.load(src_path, mmap_mode="r")
            src_arr = np.asarray(src_arr, dtype=np.float32)
            x_raw = src_arr[:8, :]
            y_raw = src_arr[8:, :]

            policy = str(rng.choice(policies, p=source_probs))
            x_aug, y_aug = augment_trace_for_storage(
                x_raw,
                y_raw,
                policy=policy,
                rng=rng,
                time_shift=augment_time_shift,
                amplitude_low=augment_amplitude_low,
                amplitude_high=augment_amplitude_high,
                noise_std_factor=augment_noise_std_factor,
            )

            aug_path = build_augmented_path(
                src_path=src_path,
                volcano_root=volcano_root,
                split_augmented_root=split_augmented_root,
                aug_index=aug_counter,
                policy=policy,
            )
            aug_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(aug_path, np.concatenate([x_aug, y_aug], axis=0).astype(np.float32))

            aug_path_str = str(aug_path.as_posix())
            combined_filepaths.append(aug_path_str)
            combined_labels.append(class_name)
            combined_label_ids.append(CLASS_TO_ID[class_name])
            is_augmented.append(True)
            augmented_paths.append(aug_path_str)
            aug_counter += 1

    return (
        np.asarray(combined_filepaths),
        np.asarray(combined_labels),
        np.asarray(combined_label_ids, dtype=np.int64),
        np.asarray(is_augmented, dtype=bool),
        np.asarray(augmented_paths),
    )


def _stratified_train_val_split_from_train(
    labels: np.ndarray,
    rng: np.random.Generator,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in the open interval (0, 1)")

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_name in sorted(np.unique(labels)):
        class_idx = np.where(labels == class_name)[0]
        rng.shuffle(class_idx)
        total = len(class_idx)

        if total == 0:
            continue
        if total == 1:
            val_count = 0
        else:
            val_count = int(round(total * val_fraction))
            val_count = max(1, min(total - 1, val_count))

        val_parts.append(class_idx[:val_count])
        train_parts.append(class_idx[val_count:])

    train_idx = (
        np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    )
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx
