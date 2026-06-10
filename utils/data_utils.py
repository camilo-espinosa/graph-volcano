from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASS_TO_ID = {"VT": 1, "LP": 2, "TR": 3, "AV": 4, "IC": 5}
VALID_CLASSES = tuple(CLASS_TO_ID.keys())

DEFAULT_AUGMENT_TARGET_COUNTS = {"AV": 1500, "IC": 1500}
DEFAULT_AUGMENT_POLICIES = {
    "AV": (("shift", 0.50), ("shift_amp", 0.25), ("shift_noise", 0.25)),
    "IC": (("shift", 0.70), ("shift_amp", 0.20), ("shift_noise", 0.10)),
}


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


def _stratified_half_split(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []

    for class_name in sorted(np.unique(labels)):
        class_idx = np.where(labels == class_name)[0]
        rng.shuffle(class_idx)

        midpoint = len(class_idx) // 2
        left_parts.append(class_idx[:midpoint])
        right_parts.append(class_idx[midpoint:])

    left_idx = np.concatenate(left_parts) if left_parts else np.empty(0, dtype=np.int64)
    right_idx = (
        np.concatenate(right_parts) if right_parts else np.empty(0, dtype=np.int64)
    )
    rng.shuffle(left_idx)
    rng.shuffle(right_idx)
    return left_idx, right_idx


def build_5x2_split_specs(
    labels: np.ndarray,
    repeats: int = 5,
    random_seed: int = 42,
) -> list[SplitSpec]:
    base_rng = np.random.default_rng(seed=random_seed)
    split_specs: list[SplitSpec] = []

    for repeat in range(1, repeats + 1):
        repeat_seed = int(base_rng.integers(0, np.iinfo(np.uint32).max))
        rng = np.random.default_rng(seed=repeat_seed)
        half_a, half_b = _stratified_half_split(labels, rng)

        split_specs.append(
            SplitSpec(
                repeat=repeat,
                split_id=1,
                train_idx=half_a,
                test_idx=half_b,
            )
        )
        split_specs.append(
            SplitSpec(
                repeat=repeat,
                split_id=2,
                train_idx=half_b,
                test_idx=half_a,
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


def expand_training_set_with_augmentation(
    train_paths: np.ndarray,
    train_labels: np.ndarray,
    train_label_ids: np.ndarray,
    volcano_root: Path,
    split_augmented_root: Path,
    random_seed: int,
    augment_target_counts: dict[str, int] | None = None,
    augment_policies: dict[str, tuple[tuple[str, float], ...]] | None = None,
    augment_time_shift: int = 1000,
    augment_amplitude_low: float = 0.85,
    augment_amplitude_high: float = 1.15,
    augment_noise_std_factor: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    augment_target_counts = augment_target_counts or DEFAULT_AUGMENT_TARGET_COUNTS
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


def _stratified_train_test_split(
    labels: np.ndarray,
    rng: np.random.Generator,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_name in sorted(np.unique(labels)):
        class_idx = np.where(labels == class_name)[0]
        rng.shuffle(class_idx)
        total = len(class_idx)

        if total == 1:
            test_count = 1
        else:
            test_count = int(round(total * test_fraction))
            test_count = max(1, min(total - 1, test_count))

        test_parts.append(class_idx[:test_count])
        train_parts.append(class_idx[test_count:])

    train_idx = (
        np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    )
    test_idx = np.concatenate(test_parts) if test_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


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


def _stratified_train_val_test_split(
    labels: np.ndarray,
    rng: np.random.Generator,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = np.asarray(
        [train_fraction, val_fraction, test_fraction], dtype=np.float64
    )
    if not np.isclose(float(np.sum(fractions)), 1.0, atol=1e-8):
        raise ValueError(
            "train_fraction + val_fraction + test_fraction must sum to 1.0"
        )

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_name in sorted(np.unique(labels)):
        class_idx = np.where(labels == class_name)[0]
        rng.shuffle(class_idx)
        total = len(class_idx)

        if total == 0:
            continue
        if total == 1:
            train_count, val_count, test_count = 1, 0, 0
        elif total == 2:
            train_count, val_count, test_count = 1, 0, 1
        else:
            counts = np.floor(fractions * total).astype(np.int64)
            remainder = int(total - int(np.sum(counts)))
            order = np.argsort(-fractions)
            for rem_idx in range(remainder):
                counts[int(order[rem_idx % 3])] += 1

            # Keep all three splits non-empty for classes with at least 3 samples.
            for split_idx in range(3):
                if counts[split_idx] == 0:
                    donor_idx = int(np.argmax(counts))
                    if counts[donor_idx] <= 1:
                        continue
                    counts[donor_idx] -= 1
                    counts[split_idx] += 1

            train_count = int(counts[0])
            val_count = int(counts[1])
            test_count = int(total - train_count - val_count)

        train_parts.append(class_idx[:train_count])
        val_parts.append(class_idx[train_count : train_count + val_count])
        test_parts.append(
            class_idx[train_count + val_count : train_count + val_count + test_count]
        )

    train_idx = (
        np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    )
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _stratified_subset_from_pool(
    pool_labels: np.ndarray,
    rng: np.random.Generator,
    ratio: float,
) -> np.ndarray:
    subset_parts: list[np.ndarray] = []

    for class_name in sorted(np.unique(pool_labels)):
        class_idx = np.where(pool_labels == class_name)[0]
        rng.shuffle(class_idx)

        if len(class_idx) == 0:
            continue

        subset_count = int(round(len(class_idx) * ratio))
        subset_count = max(1, min(len(class_idx), subset_count))
        subset_parts.append(class_idx[:subset_count])

    if not subset_parts:
        return np.empty(0, dtype=np.int64)

    subset_idx = np.concatenate(subset_parts)
    rng.shuffle(subset_idx)
    return subset_idx


def generate_cross_volcano_eval_manifests(
    data_root: Path,
    prepared_root: Path,
    target_volcano: str,
    random_seed: int = 42,
    test_fraction: float = 0.80,
    train_percentages: tuple[int, ...] = (1, 5, 10, 20),
) -> None:
    if not data_root.exists():
        raise RuntimeError(f"Data root not found: {data_root}")

    volcano_dirs = sorted(
        [
            p
            for p in data_root.iterdir()
            if p.is_dir() and p.name not in {"prepared_data", target_volcano}
        ]
    )

    if not volcano_dirs:
        raise RuntimeError(
            "No auxiliary volcano folders found for cross-volcano manifests."
        )

    cross_root = prepared_root / "cross_volcano"
    cross_root.mkdir(parents=True, exist_ok=True)

    for offset, volcano_dir in enumerate(volcano_dirs):
        all_paths, all_labels, all_label_ids = collect_volcano_samples(volcano_dir)
        rng = np.random.default_rng(seed=random_seed + 500 + offset)

        pool_idx, test_idx = _stratified_train_test_split(
            all_labels,
            rng=rng,
            test_fraction=test_fraction,
        )

        volcano_out = cross_root / volcano_dir.name
        volcano_out.mkdir(parents=True, exist_ok=True)

        save_manifest(
            volcano_out / "test_80.npz",
            filepaths=all_paths[test_idx],
            labels=all_labels[test_idx],
            label_ids=all_label_ids[test_idx],
            extra_fields={
                "volcano": np.asarray([volcano_dir.name]),
                "split_type": np.asarray(["test_80"]),
            },
        )

        pool_labels = all_labels[pool_idx]
        pool_paths = all_paths[pool_idx]
        pool_label_ids = all_label_ids[pool_idx]

        for pct in train_percentages:
            ratio = float(pct) / 20.0
            subset_local_idx = _stratified_subset_from_pool(
                pool_labels,
                rng=rng,
                ratio=ratio,
            )
            save_manifest(
                volcano_out / f"train_{pct:02d}pct.npz",
                filepaths=pool_paths[subset_local_idx],
                labels=pool_labels[subset_local_idx],
                label_ids=pool_label_ids[subset_local_idx],
                extra_fields={
                    "volcano": np.asarray([volcano_dir.name]),
                    "split_type": np.asarray([f"train_{pct:02d}pct"]),
                },
            )


import numpy as np

np.array([3068, 1892, 2360, 805, 977]) * 0.075


def generate_nvchvc_5x2_manifests(
    data_root: Path,
    prepared_root: Path,
    target_volcano: str = "NVCHVC",
    random_seed: int = 42,
    repeats: int = 5,
    val_fraction_within_train: float = 0.20,
    augment_target_counts: dict[str, int] | None = None,
    augment_policies: dict[str, tuple[tuple[str, float], ...]] | None = None,
    augment_time_shift: int = 1000,
    augment_amplitude_low: float = 0.85,
    augment_amplitude_high: float = 1.15,
    augment_noise_std_factor: float = 0.02,
) -> None:
    volcano_root = data_root / target_volcano
    if not volcano_root.exists():
        raise RuntimeError(f"Target volcano folder not found: {volcano_root}")

    all_paths, all_labels, all_label_ids = collect_volcano_samples(volcano_root)
    split_specs = build_5x2_split_specs(
        labels=all_labels,
        repeats=repeats,
        random_seed=random_seed,
    )

    nvchvc_root = prepared_root / target_volcano / "cv_5x2"
    nvchvc_root.mkdir(parents=True, exist_ok=True)

    for split_idx, spec in enumerate(split_specs):
        split_dir = nvchvc_root / f"repeat_{spec.repeat:02d}" / f"split_{spec.split_id}"
        split_dir.mkdir(parents=True, exist_ok=True)

        outer_train_paths = all_paths[spec.train_idx]
        outer_train_labels = all_labels[spec.train_idx]
        outer_train_label_ids = all_label_ids[spec.train_idx]

        split_rng = np.random.default_rng(seed=random_seed + 50000 + split_idx)
        inner_train_idx, val_idx = _stratified_train_val_split_from_train(
            outer_train_labels,
            rng=split_rng,
            val_fraction=val_fraction_within_train,
        )

        train_paths = outer_train_paths[inner_train_idx]
        train_labels = outer_train_labels[inner_train_idx]
        train_label_ids = outer_train_label_ids[inner_train_idx]
        val_paths = outer_train_paths[val_idx]
        val_labels = outer_train_labels[val_idx]
        val_label_ids = outer_train_label_ids[val_idx]

        test_paths = all_paths[spec.test_idx]
        test_labels = all_labels[spec.test_idx]
        test_label_ids = all_label_ids[spec.test_idx]

        save_manifest(
            split_dir / "train.npz",
            filepaths=train_paths,
            labels=train_labels,
            label_ids=train_label_ids,
            extra_fields={
                "repeat": np.asarray([spec.repeat], dtype=np.int64),
                "split": np.asarray([spec.split_id], dtype=np.int64),
                "volcano": np.asarray([target_volcano]),
                "val_fraction_within_train": np.asarray(
                    [val_fraction_within_train], dtype=np.float32
                ),
            },
        )
        save_manifest(
            split_dir / "val.npz",
            filepaths=val_paths,
            labels=val_labels,
            label_ids=val_label_ids,
            extra_fields={
                "repeat": np.asarray([spec.repeat], dtype=np.int64),
                "split": np.asarray([spec.split_id], dtype=np.int64),
                "volcano": np.asarray([target_volcano]),
                "val_fraction_within_train": np.asarray(
                    [val_fraction_within_train], dtype=np.float32
                ),
            },
        )
        save_manifest(
            split_dir / "test.npz",
            filepaths=test_paths,
            labels=test_labels,
            label_ids=test_label_ids,
            extra_fields={
                "repeat": np.asarray([spec.repeat], dtype=np.int64),
                "split": np.asarray([spec.split_id], dtype=np.int64),
                "volcano": np.asarray([target_volcano]),
                "val_fraction_within_train": np.asarray(
                    [val_fraction_within_train], dtype=np.float32
                ),
            },
        )

        train_aug_paths, train_aug_labels, train_aug_label_ids, is_augmented, _ = (
            expand_training_set_with_augmentation(
                train_paths=train_paths,
                train_labels=train_labels,
                train_label_ids=train_label_ids,
                volcano_root=volcano_root,
                split_augmented_root=split_dir / "augmented",
                random_seed=random_seed + 10000 + split_idx,
                augment_target_counts=augment_target_counts,
                augment_policies=augment_policies,
                augment_time_shift=augment_time_shift,
                augment_amplitude_low=augment_amplitude_low,
                augment_amplitude_high=augment_amplitude_high,
                augment_noise_std_factor=augment_noise_std_factor,
            )
        )

        save_manifest(
            split_dir / "train_aug.npz",
            filepaths=train_aug_paths,
            labels=train_aug_labels,
            label_ids=train_aug_label_ids,
            extra_fields={
                "repeat": np.asarray([spec.repeat], dtype=np.int64),
                "split": np.asarray([spec.split_id], dtype=np.int64),
                "volcano": np.asarray([target_volcano]),
                "val_fraction_within_train": np.asarray(
                    [val_fraction_within_train], dtype=np.float32
                ),
                "is_augmented": is_augmented,
            },
        )

        print(
            f"Saved repeat={spec.repeat} split={spec.split_id} | train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} train_aug={len(train_aug_paths)}"
        )


def generate_nvchvc_train_val_test_manifests(
    data_root: Path,
    prepared_root: Path,
    target_volcano: str = "NVCHVC",
    random_seed: int = 42,
    repeats: int = 1,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    augment_target_counts: dict[str, int] | None = None,
    augment_policies: dict[str, tuple[tuple[str, float], ...]] | None = None,
    augment_time_shift: int = 1000,
    augment_amplitude_low: float = 0.85,
    augment_amplitude_high: float = 1.15,
    augment_noise_std_factor: float = 0.02,
) -> None:
    volcano_root = data_root / target_volcano
    if not volcano_root.exists():
        raise RuntimeError(f"Target volcano folder not found: {volcano_root}")

    all_paths, all_labels, all_label_ids = collect_volcano_samples(volcano_root)

    split_root = prepared_root / target_volcano / "split_70_15_15"
    split_root.mkdir(parents=True, exist_ok=True)

    for repeat in range(1, repeats + 1):
        rng = np.random.default_rng(seed=random_seed + (repeat * 1000))
        train_idx, val_idx, test_idx = _stratified_train_val_test_split(
            all_labels,
            rng=rng,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

        split_dir = split_root / f"repeat_{repeat:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_paths = all_paths[train_idx]
        train_labels = all_labels[train_idx]
        train_label_ids = all_label_ids[train_idx]

        val_paths = all_paths[val_idx]
        val_labels = all_labels[val_idx]
        val_label_ids = all_label_ids[val_idx]

        test_paths = all_paths[test_idx]
        test_labels = all_labels[test_idx]
        test_label_ids = all_label_ids[test_idx]

        common_fields = {
            "repeat": np.asarray([repeat], dtype=np.int64),
            "volcano": np.asarray([target_volcano]),
            "train_fraction": np.asarray([train_fraction], dtype=np.float32),
            "val_fraction": np.asarray([val_fraction], dtype=np.float32),
            "test_fraction": np.asarray([test_fraction], dtype=np.float32),
        }

        save_manifest(
            split_dir / "train.npz",
            filepaths=train_paths,
            labels=train_labels,
            label_ids=train_label_ids,
            extra_fields={**common_fields, "split_type": np.asarray(["train"])},
        )
        save_manifest(
            split_dir / "val.npz",
            filepaths=val_paths,
            labels=val_labels,
            label_ids=val_label_ids,
            extra_fields={**common_fields, "split_type": np.asarray(["val"])},
        )
        save_manifest(
            split_dir / "test.npz",
            filepaths=test_paths,
            labels=test_labels,
            label_ids=test_label_ids,
            extra_fields={**common_fields, "split_type": np.asarray(["test"])},
        )

        train_aug_paths, train_aug_labels, train_aug_label_ids, is_augmented, _ = (
            expand_training_set_with_augmentation(
                train_paths=train_paths,
                train_labels=train_labels,
                train_label_ids=train_label_ids,
                volcano_root=volcano_root,
                split_augmented_root=split_dir / "augmented",
                random_seed=random_seed + 20000 + repeat,
                augment_target_counts=augment_target_counts,
                augment_policies=augment_policies,
                augment_time_shift=augment_time_shift,
                augment_amplitude_low=augment_amplitude_low,
                augment_amplitude_high=augment_amplitude_high,
                augment_noise_std_factor=augment_noise_std_factor,
            )
        )

        save_manifest(
            split_dir / "train_aug.npz",
            filepaths=train_aug_paths,
            labels=train_aug_labels,
            label_ids=train_aug_label_ids,
            extra_fields={
                **common_fields,
                "split_type": np.asarray(["train_aug"]),
                "is_augmented": is_augmented,
            },
        )

        print(
            f"Saved repeat={repeat} | train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} train_aug={len(train_aug_paths)}"
        )
