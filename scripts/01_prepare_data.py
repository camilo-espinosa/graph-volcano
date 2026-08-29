"""
Generate the original NVCHVC 5-fold dataset with inner validation and train-only augmentation.

Run:
    python scripts/01_prepare_data.py
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import (
    CLASS_TO_ID,
    build_stratified_kfold_specs,
    collect_volcano_samples,
    expand_training_set_with_augmentation,
    save_manifest,
    _stratified_train_val_split_from_train,
)

DATA_ROOT = PROJECT_ROOT / "data"
PREPARED_ROOT = DATA_ROOT / "prepared_data"
TARGET_VOLCANO = "NVCHVC"

# Reproducible split/augmentation generation.
RANDOM_SEED = 42
N_FOLDS = 5
VAL_FRACTION_WITHIN_TRAIN = 0.15

# AV and IC targets are inferred internally per fold as mean count of VT/LP/TR.
AUGMENT_POLICIES = {
    "AV": (("shift", 0.50), ("shift_amp", 0.25), ("shift_noise", 0.25)),
    "IC": (("shift", 0.70), ("shift_amp", 0.20), ("shift_noise", 0.10)),
}
AUGMENT_TIME_SHIFT = 1000
AUGMENT_AMPLITUDE_LOW = 0.85
AUGMENT_AMPLITUDE_HIGH = 1.15
AUGMENT_NOISE_STD_FACTOR = 0.02
BG_CLASS_NAME = "BG"
BG_CLASS_ID = 0
CLASS_TO_ID_WITH_BG = {BG_CLASS_NAME: BG_CLASS_ID, **CLASS_TO_ID}


def main() -> None:
    if not DATA_ROOT.exists():
        raise RuntimeError(f"Data root not found: {DATA_ROOT}")

    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    gitkeep_path = PREPARED_ROOT / ".gitkeep"
    gitkeep_path.touch(exist_ok=True)

    print(
        "Generating NVCHVC stratified 5-fold manifests with inner validation and augmented train splits..."
    )

    volcano_root = DATA_ROOT / TARGET_VOLCANO
    if not volcano_root.exists():
        raise RuntimeError(f"Target volcano folder not found: {volcano_root}")

    all_paths, all_labels, all_label_ids = collect_volcano_samples(
        volcano_root,
        class_to_id=CLASS_TO_ID_WITH_BG,
    )
    bg_count = int(np.sum(all_labels == BG_CLASS_NAME))
    if bg_count == 0:
        raise RuntimeError(
            f"No {BG_CLASS_NAME} .npy files found under: {volcano_root / BG_CLASS_NAME}"
        )
    print(f"Found {BG_CLASS_NAME} windows: {bg_count}")
    split_specs = build_stratified_kfold_specs(
        labels=all_labels,
        n_splits=N_FOLDS,
        random_seed=RANDOM_SEED,
    )

    nvchvc_root = PREPARED_ROOT / TARGET_VOLCANO / "cv_5fold"
    nvchvc_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, spec in enumerate(split_specs):
        split_dir = nvchvc_root / f"fold_{spec.split_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        outer_train_paths = all_paths[spec.train_idx]
        outer_train_labels = all_labels[spec.train_idx]
        outer_train_label_ids = all_label_ids[spec.train_idx]

        split_rng = np.random.default_rng(seed=RANDOM_SEED + 50000 + fold_idx)
        inner_train_idx, val_idx = _stratified_train_val_split_from_train(
            outer_train_labels,
            rng=split_rng,
            val_fraction=VAL_FRACTION_WITHIN_TRAIN,
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

        common_fields = {
            "fold": np.asarray([spec.split_id], dtype=np.int64),
            "n_splits": np.asarray([N_FOLDS], dtype=np.int64),
            "volcano": np.asarray([TARGET_VOLCANO]),
            "val_fraction_within_train": np.asarray(
                [VAL_FRACTION_WITHIN_TRAIN], dtype=np.float32
            ),
        }

        save_manifest(
            split_dir / "train.npz",
            filepaths=train_paths,
            labels=train_labels,
            label_ids=train_label_ids,
            extra_fields=common_fields,
        )

        val_perm = np.random.default_rng(seed=RANDOM_SEED + 30000 + fold_idx).permutation(
            len(val_paths)
        )
        save_manifest(
            split_dir / "val.npz",
            filepaths=val_paths[val_perm],
            labels=val_labels[val_perm],
            label_ids=val_label_ids[val_perm],
            extra_fields=common_fields,
        )

        test_perm = np.random.default_rng(seed=RANDOM_SEED + 40000 + fold_idx).permutation(
            len(test_paths)
        )
        save_manifest(
            split_dir / "test.npz",
            filepaths=test_paths[test_perm],
            labels=test_labels[test_perm],
            label_ids=test_label_ids[test_perm],
            extra_fields=common_fields,
        )

        train_aug_paths, train_aug_labels, train_aug_label_ids, is_augmented, _ = (
            expand_training_set_with_augmentation(
                train_paths=train_paths,
                train_labels=train_labels,
                train_label_ids=train_label_ids,
                volcano_root=volcano_root,
                split_augmented_root=split_dir / "augmented",
                random_seed=RANDOM_SEED + 10000 + fold_idx,
                augment_policies=AUGMENT_POLICIES,
                augment_time_shift=AUGMENT_TIME_SHIFT,
                augment_amplitude_low=AUGMENT_AMPLITUDE_LOW,
                augment_amplitude_high=AUGMENT_AMPLITUDE_HIGH,
                augment_noise_std_factor=AUGMENT_NOISE_STD_FACTOR,
            )
        )

        # Shuffle once per fold to avoid deterministic ordering blocks.
        train_mix_rng = np.random.default_rng(seed=RANDOM_SEED + 20000 + fold_idx)
        perm = train_mix_rng.permutation(len(train_aug_paths))
        train_aug_paths = train_aug_paths[perm]
        train_aug_labels = train_aug_labels[perm]
        train_aug_label_ids = train_aug_label_ids[perm]
        is_augmented = is_augmented[perm]

        save_manifest(
            split_dir / "train_aug.npz",
            filepaths=train_aug_paths,
            labels=train_aug_labels,
            label_ids=train_aug_label_ids,
            extra_fields={
                **common_fields,
                "is_augmented": is_augmented,
            },
        )

        print(
            f"Saved fold={spec.split_id}/{N_FOLDS} | train={len(train_paths)} val={len(val_paths)} "
            f"test={len(test_paths)} train_aug={len(train_aug_paths)}"
        )

    print(f"Done. Prepared data written to: {PREPARED_ROOT}")


if __name__ == "__main__":
    main()
