"""
Generate the leave-one-out cross-volcano dataset splits used by scripts 03 and 04.

Run:
    python scripts/01b_prepare_cross-volcano_data.py
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import (
    CLASS_TO_ID,
    VALID_CLASSES,
    augment_trace_for_storage,
    collect_volcano_samples,
    save_manifest,
    _stratified_train_val_split_from_train,
)

DATA_ROOT = PROJECT_ROOT / "data"
PREPARED_ROOT = DATA_ROOT / "prepared_data"

RANDOM_SEED = 42
TARGET_PER_CLASS = 1500
VAL_FRACTION_WITHIN_TRAIN = 0.15

HOLDOUT_VOLCANOES = ("VCA", "CAU", "LDM")
ALL_VOLCANOES = ("NVCHVC", "CAU", "LDM", "VCA")

AUGMENT_POLICIES = {
    "AV": (("shift", 0.50), ("shift_amp", 0.25), ("shift_noise", 0.25)),
    "IC": (("shift", 0.70), ("shift_amp", 0.20), ("shift_noise", 0.10)),
}
AUGMENT_TIME_SHIFT = 1000
AUGMENT_AMPLITUDE_LOW = 0.85
AUGMENT_AMPLITUDE_HIGH = 1.15
AUGMENT_NOISE_STD_FACTOR = 0.02


def main() -> None:
    if not DATA_ROOT.exists():
        raise RuntimeError(f"Data root not found: {DATA_ROOT}")

    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    (PREPARED_ROOT / ".gitkeep").touch(exist_ok=True)

    print("Generating leave-one-out cross-volcano manifests...")

    if not (0.0 < VAL_FRACTION_WITHIN_TRAIN < 1.0):
        raise ValueError(
            "VAL_FRACTION_WITHIN_TRAIN must be in the open interval (0, 1)"
        )

    pre_split_target_per_class = int(
        np.ceil(float(TARGET_PER_CLASS) / (1.0 - float(VAL_FRACTION_WITHIN_TRAIN)))
    )

    cross_root = PREPARED_ROOT / "cross_volcano_loo"
    cross_root.mkdir(parents=True, exist_ok=True)

    volcano_to_idx = {name: idx for idx, name in enumerate(ALL_VOLCANOES)}

    volcano_samples: dict[str, dict[str, np.ndarray]] = {}
    for volcano_name in ALL_VOLCANOES:
        volcano_root = DATA_ROOT / volcano_name
        if not volcano_root.exists():
            raise FileNotFoundError(f"Volcano folder not found: {volcano_root}")

        filepaths, labels, label_ids = collect_volcano_samples(volcano_root)
        volcano_samples[volcano_name] = {
            "filepaths": filepaths,
            "labels": labels,
            "label_ids": label_ids,
        }

    for fold_idx, held_out in enumerate(HOLDOUT_VOLCANOES, start=1):
        if held_out not in ALL_VOLCANOES:
            raise ValueError(
                f"Held-out volcano '{held_out}' is not in ALL_VOLCANOES={ALL_VOLCANOES}."
            )

        train_volcanoes = [v for v in ALL_VOLCANOES if v != held_out]
        fold_rng = np.random.default_rng(seed=RANDOM_SEED + fold_idx * 1000)

        fold_dir = cross_root / f"fold_{fold_idx:02d}_holdout_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        aug_root = fold_dir / "augmented"
        aug_root.mkdir(parents=True, exist_ok=True)

        def _infer_descriptor_path(sample_path: str) -> str:
            src = Path(str(sample_path))
            src_parts = src.parts
            hit = None
            for volcano_name in ALL_VOLCANOES:
                if volcano_name in src_parts:
                    hit = volcano_name
                    break
            if hit is None:
                return ""

            idx = src_parts.index(hit)
            rel = Path(*src_parts[idx + 1 :]).with_suffix(".npz")
            desc_path = PREPARED_ROOT / "descriptors" / hit / rel
            return str(desc_path.as_posix())

        selected_paths: list[str] = []
        selected_labels: list[str] = []
        selected_label_ids: list[int] = []
        selected_volcano_idx: list[int] = []
        selected_descriptor_paths: list[str] = []

        for class_name in VALID_CLASSES:
            class_sources: dict[str, np.ndarray] = {}
            for volcano_name in train_volcanoes:
                labels = volcano_samples[volcano_name]["labels"]
                class_sources[volcano_name] = np.where(labels == class_name)[0]

            available_volcanoes = [
                v for v in train_volcanoes if int(len(class_sources[v])) > 0
            ]
            if len(available_volcanoes) == 0:
                continue

            base = int(pre_split_target_per_class) // int(len(available_volcanoes))
            rem = int(pre_split_target_per_class) % int(len(available_volcanoes))
            alloc = {
                v: base + (1 if idx < rem else 0)
                for idx, v in enumerate(available_volcanoes)
            }

            for volcano_name in available_volcanoes:
                src_idx = class_sources[volcano_name]
                fold_rng.shuffle(src_idx)
                draw = min(int(alloc[volcano_name]), int(len(src_idx)))
                if draw <= 0:
                    continue

                picked = src_idx[:draw]
                paths_np = volcano_samples[volcano_name]["filepaths"][picked]

                selected_paths.extend([str(x) for x in paths_np.tolist()])
                selected_labels.extend([class_name] * draw)
                selected_label_ids.extend([int(CLASS_TO_ID[class_name])] * draw)
                selected_volcano_idx.extend([int(volcano_to_idx[volcano_name])] * draw)
                selected_descriptor_paths.extend(
                    [_infer_descriptor_path(str(x)) for x in paths_np.tolist()]
                )

        selected_paths_np = np.asarray(selected_paths)
        selected_labels_np = np.asarray(selected_labels)
        selected_label_ids_np = np.asarray(selected_label_ids, dtype=np.int64)
        selected_volcano_idx_np = np.asarray(selected_volcano_idx, dtype=np.int64)
        selected_descriptor_paths_np = np.asarray(selected_descriptor_paths)

        perm = fold_rng.permutation(len(selected_paths_np))
        selected_paths_np = selected_paths_np[perm]
        selected_labels_np = selected_labels_np[perm]
        selected_label_ids_np = selected_label_ids_np[perm]
        selected_volcano_idx_np = selected_volcano_idx_np[perm]
        selected_descriptor_paths_np = selected_descriptor_paths_np[perm]

        tr_idx, va_idx = _stratified_train_val_split_from_train(
            selected_labels_np,
            rng=np.random.default_rng(seed=RANDOM_SEED + fold_idx * 1000 + 77),
            val_fraction=VAL_FRACTION_WITHIN_TRAIN,
        )

        train_paths_base = selected_paths_np[tr_idx]
        train_labels_base = selected_labels_np[tr_idx]
        train_label_ids_base = selected_label_ids_np[tr_idx]
        train_volcano_idx_base = selected_volcano_idx_np[tr_idx]
        train_descriptor_paths_base = selected_descriptor_paths_np[tr_idx]

        val_paths = selected_paths_np[va_idx]
        val_labels = selected_labels_np[va_idx]
        val_label_ids = selected_label_ids_np[va_idx]
        val_volcano_idx = selected_volcano_idx_np[va_idx]
        val_descriptor_paths = selected_descriptor_paths_np[va_idx]

        train_paths = [str(x) for x in train_paths_base.tolist()]
        train_labels = [str(x) for x in train_labels_base.tolist()]
        train_label_ids = [int(x) for x in train_label_ids_base.tolist()]
        train_volcano_idx = [int(x) for x in train_volcano_idx_base.tolist()]
        train_descriptor_paths = [str(x) for x in train_descriptor_paths_base.tolist()]
        train_is_augmented = [False for _ in range(len(train_paths))]

        volcano_idx_to_name = {idx: name for name, idx in volcano_to_idx.items()}

        for class_name in VALID_CLASSES:
            class_train_idx = np.where(train_labels_base == class_name)[0]
            current_count = int(len(class_train_idx))
            shortfall = int(TARGET_PER_CLASS) - current_count
            if shortfall <= 0:
                continue

            unique_train_volcanoes = sorted(
                set(int(train_volcano_idx_base[i]) for i in class_train_idx)
            )
            can_augment = (
                class_name in {"AV", "IC"}
                and len(unique_train_volcanoes) == 1
                and len(class_train_idx) > 0
            )
            if not can_augment:
                continue

            source_volcano = volcano_idx_to_name[int(unique_train_volcanoes[0])]
            source_root = DATA_ROOT / source_volcano
            source_probs = np.array(
                [p for _, p in AUGMENT_POLICIES[class_name]], dtype=np.float32
            )
            source_probs = source_probs / np.sum(source_probs)
            policies = [name for name, _ in AUGMENT_POLICIES[class_name]]

            sampled_paths_for_class = [train_paths[i] for i in class_train_idx.tolist()]
            for aug_idx in range(shortfall):
                src_path = Path(
                    str(fold_rng.choice(np.asarray(sampled_paths_for_class)))
                )
                src_arr = np.load(src_path, mmap_mode="r")
                src_arr = np.asarray(src_arr, dtype=np.float32)
                x_raw = src_arr[:8, :]
                y_raw = src_arr[8:, :]

                policy = str(fold_rng.choice(policies, p=source_probs))
                x_aug, y_aug = augment_trace_for_storage(
                    x_raw,
                    y_raw,
                    policy=policy,
                    rng=fold_rng,
                    time_shift=AUGMENT_TIME_SHIFT,
                    amplitude_low=AUGMENT_AMPLITUDE_LOW,
                    amplitude_high=AUGMENT_AMPLITUDE_HIGH,
                    noise_std_factor=AUGMENT_NOISE_STD_FACTOR,
                )

                rel = src_path.relative_to(source_root)
                class_dir = rel.parts[0]
                out_dir = aug_root / source_volcano / class_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = (
                    f"{src_path.stem}_fold{fold_idx:02d}_{class_name}_"
                    f"aug_{aug_idx:05d}_{policy}.npy"
                )
                out_path = out_dir / out_name
                np.save(
                    out_path,
                    np.concatenate([x_aug, y_aug], axis=0).astype(np.float32),
                )

                train_paths.append(str(out_path.as_posix()))
                train_labels.append(class_name)
                train_label_ids.append(int(CLASS_TO_ID[class_name]))
                train_volcano_idx.append(int(volcano_to_idx[source_volcano]))
                train_descriptor_paths.append("")
                train_is_augmented.append(True)

        train_paths_np = np.asarray(train_paths)
        train_labels_np = np.asarray(train_labels)
        train_label_ids_np = np.asarray(train_label_ids, dtype=np.int64)
        train_volcano_idx_np = np.asarray(train_volcano_idx, dtype=np.int64)
        train_descriptor_paths_np = np.asarray(train_descriptor_paths)
        train_is_aug_np = np.asarray(train_is_augmented, dtype=bool)

        train_perm = fold_rng.permutation(len(train_paths_np))
        train_paths_np = train_paths_np[train_perm]
        train_labels_np = train_labels_np[train_perm]
        train_label_ids_np = train_label_ids_np[train_perm]
        train_volcano_idx_np = train_volcano_idx_np[train_perm]
        train_descriptor_paths_np = train_descriptor_paths_np[train_perm]
        train_is_aug_np = train_is_aug_np[train_perm]

        test_fp = volcano_samples[held_out]["filepaths"]
        test_lb = volcano_samples[held_out]["labels"]
        test_id = volcano_samples[held_out]["label_ids"]
        test_vol_idx = np.full(
            shape=(len(test_fp),),
            fill_value=int(volcano_to_idx[held_out]),
            dtype=np.int64,
        )
        test_desc = np.asarray(
            [_infer_descriptor_path(str(x)) for x in test_fp.tolist()]
        )

        common_fields = {
            "fold_id": np.asarray([fold_idx], dtype=np.int64),
            "held_out_volcano": np.asarray([held_out]),
            "train_volcanoes": np.asarray(train_volcanoes),
            "target_per_class": np.asarray([TARGET_PER_CLASS], dtype=np.int64),
            "val_fraction_within_train": np.asarray(
                [VAL_FRACTION_WITHIN_TRAIN], dtype=np.float32
            ),
            "seed": np.asarray([RANDOM_SEED], dtype=np.int64),
        }

        save_manifest(
            fold_dir / "train.npz",
            filepaths=train_paths_np,
            labels=train_labels_np,
            label_ids=train_label_ids_np,
            extra_fields={
                **common_fields,
                "volcano_idx": train_volcano_idx_np,
                "descriptor_paths": train_descriptor_paths_np,
                "is_augmented": train_is_aug_np,
            },
        )
        save_manifest(
            fold_dir / "val.npz",
            filepaths=val_paths,
            labels=val_labels,
            label_ids=val_label_ids,
            extra_fields={
                **common_fields,
                "volcano_idx": val_volcano_idx,
                "descriptor_paths": val_descriptor_paths,
                "is_augmented": np.zeros(len(val_paths), dtype=bool),
            },
        )
        save_manifest(
            fold_dir / "test.npz",
            filepaths=test_fp,
            labels=test_lb,
            label_ids=test_id,
            extra_fields={
                **common_fields,
                "volcano_idx": test_vol_idx,
                "descriptor_paths": test_desc,
                "is_augmented": np.zeros(len(test_fp), dtype=bool),
            },
        )

        print(
            f"Saved {fold_dir.name} | train={len(tr_idx)} val={len(va_idx)} test={len(test_fp)}"
        )

    print(f"Done. Prepared data written to: {PREPARED_ROOT / 'cross_volcano_loo'}")


if __name__ == "__main__":
    main()
