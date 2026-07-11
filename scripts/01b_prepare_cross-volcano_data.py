"""
Prepare progressive finetuning splits for CAU, VCA, and LDM.

For each target volcano this script creates 5 repeated stratified 80/20 splits:
- 80% held-out evaluation set (`test_80.npz`)
- 20% finetuning pool (`pool_20.npz`)

From the 20% pool it then creates nested progressive subsets for
1%, 5%, 10%, and 20% of the full volcano data, each with an 80/20
train/validation split.

Run:
    python scripts/01b_prepare_cross-volcano_data.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import (
    _stratified_train_val_split_from_train,
    collect_volcano_samples,
    save_manifest,
)
from utils.finetune_utils import split_indices_stratified

DATA_ROOT = PROJECT_ROOT / "data"
PREPARED_ROOT = DATA_ROOT / "prepared_data"

RANDOM_SEED = 42
TARGET_VOLCANOES = ("CAU", "VCA", "LDM")
N_REPEATS = 5
EVAL_FRACTION = 0.80
INNER_VAL_FRACTION = 0.20
PROGRESSIVE_SUBSET_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def _fraction_key(fraction: float) -> str:
    return f"{int(round(float(fraction) * 100)):02d}pct"


def _subset_seed(target_seed: int, repeat_idx: int, subset_fraction: float) -> int:
    return int(target_seed + repeat_idx * 1000 + int(round(subset_fraction * 1000)))


def _build_progressive_subset_indices(
    pool_labels: np.ndarray,
    subset_fraction_full: float,
    *,
    seed: int,
) -> np.ndarray:
    if not (0.0 < float(subset_fraction_full) <= 1.0):
        raise ValueError(
            f"subset_fraction_full must be in (0, 1], got {subset_fraction_full}"
        )

    pool_labels = np.asarray(pool_labels)
    rng = np.random.default_rng(seed=int(seed))
    pool_fraction = 1.0 - float(EVAL_FRACTION)
    factor = float(subset_fraction_full) / float(pool_fraction)

    selected_parts: list[np.ndarray] = []
    for class_name in sorted(np.unique(pool_labels).tolist()):
        class_idx = np.where(pool_labels == class_name)[0].astype(np.int64, copy=False)
        rng.shuffle(class_idx)

        if factor >= 1.0:
            take = int(len(class_idx))
        else:
            take = int(np.floor(float(len(class_idx)) * float(factor)))
            if take == 0 and len(class_idx) > 0:
                take = 1

        if take > 0:
            selected_parts.append(class_idx[:take])

    if len(selected_parts) == 0:
        return np.empty(0, dtype=np.int64)

    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    rng.shuffle(selected)
    return selected


def _manifest_common_fields(
    *,
    target_volcano: str,
    repeat_idx: int,
    seed: int,
    pool_fraction: float,
) -> dict[str, np.ndarray]:
    return {
        "target_volcano": np.asarray([target_volcano]),
        "repeat_id": np.asarray([repeat_idx], dtype=np.int64),
        "n_repeats": np.asarray([N_REPEATS], dtype=np.int64),
        "eval_fraction": np.asarray([EVAL_FRACTION], dtype=np.float32),
        "pool_fraction": np.asarray([pool_fraction], dtype=np.float32),
        "seed": np.asarray([seed], dtype=np.int64),
    }


def main() -> None:
    if not DATA_ROOT.exists():
        raise RuntimeError(f"Data root not found: {DATA_ROOT}")

    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    (PREPARED_ROOT / ".gitkeep").touch(exist_ok=True)

    print("Generating progressive finetuning manifests...")

    progressive_root = PREPARED_ROOT / "progressive_finetuning"
    progressive_root.mkdir(parents=True, exist_ok=True)

    volcano_samples: dict[str, dict[str, np.ndarray]] = {}
    for volcano_name in TARGET_VOLCANOES:
        volcano_root = DATA_ROOT / volcano_name
        if not volcano_root.exists():
            raise FileNotFoundError(f"Volcano folder not found: {volcano_root}")

        filepaths, labels, label_ids = collect_volcano_samples(volcano_root)
        volcano_samples[volcano_name] = {
            "filepaths": filepaths,
            "labels": labels,
            "label_ids": label_ids,
        }

    for volcano_offset, volcano_name in enumerate(TARGET_VOLCANOES):
        samples = volcano_samples[volcano_name]
        filepaths = samples["filepaths"]
        labels = samples["labels"]
        label_ids = samples["label_ids"]

        target_root = progressive_root / volcano_name
        target_root.mkdir(parents=True, exist_ok=True)

        for repeat_idx in range(1, N_REPEATS + 1):
            split_seed = RANDOM_SEED + volcano_offset * 10000 + repeat_idx * 100
            pool_idx, test_idx = split_indices_stratified(
                label_ids,
                val_frac=EVAL_FRACTION,
                seed=split_seed,
            )

            if len(pool_idx) == 0 or len(test_idx) == 0:
                raise RuntimeError(
                    f"Invalid split for {volcano_name} repeat {repeat_idx}: "
                    f"pool={len(pool_idx)} test={len(test_idx)}"
                )

            fold_dir = target_root / f"fold_{repeat_idx:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            subsets_root = fold_dir / "subsets"
            subsets_root.mkdir(parents=True, exist_ok=True)

            pool_paths = filepaths[pool_idx]
            pool_labels = labels[pool_idx]
            pool_label_ids = label_ids[pool_idx]
            test_paths = filepaths[test_idx]
            test_labels = labels[test_idx]
            test_label_ids = label_ids[test_idx]

            save_manifest(
                fold_dir / "pool_20.npz",
                filepaths=pool_paths,
                labels=pool_labels,
                label_ids=pool_label_ids,
                extra_fields={
                    **_manifest_common_fields(
                        target_volcano=volcano_name,
                        repeat_idx=repeat_idx,
                        seed=split_seed,
                        pool_fraction=1.0 - EVAL_FRACTION,
                    ),
                    "split_role": np.asarray(["pool"], dtype="U16"),
                },
            )
            save_manifest(
                fold_dir / "test_80.npz",
                filepaths=test_paths,
                labels=test_labels,
                label_ids=test_label_ids,
                extra_fields={
                    **_manifest_common_fields(
                        target_volcano=volcano_name,
                        repeat_idx=repeat_idx,
                        seed=split_seed,
                        pool_fraction=1.0 - EVAL_FRACTION,
                    ),
                    "split_role": np.asarray(["test"], dtype="U16"),
                },
            )

            subset_rows: list[dict[str, object]] = []
            for subset_fraction in PROGRESSIVE_SUBSET_FRACTIONS:
                subset_key = _fraction_key(subset_fraction)
                subset_dir = subsets_root / subset_key
                subset_dir.mkdir(parents=True, exist_ok=True)

                subset_idx = _build_progressive_subset_indices(
                    pool_labels=pool_labels,
                    subset_fraction_full=subset_fraction,
                    seed=_subset_seed(split_seed, repeat_idx, subset_fraction),
                )
                if len(subset_idx) == 0:
                    raise RuntimeError(
                        f"Empty progressive subset for {volcano_name} repeat {repeat_idx} "
                        f"subset={subset_key}"
                    )

                subset_paths = pool_paths[subset_idx]
                subset_labels = pool_labels[subset_idx]
                subset_label_ids = pool_label_ids[subset_idx]

                train_idx, val_idx = _stratified_train_val_split_from_train(
                    subset_labels,
                    rng=np.random.default_rng(
                        seed=_subset_seed(split_seed, repeat_idx, subset_fraction) + 77
                    ),
                    val_fraction=INNER_VAL_FRACTION,
                )

                train_paths = subset_paths[train_idx]
                train_labels = subset_labels[train_idx]
                train_label_ids = subset_label_ids[train_idx]
                val_paths = subset_paths[val_idx]
                val_labels = subset_labels[val_idx]
                val_label_ids = subset_label_ids[val_idx]

                common_subset_fields = {
                    **_manifest_common_fields(
                        target_volcano=volcano_name,
                        repeat_idx=repeat_idx,
                        seed=split_seed,
                        pool_fraction=1.0 - EVAL_FRACTION,
                    ),
                    "subset_key": np.asarray([subset_key]),
                    "subset_fraction_full": np.asarray(
                        [subset_fraction], dtype=np.float32
                    ),
                    "subset_fraction_of_pool": np.asarray(
                        [subset_fraction / (1.0 - EVAL_FRACTION)], dtype=np.float32
                    ),
                    "val_fraction_within_subset": np.asarray(
                        [INNER_VAL_FRACTION], dtype=np.float32
                    ),
                    "split_role": np.asarray(["train_val"], dtype="U16"),
                }

                save_manifest(
                    subset_dir / "train.npz",
                    filepaths=train_paths,
                    labels=train_labels,
                    label_ids=train_label_ids,
                    extra_fields={
                        **common_subset_fields,
                        "split_part": np.asarray(["train"], dtype="U16"),
                    },
                )
                save_manifest(
                    subset_dir / "val.npz",
                    filepaths=val_paths,
                    labels=val_labels,
                    label_ids=val_label_ids,
                    extra_fields={
                        **common_subset_fields,
                        "split_part": np.asarray(["val"], dtype="U16"),
                    },
                )

                subset_rows.append(
                    {
                        "subset_key": subset_key,
                        "subset_fraction_full": float(subset_fraction),
                        "subset_fraction_of_pool": float(
                            subset_fraction / (1.0 - EVAL_FRACTION)
                        ),
                        "subset_size": int(len(subset_idx)),
                        "train_size": int(len(train_idx)),
                        "val_size": int(len(val_idx)),
                    }
                )

            fold_summary = {
                "target_volcano": volcano_name,
                "repeat_id": int(repeat_idx),
                "seed": int(split_seed),
                "eval_fraction": float(EVAL_FRACTION),
                "pool_fraction": float(1.0 - EVAL_FRACTION),
                "n_samples_total": int(len(filepaths)),
                "n_samples_pool": int(len(pool_idx)),
                "n_samples_test": int(len(test_idx)),
                "classes_present": sorted(np.unique(labels).tolist()),
                "subset_summaries": subset_rows,
            }
            with (fold_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
                json.dump(fold_summary, f, indent=2)

            print(
                f"Saved {volcano_name} fold={repeat_idx:02d} | "
                f"pool={len(pool_idx)} test={len(test_idx)} | "
                + ", ".join(
                    f"{row['subset_key']}={row['subset_size']}" for row in subset_rows
                )
            )

    print(f"Done. Prepared data written to: {progressive_root}")


if __name__ == "__main__":
    main()
