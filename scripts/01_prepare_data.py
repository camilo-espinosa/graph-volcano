from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import generate_cross_volcano_eval_manifests
from utils.data_utils import generate_nvchvc_5x2_manifests

DATA_ROOT = PROJECT_ROOT / "data"
PREPARED_ROOT = DATA_ROOT / "prepared_data"
TARGET_VOLCANO = "NVCHVC"

# Reproducible split/augmentation generation.
RANDOM_SEED = 42
REPEATS_5X2 = 5
VAL_FRACTION_WITHIN_TRAIN = 0.20

# AV and IC are upsampled per train split until each class reaches this count.
AUGMENT_TARGET_COUNTS = {"AV": 1500, "IC": 1500}
AUGMENT_POLICIES = {
    "AV": (("shift", 0.50), ("shift_amp", 0.25), ("shift_noise", 0.25)),
    "IC": (("shift", 0.70), ("shift_amp", 0.20), ("shift_noise", 0.10)),
}
AUGMENT_TIME_SHIFT = 1000
AUGMENT_AMPLITUDE_LOW = 0.85
AUGMENT_AMPLITUDE_HIGH = 1.15
AUGMENT_NOISE_STD_FACTOR = 0.02

# Cross-volcano finetuning manifest train sizes (% of full volcano data).
CROSS_VOLCANO_TRAIN_PCTS = (1, 5, 10, 20)


def main() -> None:
    if not DATA_ROOT.exists():
        raise RuntimeError(f"Data root not found: {DATA_ROOT}")

    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    gitkeep_path = PREPARED_ROOT / ".gitkeep"
    gitkeep_path.touch(exist_ok=True)

    print(
        "Generating NVCHVC 5x2 manifests with inner validation and augmented train splits..."
    )
    generate_nvchvc_5x2_manifests(
        data_root=DATA_ROOT,
        prepared_root=PREPARED_ROOT,
        target_volcano=TARGET_VOLCANO,
        random_seed=RANDOM_SEED,
        repeats=REPEATS_5X2,
        val_fraction_within_train=VAL_FRACTION_WITHIN_TRAIN,
        augment_target_counts=AUGMENT_TARGET_COUNTS,
        augment_policies=AUGMENT_POLICIES,
        augment_time_shift=AUGMENT_TIME_SHIFT,
        augment_amplitude_low=AUGMENT_AMPLITUDE_LOW,
        augment_amplitude_high=AUGMENT_AMPLITUDE_HIGH,
        augment_noise_std_factor=AUGMENT_NOISE_STD_FACTOR,
    )

    print("Generating cross-volcano evaluation manifests...")
    generate_cross_volcano_eval_manifests(
        data_root=DATA_ROOT,
        prepared_root=PREPARED_ROOT,
        target_volcano=TARGET_VOLCANO,
        random_seed=RANDOM_SEED,
        test_fraction=0.80,
        train_percentages=CROSS_VOLCANO_TRAIN_PCTS,
    )

    print(f"Done. Prepared data written to: {PREPARED_ROOT}")


if __name__ == "__main__":
    main()
