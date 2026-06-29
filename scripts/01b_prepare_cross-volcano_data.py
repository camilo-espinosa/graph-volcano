"""
Generate the leave-one-out cross-volcano dataset splits used by scripts 03 and 04.

Run:
    python scripts/01b_prepare_cross-volcano_data.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import generate_cross_volcano_leave_one_out_manifests

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
    generate_cross_volcano_leave_one_out_manifests(
        data_root=DATA_ROOT,
        prepared_root=PREPARED_ROOT,
        holdout_volcanoes=HOLDOUT_VOLCANOES,
        all_volcanoes=ALL_VOLCANOES,
        random_seed=RANDOM_SEED,
        target_per_class=TARGET_PER_CLASS,
        val_fraction_within_train=VAL_FRACTION_WITHIN_TRAIN,
        augment_policies=AUGMENT_POLICIES,
        augment_time_shift=AUGMENT_TIME_SHIFT,
        augment_amplitude_low=AUGMENT_AMPLITUDE_LOW,
        augment_amplitude_high=AUGMENT_AMPLITUDE_HIGH,
        augment_noise_std_factor=AUGMENT_NOISE_STD_FACTOR,
        n_stations=8,
    )

    print(f"Done. Prepared data written to: {PREPARED_ROOT / 'cross_volcano_loo'}")


if __name__ == "__main__":
    main()
