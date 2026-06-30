"""
01b_edge_data.py — Offline cross-correlation edge feature precomputation.

Computes edge_attr_dynamic [N, n_station_pairs, 2] for every split manifest used in:
  - NVCHVC 5-fold CV (train, train_aug, val, test per fold)
  - Cross-volcano evaluation (test_80, train_{pct}pct per source volcano)

Edge feature layout (xcorr_feat_dim=2, matches `edge_mpnn__xcorr`):
  [:, :, 0]  peak_lag_seconds — time shift that maximises cross-correlation;
                                 encodes moveout / relative arrival delay.
  [:, :, 1]  peak_coherence   — normalised CC at that lag (clamped [-1, 1]);
                                 encodes waveform similarity / signal quality.

Note on RSAM: also saved to the same .npz as rsam [N, S] (mean absolute amplitude
per station). RSAM is a *node* feature (`use_rsam_node_feat`), not an edge feature;
it is stored here for convenience since it is free to compute alongside xcorr.

Output files mirror the manifest directory layout under an edge_data/ subfolder:
  data/prepared_data/NVCHVC/cv_5fold/fold_XX/edge_data/{train,train_aug,val,test}.npz
  data/prepared_data/cross_volcano/<VOLCANO>/edge_data/{test_80,train_XXpct,...}.npz

Each output .npz contains:
  edge_attr_dynamic : [N, n_station_pairs, 2]   float32  (edge feature for MPNN)
  rsam              : [N, S]                     float32  (node feature, separate use)

where n_station_pairs = S*(S-1) = 56 for S=8 fully-connected stations
(ordered as station_pair_index(8) from utils/edge_features.py).

Run:
  python scripts/01b_edge_data.py
  python scripts/01b_edge_data.py --force        # recompute even if file exists
  python scripts/01b_edge_data.py --chunk 256    # batch size for waveform loading
  python scripts/01b_edge_data.py --max-lag 5.0  # max cross-correlation lag (seconds)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.edge_features import batch_xcorr_features, compute_rsam, station_pair_index

# ---------------------------------------------------------------------------
# Constants matching 01_prepare_data.py
# ---------------------------------------------------------------------------
TARGET_VOLCANO = "NVCHVC"
N_FOLDS = 5
N_STATIONS = 8
SAMPLING_RATE = 100.0  # Hz

FOLD_SPLITS = ["train.npz", "train_aug.npz", "val.npz", "test.npz"]
CROSS_SPLITS_REQUIRED = ["test.npz"]  # at minimum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_waveforms_from_manifest(npz_path: Path) -> np.ndarray:
    """
    Load all waveforms referenced in a manifest .npz.

    Each .npy file in `filepaths` stores [S+C, T] where:
      arr[:S] = waveforms [S, T]
      arr[S:] = one-hot labels [C, T]

    Returns:
        waveforms: [N, S, T]  float32
    """
    with np.load(npz_path) as meta:
        filepaths = meta["filepaths"].copy()

    waveforms = []
    for fp in filepaths:
        arr = np.load(fp, mmap_mode="r")
        arr = np.asarray(arr, dtype=np.float32)
        waveforms.append(arr[:N_STATIONS, :])  # [S, T]

    return np.stack(waveforms, axis=0)  # [N, S, T]


def _process_in_chunks(
    waveforms: np.ndarray,
    chunk_size: int,
    max_lag_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute xcorr features and RSAM over [N, S, T] in memory-friendly chunks.

    Returns:
        edge_attr_dynamic : [N, n_pairs, 2]  float32
        rsam              : [N, S]           float32
    """
    n_samples = waveforms.shape[0]
    n_pairs = len(station_pair_index(N_STATIONS))

    all_edge = np.empty((n_samples, n_pairs, 2), dtype=np.float32)
    all_rsam = np.empty((n_samples, N_STATIONS), dtype=np.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = waveforms[start:end]  # [chunk, S, T]
        all_edge[start:end] = batch_xcorr_features(
            chunk, sampling_rate=SAMPLING_RATE, max_lag_seconds=max_lag_seconds
        )
        all_rsam[start:end] = compute_rsam(chunk)

    return all_edge, all_rsam


def _compute_and_save(
    manifest_path: Path,
    out_path: Path,
    chunk_size: int,
    max_lag_seconds: float | None,
    force: bool,
) -> bool:
    """
    Compute and save edge features for one manifest.

    Returns True if the file was (re)computed, False if skipped.
    """
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.relative_to(PROJECT_ROOT)} already exists.")
        return False

    if not manifest_path.exists():
        print(
            f"  [warn] Manifest not found, skipping: {manifest_path.relative_to(PROJECT_ROOT)}"
        )
        return False

    t0 = time.perf_counter()
    waveforms = _load_waveforms_from_manifest(manifest_path)
    n = waveforms.shape[0]

    edge_attr_dynamic, rsam = _process_in_chunks(waveforms, chunk_size, max_lag_seconds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        edge_attr_dynamic=edge_attr_dynamic,
        rsam=rsam,
    )

    elapsed = time.perf_counter() - t0
    print(
        f"  [ok]   {out_path.relative_to(PROJECT_ROOT)}"
        f"  N={n}  shape={edge_attr_dynamic.shape}  {elapsed:.1f}s"
    )
    return True


# ---------------------------------------------------------------------------
# Dataset-specific discovery helpers
# ---------------------------------------------------------------------------


def _iter_nvchvc_manifests(
    prepared_root: Path,
    n_folds: int,
    splits: list[str],
) -> list[tuple[Path, Path]]:
    """
    Yield (manifest_path, edge_data_output_path) pairs for NVCHVC 5-fold CV.
    """
    cv_root = prepared_root / TARGET_VOLCANO / "cv_5fold"
    pairs: list[tuple[Path, Path]] = []
    for fold in range(1, n_folds + 1):
        fold_dir = cv_root / f"fold_{fold:02d}"
        for split_name in splits:
            manifest = fold_dir / split_name
            edge_out = fold_dir / "edge_data" / split_name
            pairs.append((manifest, edge_out))
    return pairs


def _iter_cross_volcano_manifests(
    cross_root: Path,
) -> list[tuple[Path, Path]]:
    """
    Yield (manifest_path, edge_data_output_path) pairs for all cross-volcano splits.
    """
    pairs: list[tuple[Path, Path]] = []
    if not cross_root.exists():
        return pairs

    for volcano_dir in sorted(cross_root.iterdir()):
        if not volcano_dir.is_dir():
            continue
        for npz_file in sorted(volcano_dir.glob("*.npz")):
            edge_out = volcano_dir / "edge_data" / npz_file.name
            pairs.append((npz_file, edge_out))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute cross-correlation edge features for all dataset splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing edge data files.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=512,
        metavar="N",
        help="Number of samples to process per chunk (memory/speed trade-off).",
    )
    parser.add_argument(
        "--max-lag",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Maximum cross-correlation lag window in seconds. None = full window.",
    )
    parser.add_argument(
        "--no-cross",
        action="store_true",
        help="Skip cross-volcano manifests (process only NVCHVC 5-fold CV).",
    )
    parser.add_argument(
        "--no-nvchvc",
        action="store_true",
        help="Skip NVCHVC 5-fold CV manifests (process only cross-volcano).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepared_root = PROJECT_ROOT / "data" / "prepared_data"
    cross_root = prepared_root / "cross_volcano_loo"

    max_lag: float | None = args.max_lag if args.max_lag > 0 else None

    print(
        f"Edge feature precomputation\n"
        f"  chunk_size   = {args.chunk}\n"
        f"  max_lag      = {max_lag} s\n"
        f"  n_pairs      = {len(station_pair_index(N_STATIONS))}  (S={N_STATIONS} fully-connected)\n"
        f"  force        = {args.force}\n"
    )

    all_pairs: list[tuple[Path, Path]] = []

    if not args.no_nvchvc:
        nvchvc_pairs = _iter_nvchvc_manifests(prepared_root, N_FOLDS, FOLD_SPLITS)
        print(f"NVCHVC 5-fold CV: {len(nvchvc_pairs)} manifests across {N_FOLDS} folds")
        all_pairs.extend(nvchvc_pairs)

    if not args.no_cross:
        cross_pairs = _iter_cross_volcano_manifests(cross_root)
        n_volcanoes = len({p[0].parent for p in cross_pairs})
        print(
            f"Cross-volcano:    {len(cross_pairs)} manifests across {n_volcanoes} source volcano(s)"
        )
        all_pairs.extend(cross_pairs)

    if not all_pairs:
        print("Nothing to process.")
        return

    print(f"\nTotal: {len(all_pairs)} manifest(s)\n")

    computed = 0
    skipped = 0
    for manifest_path, edge_out in all_pairs:
        did_compute = _compute_and_save(
            manifest_path=manifest_path,
            out_path=edge_out,
            chunk_size=args.chunk,
            max_lag_seconds=max_lag,
            force=args.force,
        )
        if did_compute:
            computed += 1
        else:
            skipped += 1

    print(f"\nDone.  computed={computed}  skipped={skipped}  total={len(all_pairs)}")


if __name__ == "__main__":
    main()
