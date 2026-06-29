"""
edge_features.py — OFFLINE precomputation/caching of dynamic edge features and
per-station RSAM for UNet_MPNN.

This module is intentionally standalone: it is NOT imported in the training hot
path. Use it once to precompute and cache (to .npy / .npz) the inputs that the
model consumes as `edge_attr_dynamic` and `rsam`:

    edge_attr_dynamic : [B, n_station_pairs, F_dyn]  (e.g. {lag, coherence})
    rsam              : [B, S]

The station-pair ordering MUST match UNet_MPNN's edge construction, which is:
    for i in range(S):
        for j in range(S):
            if i != j: pair (i -> j)
i.e. all ordered station pairs excluding self-loops, row-major over (i, j).
Use `station_pair_index(S)` to get that ordering.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def station_pair_index(n_stations: int) -> List[Tuple[int, int]]:
    """Ordered (src, dst) station pairs matching UNet_MPNN fully_connected edges."""
    return [
        (i, j)
        for i in range(n_stations)
        for j in range(n_stations)
        if i != j
    ]


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def pairwise_xcorr_features(
    waveforms: np.ndarray,
    sampling_rate: float = 100.0,
    max_lag_seconds: float | None = None,
) -> np.ndarray:
    """
    Compute two physically interpretable features for every ordered station pair
    via FFT cross-correlation. Run OFFLINE.

    Feature layout (axis=-1):
      [0] peak_lag_seconds  — time shift that maximises cross-correlation;
                              encodes moveout / relative arrival delay.
      [1] peak_coherence    — normalised CC value at that lag (clamped to [-1,1]);
                              encodes waveform similarity / signal quality.

    These two values map directly to `xcorr_feat_dim=2` in `edge_mpnn__xcorr`.

    Args:
        waveforms: [S, T] single-window multistation waveforms (one event).
        sampling_rate: samples per second.
        max_lag_seconds: optional clamp on the searched lag window.

    Returns:
        [n_station_pairs, 2] array of (peak_lag_seconds, peak_coherence),
        ordered to match station_pair_index(S).
    """
    if waveforms.ndim != 2:
        raise ValueError(f"waveforms must be [S, T]. Got shape: {waveforms.shape}")
    n_stations, n_samples = waveforms.shape

    # Zero-mean, unit-norm per station for a normalized cross-correlation.
    x = waveforms.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    x = x / norms

    nfft = _next_pow2(2 * n_samples - 1)
    fft = np.fft.rfft(x, n=nfft, axis=1)

    pairs = station_pair_index(n_stations)
    out = np.zeros((len(pairs), 2), dtype=np.float32)

    max_lag_samples = (
        None
        if max_lag_seconds is None
        else int(round(max_lag_seconds * sampling_rate))
    )

    for k, (i, j) in enumerate(pairs):
        cc = np.fft.irfft(fft[i] * np.conj(fft[j]), n=nfft)
        cc = np.concatenate([cc[-(n_samples - 1):], cc[:n_samples]])
        lags = np.arange(-(n_samples - 1), n_samples)

        if max_lag_samples is not None:
            keep = np.abs(lags) <= max_lag_samples
            cc = cc[keep]
            lags = lags[keep]

        best = int(np.argmax(cc))
        out[k, 0] = lags[best] / sampling_rate
        out[k, 1] = float(np.clip(cc[best], -1.0, 1.0))

    return out


def batch_xcorr_features(
    waveforms_batch: np.ndarray,
    sampling_rate: float = 100.0,
    max_lag_seconds: float | None = None,
) -> np.ndarray:
    """
    Vectorize pairwise_xcorr_features over a batch.

    Args:
        waveforms_batch: [B, S, T]

    Returns:
        [B, n_station_pairs, 2]
    """
    if waveforms_batch.ndim != 3:
        raise ValueError(
            f"waveforms_batch must be [B, S, T]. Got: {waveforms_batch.shape}"
        )
    feats = [
        pairwise_xcorr_features(w, sampling_rate, max_lag_seconds)
        for w in waveforms_batch
    ]
    return np.stack(feats, axis=0)


def compute_rsam(
    waveforms_batch: np.ndarray,
) -> np.ndarray:
    """
    Per-station RSAM (mean absolute amplitude) for a batch. Run OFFLINE.

    Args:
        waveforms_batch: [B, S, T]

    Returns:
        [B, S]
    """
    if waveforms_batch.ndim != 3:
        raise ValueError(
            f"waveforms_batch must be [B, S, T]. Got: {waveforms_batch.shape}"
        )
    return np.abs(waveforms_batch).mean(axis=2).astype(np.float32)


def cache_edge_features(
    out_path: str,
    waveforms_batch: np.ndarray,
    sampling_rate: float = 100.0,
    max_lag_seconds: float | None = None,
    compute_rsam_feat: bool = True,
) -> None:
    """
    Precompute and cache edge_attr_dynamic (+ optional rsam) to an .npz file.

    edge_attr_dynamic[:, :, 0] = peak_lag_seconds  (moveout / arrival delay)
    edge_attr_dynamic[:, :, 1] = peak_coherence    (waveform similarity)
    rsam is a node feature (mean absolute amplitude), stored separately.

    Loadable later with:
        data = np.load(out_path)
        edge_attr_dynamic = data["edge_attr_dynamic"]  # [B, n_pairs, 2]
        rsam = data["rsam"]                             # [B, S] (if present)
    """
    edge_attr_dynamic = batch_xcorr_features(
        waveforms_batch, sampling_rate, max_lag_seconds
    )
    payload = {"edge_attr_dynamic": edge_attr_dynamic}
    if compute_rsam_feat:
        payload["rsam"] = compute_rsam(waveforms_batch)
    np.savez_compressed(out_path, **payload)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = rng.standard_normal((2, 8, 8192)).astype(np.float32)

    pairs = station_pair_index(8)
    assert len(pairs) == 8 * 7

    feats = batch_xcorr_features(demo, sampling_rate=100.0, max_lag_seconds=5.0)
    assert feats.shape == (2, len(pairs), 2), feats.shape

    rsam = compute_rsam(demo)
    assert rsam.shape == (2, 8), rsam.shape

    print(
        f"[edge_features] xcorr {feats.shape}, rsam {rsam.shape}, "
        f"n_pairs={len(pairs)} OK"
    )
