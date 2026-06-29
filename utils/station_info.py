from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

LonLat = Tuple[float, float]
StationMap = Dict[str, LonLat]


STATIONS_BY_VOLCANO: Dict[str, StationMap] = {
    "CAU": {
        "PHU": (-72.15, -40.61),
        "FUT": (-72.31, -40.38),
        "VRA": (-72.29, -40.56),
        "CAU": (-72.16, -40.62),
        "PIU": (-72.27, -40.53),
    },
    "NVCHVC": {
        "FRE": (-71.39, -36.87),
        "SHG": (-71.38, -36.88),
        "NBL": (-71.38, -36.82),
        "SHA": (-71.36, -36.80),
        "FU2": (-71.34, -36.90),
        "CHS": (-71.34, -36.87),
        "LBN": (-71.38, -36.85),
        "PLA": (-71.45, -36.83),
    },
    "LDM": {
        "PUE": (-70.45, -36.05),
        "MAU": (-70.53, -36.06),
        "NIE": (-70.56, -36.10),
        "COL": (-70.49, -36.11),
        "BOB": (-70.55, -36.01),
        "ARA": (-70.75, -35.80),
        "CLP": (-70.59, -35.95),
    },
    "VCA": {
        "VT2": (-71.53, -39.59),
        "CHP": (-71.94, -39.49),
        "KIK": (-71.87, -39.42),
        "VN2": (-71.95, -39.40),
        "TRA": (-71.89, -39.44),
        "CVI": (-71.94, -39.43),
        "PCH": (-71.82, -39.46),
        "MRN": (-71.95, -39.41),
    },
}


CRATER_BY_VOLCANO: Dict[str, LonLat] = {
    "NVCHVC": (-71.37667, -36.86333),
    "VCA": (-71.93, -39.42),
    "LDM": (-70.492, -36.058),
    "CAU": (-72.117, -40.590),
}


DEFAULT_VOLCANO_ORDER: Tuple[str, ...] = tuple(STATIONS_BY_VOLCANO.keys())
VOLCANO_TO_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(DEFAULT_VOLCANO_ORDER)
}


def get_station_coords(volcano_name: str) -> StationMap:
    if volcano_name not in STATIONS_BY_VOLCANO:
        known = sorted(STATIONS_BY_VOLCANO.keys())
        raise KeyError(
            f"Unknown volcano station metadata: {volcano_name}. Known: {known}"
        )
    return STATIONS_BY_VOLCANO[volcano_name]


def get_crater_coords(volcano_name: str) -> LonLat:
    if volcano_name not in CRATER_BY_VOLCANO:
        known = sorted(CRATER_BY_VOLCANO.keys())
        raise KeyError(
            f"Unknown volcano crater metadata: {volcano_name}. Known: {known}"
        )
    return CRATER_BY_VOLCANO[volcano_name]


def get_default_volcano_to_index() -> Dict[str, int]:
    return dict(VOLCANO_TO_INDEX)


def get_default_volcano_order() -> Tuple[str, ...]:
    return tuple(DEFAULT_VOLCANO_ORDER)


def get_volcano_index(volcano_name: str) -> int:
    if volcano_name not in VOLCANO_TO_INDEX:
        known = sorted(VOLCANO_TO_INDEX.keys())
        raise KeyError(
            f"Unknown volcano for index mapping: {volcano_name}. Known: {known}"
        )
    return int(VOLCANO_TO_INDEX[volcano_name])


def infer_volcano_name_from_path(path_like: str | Path) -> str:
    path = Path(path_like)
    path_parts_upper = {part.upper() for part in path.parts}
    for volcano_name in DEFAULT_VOLCANO_ORDER:
        if volcano_name.upper() in path_parts_upper:
            return volcano_name
    known = list(DEFAULT_VOLCANO_ORDER)
    raise KeyError(
        f"Could not infer volcano name from path: {path}. Expected one of: {known}"
    )


def _build_geometry_for_volcano(volcano_name: str, n_stations: int) -> np.ndarray:
    station_coords = list(get_station_coords(volcano_name).items())
    crater_coords = get_crater_coords(volcano_name)

    if len(station_coords) > n_stations:
        station_coords = station_coords[:n_stations]
    elif len(station_coords) < n_stations:
        missing = n_stations - len(station_coords)
        station_coords.extend([(f"PAD_{i:02d}", crater_coords) for i in range(missing)])

    lat_mean = float(np.mean([coords[1] for _, coords in station_coords]))
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mean))
    km_per_deg_lat = 111.0

    coords_array = np.array(
        [
            (
                (lon - crater_coords[0]) * km_per_deg_lon,
                (lat - crater_coords[1]) * km_per_deg_lat,
            )
            for _, (lon, lat) in station_coords
        ],
        dtype=np.float32,
    )
    dist_to_crater = np.linalg.norm(coords_array, axis=1, keepdims=True).astype(
        np.float32
    )

    network_xy = np.zeros((1, 2), dtype=np.float32)
    network_dist = np.zeros((1, 1), dtype=np.float32)

    xy_full = np.concatenate([coords_array, network_xy], axis=0)
    dist_full = np.concatenate([dist_to_crater, network_dist], axis=0)
    xy_norm = float(np.linalg.norm(xy_full) + 1e-6)
    dist_norm = float(np.linalg.norm(dist_full) + 1e-6)

    geom = np.concatenate([xy_full / xy_norm, dist_full / dist_norm], axis=1)
    return geom.astype(np.float32)


def build_volcano_geometry_bank(
    n_stations: int,
    volcano_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, Dict[str, int], tuple[str, ...]]:
    order = tuple(volcano_order) if volcano_order is not None else DEFAULT_VOLCANO_ORDER
    volcano_to_idx = {name: idx for idx, name in enumerate(order)}
    geom_bank = np.stack(
        [_build_geometry_for_volcano(name, n_stations=n_stations) for name in order],
        axis=0,
    )
    return geom_bank, volcano_to_idx, order


# Backward-compatible aliases.
stations = STATIONS_BY_VOLCANO
volcanoes = CRATER_BY_VOLCANO
