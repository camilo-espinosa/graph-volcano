from __future__ import annotations

from typing import Dict, Tuple


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


# Backward-compatible aliases.
stations = STATIONS_BY_VOLCANO
volcanoes = CRATER_BY_VOLCANO
