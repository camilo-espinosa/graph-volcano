from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.MuSSeg import MuSSeg, StationAttentionBlock
from utils.station_info import get_crater_coords, get_station_coords


def _expected_station_dist(volcano_name: str) -> torch.Tensor:
    station_coords = list(get_station_coords(volcano_name).items())
    crater_coords = get_crater_coords(volcano_name)

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
    dist_to_crater = np.linalg.norm(coords_array, axis=1).astype(np.float32)
    n_stations = int(dist_to_crater.shape[0])
    ranks = np.empty(n_stations, dtype=np.float32)
    ranks[np.argsort(dist_to_crater)] = np.arange(n_stations, dtype=np.float32)
    normalized = 1.0 - ranks / n_stations + 1.0 / n_stations
    return torch.from_numpy(normalized[:, None].astype(np.float32))


class MuSSegDistanceFeaturesTest(unittest.TestCase):
    def test_station_attention_zero_bias_matches_default(self) -> None:
        torch.manual_seed(0)
        block = StationAttentionBlock(channels=8, heads=4)
        x = torch.randn(2, 4, 8, 16)
        zero_bias = torch.zeros(4, 4)

        expected = block(x)
        actual = block(x, dist_bias=zero_bias)

        torch.testing.assert_close(actual, expected)

    def test_distance_feature_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "volcano_name is required"):
            MuSSeg(
                shared_station_encoder=True,
                station_interaction="late_attention",
                use_distance_attn_bias=True,
            )

        with self.assertRaisesRegex(
            ValueError, "requires station_interaction='late_attention'"
        ):
            MuSSeg(
                shared_station_encoder=True,
                use_distance_attn_bias=True,
                use_distance_bottleneck_emb=False,
                volcano_name="NVCHVC",
            )

    def test_station_distance_buffer_and_permutation(self) -> None:
        model = MuSSeg(
            shared_station_encoder=True,
            station_interaction="late_attention",
            use_distance_attn_bias=True,
            use_distance_bottleneck_emb=True,
            volcano_name="NVCHVC",
        )

        torch.testing.assert_close(model.station_dist, _expected_station_dist("NVCHVC"))

        original = model.station_dist.clone()
        perm = torch.tensor([2, 0, 1, 3, 4, 5, 7, 6], dtype=torch.long)
        model.permute_stations(perm)
        torch.testing.assert_close(model.station_dist, original[perm])

    def test_forward_shared_with_distance_features(self) -> None:
        model = MuSSeg(
            in_channels=8,
            classes=6,
            depth=5,
            filters_root=32,
            bottleneck_attention=True,
            shared_station_encoder=True,
            station_interaction="late_attention",
            use_distance_attn_bias=True,
            use_distance_bottleneck_emb=True,
            volcano_name="NVCHVC",
        )

        x = torch.randn(2, 8, 1, 256)
        y = model(x)

        self.assertEqual(y.ndim, 3)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 6)

    def test_model_registry_distance_variants(self) -> None:
        phase_no_module = types.ModuleType("models.PhaseNO")

        class PhaseNO:  # pragma: no cover - import stub only
            pass

        phase_no_module.PhaseNO = PhaseNO
        prior_phase_no = sys.modules.get("models.PhaseNO")
        prior_registry = sys.modules.pop("utils.model_registry", None)
        sys.modules["models.PhaseNO"] = phase_no_module

        try:
            model_registry = importlib.import_module("utils.model_registry")
        finally:
            if prior_phase_no is not None:
                sys.modules["models.PhaseNO"] = prior_phase_no
            else:
                sys.modules.pop("models.PhaseNO", None)
            if prior_registry is not None:
                sys.modules["utils.model_registry"] = prior_registry

        specs = {
            key: model_registry.MODEL_REGISTRY[key]
            for key in (
                "musseg_pi_se_lsa_ba_dist_attn",
                "musseg_pi_se_lsa_ba_dist_emb",
                "musseg_pi_se_lsa_ba_dist_both",
            )
        }

        self.assertEqual(
            specs["musseg_pi_se_lsa_ba_dist_attn"]["model_kwargs"]["volcano_name"],
            "NVCHVC",
        )
        self.assertTrue(
            specs["musseg_pi_se_lsa_ba_dist_attn"]["model_kwargs"][
                "use_distance_attn_bias"
            ]
        )
        self.assertTrue(
            specs["musseg_pi_se_lsa_ba_dist_emb"]["model_kwargs"][
                "use_distance_bottleneck_emb"
            ]
        )
        self.assertTrue(
            specs["musseg_pi_se_lsa_ba_dist_both"]["model_kwargs"][
                "use_distance_attn_bias"
            ]
        )
        self.assertTrue(
            specs["musseg_pi_se_lsa_ba_dist_both"]["model_kwargs"][
                "use_distance_bottleneck_emb"
            ]
        )
        self.assertEqual(
            specs["musseg_pi_se_lsa_ba_dist_attn"]["sort_order"],
            61,
        )
        self.assertEqual(
            specs["musseg_pi_se_lsa_ba_dist_emb"]["sort_order"],
            62,
        )
        self.assertEqual(
            specs["musseg_pi_se_lsa_ba_dist_both"]["sort_order"],
            63,
        )


if __name__ == "__main__":
    unittest.main()
