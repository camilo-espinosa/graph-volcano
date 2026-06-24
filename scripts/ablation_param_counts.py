"""Instantiate all ablation variants and print parameter counts.

Usage:
    python scripts/ablation_param_counts.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.UNet_GraphSAGE import UNet_GraphSAGE

GRAPH_LEVELS = [3, 4]
V5_FULL_KWARGS = {
    "graph_levels": GRAPH_LEVELS,
    "attention_pool_mode": "bottleneck_only",
    "use_bottleneck_attention": True,
    "graph_norm_type": "graphnorm",
    "node_feature_mode": "geometry",
    "graph_backend": "graphsage",
    "use_message_passing": True,
    "virtual_node_pool_mode": "learned",
    "bottleneck_virtual_node_pool_mode": "learned",
    "use_skip_graph": True,
    "init_features": 16,
    "depth": 5,
}

ABLATION_MODEL_KWARGS = {
    "v5_full_with_level_2": {
        **V5_FULL_KWARGS,
        "graph_levels": [2, 3, 4],
    },
    "v5_full_bigger_model": {
        **V5_FULL_KWARGS,
        "init_features": 24,
    },
    "v5_full_all_levels": {
        **V5_FULL_KWARGS,
        "attention_pool_mode": "all_levels",
    },
    "v5_full": {
        **V5_FULL_KWARGS,
    },
    "ablation_2_mlp_backend": {
        **V5_FULL_KWARGS,
        "graph_backend": "mlp",
    },
    "ablation_3_no_message_passing": {
        **V5_FULL_KWARGS,
        "use_message_passing": False,
    },
    "ablation_4_no_bottleneck_attention": {
        **V5_FULL_KWARGS,
        "use_bottleneck_attention": False,
    },
    "ablation_5_no_norm": {
        **V5_FULL_KWARGS,
        "graph_norm_type": "none",
    },
    "ablation_6_batchnorm": {
        **V5_FULL_KWARGS,
        "graph_norm_type": "batchnorm",
    },
    "ablation_7_mean_virtual_node_pool": {
        **V5_FULL_KWARGS,
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
    },
    "ablation_8_graph_only_bottleneck": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
    },
    "ablation_9_no_skip_graph": {
        **V5_FULL_KWARGS,
        "use_skip_graph": False,
    },
    "ablation_10_learned_station_embedding_only": {
        **V5_FULL_KWARGS,
        "node_feature_mode": "learned_station_embedding",
        "station_embedding_dim": 3,
    },
    "only_bottleneck_attention": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "use_bottleneck_attention": True,
        "use_skip_graph": False,
        "use_message_passing": False,
    },
    "only_graph_no_attention": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "use_bottleneck_attention": False,
        "graph_norm_type": "none",
    },
    "leaner_model": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "graph_norm_type": "none",
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
        "use_skip_graph": False,
    },
    "leanest_model": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "graph_norm_type": "none",
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
        "use_skip_graph": False,
        "node_feature_mode": "learned_station_embedding",
    },
}


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    rows: list[tuple[str, int, int]] = []

    for ablation_name, kwargs in ABLATION_MODEL_KWARGS.items():
        model = UNet_GraphSAGE(in_channels=1, out_channels=6, **kwargs)
        total, trainable = count_parameters(model)
        rows.append((ablation_name, total, trainable))

    name_width = max(len("ablation"), *(len(name) for name, _, _ in rows))
    total_width = len("total_params")
    trainable_width = len("trainable_params")

    header = (
        f"{'ablation':<{name_width}}  "
        f"{'total_params':>{total_width}}  "
        f"{'trainable_params':>{trainable_width}}"
    )
    print(header)
    print("-" * len(header))

    for name, total, trainable in rows:
        print(
            f"{name:<{name_width}}  "
            f"{total:>{total_width},}  "
            f"{trainable:>{trainable_width},}"
        )


if __name__ == "__main__":
    main()
