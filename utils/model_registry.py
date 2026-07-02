"""Central registry for model families, checkpoints, and ablation variants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models.UNet import UNet
from models.UNet_bottleneck_attention import UNetBottleneckAttention
from models.UNet_GraphSAGE import UNet_GraphSAGE
from models.UNet_MPNN import UNet_MPNN

UNET_BASE_KWARGS: dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 6,
    "init_features": 16,
    "depth": 5,
}

GRAPHSAGE_BASE_KWARGS: dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 6,
    "graph_levels": [3, 4],
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

MPNN_BASE_KWARGS: dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 6,
    "graph_topology": "fully_connected",
    "edge_feature_mode": "delta_pos",
    "node_feature_mode": "geometry",
    "graph_levels": [],
    "use_skip_graph": False,
    "use_bottleneck_attention": True,
    "graph_norm": "none",
}


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "edge_mpnn__encoder": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "graph_levels": [0, 1, 2],
        },
        "batch_size": 4,
        "sort_order": 100,
        "enabled": True,
        "aliases": (),
    },        
    "edge_mpnn__early_l2": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "graph_levels": [2],
        },
        "batch_size": 4,
        "sort_order": 50,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__early_l1": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "graph_levels": [1],
        },
        "batch_size": 4,
        "sort_order": 60,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__no_edge_feats": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "edge_feature_mode": "none",
        },
        "batch_size": 10,
        "sort_order": 90,
        "enabled": True,
        "aliases": (),
    },

    "edge_mpnn__xcorr": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "edge_feature_mode": "delta_pos_xcorr",
            "xcorr_feat_dim": 2,
        },
        "batch_size": 10,
        "sort_order": 120,
        "enabled": True,
        "aliases": (),
    },
    "unet": {
        "family": "unet",
        "trainer_kind": "unet_2d",
        "display_name": "UNet",
        "model_cls": UNet,
        "model_kwargs": deepcopy(UNET_BASE_KWARGS),
        "batch_size": 24,
        "sort_order": 10,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__bottleneck": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": deepcopy(MPNN_BASE_KWARGS),
        "batch_size": 10,
        "sort_order": 80,
        "enabled": True,
        "aliases": (),
    },
    "ablation_5_no_norm": {
        "family": "graphsage",
        "trainer_kind": "graph",
        "display_name": "UNet_GraphSAGE",
        "model_cls": UNet_GraphSAGE,
        "model_kwargs": {
            **GRAPHSAGE_BASE_KWARGS,
            "graph_norm_type": "none",
        },
        "batch_size": 10,
        "sort_order": 50,
        "enabled": True,
        "aliases": (),
    },
    "unet_bottleneck_attention": {
        "family": "unet",
        "trainer_kind": "unet_2d",
        "display_name": "UNetBottleneckAttention",
        "model_cls": UNetBottleneckAttention,
        "model_kwargs": {
            **UNET_BASE_KWARGS,
            "bottleneck_attn_heads": 4,
            "bottleneck_attn_dropout": 0.0,
            "bottleneck_attn_ff_mult": 2,
        },
        "batch_size": 24,
        "sort_order": 20,
        "enabled": True,
        "aliases": (),
    },
    "ablation_2_mlp_backend": {
        "family": "graphsage",
        "trainer_kind": "graph",
        "display_name": "UNet_GraphSAGE_Ablation2_MLPBackend",
        "model_cls": UNet_GraphSAGE,
        "model_kwargs": {
            **GRAPHSAGE_BASE_KWARGS,
            "graph_backend": "mlp",
        },
        "batch_size": 10,
        "sort_order": 20,
        "enabled": True,
        "aliases": (),
    },
    "ablation_3_no_message_passing": {
        "family": "graphsage",
        "trainer_kind": "graph",
        "display_name": "UNet_GraphSAGE_Ablation3_NoMessagePassing",
        "model_cls": UNet_GraphSAGE,
        "model_kwargs": {
            **GRAPHSAGE_BASE_KWARGS,
            "use_message_passing": False,
        },
        "batch_size": 10,
        "sort_order": 30,
        "enabled": True,
        "aliases": (),
    },
    "ablation_4_no_bottleneck_attention": {
        "family": "graphsage",
        "trainer_kind": "graph",
        "display_name": "UNet_GraphSAGE",
        "model_cls": UNet_GraphSAGE,
        "model_kwargs": {
            **GRAPHSAGE_BASE_KWARGS,
            "use_bottleneck_attention": False,
        },
        "batch_size": 10,
        "sort_order": 40,
        "enabled": True,
        "aliases": (),
    },
    "only_graph_no_attention": {
        "family": "graphsage",
        "trainer_kind": "graph",
        "display_name": "UNet_GraphSAGE",
        "model_cls": UNet_GraphSAGE,
        "model_kwargs": {
            **GRAPHSAGE_BASE_KWARGS,
            "graph_levels": [],
            "use_bottleneck_attention": False,
            "graph_norm_type": "none",
        },
        "batch_size": 10,
        "sort_order": 120,
        "enabled": True,
        "aliases": (),
    },
    "ablation_11_no_node_features": {
        "family": "graphsage",
        "trainer_kind": "graph",
        "display_name": "UNet_GraphSAGE",
        "model_cls": UNet_GraphSAGE,
        "model_kwargs": {
            **GRAPHSAGE_BASE_KWARGS,
            "node_feature_mode": "none",
        },
        "batch_size": 10,
        "sort_order": 150,
        "enabled": True,
        "aliases": (),
    },
    "pairwise_conv2d__l0": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "pairwise_conv_levels": [0],
            "pairwise_conv_kernel": 9,
        },
        "batch_size": 8,
        "sort_order": 10,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__early_l0": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "graph_levels": [0],
        },
        "batch_size": 8,
        "sort_order": 20,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__aggr_max": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "mpnn_aggr": "max",
        },
        "batch_size": 10,
        "sort_order": 30,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__layers_4": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "mpnn_layers": 4,
        },
        "batch_size": 10,
        "sort_order": 40,
        "enabled": True,
        "aliases": (),
    },

    "edge_mpnn__star_topology": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "graph_topology": "star",
        },
        "batch_size": 10,
        "sort_order": 110,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__no_spatial_info": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "edge_feature_mode": "none",
            "node_feature_mode": "none",
        },
        "batch_size": 10,
        "sort_order": 130,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__rsam": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "use_rsam_node_feat": True,
        },
        "batch_size": 8,
        "sort_order": 140,
        "enabled": True,
        "aliases": (),
    },
    "edge_mpnn__no_attention": {
        "family": "mpnn",
        "trainer_kind": "graph",
        "display_name": "UNet_MPNN",
        "model_cls": UNet_MPNN,
        "model_kwargs": {
            **MPNN_BASE_KWARGS,
            "use_bottleneck_attention": False,
        },
        "batch_size": 8,
        "sort_order": 150,
        "enabled": True,
        "aliases": (),
    },
    # "leaner_model": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "graph_levels": [],
    #         "graph_norm_type": "none",
    #         "virtual_node_pool_mode": "mean",
    #         "bottleneck_virtual_node_pool_mode": "mean",
    #         "use_skip_graph": False,
    #     },
    #     "batch_size": 10,
    #     "sort_order": 130,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "leanest_model": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "graph_levels": [],
    #         "graph_norm_type": "none",
    #         "virtual_node_pool_mode": "mean",
    #         "bottleneck_virtual_node_pool_mode": "mean",
    #         "use_skip_graph": False,
    #         "node_feature_mode": "learned_station_embedding",
    #     },
    #     "batch_size": 10,
    #     "sort_order": 140,
    #     "enabled": False,
    #     "aliases": (),
    # },
    # "ablation_6_batchnorm": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "graph_norm_type": "batchnorm",
    #     },
    #     "batch_size": 18,
    #     "sort_order": 60,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "ablation_7_mean_virtual_node_pool": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE_Ablation7_MeanVirtualNodePooling",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "virtual_node_pool_mode": "mean",
    #         "bottleneck_virtual_node_pool_mode": "mean",
    #     },
    #     "batch_size": 14,
    #     "sort_order": 70,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "ablation_8_graph_only_bottleneck": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "graph_levels": [],
    #     },
    #     "batch_size": 10,
    #     "sort_order": 80,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "ablation_9_no_skip_graph": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE_Ablation9_NoSkipGraph",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "use_skip_graph": False,
    #     },
    #     "batch_size": 18,
    #     "sort_order": 90,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "ablation_10_learned_station_embedding_only": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "node_feature_mode": "learned_station_embedding",
    #         "station_embedding_dim": 3,
    #     },
    #     "batch_size": 18,
    #     "sort_order": 100,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "only_bottleneck_attention": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "graph_levels": [],
    #         "use_skip_graph": False,
    #         "use_message_passing": False,
    #     },
    #     "batch_size": 10,
    #     "sort_order": 110,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "v5_full": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": deepcopy(GRAPHSAGE_BASE_KWARGS),
    #     "batch_size": 18,
    #     "sort_order": 10,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "v5_full_with_level_2": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "graph_levels": [2, 3, 4],
    #     },
    #     "batch_size": 18,
    #     "sort_order": 11,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "v5_full_bigger_model": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "init_features": 24,
    #     },
    #     "batch_size": 18,
    #     "sort_order": 12,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "v5_full_all_levels": {
    #     "family": "graphsage",
    #     "trainer_kind": "graph",
    #     "display_name": "UNet_GraphSAGE",
    #     "model_cls": UNet_GraphSAGE,
    #     "model_kwargs": {
    #         **GRAPHSAGE_BASE_KWARGS,
    #         "attention_pool_mode": "all_levels",
    #     },
    #     "batch_size": 18,
    #     "sort_order": 13,
    #     "enabled": True,
    #     "aliases": (),
    # },
}


def get_model_spec(model_key: str) -> dict[str, Any]:
    if model_key in MODEL_REGISTRY:
        return deepcopy(MODEL_REGISTRY[model_key])
    for spec in MODEL_REGISTRY.values():
        if model_key in spec.get("aliases", ()):
            return deepcopy(spec)
    raise KeyError(
        f"Unknown model key '{model_key}'. Available: {sorted(MODEL_REGISTRY.keys())}"
    )


def list_model_specs(
    *,
    family: str | None = None,
    trainer_kind: str | None = None,
    preserve_order: bool = False,
) -> dict[str, dict[str, Any]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for key, spec in MODEL_REGISTRY.items():
        if family is not None and spec["family"] != family:
            continue
        if trainer_kind is not None and spec["trainer_kind"] != trainer_kind:
            continue
        if not spec.get("enabled", True):
            continue
        items.append((key, deepcopy(spec)))
    if not preserve_order:
        items.sort(key=lambda item: (item[1]["family"], item[1]["sort_order"], item[0]))
    return dict(items)


def build_model_from_spec(model_key: str, n_classes: int = 6, **overrides):
    spec = get_model_spec(model_key)
    model_cls = spec["model_cls"]
    kwargs = dict(spec["model_kwargs"])
    kwargs.update(overrides)
    kwargs.setdefault("out_channels", n_classes)
    kwargs.setdefault("in_channels", 1)
    return model_cls(**kwargs)


MODEL_SPECS = {
    key: {
        "display_name": spec["display_name"],
        "model_cls": spec["model_cls"],
        "model_kwargs": deepcopy(spec["model_kwargs"]),
        "family": spec["family"],
        "trainer_kind": spec["trainer_kind"],
        "batch_size": spec["batch_size"],
        "sort_order": spec["sort_order"],
        "enabled": spec["enabled"],
    }
    for key, spec in MODEL_REGISTRY.items()
}
