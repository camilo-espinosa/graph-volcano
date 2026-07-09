"""Central registry for model families, checkpoints, and ablation variants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models.UNet import UNet
from models.PhaseNet import PhaseNet
from models.PhaseNet_bottleneck_attention import PhaseNetBottleneckAttention
from models.UNet_bottleneck_attention import UNetBottleneckAttention


UNET_BASE_KWARGS: dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 6,
    "init_features": 16,
    "depth": 5,
    "feature_dropout": 0.2,
}

PHASENET_BASE_KWARGS: dict[str, Any] = {
    "in_channels": 8,
    "classes": 6,
    "depth": 5,
    "kernel_size": 7,
    "stride": 2,
    "filters_root": 32,
    "norm": "std",
    "feature_dropout": 0.2,
}

PHASENET_BOTTLENECK_ATTENTION_BASE_KWARGS: dict[str, Any] = {
    **PHASENET_BASE_KWARGS,
    "bottleneck_attn_heads": 4,
    "bottleneck_attn_dropout": 0.2,
    "bottleneck_attn_ff_mult": 2,
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
    "bottleneck_attn_dropout": 0.2,
}

MPNN_BASE_KWARGS: dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 6,
    "init_features": 16,
    "depth": 5,
    "n_stations": 8,
    "station_fusion_levels": [0, 1, 2],
    "fusion_kernel": 9,
    "readout_mode": "attention",
    "use_bottleneck_attention": True,
    "feature_dropout": 0.2,
    "bottleneck_attn_dropout": 0.2,

}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "phasenet_bottleneck_attention": {
        "family": "phasenet",
        "trainer_kind": "1d",
        "display_name": "PhaseNetBottleneckAttention",
        "model_cls": PhaseNetBottleneckAttention,
        "model_kwargs": deepcopy(PHASENET_BOTTLENECK_ATTENTION_BASE_KWARGS),
        "batch_size": 64,
        "sort_order": 6,
        "enabled": True,
        "aliases": ("PhaseNetBottleneckAttention",),
    },    
    "phasenet": {
        "family": "phasenet",
        "trainer_kind": "1d",
        "display_name": "PhaseNet",
        "model_cls": PhaseNet,
        "model_kwargs": deepcopy(PHASENET_BASE_KWARGS),
        "batch_size": 64,
        "sort_order": 5,
        "enabled": True,
        "aliases": ("PhaseNet",),
    },
    # "unet_bottleneck_attention": {
    #     "family": "unet",
    #     "trainer_kind": "2d",
    #     "display_name": "UNetBottleneckAttention",
    #     "model_cls": UNetBottleneckAttention,
    #     "model_kwargs": {
    #         **UNET_BASE_KWARGS,
    #         "bottleneck_attn_heads": 4,
    #         "bottleneck_attn_dropout": 0.2,
    #         "bottleneck_attn_ff_mult": 2,
    #     },
    #     "batch_size": 32,
    #     "sort_order": 20,
    #     "enabled": True,
    #     "aliases": (),
    # },
    "unet": {
        "family": "unet",
        "trainer_kind": "2d",
        "display_name": "UNet",
        "model_cls": UNet,
        "model_kwargs": deepcopy(UNET_BASE_KWARGS),
        "batch_size": 32,
        "sort_order": 10,
        "enabled": True,
        "aliases": (),
    },    
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
    if spec["family"] == "phasenet":
        kwargs.setdefault("classes", n_classes)
        kwargs.setdefault("in_channels", 8)
    else:
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
