"""Central registry for the active model definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models.PhaseNO import PhaseNO
from models.PhaseNet import PhaseNet
from models.PhaseNet_bottleneck_attention import PhaseNetBottleneckAttention
from models.MuSSeg import MuSSeg
from models.PhaseNet_permutation_invariant import PhaseNetPermutationInvariant
from models.UNet import UNet
from models.UNet_bottleneck_attention import UNetBottleneckAttention

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

PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS: dict[str, Any] = {
    **PHASENET_BASE_KWARGS,
    "filters_root": 32,
    "bottleneck_attention": False,
    "shared_station_encoder": False,
    "station_interaction": "none",
    "pairconv_levels": [],
    "pairconv_aggregation": "sum",
    "pairconv_ratio": 1.0,
    "station_attention_levels": [],
    "bottleneck_attn_heads": 4,
    "bottleneck_attn_dropout": 0.2,
    "bottleneck_attn_ff_mult": 2,
    "station_attn_heads": 4,
    "station_attn_dropout": 0.2,
    "station_attn_ff_mult": 2,
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # "phasenet": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "display_name": "PhaseNet",
    #     "model_cls": PhaseNet,
    #     "model_kwargs": deepcopy(PHASENET_BASE_KWARGS),
    #     "batch_size": 64,
    #     "sort_order": 10,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "phasenet_bottleneck_attention": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "display_name": "PhaseNetBottleneckAttention",
    #     "model_cls": PhaseNetBottleneckAttention,
    #     "model_kwargs": deepcopy(PHASENET_BOTTLENECK_ATTENTION_BASE_KWARGS),
    #     "batch_size": 64,
    #     "sort_order": 13,
    #     "enabled": True,
    #     "aliases": ("phasenet_ba",),
    # },
    # "phasenet_ba_fr24": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "display_name": "PhaseNetBottleneckAttention_FR24",
    #     "model_cls": PhaseNetBottleneckAttention,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_BOTTLENECK_ATTENTION_BASE_KWARGS),
    #         "filters_root": 24,
    #     },
    #     "batch_size": 64,
    #     "sort_order": 30,
    #     "enabled": True,
    #     "aliases": (),
    # },
    "musseg_pi_se_ba": {
        "family": "phasenet",
        "trainer_kind": "1d",
        "display_name": "MuSSeg_PI_SE_BA",
        "model_cls": MuSSeg,
        "model_kwargs": {
            **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
            "bottleneck_attention": True,
            "shared_station_encoder": True,
        },
        "batch_size": 20,
        "sort_order": 20,
        "enabled": True,
        "aliases": (),
    },
    "musseg_fr24": {
        "family": "phasenet",
        "trainer_kind": "1d",
        "display_name": "MuSSeg_FR24",
        "model_cls": MuSSeg,
        "model_kwargs": {
            **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
            "filters_root": 24,
            "bottleneck_attention": True,
            "shared_station_encoder": False,
        },
        "batch_size": 16,
        "sort_order": 25,
        "enabled": True,
        "aliases": (),
    },
    "musseg_pi_se_lpc_sum_ba": {
        "family": "phasenet",
        "trainer_kind": "1d",
        "display_name": "MuSSeg_PI_SE_LatePairConvSum_BA",
        "model_cls": MuSSeg,
        "model_kwargs": {
            **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
            "bottleneck_attention": True,
            "shared_station_encoder": True,
            "station_interaction": "late_pairconv",
            "pairconv_aggregation": "sum",
            "pairconv_ratio": 0.25,
        },
        "batch_size": 12,
        "sort_order": 40,
        "enabled": True,
        "aliases": (),
    },
    "musseg_pi_se_lpc_attn_ba": {
        "family": "phasenet",
        "trainer_kind": "1d",
        "display_name": "MuSSeg_PI_SE_LatePairConvAttention_BA",
        "model_cls": MuSSeg,
        "model_kwargs": {
            **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
            "bottleneck_attention": True,
            "shared_station_encoder": True,
            "station_interaction": "late_pairconv",
            "pairconv_aggregation": "attention",
            "pairconv_ratio": 0.25,
        },
        "batch_size": 8,
        "sort_order": 50,
        "enabled": True,
        "aliases": (),
    },
    "musseg_pi_se_lsa_ba": {
        "family": "phasenet",
        "trainer_kind": "1d",
        "display_name": "MuSSeg_PI_SE_LateStationAttention_BA",
        "model_cls": MuSSeg,
        "model_kwargs": {
            **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
            "bottleneck_attention": True,
            "shared_station_encoder": True,
            "station_interaction": "late_attention",
        },
        "batch_size": 12,
        "sort_order": 60,
        "enabled": True,
        "aliases": (),
    },
    # "phaseno": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "display_name": "PhaseNO",
    #     "model_cls": PhaseNO,
    #     "model_kwargs": {
    #         "in_channels": 8,
    #         "classes": 6,
    #         "modes": 24,
    #         "width": 48,
    #     },
    #     "batch_size": 8,
    #     "sort_order": 70,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "unet": {
    #     "family": "unet",
    #     "trainer_kind": "2d",
    #     "display_name": "UNet",
    #     "model_cls": UNet,
    #     "model_kwargs": {
    #         "in_channels": 1,
    #         "out_channels": 6,
    #         "init_features": 16,
    #         "depth": 5,
    #     },
    #     "batch_size": 32,
    #     "sort_order": 5,
    #     "enabled": True,
    #     "aliases": (),
    # },
    # "unet_attention": {
    #     "family": "unet",
    #     "trainer_kind": "2d",
    #     "display_name": "UNetBottleneckAttention",
    #     "model_cls": UNetBottleneckAttention,
    #     "model_kwargs": {
    #         "in_channels": 1,
    #         "out_channels": 6,
    #         "init_features": 16,
    #         "depth": 5,
    #         "bottleneck_attn_heads": 4,
    #         "bottleneck_attn_dropout": 0.2,
    #         "bottleneck_attn_ff_mult": 2,
    #         "feature_dropout": 0.2,
    #     },
    #     "batch_size": 32,
    #     "sort_order": 6,
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
