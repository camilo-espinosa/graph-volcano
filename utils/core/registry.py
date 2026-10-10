"""Central registry for the active model definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models.PhaseNet import PhaseNet
from models.PhaseNet_bottleneck_attention import PhaseNetBottleneckAttention
from models.MuSSeg import MuSSeg
from models.MuSSED import MuSSED
from models.MuSSED_FB import MuSSEDFB
from models.MuSSED_TCN import MuSSEDTCN
from models.MuSSED_v2 import MuSSED as MuSSEDv2
from models.UNet import UNet
from models.UNet_bottleneck_attention import UNetBottleneckAttention

MUSSED_DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "class": 4.0,
    "bbox": 2.5,
    "giou": 2.5,
    "confidence": 2.0,
}

# Shared defaults for event-detection matching used by training/evaluation/reporting.
EVENT_DETECTION_EVAL_DEFAULTS: dict[str, float | str] = {
    "match_iou_threshold": 0.3,
    "matching_strategy": "iou",  # one of: iou, overlap_recall, dual
    "overlap_recall_threshold": 0.8,
}


def _detection_spec(model_cls, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": "event_detection",
        "trainer_kind": "event_detection",
        "model_cls": model_cls,
        "model_kwargs": model_kwargs,
        "batch_size": 16,
    }


def _baseline_mussed(key: str) -> tuple[str, type, dict[str, Any]]:
    _, root, depth = key.split("_")
    filters_root = int(root[1:])
    layers = int(depth[1:])
    return (
        key,
        MuSSED,
        {
            "num_queries": 1,
            "use_query_cross_attention": False,
            "filters_root": filters_root,
            "depth": layers,
            "stride": 2,
            "dilation": 1,
            "memory_levels": (0, 1) if layers == 3 else (0, 1, 2),
        },
    )


def _tcn_config(channels: int, blocks: int) -> tuple[str, type, dict[str, Any]]:
    return (
        f"MuSSED_TCN_c{channels}_b{blocks}",
        MuSSEDTCN,
        {"channels": channels, "num_blocks": blocks, "num_queries": 1},
    )


def _filterbank_config(
    bands: int, filters_root: int
) -> tuple[str, type, dict[str, Any]]:
    return (
        f"MuSSED_FB_f{bands}_r{filters_root}",
        MuSSEDFB,
        {
            "num_bands": bands,
            "filters_root": filters_root,
            "num_queries": 1,
        },
    )


def _v2_config(root: int, depth: int) -> tuple[str, type, dict[str, Any]]:
    return (
        f"MuSSEDv2_r{root}_d{depth}",
        MuSSEDv2,
        {
            "num_queries": 1,
            "use_query_cross_attention": False,
            "filters_root": root,
            "depth": depth,
            "stride": 2,
            "dilation": 1,
            "memory_levels": (0, 1, 2),
        },
    )


def _ordered_detection_specs() -> dict[str, dict[str, Any]]:
    medium = [
        _baseline_mussed("MuSSED_r32_d4"),
        _tcn_config(48, 10),
        _filterbank_config(16, 48),
        _v2_config(32, 4),
    ]
    light = [
        _baseline_mussed("MuSSED_r24_d3"),
        _tcn_config(32, 8),
        _filterbank_config(8, 24),
        _baseline_mussed("MuSSED_r32_d3"),
        _tcn_config(32, 10),
        _filterbank_config(8, 48),
        _baseline_mussed("MuSSED_r24_d4"),
        _tcn_config(32, 12),
        _filterbank_config(16, 24),
        _baseline_mussed("MuSSED_r48_d3"),
        _tcn_config(48, 8),
        _filterbank_config(32, 24),
    ]
    large = [
        _baseline_mussed("MuSSED_r48_d4"),
        _tcn_config(48, 12),
        _filterbank_config(32, 48),
        _baseline_mussed("MuSSED_r24_d5"),
        _tcn_config(64, 8),
        _baseline_mussed("MuSSED_r64_d3"),
        _v2_config(48, 5),
        _baseline_mussed("MuSSED_r32_d5"),
        _tcn_config(64, 10),
        _baseline_mussed("MuSSED_r24_d6"),
        _baseline_mussed("MuSSED_r64_d4"),
        _tcn_config(64, 12),
        _baseline_mussed("MuSSED_r48_d5"),
        _baseline_mussed("MuSSED_r32_d6"),
        _baseline_mussed("MuSSED_r48_d6"),
        _baseline_mussed("MuSSED_r64_d5"),
        _baseline_mussed("MuSSED_r64_d6"),
    ]

    specs: dict[str, dict[str, Any]] = {}
    for key, model_cls, model_kwargs in medium + light + large:
        if key in specs:
            raise ValueError(f"Duplicate event-detection model key: {key}")
        specs[key] = _detection_spec(model_cls, model_kwargs)
    return specs


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # Disabled during the Test-1 MuSSED sweep. Restore these entries when
    # running the other evaluation protocols.
    # "unet": {
    #     "family": "unet",
    #     "trainer_kind": "segmentation",
    #     "model_cls": UNet,
    #     "model_kwargs": {
    #         "in_channels": 1,
    #         "out_channels": 6,
    #         "init_features": 16,
    #         "depth": 5,
    #     },
    #     "batch_size": 32,
    # },
    # "unet_attention": {
    #     "family": "unet",
    #     "trainer_kind": "segmentation",
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
    # },
    # "phasenet": {
    #     "family": "phasenet",
    #     "trainer_kind": "segmentation",
    #     "model_cls": PhaseNet,
    #     "model_kwargs": {},
    #     "batch_size": 64,
    # },
    # "MuSSED": {
    #     "family": "event_detection",
    #     "trainer_kind": "event_detection",
    #     "model_cls": MuSSED,
    #     "model_kwargs": {"num_queries": 1, "use_query_cross_attention": False},
    #     "batch_size": 16,
    # },
    # "MuSSeg": {
    #     "family": "phasenet",
    #     "trainer_kind": "segmentation",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {},
    #     "batch_size": 32,
    # },
    # "phasenet_bottleneck_attention": {
    #     "family": "phasenet",
    #     "trainer_kind": "segmentation",
    #     "model_cls": PhaseNetBottleneckAttention,
    #     "model_kwargs": {},
    #     "batch_size": 64,
    # },
    **_ordered_detection_specs(),
}


def get_model_spec(model_key: str) -> dict[str, Any]:
    if model_key in MODEL_REGISTRY:
        return deepcopy(MODEL_REGISTRY[model_key])
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
        items.append((key, deepcopy(spec)))
    if not preserve_order:
        items.sort(key=lambda item: (item[1]["family"], item[0]))
    return dict(items)


def build_model_from_spec(model_key: str, n_classes: int = 6, **overrides):
    spec = get_model_spec(model_key)
    model_cls = spec["model_cls"]
    kwargs = dict(spec["model_kwargs"])
    kwargs.update(overrides)
    if spec["family"] == "phasenet":
        kwargs.setdefault("classes", n_classes)
        kwargs.setdefault("in_channels", 8)
    elif spec["family"] == "unet":
        kwargs.setdefault("out_channels", n_classes)
        kwargs.setdefault("in_channels", 1)
    else:
        pass
    return model_cls(**kwargs)


MODEL_SPECS = {
    key: {
        "model_cls": spec["model_cls"],
        "model_kwargs": deepcopy(spec["model_kwargs"]),
        "family": spec["family"],
        "trainer_kind": spec["trainer_kind"],
        "batch_size": spec["batch_size"],
        "loss_weights": deepcopy(spec.get("loss_weights", {})),
        "eval_matching": deepcopy(
            spec.get("eval_matching", EVENT_DETECTION_EVAL_DEFAULTS)
        ),
    }
    for key, spec in MODEL_REGISTRY.items()
}
