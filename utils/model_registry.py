"""Central registry for the active model definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models.PhaseNet import PhaseNet
from models.PhaseNet_bottleneck_attention import PhaseNetBottleneckAttention
from models.MuSSeg import MuSSeg
from models.MuSSED import MuSSED
from models.UNet import UNet
from models.UNet_bottleneck_attention import UNetBottleneckAttention

MUSSED_DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "class": 4.0,
    "bbox": 2.0,
    "giou": 2.0,
    "confidence": 2.0,
}

# Shared defaults for event-detection matching used by training/evaluation/reporting.
EVENT_DETECTION_EVAL_DEFAULTS: dict[str, float | str] = {
    "match_iou_threshold": 0.3,
    "matching_strategy": "iou",  # one of: iou, overlap_recall, dual
    "overlap_recall_threshold": 0.8,
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
   "MuSSED_k3": {
        "family": "event_detection",
        "trainer_kind": "event_detection",
        "model_cls": MuSSED,
        "model_kwargs": {},
        "batch_size": 14,
    },    
   "MuSSED_k127": {
        "family": "event_detection",
        "trainer_kind": "event_detection",
        "model_cls": MuSSED,
        "model_kwargs": {"query_head_kernel_size":127},
        "batch_size": 14,
    },    
#     "MuSSeg": {
#         "family": "phasenet",
#         "trainer_kind": "1d",
#         "model_cls": MuSSeg,
#         "model_kwargs": {},
#         "batch_size": 32,
#     },
#    # ---- Tier B: baselines ----
#     "unet": {
#         "family": "unet",
#         "trainer_kind": "2d",
#         "model_cls": UNet,
#         "model_kwargs": {
#             "in_channels": 1,
#             "out_channels": 6,
#             "init_features": 16,
#             "depth": 5,
#         },
#         "batch_size": 32,
#     },
#     "unet_attention": {
#         "family": "unet",
#         "trainer_kind": "2d",
#         "model_cls": UNetBottleneckAttention,
#         "model_kwargs": {
#             "in_channels": 1,
#             "out_channels": 6,
#             "init_features": 16,
#             "depth": 5,
#             "bottleneck_attn_heads": 4,
#             "bottleneck_attn_dropout": 0.2,
#             "bottleneck_attn_ff_mult": 2,
#             "feature_dropout": 0.2,
#         },
#         "batch_size": 32,
#     },
#     "phasenet": {
#         "family": "phasenet",
#         "trainer_kind": "1d",
#         "model_cls": PhaseNet,
#         "model_kwargs": {},
#         "batch_size": 64,
#     },
#     "phasenet_bottleneck_attention": {
#         "family": "phasenet",
#         "trainer_kind": "1d",
#         "model_cls": PhaseNetBottleneckAttention,
#         "model_kwargs": {},
#         "batch_size": 64,
#     },
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
        "eval_matching": deepcopy(spec.get("eval_matching", EVENT_DETECTION_EVAL_DEFAULTS)),
    }
    for key, spec in MODEL_REGISTRY.items()
}
