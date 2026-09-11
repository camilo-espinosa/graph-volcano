from __future__ import annotations

import numpy as np
import torch


def split_indices_stratified(
    label_ids: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")

    rng = np.random.default_rng(int(seed))
    label_ids = np.asarray(label_ids).astype(np.int64)
    n = int(label_ids.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 samples for 85/15 split.")

    all_indices = np.arange(n, dtype=np.int64)
    val_indices = []

    for c in sorted(np.unique(label_ids).tolist()):
        class_idx = all_indices[label_ids == c]
        class_count = int(class_idx.shape[0])
        rng.shuffle(class_idx)

        if class_count <= 1:
            take = 0
        else:
            take = max(1, int(round(class_count * val_frac)))
            if take >= class_count:
                take = class_count - 1

        if take > 0:
            val_indices.extend(class_idx[:take].tolist())

    val_indices = np.array(sorted(set(val_indices)), dtype=np.int64)

    if val_indices.size == 0:
        val_indices = np.array([int(rng.integers(0, n))], dtype=np.int64)

    train_mask = np.ones(n, dtype=bool)
    train_mask[val_indices] = False
    train_indices = all_indices[train_mask]

    if train_indices.size == 0 or val_indices.size == 0:
        raise RuntimeError(
            f"Invalid split produced empty set: n_train={train_indices.size}, n_val={val_indices.size}"
        )

    return train_indices, val_indices


def apply_finetune_protocol(model: torch.nn.Module, protocol_key: str) -> None:
    if protocol_key == "protocol_a_all_weights":
        for p in model.parameters():
            p.requires_grad = True
        return

    if protocol_key != "protocol_b_decoder_only":
        raise ValueError(f"Unknown protocol_key: {protocol_key}")

    for p in model.parameters():
        p.requires_grad = False

    def _resolve_required_module(
        aliases: tuple[str, ...],
        *,
        role_name: str,
        layout_name: str,
    ) -> torch.nn.Module:
        for alias in aliases:
            module = getattr(model, alias, None)
            if module is not None:
                return module
        alias_text = ", ".join(aliases)
        raise AttributeError(
            "Model is missing required decoder component for protocol B: "
            f"role='{role_name}', tried aliases=[{alias_text}], layout='{layout_name}', "
            f"model_class='{model.__class__.__name__}'"
        )

    # Supported decoder-only layouts:
    # - UNet family: upconv_list + decoder_list + conv
    # - PhaseNet/MuSSeg families: up_branch + out
    decoder_layouts: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
        (
            "unet_like",
            (
                ("upconv_list",),
                ("decoder_list",),
                ("conv", "conv_final"),
            ),
        ),
        (
            "phasenet_like",
            (
                ("up_branch",),
                ("out", "conv_final", "conv"),
            ),
        ),
    )

    selected_layout_name: str | None = None
    selected_modules: list[torch.nn.Module] = []

    for layout_name, required_roles in decoder_layouts:
        modules_for_layout: list[torch.nn.Module] = []
        layout_ok = True
        for aliases in required_roles:
            module = None
            for alias in aliases:
                candidate = getattr(model, alias, None)
                if candidate is not None:
                    module = candidate
                    break
            if module is None:
                layout_ok = False
                break
            modules_for_layout.append(module)

        if layout_ok:
            selected_layout_name = layout_name
            selected_modules = modules_for_layout
            break

    if selected_layout_name is None:
        supported_layouts = ", ".join(name for name, _ in decoder_layouts)
        raise AttributeError(
            "Unsupported model layout for protocol_b_decoder_only. "
            f"model_class='{model.__class__.__name__}', "
            f"supported_layouts=[{supported_layouts}]"
        )

    # Re-resolve through the strict helper so missing components raise actionable errors.
    if selected_layout_name == "unet_like":
        selected_modules = [
            _resolve_required_module(
                ("upconv_list",),
                role_name="decoder_upsampling",
                layout_name=selected_layout_name,
            ),
            _resolve_required_module(
                ("decoder_list",),
                role_name="decoder_blocks",
                layout_name=selected_layout_name,
            ),
            _resolve_required_module(
                ("conv", "conv_final"),
                role_name="decoder_head",
                layout_name=selected_layout_name,
            ),
        ]
    elif selected_layout_name == "phasenet_like":
        selected_modules = [
            _resolve_required_module(
                ("up_branch",),
                role_name="decoder_upsampling",
                layout_name=selected_layout_name,
            ),
            _resolve_required_module(
                ("out", "conv_final", "conv"),
                role_name="decoder_head",
                layout_name=selected_layout_name,
            ),
        ]
    else:
        raise RuntimeError(
            f"Unhandled decoder layout selection: {selected_layout_name}"
        )

    for module in selected_modules:
        for p in module.parameters():
            p.requires_grad = True

    trainable, total = trainable_parameter_count(model)
    if trainable <= 0 or total <= 0:
        raise RuntimeError(
            "protocol_b_decoder_only produced an invalid parameter state: "
            f"trainable={trainable}, total={total}, "
            f"model_class='{model.__class__.__name__}', layout='{selected_layout_name}'"
        )


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total
