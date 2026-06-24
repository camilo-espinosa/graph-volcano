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

    for module_name in ["upconv_list", "decoder_list", "conv_final"]:
        module = getattr(model, module_name, None)
        if module is None:
            raise AttributeError(
                f"Model missing expected decoder module '{module_name}' for protocol B"
            )
        for p in module.parameters():
            p.requires_grad = True


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total
