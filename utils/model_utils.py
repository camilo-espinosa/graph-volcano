from pathlib import Path
from typing import Optional

import torch

from models.PhaseNet import PhaseNet
from models.UNet import UNet
from models.UNet_GraphSAGE import (
    UNet_GraphSAGE,
    UNet_GraphSAGE_Ablation2_MLPBackend,
    UNet_GraphSAGE_Ablation3_NoMessagePassing,
    UNet_GraphSAGE_Ablation7_MeanVirtualNodePooling,
    UNet_GraphSAGE_Ablation9_NoSkipGraph,
)


def build_model(arch: str, n_classes: int = 6, **model_kwargs):
    if arch == "UNet":
        return UNet(in_channels=1, out_channels=n_classes, init_features=16, depth=5)
    if arch == "PhaseNet":
        return PhaseNet(
            in_channels=8,
            classes=n_classes,
            depth=5,
            kernel_size=7,
            stride=2,
            norm="std",
            filters_root=32,
        )
    if arch == "UNet_GraphSAGE":
        return UNet_GraphSAGE(
            in_channels=1,
            out_channels=n_classes,
            init_features=16,
            depth=5,
        )
    if arch == "UNet_GraphSAGE_Ablation2_MLPBackend":
        return UNet_GraphSAGE_Ablation2_MLPBackend(
            in_channels=1,
            out_channels=n_classes,
            init_features=16,
            depth=5,
            **model_kwargs,
        )
    if arch == "UNet_GraphSAGE_Ablation3_NoMessagePassing":
        return UNet_GraphSAGE_Ablation3_NoMessagePassing(
            in_channels=1,
            out_channels=n_classes,
            init_features=16,
            depth=5,
            **model_kwargs,
        )
    if arch == "UNet_GraphSAGE_Ablation7_MeanVirtualNodePooling":
        return UNet_GraphSAGE_Ablation7_MeanVirtualNodePooling(
            in_channels=1,
            out_channels=n_classes,
            init_features=16,
            depth=5,
            **model_kwargs,
        )
    if arch == "UNet_GraphSAGE_Ablation9_NoSkipGraph":
        return UNet_GraphSAGE_Ablation9_NoSkipGraph(
            in_channels=1,
            out_channels=n_classes,
            init_features=16,
            depth=5,
            **model_kwargs,
        )

    # Backward-compatible aliases from older model naming.
    if arch in {
        "UNet_GraphSAGE_v2",
        "UNet_GraphSAGE_v3",
        "UNet_GraphSAGE_v4",
        "UNet_GraphSAGE_v5",
        "UNet_GraphSAGE_v6",
    }:
        return UNet_GraphSAGE(
            in_channels=1,
            out_channels=n_classes,
            init_features=16,
            depth=5,
            **model_kwargs,
        )
    raise ValueError(
        f"Unsupported arch '{arch}'. Allowed: 'UNet', 'PhaseNet', 'UNet_GraphSAGE', 'UNet_GraphSAGE_Ablation2_MLPBackend', 'UNet_GraphSAGE_Ablation3_NoMessagePassing', 'UNet_GraphSAGE_Ablation7_MeanVirtualNodePooling', 'UNet_GraphSAGE_Ablation9_NoSkipGraph', and legacy aliases 'UNet_GraphSAGE_v2'..'UNet_GraphSAGE_v6'."
    )


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Path) -> bool:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return True


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    checkpoint_path: Path,
    best_val_loss: Optional[float] = None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    torch.save(payload, checkpoint_path)
