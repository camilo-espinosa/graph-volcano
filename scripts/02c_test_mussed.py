"""
02c_test_mussed.py

Training script for MuSSED (Multi-Station Seismic Event Detection).

Implements:
- Phase 2 training infrastructure (losses, matching, metrics)
- Phase 3 training loop
- Full training pipeline with validation and checkpointing

Features:
- Displays batch progress during training (Epoch X [Batch Y/Z])
- Tracks Mean F1 Score and Mean IoU on validation set
- Automatic best model checkpointing based on mAP

Usage:
    python scripts/02c_test_mussed.py                    # Uses NVCHVC fold_01 defaults
    python scripts/02c_test_mussed.py --epochs 100       # Custom epochs
    python scripts/02c_test_mussed.py --batch-size 16    # Custom batch size
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.MuSSED import MuSSED
from utils.musseg_utils import build_musseg_dataloader
from utils.trainer_detr_1d import DETRTrainer

# ================================ CONFIG ================================
# Best ablation from model registry: musseg_d4_r16_s222_k127
# Configured as encoder for MuSSED with late attention and bottleneck attention
# Note: use_temporal_projection=False passes bottleneck features directly to decoder
#       with only positional encoding (more efficient)

MUSSED_CONFIG = {
    # Encoder (best MuSSeg ablation: musseg_d4_r16_s222_k127)
    "num_classes": 6,
    "depth": 4,
    "kernel_size": 127,
    "stride": [2, 2, 2],
    "dilation": [1, 1, 1, 1],
    "filters_root": 16,  # encoder output channels = 16 * 2^3 = 128
    "bottleneck_attention": True,
    # Station interaction: always late attention at the final encoder level
    "bottleneck_attn_heads": 4,
    "bottleneck_attn_ff_mult": 2,
    "station_attn_heads": 4,
    "station_attn_ff_mult": 2,
    # Detection head
    "num_queries": 10,
    "query_dim": 128,  # Must match encoder output channels when use_temporal_projection=False
    "hidden_dim": 256,
    "num_decoder_heads": 4,
    "num_decoder_layers": 2,
    "decoder_dropout": 0.1,
    # Decoder optimization: skip unnecessary projection layer
    "use_temporal_projection": False,
    # Constrained interval regression (0 <= start <= center <= end <= 1)
    "constrain_intervals": True,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ================================ UTILITIES ================================


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main(args):
    """
    Main training function.
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MuSSED Training")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Output directory: {output_dir}")

    # Build data loaders
    print("\nBuilding data loaders...")
    if not Path(args.train_npz).exists():
        raise FileNotFoundError(f"Train NPZ not found: {args.train_npz}")

    train_dataset, train_loader = build_musseg_dataloader(
        Path(args.train_npz),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
        num_classes=6,
        station_rows=None,
        use_zero_mask=True,
        scramble_stations=False,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Train batches: {len(train_loader)}")

    val_loader = None
    if args.val_npz is not None and Path(args.val_npz).exists():
        val_dataset, val_loader = build_musseg_dataloader(
            Path(args.val_npz),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=DEVICE == "cuda",
            num_classes=6,
            station_rows=None,
            use_zero_mask=True,
            scramble_stations=False,
        )
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Val batches: {len(val_loader)}")

    # Build model
    print("\nBuilding MuSSED model...")
    model = MuSSED(**MUSSED_CONFIG)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Build trainer
    print("\nSetting up trainer...")
    trainer = DETRTrainer(model, device=DEVICE, dtype=DTYPE)
    trainer.setup_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )

    # Save config to output directory
    config_dict = {
        "model_config": MUSSED_CONFIG,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "confidence_threshold": args.confidence_threshold,
        },
        "device": DEVICE,
        "dtype": str(DTYPE),
        "timestamp": datetime.now().isoformat(),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved config to {config_path}")

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        output_dir=output_dir,
        checkpoint_interval=args.checkpoint_interval,
        confidence_threshold=args.confidence_threshold,
        lines_per_epoch=args.lines_per_epoch,
    )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    return history


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MuSSED (DETR-based seismic event detection) model"
    )

    # Data
    parser.add_argument(
        "--train-npz",
        type=str,
        default="data/prepared_data/NVCHVC/cv_5fold/fold_01/train.npz",
        help="Path to training data NPZ file",
    )
    parser.add_argument(
        "--val-npz",
        type=str,
        default="data/prepared_data/NVCHVC/cv_5fold/fold_01/val.npz",
        help="Path to validation data NPZ file (optional)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization weight decay",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of linear warmup steps",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for evaluation",
    )
    parser.add_argument(
        "--lines-per-epoch",
        type=int,
        default=5,
        help="Number of progress lines to print per epoch",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mussed",
        help="Output directory for checkpoints and logs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    history = main(args)
    print("Done!")
