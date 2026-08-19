"""
Training orchestration for MuSSED (DETR-based event detection).

Handles:
- Data loading from segmentation format
- Forward passes with loss computation
- Validation and metrics
- Checkpointing
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.MuSSED import MuSSED
from utils.detection_prediction_utils import normalize_prediction_intervals
from utils.musseg_utils import MuSSegWindowDataset, musseg_collate_fn, MuSSegBatch
from utils.event_targets import batch_segmentation_to_events
from utils.detr_event_loss import DETREventLoss
from utils.event_detection_metrics import EventDetectionMetrics


class DETRTrainer:
    """
    Trainer for MuSSED event detection model.
    """

    def __init__(
        self,
        model: MuSSED,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            model: MuSSED model instance
            device: Device to train on ("cpu", "cuda", etc.)
            dtype: Data type for tensors
        """
        self.model = model
        self.device = device
        self.dtype = dtype

        self.model.to(device).type(dtype)

        self.loss_fn = DETREventLoss(num_classes=6)
        self.metrics_fn = EventDetectionMetrics(num_classes=6)

        self.optimizer = None
        self.scheduler = None

    def setup_optimizer(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
    ):
        """
        Setup optimizer and learning rate scheduler.

        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            warmup_steps: Number of warmup steps (linear warmup)
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Linear warmup + cosine decay
        def lr_schedule(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            else:
                return 0.5 * (
                    1 + np.cos(np.pi * (step - warmup_steps) / (1000 - warmup_steps))
                )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_schedule)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = 0,
        lines_per_epoch: int = 5,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Epoch number (for logging)
            lines_per_epoch: Number of progress lines to print per epoch

        Returns:
            Dict with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_loss_class = 0.0
        total_loss_bbox = 0.0

        num_batches = len(train_loader)
        print_interval = max(1, num_batches // lines_per_epoch)

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            predictions = self.model(batch.x)

            # Convert segmentation labels to events
            targets = batch_segmentation_to_events(batch.y_onehot, normalize=True)

            # Compute loss
            loss_dict = self.loss_fn(predictions, targets)

            loss = loss_dict["loss_total"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_loss_class += loss_dict["loss_class"].item()
            total_loss_bbox += loss_dict["loss_bbox"].item()

            # Print progress
            if (batch_idx + 1) % print_interval == 0 or batch_idx == 0:
                print(
                    f"  [Epoch {epoch}] Batch {batch_idx + 1}/{num_batches} | "
                    f"Loss: {loss.item():.4f}"
                )

        # Average over batch
        return {
            "loss_total": total_loss / num_batches,
            "loss_class": total_loss_class / num_batches,
            "loss_bbox": total_loss_bbox / num_batches,
        }

    def evaluate(
        self,
        val_loader: DataLoader,
        epoch: int = 0,
        lines_per_epoch: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader
            epoch: Epoch number (for logging)
            lines_per_epoch: Number of progress lines to print per epoch

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0.0

        all_predictions_dict = {
            "class_logits": [],
            "center": [],
            "start": [],
            "end": [],
            "duration": [],
        }
        all_targets = []

        num_batches = len(val_loader)
        print_interval = max(1, num_batches // lines_per_epoch)

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = self._move_batch_to_device(batch)

                # Forward pass
                predictions = normalize_prediction_intervals(self.model(batch.x))

                # Convert segmentation labels to events
                targets = batch_segmentation_to_events(batch.y_onehot, normalize=True)
                all_targets.extend(targets)

                # Compute loss
                loss_dict = self.loss_fn(predictions, targets)
                total_loss += loss_dict["loss_total"].item()

                # Collect predictions for metrics
                for key in all_predictions_dict:
                    all_predictions_dict[key].append(
                        predictions[key].detach().cpu().numpy()
                    )

                # Print progress
                if (batch_idx + 1) % print_interval == 0 or batch_idx == 0:
                    print(f"  [Epoch {epoch}] Val Batch {batch_idx + 1}/{num_batches}")

        # Concatenate predictions
        predictions_for_metrics = {}
        for key in all_predictions_dict:
            predictions_for_metrics[key] = np.concatenate(
                all_predictions_dict[key], axis=0
            )

        # Compute metrics
        metrics_dict = self.metrics_fn.evaluate_batch(
            predictions_for_metrics,
            all_targets,
        )

        # Compute F1 at IoU=0.5 and mean IoU
        f1_score = self.metrics_fn.compute_f1(
            predictions_for_metrics,
            all_targets,
            iou_threshold=0.3,
        )
        metrics_dict["F1@0.3"] = f1_score

        # Add loss to metrics
        metrics_dict["loss_val"] = total_loss / num_batches

        return metrics_dict

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int = 10,
        output_dir: Path | str = "results/mussed",
        checkpoint_interval: int = 1,
        lines_per_epoch: int = 5,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            output_dir: Directory to save checkpoints and logs
            checkpoint_interval: Save checkpoint every N epochs
            lines_per_epoch: Number of progress lines to print per epoch

        Returns:
            Dict with training history
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "train_loss_class": [],
            "train_loss_bbox": [],
            "val_metrics": [],
        }

        best_map = 0.0
        best_epoch = 0

        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {output_dir}")

        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(
                train_loader, epoch=epoch + 1, lines_per_epoch=lines_per_epoch
            )
            history["train_loss"].append(train_metrics["loss_total"])
            history["train_loss_class"].append(train_metrics["loss_class"])
            history["train_loss_bbox"].append(train_metrics["loss_bbox"])

            print(
                f"[Epoch {epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_metrics['loss_total']:.4f} | "
                f"Class: {train_metrics['loss_class']:.4f} | "
                f"BBox: {train_metrics['loss_bbox']:.4f}"
            )

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(
                    val_loader,
                    epoch=epoch + 1,
                    lines_per_epoch=lines_per_epoch,
                )
                history["val_metrics"].append(val_metrics)

                current_map = val_metrics.get("mAP", 0.0)
                f1_score = val_metrics.get("F1@0.3", 0.0)
                mean_iou = np.mean(
                    [
                        val_metrics.get(f"mAP@{t:.1f}", 0.0)
                        for t in self.metrics_fn.iou_thresholds
                    ]
                )

                print(
                    f"  Val Loss: {val_metrics['loss_val']:.4f} | "
                    f"mAP: {current_map:.4f} | F1@0.3: {f1_score:.4f} | Mean IoU: {mean_iou:.4f}"
                )

                # Save best checkpoint
                if current_map > best_map:
                    best_map = current_map
                    best_epoch = epoch
                    best_checkpoint_path = output_dir / "best_model.pt"
                    self.save_checkpoint(best_checkpoint_path)
                    print(f"  → Saved best model to {best_checkpoint_path}")

            # Save periodic checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path)

        # Save final model
        final_path = output_dir / "final_model.pt"
        self.save_checkpoint(final_path)

        # Save history
        history_path = output_dir / "history.json"
        self._save_history(history, history_path)

        print(f"\nTraining complete!")
        print(f"Best mAP: {best_map:.4f} at epoch {best_epoch + 1}")
        print(f"Models saved to {output_dir}")

        return history

    def save_checkpoint(self, path: Path | str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": (
                    self.optimizer.state_dict() if self.optimizer is not None else None
                ),
                "scheduler_state": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "timestamp": datetime.now().isoformat(),
            },
            path,
        )

    def load_checkpoint(self, path: Path | str):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        if checkpoint["optimizer_state"] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

    def _move_batch_to_device(self, batch: MuSSegBatch) -> MuSSegBatch:
        """Move batch to device."""
        return MuSSegBatch(
            x=batch.x.to(self.device, dtype=self.dtype),
            y_onehot=batch.y_onehot.to(self.device, dtype=self.dtype),
            y_label=batch.y_label.to(self.device),
            station_mask=batch.station_mask.to(self.device),
        )

    @staticmethod
    def _save_history(history: Dict[str, list], path: Path | str):
        """Save training history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, values in history.items():
            if key == "val_metrics":
                history_serializable[key] = values
            else:
                history_serializable[key] = [float(v) for v in values]

        with open(path, "w") as f:
            json.dump(history_serializable, f, indent=2)
