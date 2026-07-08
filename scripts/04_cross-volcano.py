"""
Leave-one-out cross-volcano training and evaluation.

Run:
    python scripts/04_cross-volcano.py

Folds:
- train on NVCHVC+CAU+LDM, test on VCA
- train on NVCHVC+VCA+LDM, test on CAU
- train on NVCHVC+CAU+VCA, test on LDM

Data input is expected under:
    data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/{train,val,test}.npz
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.fold_io_utils import append_row_csv
from utils.model_registry import MODEL_SPECS

# MODEL_SPECS = dict(reversed(list(MODEL_SPECS.items())))

from utils.script_common import parse_csv_selection, resolve_project_path
from utils.train_utils import (
    BalancedBatchSampler,
    CrossVolcanoLOODataset,
    UNetPatchDataset,
    cleanup_gpu_cache,
    cm_eval,
    combined_dice_ce_loss,
    combined_dice_ce_loss_2d,
    compute_event_f1_iou_graphsage,
    compute_summary,
    f1_score_from_confusion_matrix,
    predicted_from_output,
    save_confusion_matrix_image,
    save_event_plot_payloads,
)
from utils import data_utils

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / "cross_volcano_loo"
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "results" / "experiments" / "complete_experiment"
)
DEFAULT_OUTPUT_NAME = "cross_volcano_leave_one_out"
FOLD_GLOB = "fold_*_holdout_*"
CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
ALL_CLASS_NAMES = ["BG", "VT", "LP", "TR", "AV", "IC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-out cross-volcano training and evaluation for all model families."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root folder with fold_XX_holdout_<VOLCANO> train/val/test manifests.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Base experiment root used for output folder placement.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <experiment-root>/cross_volcano_leave_one_out",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys. Default: all enabled model registry keys.",
    )
    parser.add_argument(
        "--family",
        choices=["all", "unet", "graphsage", "mpnn"],
        default="all",
        help="Optional model family filter.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=5e-6)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-confusion-matrices", action="store_true")
    parser.add_argument(
        "--max-event-plots-per-class",
        type=int,
        default=20,
        help="Max misclassified examples saved per true class on best-val-F1 epochs.",
    )
    return parser.parse_args()


class GraphForwardWrapper(torch.nn.Module):
    """Compatibility wrapper for graph-family models."""

    def __init__(
        self,
        base_model: torch.nn.Module,
    ):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor, **kwargs):
        return self.base_model(x, **kwargs)


def _extract_graph_batch(
    batch,
    device: torch.device,
):
    xb = batch[0].to(device)
    y_onehot = batch[1].to(device)
    forward_kwargs: dict[str, Any] = {}
    return xb, y_onehot, forward_kwargs


def _evaluate_unet(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dice_weight: float,
    ce_weight: float,
) -> tuple[list[float], float, list[float], float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.inference_mode():
        for xb, y_onehot, _ in loader:
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)
            out = model(xb)
            loss, _, _ = combined_dice_ce_loss_2d(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=dice_weight,
                ce_weight=ce_weight,
            )
            total_loss += float(loss.item())
            n_batches += 1
            del xb, y_onehot, out, loss

    mean_loss = float(total_loss / n_batches) if n_batches > 0 else 0.0
    cm = cm_eval(
        model=model,
        dataloader=loader,
        device=device,
        len_window=8192,
        im_size=256,
        clases_list={1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"},
        t_bg=0,
        t_cl=0,
    )
    f1_scores, _, _ = f1_score_from_confusion_matrix(cm)
    support = np.sum(cm, axis=1)
    active_mask = support > 0
    mean_f1 = (
        float(np.mean([f1_scores[i] for i, active in enumerate(active_mask) if active]))
        if np.any(active_mask)
        else 0.0
    )
    iou_per_class = []
    for c in range(len(CLASS_NAMES)):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        denom = tp + fp + fn
        iou_per_class.append(float(tp / denom) if denom > 0 else 0.0)
    mean_iou = (
        float(
            np.mean(
                [iou_per_class[i] for i, active in enumerate(active_mask) if active]
            )
        )
        if np.any(active_mask)
        else 0.0
    )
    return (
        [float(x) for x in f1_scores],
        mean_f1,
        [float(x) for x in iou_per_class],
        mean_iou,
        mean_loss,
        cm,
    )


def _filter_payloads_per_class(
    payloads: list[dict],
    max_per_class: int,
) -> list[dict]:
    """Keep at most max_per_class misclassified payloads per true class."""
    counts: dict[str, int] = {}
    filtered = []
    for p in payloads:
        key = p["true_name"]
        if counts.get(key, 0) < max_per_class:
            filtered.append(p)
            counts[key] = counts.get(key, 0) + 1
    return filtered


def _collect_unet_misclassified_plots(
    model: torch.nn.Module,
    npz_path: Path,
    device: torch.device,
    max_per_class: int,
    class_names: list[str],
) -> list[dict]:
    """Collect misclassified UNet examples (one sample at a time) for event plot saving."""
    class_map = {1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"}
    ds = UNetPatchDataset(npz_path, return_debug=True)
    payloads: list[dict] = []
    counts: dict[int, int] = {}
    event_classes = list(class_map.keys())

    model.eval()
    with torch.inference_mode():
        for i in range(len(ds)):
            if all(counts.get(int(c), 0) >= max_per_class for c in event_classes):
                break
            x_unet, _y_onehot_2d, _y_idx, x_raw, y_raw, _x_used, _y_used, _aug_meta = ds[i]
            true_class = int(np.argmax(y_raw[1:].sum(axis=1))) + 1
            if counts.get(true_class, 0) >= max_per_class:
                continue
            xb = x_unet.unsqueeze(0).to(device)
            out = model(xb)
            out_probs = torch.softmax(out, dim=1).detach().cpu()
            out_unstacked = data_utils.activation_unstacking(
                out_probs, len_window=8192, N=256, n_classes=6
            )
            out_np = out_unstacked[0].numpy()
            del xb, out, out_probs, out_unstacked

            pred_class, _, _, _ = predicted_from_output(out_np, class_map)
            if int(pred_class) == true_class:
                continue

            max_indices = np.argmax(out_np, axis=0)
            processed_out = np.eye(out_np.shape[0], dtype=np.float32)[max_indices].T
            payloads.append(
                {
                    "sample_global_idx": i,
                    "true_evt": true_class,
                    "pred_evt": int(pred_class),
                    "true_name": class_names[true_class],
                    "pred_name": class_names[int(pred_class)],
                    "x_raw": x_raw,
                    "out_np": out_np,
                    "processed_out": processed_out,
                    "y_onehot": y_raw,
                }
            )
            counts[true_class] = counts.get(true_class, 0) + 1

    return payloads


def main() -> None:
    args = parse_args()

    data_root = resolve_project_path(args.data_root, PROJECT_ROOT)
    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    output_dir = (
        resolve_project_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir is not None
        else (experiment_root / DEFAULT_OUTPUT_NAME)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted([p for p in data_root.glob(FOLD_GLOB) if p.is_dir()])
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No fold directories found under: {data_root}")
    for fold_dir in fold_dirs:
        for name in ["train.npz", "val.npz", "test.npz"]:
            p = fold_dir / name
            if not p.exists():
                raise FileNotFoundError(f"Missing manifest file: {p}")

    specs = {
        k: v
        for k, v in MODEL_SPECS.items()
        if v.get("enabled", True)
        and (args.family == "all" or str(v["family"]) == args.family)
    }
    selected_models = parse_csv_selection(args.models, list(specs.keys()), "models")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "lr_final": float(args.lr_final),
        "early_stop_patience": int(args.early_stop_patience),
        "dice_weight": float(args.dice_weight),
        "ce_weight": float(args.ce_weight),
        "batch_size_override": (
            None if args.batch_size is None else int(args.batch_size)
        ),
        "selected_models": selected_models,
        "fold_dirs": [str(p) for p in fold_dirs],
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 90)
    print("Cross-volcano leave-one-out training")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Models ({len(selected_models)}): {selected_models}")
    print("=" * 90)

    fieldnames = [
        "model_key",
        "family",
        "fold",
        "held_out_volcano",
        "n_train",
        "n_val",
        "n_test",
        "best_epoch",
        "best_val_mean_f1",
        "test_loss",
        "test_mean_f1",
        "test_mean_iou",
        "elapsed_seconds",
    ]
    for cls in CLASS_NAMES:
        fieldnames.append(f"test_f1_{cls}")
        fieldnames.append(f"test_iou_{cls}")

    rows: list[dict[str, Any]] = []

    for model_key in selected_models:
        spec = MODEL_SPECS[model_key]
        family = str(spec["family"])
        trainer_kind = str(spec["trainer_kind"])
        model_kwargs = dict(spec["model_kwargs"])
        batch_size = (
            int(args.batch_size)
            if args.batch_size is not None
            else int(spec["batch_size"])
        )

        print(f"\n[MODEL] {model_key} family={family} batch_size={batch_size}")

        for fold_idx, fold_dir in enumerate(fold_dirs, start=1):
            train_npz = fold_dir / "train.npz"
            val_npz = fold_dir / "val.npz"
            test_npz = fold_dir / "test.npz"
            with np.load(test_npz) as data:
                if "held_out_volcano" in data:
                    held_out = str(data["held_out_volcano"][0])
                else:
                    name = test_npz.parent.name
                    held_out = (
                        name.split("holdout_")[-1] if "holdout_" in name else "UNKNOWN"
                    )

            print(f"  [FOLD {fold_idx}] Testing on: {held_out}")

            run_start = time.time()
            fold_out = output_dir / model_key / fold_dir.name
            ckpt_dir = fold_out / "checkpoints"
            reports_dir = fold_out / "reports"
            cm_dir = fold_out / "confusion_matrices"
            val_event_plots_dir = fold_out / "val_event_plots"
            for p in [ckpt_dir, reports_dir, cm_dir, val_event_plots_dir]:
                p.mkdir(parents=True, exist_ok=True)

            if trainer_kind == "2d":
                model = spec["model_cls"](
                    in_channels=1,
                    out_channels=6,
                    **{
                        k: v
                        for k, v in model_kwargs.items()
                        if k not in {"in_channels", "out_channels"}
                    },
                ).to(device)

                train_ds = UNetPatchDataset(train_npz)
                val_ds = UNetPatchDataset(val_npz)
                test_ds = UNetPatchDataset(test_npz)

                print(
                    f"    [DATA] Train: {len(train_ds)} patches | Val: {len(val_ds)} patches | Test: {len(test_ds)} patches"
                )

                train_sampler = BalancedBatchSampler(
                    train_ds.label_ids, batch_size=batch_size
                )
                train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                wrapper = model
            else:
                if family == "phasenet":
                    model = spec["model_cls"](**model_kwargs).to(device)
                else:
                    model = spec["model_cls"](
                        n_stations=8,
                        in_channels=1,
                        out_channels=6,
                        **{
                            k: v
                            for k, v in model_kwargs.items()
                            if k not in {"n_stations", "in_channels", "out_channels"}
                        },
                    ).to(device)

                train_ds = CrossVolcanoLOODataset(
                    train_npz,
                    descriptor_names=None,
                    return_volcano_idx=False,
                )
                val_ds = CrossVolcanoLOODataset(
                    val_npz,
                    descriptor_names=None,
                    return_volcano_idx=False,
                )
                test_ds = CrossVolcanoLOODataset(
                    test_npz,
                    descriptor_names=None,
                    return_volcano_idx=False,
                )

                print(
                    f"    [DATA] Train: {len(train_ds)} events | Val: {len(val_ds)} events | Test: {len(test_ds)} events"
                )

                train_sampler = BalancedBatchSampler(
                    train_ds.label_ids, batch_size=batch_size
                )
                train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                wrapper = GraphForwardWrapper(
                    model,
                ).to(device)

            optimizer = optim.Adam(
                [p for p in wrapper.parameters() if p.requires_grad],
                lr=float(args.lr),
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(args.epochs / 5)),
                eta_min=float(args.lr_final),
            )

            best_val_mean_f1 = float("-inf")
            best_epoch = -1
            no_improve = 0

            metrics_rows = []

            print(
                f"    [TRAINING] Starting {int(args.epochs)} epochs | lr={float(args.lr):.2e} -> {float(args.lr_final):.2e}"
            )

            for epoch in range(int(args.epochs)):
                wrapper.train()
                train_loss = 0.0

                if trainer_kind == "2d":
                    for batch_idx, (xb, y_onehot, _) in enumerate(train_loader, 1):
                        xb = xb.to(device)
                        y_onehot = y_onehot.to(device)

                        optimizer.zero_grad(set_to_none=True)
                        out = wrapper(xb)
                        loss, _, _ = combined_dice_ce_loss_2d(
                            out,
                            y_onehot,
                            class_weights=None,
                            dice_weight=float(args.dice_weight),
                            ce_weight=float(args.ce_weight),
                        )
                        loss.backward()
                        optimizer.step()
                        train_loss += float(loss.item())
                        del xb, y_onehot, out, loss

                        if (
                            batch_idx % max(1, len(train_loader) // 3) == 0
                            or batch_idx == 1
                        ):
                            print(
                                f"      epoch={epoch+1:3d} batch={batch_idx:3d}/{len(train_loader):3d}"
                            )
                else:
                    for batch_idx, batch in enumerate(train_loader, 1):
                        xb, y_onehot, forward_kwargs = _extract_graph_batch(
                            batch, device
                        )

                        optimizer.zero_grad(set_to_none=True)
                        out = wrapper(xb, **forward_kwargs)
                        loss, _, _ = combined_dice_ce_loss(
                            out,
                            y_onehot,
                            class_weights=None,
                            dice_weight=float(args.dice_weight),
                            ce_weight=float(args.ce_weight),
                        )
                        loss.backward()
                        optimizer.step()
                        train_loss += float(loss.item())
                        del xb, y_onehot, out, loss

                        if (
                            batch_idx % max(1, len(train_loader) // 3) == 0
                            or batch_idx == 1
                        ):
                            print(
                                f"      epoch={epoch+1:3d} batch={batch_idx:3d}/{len(train_loader):3d}"
                            )

                scheduler.step()

                if trainer_kind == "2d":
                    (
                        val_f1_per_class,
                        val_mean_f1,
                        val_iou_per_class,
                        val_mean_iou,
                        val_loss,
                        val_cm,
                    ) = _evaluate_unet(
                        wrapper,
                        val_loader,
                        device,
                        dice_weight=float(args.dice_weight),
                        ce_weight=float(args.ce_weight),
                    )
                else:
                    (
                        val_f1_per_class,
                        val_mean_f1,
                        val_iou_per_class,
                        val_mean_iou,
                        _val_iou_all,
                        _val_mean_iou_all,
                        val_loss,
                        val_event_payloads,
                        val_cm,
                    ) = compute_event_f1_iou_graphsage(
                        wrapper,
                        val_loader,
                        device,
                        descriptor_names=None,
                        return_cm=True,
                        return_val_loss=True,
                        return_event_plot_payloads=True,
                        save_event_plots=False,
                        max_event_plots=int(args.max_event_plots_per_class) * len(CLASS_NAMES),
                        epoch=epoch,
                    )

                improved = float(val_mean_f1) > float(best_val_mean_f1)
                if improved:
                    best_val_mean_f1 = float(val_mean_f1)
                    best_epoch = int(epoch)
                    no_improve = 0
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": float(val_loss),
                            "val_mean_f1": float(val_mean_f1),
                        },
                        ckpt_dir / "best_f1.pt",
                    )
                    save_confusion_matrix_image(
                        cm=val_cm,
                        labels=CLASS_NAMES,
                        out_path=cm_dir / f"val_cm_best_f1_epoch_{epoch:03d}.png",
                        title=f"Val CM - {model_key} fold {fold_idx} epoch {epoch}",
                    )
                    if trainer_kind == "2d":
                        _best_plots = _collect_unet_misclassified_plots(
                            model=model,
                            npz_path=val_npz,
                            device=device,
                            max_per_class=int(args.max_event_plots_per_class),
                            class_names=ALL_CLASS_NAMES,
                        )
                    else:
                        _best_plots = _filter_payloads_per_class(
                            val_event_payloads,
                            max_per_class=int(args.max_event_plots_per_class),
                        )
                    save_event_plot_payloads(_best_plots, val_event_plots_dir, epoch=epoch)
                    print(f"      \u2713 epoch={epoch+1:3d} val_f1={val_mean_f1:.4f} [BEST]")
                else:
                    no_improve += 1
                    if (epoch + 1) % max(
                        1, int(args.epochs) // 5
                    ) == 0 or no_improve >= int(args.early_stop_patience) - 3:
                        print(
                            f"      · epoch={epoch+1:3d} val_f1={val_mean_f1:.4f} (no improve: {no_improve}/{args.early_stop_patience})"
                        )

                metrics_rows.append(
                    {
                        "epoch": int(epoch),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "val_mean_f1": float(val_mean_f1),
                        "val_mean_iou": float(val_mean_iou),
                    }
                )

                if no_improve >= int(args.early_stop_patience):
                    break

                del val_cm
                if trainer_kind != "2d":
                    del val_event_payloads
                cleanup_gpu_cache()

            pd.DataFrame(metrics_rows).to_csv(
                reports_dir / "training_metrics.csv",
                index=False,
                encoding="utf-8-sig",
                sep=";",
                decimal=",",
            )

            best_ckpt_path = ckpt_dir / "best_f1.pt"
            if not best_ckpt_path.exists():
                raise RuntimeError(f"Missing best checkpoint: {best_ckpt_path}")

            best_ckpt = torch.load(
                best_ckpt_path, map_location=device, weights_only=False
            )
            model.load_state_dict(best_ckpt["model_state_dict"])
            wrapper.eval()

            if trainer_kind == "2d":
                (
                    test_f1_per_class,
                    test_mean_f1,
                    test_iou_per_class,
                    test_mean_iou,
                    test_loss,
                    test_cm,
                ) = _evaluate_unet(
                    wrapper,
                    test_loader,
                    device,
                    dice_weight=float(args.dice_weight),
                    ce_weight=float(args.ce_weight),
                )
            else:
                (
                    test_f1_per_class,
                    test_mean_f1,
                    test_iou_per_class,
                    test_mean_iou,
                    _test_iou_all,
                    _test_mean_iou_all,
                    test_loss,
                    test_cm,
                ) = compute_event_f1_iou_graphsage(
                    wrapper,
                    test_loader,
                    device,
                    descriptor_names=None,
                    return_cm=True,
                    return_val_loss=True,
                    return_event_plot_payloads=False,
                    save_event_plots=False,
                    max_event_plots=0,
                    epoch=None,
                )

            elapsed = float(time.time() - run_start)

            row = {
                "model_key": model_key,
                "family": family,
                "fold": int(fold_idx),
                "held_out_volcano": held_out,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "best_epoch": int(best_epoch),
                "best_val_mean_f1": float(best_val_mean_f1),
                "test_loss": float(test_loss),
                "test_mean_f1": float(test_mean_f1),
                "test_mean_iou": float(test_mean_iou),
                "elapsed_seconds": elapsed,
            }
            for idx, cls in enumerate(CLASS_NAMES):
                row[f"test_f1_{cls}"] = float(test_f1_per_class[idx])
                row[f"test_iou_{cls}"] = float(test_iou_per_class[idx])

            rows.append(row)
            append_row_csv(
                output_dir / "cross_volcano_fold_metrics.csv",
                row=row,
                fieldnames=fieldnames,
            )

            with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

            print(
                f"  fold={fold_idx:02d} held_out={held_out} "
                f"best_epoch={best_epoch} val_f1={best_val_mean_f1:.4f} "
                f"test_f1={test_mean_f1:.4f} test_iou={test_mean_iou:.4f}"
            )
            print(f"    -> Saved to: {fold_out}")

            del train_loader, val_loader, test_loader
            del train_ds, val_ds, test_ds
            del wrapper, model, best_ckpt, test_cm, optimizer, scheduler
            cleanup_gpu_cache()
            gc.collect()

    all_fold_summaries = sorted(
        output_dir.glob("*/fold_*_holdout_*/reports/fold_summary.json")
    )
    if len(all_fold_summaries) == 0:
        raise RuntimeError(
            "No fold summaries found under output directory. " "Nothing to aggregate."
        )

    all_rows: list[dict[str, Any]] = []
    for summary_path in all_fold_summaries:
        with summary_path.open("r", encoding="utf-8") as f:
            all_rows.append(json.load(f))

    df = pd.DataFrame(all_rows)
    df.to_csv(
        output_dir / "cross_volcano_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    grouped = df.groupby(["model_key", "family", "held_out_volcano"], sort=True)
    summary_rows = []
    for (model_key, family, held_out), grp in grouped:
        row = {
            "model_key": str(model_key),
            "family": str(family),
            "held_out_volcano": str(held_out),
            "n_folds": int(len(grp)),
        }
        for col in ["test_mean_f1", "test_mean_iou", "test_loss", "best_val_mean_f1"]:
            stats = compute_summary(grp[col].astype(float).tolist())
            row[f"{col}_mean"] = float(stats["mean"])
            row[f"{col}_std"] = float(stats["std"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["held_out_volcano", "test_mean_f1_mean"],
        ascending=[True, False],
    )
    summary_df.to_csv(
        output_dir / "cross_volcano_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    print(
        f"Aggregated {len(all_rows)} fold summaries from {len(all_fold_summaries)} report files."
    )

    print("=" * 90)
    print("Cross-volcano leave-one-out run complete")
    print(f"Fold metrics: {output_dir / 'cross_volcano_fold_metrics.csv'}")
    print(f"Summary: {output_dir / 'cross_volcano_summary.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
