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

from utils.edge_features import batch_xcorr_features, compute_rsam
from utils.fold_io_utils import append_row_csv
from utils.model_registry import MODEL_SPECS
from utils.script_common import parse_csv_selection, resolve_project_path
from utils.station_info import build_volcano_geometry_bank
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
    extract_descriptor_tensor,
    f1_score_from_confusion_matrix,
)

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-confusion-matrices", action="store_true")
    return parser.parse_args()


class GraphForwardWrapper(torch.nn.Module):
    """Inject xcorr/rsam kwargs on demand while preserving model signature expectations."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        needs_xcorr: bool,
        needs_rsam: bool,
        sampling_rate: float = 100.0,
        max_lag_seconds: float = 5.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.needs_xcorr = bool(needs_xcorr)
        self.needs_rsam = bool(needs_rsam)
        self.sampling_rate = float(sampling_rate)
        self.max_lag_seconds = float(max_lag_seconds)

        self.use_envelope = getattr(base_model, "use_envelope", False)
        self.num_descriptors = int(getattr(base_model, "num_descriptors", 0))

    def forward(self, x: torch.Tensor, **kwargs):
        if self.needs_xcorr or self.needs_rsam:
            waveforms_np = x.detach().cpu().numpy()
            if self.needs_xcorr:
                edge_attr_dynamic_np = batch_xcorr_features(
                    waveforms_np,
                    sampling_rate=self.sampling_rate,
                    max_lag_seconds=self.max_lag_seconds,
                )
                kwargs["edge_attr_dynamic"] = torch.from_numpy(edge_attr_dynamic_np).to(
                    device=x.device,
                    dtype=x.dtype,
                )
            if self.needs_rsam:
                rsam_np = compute_rsam(waveforms_np)
                kwargs["rsam"] = torch.from_numpy(rsam_np).to(
                    device=x.device,
                    dtype=x.dtype,
                )

        return self.base_model(x, **kwargs)


def _extract_graph_batch(
    batch,
    device: torch.device,
    model: torch.nn.Module,
):
    xb = batch[0].to(device)
    y_onehot = batch[1].to(device)

    descriptor_payload = None
    volcano_idx = None
    if len(batch) > 3 and isinstance(batch[3], dict):
        descriptor_payload = batch[3]
    elif len(batch) > 3 and torch.is_tensor(batch[3]):
        volcano_idx = batch[3].to(device).long()

    if len(batch) > 4 and torch.is_tensor(batch[4]):
        volcano_idx = batch[4].to(device).long()

    forward_kwargs: dict[str, Any] = {}

    if volcano_idx is not None:
        forward_kwargs["volcano_idx"] = volcano_idx

    if getattr(model, "use_envelope", False):
        if descriptor_payload is None or "envelope" not in descriptor_payload:
            raise ValueError(
                "Model requires envelope but batch has no descriptor payload with 'envelope'."
            )
        env = descriptor_payload["envelope"]
        if not torch.is_tensor(env):
            env = torch.as_tensor(env)
        if env.ndim == 4 and env.shape[1] == 1:
            env = env[:, 0, :, :]
        elif env.ndim == 4 and env.shape[2] == 1:
            env = env[:, :, 0, :]
        env = env.to(device=xb.device, dtype=xb.dtype)
        forward_kwargs["envelope"] = env

    model_num_desc = int(getattr(model, "num_descriptors", 0))
    if model_num_desc > 0:
        descriptor_names = getattr(model, "descriptor_names", None)
        if descriptor_names is None:
            raise ValueError(
                "Model has num_descriptors > 0 but does not expose descriptor_names."
            )
        descriptors = extract_descriptor_tensor(
            descriptor_payload,
            descriptor_names,
            xb,
        )
        forward_kwargs["descriptors"] = descriptors

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

    iou_per_class = []
    for c in range(len(CLASS_NAMES)):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        denom = tp + fp + fn
        iou_per_class.append(float(tp / denom) if denom > 0 else 0.0)

    return (
        [float(x) for x in f1_scores],
        float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0,
        [float(x) for x in iou_per_class],
        float(np.mean(iou_per_class)) if len(iou_per_class) > 0 else 0.0,
        mean_loss,
        cm,
    )


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
    selected_models = parse_csv_selection(args.models, sorted(specs.keys()), "models")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    volcano_geom_bank, volcano_name_to_idx, volcano_order = build_volcano_geometry_bank(
        n_stations=8
    )

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
        "volcano_order": list(volcano_order),
        "volcano_name_to_idx": {k: int(v) for k, v in volcano_name_to_idx.items()},
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

            run_start = time.time()
            fold_out = output_dir / model_key / fold_dir.name
            ckpt_dir = fold_out / "checkpoints"
            reports_dir = fold_out / "reports"
            cm_dir = fold_out / "confusion_matrices"
            for p in [ckpt_dir, reports_dir, cm_dir]:
                p.mkdir(parents=True, exist_ok=True)

            if family == "unet":
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

                train_sampler = BalancedBatchSampler(
                    train_ds.label_ids, batch_size=batch_size
                )
                train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                wrapper = model
            else:
                model = spec["model_cls"](
                    n_stations=8,
                    in_channels=1,
                    out_channels=6,
                    volcano_geom_nodes=volcano_geom_bank,
                    **{
                        k: v
                        for k, v in model_kwargs.items()
                        if k not in {"n_stations", "in_channels", "out_channels"}
                    },
                ).to(device)

                descriptor_names = getattr(model, "descriptor_names", None)
                model_num_desc = int(getattr(model, "num_descriptors", 0))

                train_ds = CrossVolcanoLOODataset(
                    train_npz,
                    descriptor_names=descriptor_names if model_num_desc > 0 else None,
                    return_volcano_idx=True,
                    volcano_name_to_idx=volcano_name_to_idx,
                )
                val_ds = CrossVolcanoLOODataset(
                    val_npz,
                    descriptor_names=descriptor_names if model_num_desc > 0 else None,
                    return_volcano_idx=True,
                    volcano_name_to_idx=volcano_name_to_idx,
                )
                test_ds = CrossVolcanoLOODataset(
                    test_npz,
                    descriptor_names=descriptor_names if model_num_desc > 0 else None,
                    return_volcano_idx=True,
                    volcano_name_to_idx=volcano_name_to_idx,
                )

                train_sampler = BalancedBatchSampler(
                    train_ds.label_ids, batch_size=batch_size
                )
                train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                wrapper = GraphForwardWrapper(
                    model,
                    needs_xcorr=model_kwargs.get("edge_feature_mode")
                    == "delta_pos_xcorr",
                    needs_rsam=bool(model_kwargs.get("use_rsam_node_feat", False)),
                ).to(device)

            optimizer = optim.Adam(
                [p for p in wrapper.parameters() if p.requires_grad],
                lr=float(args.lr),
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(args.epochs / 4)),
                eta_min=float(args.lr_final),
            )

            best_val_mean_f1 = float("-inf")
            best_epoch = -1
            no_improve = 0

            metrics_rows = []

            for epoch in range(int(args.epochs)):
                wrapper.train()
                train_loss = 0.0

                if family == "unet":
                    for xb, y_onehot, _ in train_loader:
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
                else:
                    for batch in train_loader:
                        xb, y_onehot, forward_kwargs = _extract_graph_batch(
                            batch, device, model
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

                scheduler.step()

                if family == "unet":
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
                    descriptor_names = getattr(model, "descriptor_names", None)
                    (
                        val_f1_per_class,
                        val_mean_f1,
                        val_iou_per_class,
                        val_mean_iou,
                        _val_iou_all,
                        _val_mean_iou_all,
                        val_loss,
                        val_cm,
                    ) = compute_event_f1_iou_graphsage(
                        wrapper,
                        val_loader,
                        device,
                        descriptor_names=(
                            list(descriptor_names)
                            if descriptor_names is not None
                            else None
                        ),
                        return_cm=True,
                        return_val_loss=True,
                        return_event_plot_payloads=False,
                        save_event_plots=False,
                        max_event_plots=0,
                        epoch=None,
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
                else:
                    no_improve += 1

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

            if family == "unet":
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
                descriptor_names = getattr(model, "descriptor_names", None)
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
                    descriptor_names=(
                        list(descriptor_names) if descriptor_names is not None else None
                    ),
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
