"""Progressive finetuning workflow for the target volcano splits."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.active_eval_utils import load_checkpoint_into_model
from utils.fold_io_utils import append_row_csv
from utils.finetune_utils import apply_finetune_protocol
from utils.model_registry import MODEL_SPECS, build_model_from_spec, get_model_spec
from utils.script_common import parse_csv_selection, resolve_project_path
from utils.train_utils import (
    BalancedBatchSampler,
    MultiStation1DDataset,
    UNetPatchDataset,
    cleanup_gpu_cache,
    combined_dice_ce_loss,
    combined_dice_ce_loss_2d,
    compute_event_f1_iou_multistation,
    compute_summary,
    evaluate_unet_model,
    save_confusion_matrix_image,
)

DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / "progressive_finetuning"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "complete_experiment"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "progressive_finetuning"

DEFAULT_PROTOCOL = "protocol_a_all_weights"
DEFAULT_SUBSET_KEYS = ["01pct", "05pct", "10pct", "20pct"]
DEFAULT_FOLDS = [1, 2, 3, 4, 5]
DEFAULT_TARGETS = ["CAU", "VCA", "LDM"]
SUBSET_FIXED_LR = {
    "01pct": 1e-5,
    "1pct": 1e-5,
    "05pct": 5e-5,
    "5pct": 5e-5,
    "10pct": 1e-4,
    "20pct": 1e-4,
}

CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
EVENT_CLASS_MAP = {
    1.0: "VT",
    2.0: "LP",
    3.0: "TR",
    4.0: "AV",
    5.0: "IC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Progressive finetuning on the target-volcano repeated 80/20 splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root folder containing progressive_finetuning/<volcano>/fold_XX manifests.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Base experiment root that contains the source ablation checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for finetuned checkpoints and reports.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys. Default: all active registry entries.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated target volcano names. Default: CAU,VCA,LDM.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated repeat ids. Default: 1,2,3,4,5.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="Comma-separated subset keys. Default: 01pct,05pct,10pct,20pct.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=DEFAULT_PROTOCOL,
        help="Finetuning protocol key from utils.finetune_utils.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--len-window", type=int, default=8192)
    parser.add_argument("--im-size", type=int, default=256)
    parser.add_argument(
        "--log-batches",
        type=int,
        default=4,
        help="Number of train-batch progress logs per epoch (>=0).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Re-run runs that already exist in metrics/artifacts.",
    )
    parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help="Skip model keys when the source checkpoint cannot be found.",
    )
    return parser.parse_args()


def discover_targets(data_root: Path) -> list[str]:
    targets: list[str] = []
    if not data_root.exists():
        raise FileNotFoundError(f"Progressive finetuning root not found: {data_root}")

    for volcano_dir in sorted(data_root.iterdir()):
        if not volcano_dir.is_dir():
            continue
        fold_dirs = [p for p in volcano_dir.glob("fold_*") if p.is_dir()]
        if any((fold_dir / "test_80.npz").exists() for fold_dir in fold_dirs):
            targets.append(volcano_dir.name)

    if len(targets) == 0:
        raise FileNotFoundError(
            f"No target volcano folders with test_80.npz found under: {data_root}"
        )

    return targets


def _discover_fold_dirs(target_root: Path) -> list[Path]:
    fold_dirs = []
    for fold_dir in sorted(target_root.glob("fold_*")):
        if fold_dir.is_dir() and (fold_dir / "test_80.npz").exists():
            fold_dirs.append(fold_dir)
    return fold_dirs


def _discover_subset_keys(fold_dir: Path) -> list[str]:
    subsets_root = fold_dir / "subsets"
    if not subsets_root.exists():
        return []
    return [p.name for p in sorted(subsets_root.iterdir()) if p.is_dir()]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fixed_lr_for_subset(subset_key: str) -> float:
    key = str(subset_key).strip().lower()
    if key not in SUBSET_FIXED_LR:
        raise ValueError(
            f"No fixed LR mapping found for subset '{subset_key}'. "
            f"Supported keys: {sorted(SUBSET_FIXED_LR.keys())}"
        )
    return float(SUBSET_FIXED_LR[key])


def _run_key(
    model_key: str,
    target_volcano: str,
    repeat_idx: int,
    subset_key: str,
) -> tuple[str, str, int, str]:
    return (str(model_key), str(target_volcano), int(repeat_idx), str(subset_key))


def _order_targets_cau_first(targets: list[str]) -> list[str]:
    ordered = [str(t) for t in targets]
    if "CAU" in ordered:
        ordered = ["CAU"] + [t for t in ordered if t != "CAU"]
    return ordered


def _load_completed_run_keys(csv_path: Path) -> set[tuple[str, str, int, str]]:
    if not csv_path.exists():
        return set()

    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8-sig")
    required = {"model_key", "target_volcano", "repeat_id", "subset_key"}
    if not required.issubset(df.columns):
        return set()

    keys: set[tuple[str, str, int, str]] = set()
    for _, row in df.iterrows():
        try:
            keys.add(
                _run_key(
                    model_key=str(row["model_key"]),
                    target_volcano=str(row["target_volcano"]),
                    repeat_idx=int(row["repeat_id"]),
                    subset_key=str(row["subset_key"]),
                )
            )
        except (TypeError, ValueError):
            continue
    return keys


def _load_existing_metrics_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8-sig")
    return df.to_dict(orient="records")


def _load_existing_run_summary_if_complete(run_root: Path) -> dict | None:
    summary_path = run_root / "reports" / "fold_summary.json"
    best_ckpt_path = run_root / "checkpoints" / "best_f1.pt"
    if summary_path.exists() and best_ckpt_path.exists():
        return _load_json(summary_path)
    return None


def _find_best_source_checkpoint(
    model_key: str, experiment_root: Path
) -> tuple[Path, dict]:
    model_root = experiment_root / "ablations" / model_key
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    candidates: list[tuple[float, str, Path, dict]] = []
    for fold_dir in sorted(model_root.glob("fold_*")):
        ckpt_path = fold_dir / "checkpoints" / "best_f1.pt"
        if not ckpt_path.exists():
            continue

        score = float("-inf")
        summary_path = fold_dir / "reports" / "fold_summary.json"
        if summary_path.exists():
            summary = _load_json(summary_path)
            score = float(summary.get("best_val_mean_f1", score))
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            score = float(ckpt.get("val_mean_f1", ckpt.get("best_val_mean_f1", score)))

        candidates.append((score, fold_dir.name, ckpt_path, {"score": score}))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No source checkpoints found under {model_root}. Expected fold_*/checkpoints/best_f1.pt"
        )

    best_score, best_fold_name, best_ckpt_path, meta = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    meta.update({"source_fold": best_fold_name, "score": float(best_score)})
    return best_ckpt_path, meta


def _load_base_model(
    model_key: str, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    spec = get_model_spec(model_key)
    model = build_model_from_spec(model_key, n_classes=6).to(device)
    load_checkpoint_into_model(
        model,
        checkpoint_path,
        device,
        trainer_kind=str(spec["trainer_kind"]),
    )
    return model


def _prepare_dataloaders(
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    batch_size: int,
    trainer_kind: str,
) -> tuple[object, object, object, DataLoader, DataLoader, DataLoader]:
    if str(trainer_kind) == "2d":
        train_ds = UNetPatchDataset(train_npz)
        val_ds = UNetPatchDataset(val_npz)
        test_ds = UNetPatchDataset(test_npz)
    else:
        train_ds = MultiStation1DDataset(train_npz)
        val_ds = MultiStation1DDataset(val_npz)
        test_ds = MultiStation1DDataset(test_npz)

    effective_batch_size = max(1, min(int(batch_size), len(train_ds)))
    train_sampler = BalancedBatchSampler(
        train_ds.label_ids,
        batch_size=effective_batch_size,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=effective_batch_size, shuffle=False)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def _train_one_run(
    *,
    model_key: str,
    base_checkpoint: Path,
    source_fold: str,
    trainer_kind: str,
    target_volcano: str,
    repeat_idx: int,
    subset_key: str,
    subset_dir: Path,
    test_npz: Path,
    device: torch.device,
    config: dict,
    output_dir: Path,
) -> dict:
    train_npz = subset_dir / "train.npz"
    val_npz = subset_dir / "val.npz"
    if not train_npz.exists() or not val_npz.exists():
        raise FileNotFoundError(
            f"Missing subset manifests for {target_volcano} fold={repeat_idx:02d} subset={subset_key}"
        )

    model = _load_base_model(model_key, base_checkpoint, device)
    apply_finetune_protocol(model, config["protocol"])

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = (
        _prepare_dataloaders(
            train_npz=train_npz,
            val_npz=val_npz,
            test_npz=test_npz,
            batch_size=int(config["batch_size"]),
            trainer_kind=str(trainer_kind),
        )
    )

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["lr"]),
    )

    run_root = (
        output_dir / model_key / target_volcano / f"fold_{repeat_idx:02d}" / subset_key
    )
    checkpoints_dir = run_root / "checkpoints"
    reports_dir = run_root / "reports"
    cm_dir = run_root / "confusion_matrices"
    for p in (checkpoints_dir, reports_dir, cm_dir):
        p.mkdir(parents=True, exist_ok=True)

    best_val_mean_f1 = float("-inf")
    best_epoch = -1
    best_val_loss = float("inf")
    no_improve = 0
    metrics_rows = []

    trainable_params = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    total_params = int(sum(p.numel() for p in model.parameters()))
    print(
        "[RUN] "
        f"model={model_key} trainer={trainer_kind} target={target_volcano} "
        f"fold={repeat_idx:02d} subset={subset_key}"
    )
    print(
        "[RUN] "
        f"source_fold={source_fold} batch_size={int(config['batch_size'])} "
        f"epochs={int(config['epochs'])} lr={float(config['lr']):.2e}"
    )
    print(
        "[RUN] "
        f"n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_ds)} "
        f"params(trainable/total)={trainable_params:,}/{total_params:,}"
    )

    run_start = time.time()
    for epoch in range(int(config["epochs"])):
        model.train()
        train_loss = 0.0
        batch_count = 0
        n_train_batches = max(1, len(train_loader))
        log_batch_points = max(0, int(config.get("log_batches", 4)))
        if log_batch_points <= 0:
            batch_log_indices: set[int] = set()
        elif log_batch_points >= n_train_batches:
            batch_log_indices = set(range(1, n_train_batches + 1))
        else:
            # Evenly-spaced logging points across the epoch, including final batch.
            batch_log_indices = set(
                int(x) for x in np.linspace(1, n_train_batches, num=log_batch_points)
            )
            batch_log_indices.add(n_train_batches)

        for batch_idx, (xb, y_onehot, _) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if str(trainer_kind) == "2d":
                loss, _, _ = combined_dice_ce_loss_2d(
                    out,
                    y_onehot,
                    class_weights=None,
                    dice_weight=float(config["dice_weight"]),
                    ce_weight=float(config["ce_weight"]),
                )
            else:
                loss, _, _ = combined_dice_ce_loss(
                    out,
                    y_onehot,
                    class_weights=None,
                    dice_weight=float(config["dice_weight"]),
                    ce_weight=float(config["ce_weight"]),
                )
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            batch_count += 1

            if batch_idx in batch_log_indices:
                running_loss = float(train_loss / max(1, batch_count))
                progress_pct = 100.0 * float(batch_idx) / float(n_train_batches)
                print(
                    f"epoch={epoch + 1:03d} batch={batch_idx:04d}/{n_train_batches:04d} "
                    f"progress={progress_pct:5.1f}% loss={float(loss.item()):.4f} "
                    f"running_loss={running_loss:.4f}"
                )

        mean_train_loss = float(train_loss / batch_count) if batch_count > 0 else 0.0

        if str(trainer_kind) == "2d":
            (
                val_f1_per_class,
                val_mean_f1,
                val_iou_per_class,
                val_mean_iou,
                val_loss,
                val_cm,
            ) = evaluate_unet_model(
                model=model,
                dataloader=val_loader,
                device=device,
                len_window=int(config["len_window"]),
                im_size=int(config["im_size"]),
                config=config,
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
                val_cm,
            ) = compute_event_f1_iou_multistation(
                model,
                val_loader,
                device,
                descriptor_names=None,
                return_cm=True,
                return_val_loss=True,
                return_event_plot_payloads=False,
                save_event_plots=False,
                max_event_plots=0,
                epoch=epoch,
            )

        improved = float(val_mean_f1) > float(best_val_mean_f1)
        if improved:
            best_val_mean_f1 = float(val_mean_f1)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            no_improve = 0
            best_ckpt_out = checkpoints_dir / "best_f1.pt"
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": float(val_loss),
                    "val_mean_f1": float(val_mean_f1),
                    "model_key": model_key,
                    "target_volcano": target_volcano,
                    "repeat_id": int(repeat_idx),
                    "subset_key": subset_key,
                    "base_checkpoint": str(base_checkpoint),
                },
                best_ckpt_out,
            )
            best_cm_out = cm_dir / f"val_cm_best_f1_epoch_{epoch:03d}.png"
            save_confusion_matrix_image(
                cm=val_cm,
                labels=CLASS_NAMES,
                out_path=best_cm_out,
                title=f"Val CM - {model_key} - {target_volcano} - fold {repeat_idx:02d} - {subset_key}",
            )
        else:
            no_improve += 1

        print(
            f"epoch={epoch + 1:03d}/{int(config['epochs']):03d} "
            f"train_loss={mean_train_loss:.4f} val_loss={float(val_loss):.4f} "
            f"val_mean_f1={float(val_mean_f1):.4f} val_mean_iou={float(val_mean_iou):.4f} "
            f"best_val_mean_f1={float(best_val_mean_f1):.4f} "
            f"no_improve={no_improve}/{int(config['early_stop_patience'])}"
        )

        metrics_rows.append(
            {
                "epoch": int(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(mean_train_loss),
                "val_loss": float(val_loss),
                "val_mean_f1": float(val_mean_f1),
                "val_mean_iou": float(val_mean_iou),
            }
        )

        if no_improve >= int(config["early_stop_patience"]):
            print(
                "[EARLY-STOP] "
                f"Stopping at epoch={epoch + 1:03d} after "
                f"{no_improve} epochs without val_mean_f1 improvement."
            )
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

    best_ckpt_path = checkpoints_dir / "best_f1.pt"
    if not best_ckpt_path.exists():
        raise RuntimeError(f"Missing best checkpoint: {best_ckpt_path}")

    print(f"[LOAD] loading best checkpoint for test evaluation: {best_ckpt_path}")

    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    if str(trainer_kind) == "2d":
        (
            test_f1_per_class,
            test_mean_f1,
            test_iou_per_class,
            test_mean_iou,
            test_loss,
            test_cm,
        ) = evaluate_unet_model(
            model=model,
            dataloader=test_loader,
            device=device,
            len_window=int(config["len_window"]),
            im_size=int(config["im_size"]),
            config=config,
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
        ) = compute_event_f1_iou_multistation(
            model,
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
        "target_volcano": target_volcano,
        "repeat_id": int(repeat_idx),
        "subset_key": subset_key,
        "source_checkpoint": str(base_checkpoint),
        "source_fold": str(source_fold),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_f1": float(best_val_mean_f1),
        "test_loss": float(test_loss),
        "test_mean_f1": float(test_mean_f1),
        "test_mean_iou": float(test_mean_iou),
        "elapsed_seconds": elapsed,
    }
    for idx, cls in enumerate(CLASS_NAMES):
        row[f"val_f1_{cls}"] = float(val_f1_per_class[idx])
        row[f"val_iou_{cls}"] = float(val_iou_per_class[idx])
        row[f"test_f1_{cls}"] = float(test_f1_per_class[idx])
        row[f"test_iou_{cls}"] = float(test_iou_per_class[idx])

    with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(f"[SAVE] fold summary={reports_dir / 'fold_summary.json'}")

    test_cm_out = cm_dir / "test_cm_best_f1.png"
    save_confusion_matrix_image(
        cm=test_cm,
        labels=CLASS_NAMES,
        out_path=test_cm_out,
        title=f"Test CM - {model_key} - {target_volcano} - fold {repeat_idx:02d} - {subset_key}",
    )
    print(f"[SAVE] test confusion matrix={test_cm_out}")
    print(
        "[DONE] "
        f"model={model_key} target={target_volcano} fold={repeat_idx:02d} subset={subset_key} "
        f"best_epoch={int(best_epoch) + 1} best_val_mean_f1={float(best_val_mean_f1):.4f} "
        f"test_mean_f1={float(test_mean_f1):.4f} test_mean_iou={float(test_mean_iou):.4f} "
        f"test_loss={float(test_loss):.4f} elapsed={elapsed:.1f}s"
    )

    del model, best_ckpt, test_cm, optimizer, train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds
    cleanup_gpu_cache()
    gc.collect()
    return row


def main() -> None:
    args = parse_args()

    data_root = resolve_project_path(args.data_root, PROJECT_ROOT)
    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    output_dir = resolve_project_path(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_targets = discover_targets(data_root)
    available_models = [k for k, v in MODEL_SPECS.items() if v.get("enabled", True)]
    available_models.reverse()
    selected_models = parse_csv_selection(args.models, available_models, "models")
    selected_targets = parse_csv_selection(args.targets, available_targets, "targets")
    selected_targets = _order_targets_cau_first(selected_targets)
    selected_folds = [
        int(x)
        for x in parse_csv_selection(
            args.folds, [str(x) for x in DEFAULT_FOLDS], "folds"
        )
    ]
    selected_subsets = parse_csv_selection(
        args.subsets,
        list(DEFAULT_SUBSET_KEYS),
        "subsets",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "experiment_root": str(experiment_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "lr": float(args.lr),
        "lr_final": float(args.lr_final),
        "dice_weight": float(args.dice_weight),
        "ce_weight": float(args.ce_weight),
        "len_window": int(args.len_window),
        "im_size": int(args.im_size),
        "log_batches": int(max(0, args.log_batches)),
        "protocol": str(args.protocol),
        "rerun_completed": bool(args.rerun_completed),
        "models": selected_models,
        "targets": selected_targets,
        "folds": selected_folds,
        "subsets": selected_subsets,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    best_source_checkpoints: dict[str, dict] = {}
    for model_key in selected_models:
        try:
            ckpt_path, meta = _find_best_source_checkpoint(model_key, experiment_root)
        except FileNotFoundError:
            if args.allow_missing_checkpoints:
                print(f"[WARN] Missing source checkpoint for {model_key}; skipping.")
                continue
            raise

        best_source_checkpoints[model_key] = {
            "checkpoint_path": str(ckpt_path),
            "source_fold": str(meta.get("source_fold", "")),
            "score": float(meta.get("score", float("nan"))),
            "batch_size": int(MODEL_SPECS[model_key]["batch_size"]),
        }

    with (output_dir / "source_checkpoints.json").open("w", encoding="utf-8") as f:
        json.dump(best_source_checkpoints, f, indent=2)

    print("=" * 90)
    print("Progressive finetuning")
    print(f"Data root: {data_root}")
    print(f"Experiment root: {experiment_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Models ({len(selected_models)}): {selected_models}")
    print(f"Targets ({len(selected_targets)}): {selected_targets}")
    print(f"Folds: {selected_folds}")
    print(f"Subsets: {selected_subsets}")
    print("=" * 90)

    fieldnames = [
        "model_key",
        "target_volcano",
        "repeat_id",
        "subset_key",
        "source_checkpoint",
        "source_fold",
        "n_train",
        "n_val",
        "n_test",
        "best_epoch",
        "best_val_loss",
        "best_val_mean_f1",
        "test_loss",
        "test_mean_f1",
        "test_mean_iou",
        "elapsed_seconds",
    ]
    for cls in CLASS_NAMES:
        fieldnames.append(f"val_f1_{cls}")
        fieldnames.append(f"val_iou_{cls}")
        fieldnames.append(f"test_f1_{cls}")
        fieldnames.append(f"test_iou_{cls}")

    metrics_csv_path = output_dir / "progressive_finetuning_metrics.csv"
    completed_run_keys: set[tuple[str, str, int, str]] = set()
    rows: list[dict[str, object]] = []
    if not args.rerun_completed:
        rows.extend(_load_existing_metrics_rows(metrics_csv_path))
        completed_run_keys = _load_completed_run_keys(metrics_csv_path)
        if len(completed_run_keys) > 0:
            print(
                f"Resuming run: found {len(completed_run_keys)} completed runs in existing metrics CSV."
            )

    for target_volcano in selected_targets:
        target_root = data_root / target_volcano
        fold_dirs = _discover_fold_dirs(target_root)
        fold_dir_map = {int(p.name.split("_")[-1]): p for p in fold_dirs}

        for model_key in selected_models:
            if model_key not in best_source_checkpoints:
                print(f"Skipping {model_key}: no source checkpoint discovered.")
                continue

            source_info = best_source_checkpoints[model_key]
            source_checkpoint = Path(source_info["checkpoint_path"])

            if model_key not in MODEL_SPECS:
                raise KeyError(f"Unknown model key: {model_key}")
            spec = get_model_spec(model_key)
            trainer_kind = str(spec["trainer_kind"])
            if trainer_kind not in {"1d", "2d"}:
                raise ValueError(
                    f"Progressive finetuning currently supports only 1d/2d models; got {model_key} ({trainer_kind})"
                )

            for repeat_idx in selected_folds:
                if repeat_idx not in fold_dir_map:
                    raise FileNotFoundError(
                        f"Missing fold_{repeat_idx:02d} for target {target_volcano} under {target_root}"
                    )

                fold_dir = fold_dir_map[repeat_idx]
                test_npz = fold_dir / "test_80.npz"
                if not test_npz.exists():
                    raise FileNotFoundError(f"Missing test manifest: {test_npz}")

                fold_subset_keys = _discover_subset_keys(fold_dir)
                available_subset_keys = [
                    k for k in selected_subsets if k in fold_subset_keys
                ]
                if len(available_subset_keys) == 0:
                    raise FileNotFoundError(
                        f"No selected subset manifests found under {fold_dir / 'subsets'}"
                    )

                skipped_subsets = 0
                for subset_key in available_subset_keys:
                    run_key = _run_key(
                        model_key, target_volcano, repeat_idx, subset_key
                    )
                    run_root = (
                        output_dir
                        / model_key
                        / target_volcano
                        / f"fold_{repeat_idx:02d}"
                        / subset_key
                    )

                    if not args.rerun_completed and run_key in completed_run_keys:
                        skipped_subsets += 1
                        print(
                            f"[SKIP] {model_key} {target_volcano} fold={repeat_idx:02d} subset={subset_key}: already in metrics CSV"
                        )
                        continue

                    if not args.rerun_completed:
                        existing_summary = _load_existing_run_summary_if_complete(
                            run_root
                        )
                        if existing_summary is not None:
                            skipped_subsets += 1
                            rows.append(existing_summary)
                            completed_run_keys.add(run_key)
                            append_row_csv(
                                metrics_csv_path,
                                row=existing_summary,
                                fieldnames=fieldnames,
                            )
                            print(
                                f"[SKIP] {model_key} {target_volcano} fold={repeat_idx:02d} subset={subset_key}: existing checkpoint/report found"
                            )
                            continue

                    subset_dir = fold_dir / "subsets" / subset_key
                    fixed_subset_lr = _fixed_lr_for_subset(subset_key)
                    run_config = {
                        "protocol": str(args.protocol),
                        "epochs": int(args.epochs),
                        "early_stop_patience": int(args.early_stop_patience),
                        "lr": float(fixed_subset_lr),
                        "lr_final": float(args.lr_final),
                        "batch_size": int(
                            args.batch_size
                            or int(
                                source_info.get(
                                    "batch_size", MODEL_SPECS[model_key]["batch_size"]
                                )
                            )
                        ),
                        "dice_weight": float(args.dice_weight),
                        "ce_weight": float(args.ce_weight),
                        "len_window": int(args.len_window),
                        "im_size": int(args.im_size),
                        "log_batches": int(max(0, args.log_batches)),
                    }

                    try:
                        row = _train_one_run(
                            model_key=model_key,
                            base_checkpoint=source_checkpoint,
                            source_fold=str(source_info.get("source_fold", "")),
                            trainer_kind=trainer_kind,
                            target_volcano=target_volcano,
                            repeat_idx=repeat_idx,
                            subset_key=subset_key,
                            subset_dir=subset_dir,
                            test_npz=test_npz,
                            device=device,
                            config=run_config,
                            output_dir=output_dir,
                        )
                    except FileNotFoundError:
                        if args.allow_missing_checkpoints:
                            print(
                                f"Skipping {model_key} {target_volcano} fold={repeat_idx:02d} subset={subset_key}: missing source or data"
                            )
                            continue
                        raise

                    rows.append(row)
                    append_row_csv(
                        metrics_csv_path,
                        row=row,
                        fieldnames=fieldnames,
                    )
                    completed_run_keys.add(run_key)

                if skipped_subsets == len(available_subset_keys):
                    print(
                        f"[SKIP] {model_key} {target_volcano} fold={repeat_idx:02d}: all selected subsets already completed"
                    )

    if len(rows) == 0:
        raise RuntimeError("No progressive finetuning runs completed.")

    df = pd.DataFrame(rows)
    df.to_csv(
        metrics_csv_path,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    summary_rows = []
    grouped = df.groupby(["model_key", "target_volcano", "subset_key"], sort=True)
    for (model_key, target_volcano, subset_key), grp in grouped:
        row = {
            "model_key": str(model_key),
            "target_volcano": str(target_volcano),
            "subset_key": str(subset_key),
            "n_runs": int(len(grp)),
        }
        for col in ["test_mean_f1", "test_mean_iou", "test_loss", "best_val_mean_f1"]:
            stats = compute_summary(grp[col].astype(float).tolist())
            row[f"{col}_mean"] = float(stats["mean"])
            row[f"{col}_std"] = float(stats["std"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["target_volcano", "subset_key", "test_mean_f1_mean"],
        ascending=[True, True, False],
    )
    summary_df.to_csv(
        output_dir / "progressive_finetuning_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    print("=" * 90)
    print("Progressive finetuning complete")
    print(f"Run metrics: {output_dir / 'progressive_finetuning_metrics.csv'}")
    print(f"Summary: {output_dir / 'progressive_finetuning_summary.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
