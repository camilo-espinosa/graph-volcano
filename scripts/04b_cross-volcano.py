"""
Leave-one-out cross-volcano evaluation only (no retraining).

This script loads existing checkpoints and evaluates them on their corresponding
hold-out test sets.

Typical run:
    python scripts/04b_cross-volcano.py

Default checkpoint roots checked (in this order):
- results/experiments/complete_experiment/leave-one-out
- results/experiments/complete_experiment/cross_volcano_leave_one_out

Expected weight layout:
    <weights_root>/<model_key>/fold_XX_holdout_<VOLCANO>/checkpoints/best_f1.pt

Expected data layout:
    data/prepared_data/cross_volcano_loo/fold_XX_holdout_<VOLCANO>/test.npz
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.PhaseNet import PhaseNet
from utils.edge_features import compute_rsam
from utils.fold_io_utils import append_row_csv
from utils.model_registry import MODEL_SPECS
from utils.script_common import parse_csv_selection, resolve_project_path
from utils.station_info import build_volcano_geometry_bank, infer_volcano_name_from_path
from utils.train_utils import (
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
    predicted_from_output,
    save_confusion_matrix_image,
)

MODEL_SPECS = dict(reversed(list(MODEL_SPECS.items())))

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / "cross_volcano_loo"
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "results" / "experiments" / "complete_experiment"
)
DEFAULT_OUTPUT_NAME = "cross_volcano_leave_one_out_eval_only"
FOLD_GLOB = "fold_*_holdout_*"
CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
ALL_CLASS_NAMES = ["BG", "VT", "LP", "TR", "AV", "IC"]
EVENT_CLASS_TO_ID = {"VT": 1, "LP": 2, "TR": 3, "AV": 4, "IC": 5}
EVENT_ID_TO_NAME = {v: k for k, v in EVENT_CLASS_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate leave-one-out cross-volcano checkpoints on test splits only "
            "(no training)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root folder with fold_XX_holdout_<VOLCANO> manifests.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Base experiment root used to resolve defaults.",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=None,
        help=(
            "Root with trained leave-one-out checkpoints. Defaults to "
            "<experiment-root>/leave-one-out, then fallback to "
            "<experiment-root>/cross_volcano_leave_one_out."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output folder. Default: "
            "<experiment-root>/cross_volcano_leave_one_out_eval_only"
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys (alias: ablation names).",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=None,
        help="Alias for --models.",
    )
    parser.add_argument(
        "--family",
        choices=["all", "unet", "graphsage", "mpnn", "phasenet"],
        default="all",
        help="Optional model family filter.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=2,
        help="Random test examples to save per event class (VT/LP/TR/AV/IC).",
    )
    parser.add_argument(
        "--save-confusion-matrices",
        action="store_true",
        help="Save confusion matrix image for each model/fold.",
    )
    parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help="Skip model/fold when best_f1.pt is missing instead of failing.",
    )
    return parser.parse_args()


class GraphForwardWrapper(torch.nn.Module):
    """Inject RSAM kwargs on demand while preserving model signature expectations."""

    def __init__(self, base_model: torch.nn.Module, needs_rsam: bool):
        super().__init__()
        self.base_model = base_model
        self.needs_rsam = bool(needs_rsam)

        self.use_envelope = getattr(base_model, "use_envelope", False)
        self.num_descriptors = int(getattr(base_model, "num_descriptors", 0))

    def forward(self, x: torch.Tensor, **kwargs):
        if self.needs_rsam:
            waveforms_np = x.detach().cpu().numpy()
            rsam_np = compute_rsam(waveforms_np)
            kwargs["rsam"] = torch.from_numpy(rsam_np).to(
                device=x.device,
                dtype=x.dtype,
            )

        return self.base_model(x, **kwargs)


def _resolve_weights_root(args: argparse.Namespace, experiment_root: Path) -> Path:
    if args.weights_root is not None:
        return resolve_project_path(args.weights_root, PROJECT_ROOT)

    primary = experiment_root / "leave-one-out"
    fallback = experiment_root / "cross_volcano_leave_one_out"
    if primary.exists():
        return primary
    return fallback


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

    if descriptor_payload is not None and "edge_attr_dynamic" in descriptor_payload:
        edge_attr_dynamic = descriptor_payload["edge_attr_dynamic"]
        if not torch.is_tensor(edge_attr_dynamic):
            edge_attr_dynamic = torch.as_tensor(edge_attr_dynamic)
        forward_kwargs["edge_attr_dynamic"] = edge_attr_dynamic.to(
            device=xb.device,
            dtype=xb.dtype,
        )

    return xb, y_onehot, forward_kwargs


def _extract_single_graph_sample(
    sample,
    device: torch.device,
    model: torch.nn.Module,
):
    x = sample[0]
    y_onehot = sample[1]
    y_label = sample[2]

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if not torch.is_tensor(y_onehot):
        y_onehot = torch.as_tensor(y_onehot)
    if not torch.is_tensor(y_label):
        y_label = torch.as_tensor(y_label)

    xb = x.unsqueeze(0).to(device)
    yb = y_onehot.unsqueeze(0).to(device)

    descriptor_payload = None
    volcano_idx = None
    if len(sample) > 3 and isinstance(sample[3], dict):
        descriptor_payload = sample[3]
    elif len(sample) > 3 and torch.is_tensor(sample[3]):
        volcano_idx = sample[3].view(1).to(device).long()

    if len(sample) > 4 and torch.is_tensor(sample[4]):
        volcano_idx = sample[4].view(1).to(device).long()

    forward_kwargs: dict[str, Any] = {}
    if volcano_idx is not None:
        forward_kwargs["volcano_idx"] = volcano_idx

    if getattr(model, "use_envelope", False):
        if descriptor_payload is None or "envelope" not in descriptor_payload:
            raise ValueError(
                "Model requires envelope but sample has no descriptor payload with 'envelope'."
            )
        env = descriptor_payload["envelope"]
        if not torch.is_tensor(env):
            env = torch.as_tensor(env)
        env = env.unsqueeze(0)
        if env.ndim == 4 and env.shape[1] == 1:
            env = env[:, 0, :, :]
        elif env.ndim == 4 and env.shape[2] == 1:
            env = env[:, :, 0, :]
        forward_kwargs["envelope"] = env.to(device=xb.device, dtype=xb.dtype)

    model_num_desc = int(getattr(model, "num_descriptors", 0))
    if model_num_desc > 0:
        descriptor_names = getattr(model, "descriptor_names", None)
        if descriptor_names is None:
            raise ValueError(
                "Model has num_descriptors > 0 but does not expose descriptor_names."
            )

        payload_batched = {}
        if descriptor_payload is not None:
            for k, v in descriptor_payload.items():
                tv = v if torch.is_tensor(v) else torch.as_tensor(v)
                payload_batched[k] = tv.unsqueeze(0)

        descriptors = extract_descriptor_tensor(payload_batched, descriptor_names, xb)
        forward_kwargs["descriptors"] = descriptors

    if descriptor_payload is not None and "edge_attr_dynamic" in descriptor_payload:
        edge_attr_dynamic = descriptor_payload["edge_attr_dynamic"]
        if not torch.is_tensor(edge_attr_dynamic):
            edge_attr_dynamic = torch.as_tensor(edge_attr_dynamic)
        edge_attr_dynamic = edge_attr_dynamic.unsqueeze(0)
        forward_kwargs["edge_attr_dynamic"] = edge_attr_dynamic.to(
            device=xb.device,
            dtype=xb.dtype,
        )

    return xb, yb, y_label.view(1).long().to(device), forward_kwargs


def _canonicalize_dataset_volcano_indices(
    dataset,
    *,
    volcano_name_to_idx: dict[str, int],
    model_key: str,
    split_name: str,
) -> None:
    """Remap dataset.sample_volcano_idx to the active volcano_name_to_idx mapping."""
    if not hasattr(dataset, "filepaths") or not hasattr(dataset, "sample_volcano_idx"):
        return

    inferred = np.asarray(
        [
            int(volcano_name_to_idx[infer_volcano_name_from_path(str(fp))])
            for fp in dataset.filepaths
        ],
        dtype=np.int64,
    )

    current = getattr(dataset, "sample_volcano_idx", None)
    if current is None:
        dataset.sample_volcano_idx = inferred
        return

    current_arr = np.asarray(current, dtype=np.int64)
    if current_arr.shape != inferred.shape:
        raise ValueError(
            f"[{model_key}][{split_name}] sample_volcano_idx shape {current_arr.shape} "
            f"does not match inferred shape {inferred.shape}."
        )

    if not np.array_equal(current_arr, inferred):
        dataset.sample_volcano_idx = inferred

    del current_arr, inferred


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


def _stable_text_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _select_balanced_indices(
    label_ids: np.ndarray,
    examples_per_class: int,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    for class_name, class_id in EVENT_CLASS_TO_ID.items():
        idx = np.where(np.asarray(label_ids, dtype=np.int64) == int(class_id))[0]
        if len(idx) == 0:
            selected[int(class_id)] = []
            continue
        n_pick = min(int(examples_per_class), int(len(idx)))
        chosen = rng.choice(idx, size=n_pick, replace=False)
        selected[int(class_id)] = [int(x) for x in chosen.tolist()]
    return selected


def _save_validation_event_plot_with_truth(
    x_raw: np.ndarray,
    out_np: np.ndarray,
    processed_out: np.ndarray,
    true_onehot: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    class_names = ALL_CLASS_NAMES
    class_colors = {
        "BG": "#808080",
        "VT": "#df8d5e",
        "LP": "#2ca02c",
        "TR": "#d62728",
        "AV": "#9467bd",
        "IC": "#8c564b",
    }

    t = np.arange(x_raw.shape[1], dtype=np.int64)
    n_stations = min(8, int(x_raw.shape[0]))

    fig, axes = plt.subplots(
        n_stations + 3,
        1,
        figsize=(14, 13),
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )

    for i in range(n_stations):
        ax = axes[i]
        ax.plot(t, x_raw[i], lw=0.7, color="black")
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(f"S{i+1}", rotation=0, labelpad=12, fontsize=8)
        ax.set_xticks([])
        ax.margins(x=0)

    ax_raw = axes[n_stations]
    for c, cname in enumerate(class_names):
        ax_raw.plot(t, out_np[c], lw=1.0, color=class_colors[cname], label=cname)
    ax_raw.set_ylim(-0.2, 1.2)
    ax_raw.set_ylabel("raw", rotation=0, labelpad=18, fontsize=9)
    ax_raw.legend(loc="upper right", ncol=6, fontsize=8, frameon=False)
    ax_raw.margins(x=0)

    ax_proc = axes[n_stations + 1]
    for c, cname in enumerate(class_names):
        ax_proc.plot(
            t,
            processed_out[c],
            lw=1.0,
            color=class_colors[cname],
            label=cname,
        )
    ax_proc.set_ylim(-0.2, 1.2)
    ax_proc.set_ylabel("argmax", rotation=0, labelpad=18, fontsize=9)
    ax_proc.margins(x=0)

    ax_true = axes[n_stations + 2]
    for c, cname in enumerate(class_names):
        ax_true.plot(
            t,
            true_onehot[c],
            lw=1.0,
            color=class_colors[cname],
            label=cname,
        )
    ax_true.set_ylim(-0.2, 1.2)
    ax_true.set_ylabel("true", rotation=0, labelpad=18, fontsize=9)
    ax_true.set_xlabel("sample")
    ax_true.margins(x=0)

    for ax in axes[n_stations:]:
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_graph_examples(
    model: torch.nn.Module,
    wrapper: torch.nn.Module,
    test_ds: CrossVolcanoLOODataset,
    out_dir: Path,
    examples_per_class: int,
    device: torch.device,
    seed_value: int,
    model_key: str,
    fold_name: str,
) -> dict[str, Any]:
    class_ids = np.asarray(test_ds.label_ids, dtype=np.int64)
    rng = np.random.default_rng(int(seed_value))
    selected = _select_balanced_indices(class_ids, examples_per_class, rng)

    saved = {name: 0 for name in CLASS_NAMES}
    missing = []

    wrapper.eval()
    with torch.inference_mode():
        for class_id, indices in selected.items():
            class_name = EVENT_ID_TO_NAME[int(class_id)]
            if len(indices) == 0:
                missing.append(class_name)
                continue

            for sample_idx in indices:
                sample = test_ds[int(sample_idx)]
                xb, yb, y_label_b, forward_kwargs = _extract_single_graph_sample(
                    sample,
                    device,
                    model,
                )

                out = wrapper(xb, **forward_kwargs)
                if out.ndim == 4:
                    probs = torch.softmax(out, dim=2).mean(dim=1)
                elif out.ndim == 3:
                    probs = torch.softmax(out, dim=1)
                else:
                    raise ValueError(
                        f"Unexpected model output shape {tuple(out.shape)} while generating examples."
                    )

                out_np = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)
                max_indices = np.argmax(out_np, axis=0)
                processed_out = np.eye(out_np.shape[0], dtype=np.float32)[max_indices].T
                true_onehot = (
                    yb[0].detach().cpu().numpy().astype(np.float32, copy=False)
                )
                x_raw = xb[0].detach().cpu().numpy().astype(np.float32, copy=False)

                pred_evt, pred_name, _, _ = predicted_from_output(
                    out_np,
                    clases_ovdas={
                        1.0: "VT",
                        2.0: "LP",
                        3.0: "TR",
                        4.0: "AV",
                        5.0: "IC",
                    },
                    t_bg=0,
                    t_cl=0,
                )
                true_evt = int(y_label_b[0].detach().cpu().item())
                true_name = EVENT_ID_TO_NAME.get(true_evt, f"C{true_evt}")

                sample_out = (
                    out_dir
                    / class_name
                    / (
                        f"sample_{int(sample_idx):05d}_"
                        f"true_{true_name}_pred_{pred_name}.png"
                    )
                )
                title = (
                    f"{model_key} | {fold_name} | sample={int(sample_idx)} | "
                    f"true={true_name}({true_evt}) pred={pred_name}({int(pred_evt)})"
                )
                _save_validation_event_plot_with_truth(
                    x_raw=x_raw,
                    out_np=out_np,
                    processed_out=processed_out,
                    true_onehot=true_onehot,
                    out_path=sample_out,
                    title=title,
                )
                saved[class_name] += 1

                del sample, xb, yb, y_label_b, out, probs

    manifest = {
        "model_key": model_key,
        "fold": fold_name,
        "examples_per_class_requested": int(examples_per_class),
        "saved_per_class": saved,
        "missing_classes": missing,
        "selected_indices": {
            EVENT_ID_TO_NAME[int(k)]: [int(x) for x in v] for k, v in selected.items()
        },
    }
    with (out_dir / "examples_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    args = parse_args()

    if args.models is not None and args.ablations is not None:
        raise ValueError("Use only one of --models or --ablations.")

    data_root = resolve_project_path(args.data_root, PROJECT_ROOT)
    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    weights_root = _resolve_weights_root(args, experiment_root)
    output_dir = (
        resolve_project_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir is not None
        else (experiment_root / DEFAULT_OUTPUT_NAME)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not weights_root.exists():
        raise FileNotFoundError(
            f"Weights root does not exist: {weights_root}. "
            "Set --weights-root explicitly."
        )

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
    raw_models = args.models if args.models is not None else args.ablations
    selected_models = parse_csv_selection(raw_models, list(specs.keys()), "models")

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
        "experiment_root": str(experiment_root),
        "weights_root": str(weights_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "batch_size_override": (
            None if args.batch_size is None else int(args.batch_size)
        ),
        "dice_weight": float(args.dice_weight),
        "ce_weight": float(args.ce_weight),
        "examples_per_class": int(args.examples_per_class),
        "selected_models": selected_models,
        "fold_dirs": [str(p) for p in fold_dirs],
        "volcano_order": list(volcano_order),
        "volcano_name_to_idx": {k: int(v) for k, v in volcano_name_to_idx.items()},
        "save_confusion_matrices": bool(args.save_confusion_matrices),
        "allow_missing_checkpoints": bool(args.allow_missing_checkpoints),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 90)
    print("Cross-volcano leave-one-out EVAL ONLY (no training)")
    print(f"Data root: {data_root}")
    print(f"Weights root: {weights_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Models ({len(selected_models)}): {selected_models}")
    print("=" * 90)

    fieldnames = [
        "model_key",
        "family",
        "fold",
        "held_out_volcano",
        "n_test",
        "checkpoint",
        "test_loss",
        "test_mean_f1",
        "test_mean_iou",
        "elapsed_seconds",
        "examples_saved_total",
        "examples_missing_classes",
    ]
    for cls in CLASS_NAMES:
        fieldnames.append(f"test_f1_{cls}")
        fieldnames.append(f"test_iou_{cls}")
    for cls in ALL_CLASS_NAMES:
        fieldnames.append(f"test_iou_all_{cls}")

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
            test_npz = fold_dir / "test.npz"
            fold_name = fold_dir.name

            with np.load(test_npz) as data:
                if "held_out_volcano" in data:
                    held_out = str(data["held_out_volcano"][0])
                else:
                    held_out = (
                        fold_name.split("holdout_")[-1]
                        if "holdout_" in fold_name
                        else "UNKNOWN"
                    )

            run_start = time.time()
            fold_out = output_dir / model_key / fold_name
            cm_dir = fold_out / "confusion_matrices"
            reports_dir = fold_out / "reports"
            examples_dir = fold_out / "test_event_examples"
            for p in [cm_dir, reports_dir, examples_dir]:
                p.mkdir(parents=True, exist_ok=True)

            ckpt_path = (
                weights_root / model_key / fold_name / "checkpoints" / "best_f1.pt"
            )
            if not ckpt_path.exists():
                msg = f"Missing checkpoint: {ckpt_path}"
                if args.allow_missing_checkpoints:
                    print(f"  [SKIP] {msg}")
                    continue
                raise FileNotFoundError(msg)

            print(f"  [FOLD {fold_idx}] held_out={held_out} checkpoint={ckpt_path}")

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

                best_ckpt = torch.load(
                    ckpt_path, map_location=device, weights_only=False
                )
                model.load_state_dict(best_ckpt["model_state_dict"])
                model.eval()

                test_ds = UNetPatchDataset(test_npz)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                (
                    test_f1_per_class,
                    test_mean_f1,
                    test_iou_per_class,
                    test_mean_iou,
                    test_loss,
                    test_cm,
                ) = _evaluate_unet(
                    model,
                    test_loader,
                    device,
                    dice_weight=float(args.dice_weight),
                    ce_weight=float(args.ce_weight),
                )

                # For 2D patch UNet variants, event plots with 1D temporal activations are
                # not directly comparable to graph/1D outputs.
                examples_manifest = {
                    "model_key": model_key,
                    "fold": fold_name,
                    "examples_per_class_requested": int(args.examples_per_class),
                    "saved_per_class": {name: 0 for name in CLASS_NAMES},
                    "missing_classes": list(CLASS_NAMES),
                    "note": (
                        "Skipped random event plots for trainer_kind=2d "
                        "because output is patch-stacked 2D."
                    ),
                }
                with (examples_dir / "examples_manifest.json").open(
                    "w", encoding="utf-8"
                ) as f:
                    json.dump(examples_manifest, f, indent=2)

                iou_all_classes = [np.nan] * len(ALL_CLASS_NAMES)

            else:
                if spec["model_cls"] is PhaseNet:
                    model = spec["model_cls"](**model_kwargs).to(device)
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

                best_ckpt = torch.load(
                    ckpt_path, map_location=device, weights_only=False
                )
                model.load_state_dict(best_ckpt["model_state_dict"])
                model.eval()

                descriptor_names = getattr(model, "descriptor_names", None)
                model_num_desc = int(getattr(model, "num_descriptors", 0))
                needs_xcorr = model_kwargs.get("edge_feature_mode") == "delta_pos_xcorr"

                def _edge_npz(npz_path: Path) -> Path:
                    return npz_path.parent / "edge_data" / npz_path.name

                test_ds = CrossVolcanoLOODataset(
                    test_npz,
                    descriptor_names=descriptor_names if model_num_desc > 0 else None,
                    edge_data_npz=_edge_npz(test_npz) if needs_xcorr else None,
                    return_volcano_idx=True,
                    volcano_name_to_idx=volcano_name_to_idx,
                )
                _canonicalize_dataset_volcano_indices(
                    test_ds,
                    volcano_name_to_idx=volcano_name_to_idx,
                    model_key=model_key,
                    split_name="test",
                )

                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                wrapper = GraphForwardWrapper(
                    model,
                    needs_rsam=bool(model_kwargs.get("use_rsam_node_feat", False)),
                ).to(device)

                (
                    test_f1_per_class,
                    test_mean_f1,
                    test_iou_per_class,
                    test_mean_iou,
                    iou_all_classes,
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

                per_fold_seed = (
                    int(args.seed) + int(fold_idx) + _stable_text_int(model_key) + 17
                )
                examples_manifest = _save_graph_examples(
                    model=model,
                    wrapper=wrapper,
                    test_ds=test_ds,
                    out_dir=examples_dir,
                    examples_per_class=int(args.examples_per_class),
                    device=device,
                    seed_value=per_fold_seed,
                    model_key=model_key,
                    fold_name=fold_name,
                )

                del wrapper

            if args.save_confusion_matrices:
                save_confusion_matrix_image(
                    test_cm,
                    labels=CLASS_NAMES,
                    out_path=cm_dir / "test_confusion_matrix.png",
                    title=f"{model_key} | {fold_name} | holdout={held_out}",
                )

            elapsed = float(time.time() - run_start)
            missing_classes = examples_manifest.get("missing_classes", [])
            saved_per_class = examples_manifest.get("saved_per_class", {})
            examples_saved_total = int(sum(int(v) for v in saved_per_class.values()))

            row = {
                "model_key": model_key,
                "family": family,
                "fold": int(fold_idx),
                "held_out_volcano": held_out,
                "n_test": int(len(test_ds)),
                "checkpoint": str(ckpt_path),
                "test_loss": float(test_loss),
                "test_mean_f1": float(test_mean_f1),
                "test_mean_iou": float(test_mean_iou),
                "elapsed_seconds": elapsed,
                "examples_saved_total": int(examples_saved_total),
                "examples_missing_classes": "|".join(str(x) for x in missing_classes),
            }
            for idx, cls in enumerate(CLASS_NAMES):
                row[f"test_f1_{cls}"] = float(test_f1_per_class[idx])
                row[f"test_iou_{cls}"] = float(test_iou_per_class[idx])
            for idx, cls in enumerate(ALL_CLASS_NAMES):
                row[f"test_iou_all_{cls}"] = float(iou_all_classes[idx])

            rows.append(row)
            append_row_csv(
                output_dir / "cross_volcano_eval_fold_metrics.csv",
                row=row,
                fieldnames=fieldnames,
            )

            with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

            print(
                f"  fold={fold_idx:02d} held_out={held_out} "
                f"test_f1={test_mean_f1:.4f} test_iou={test_mean_iou:.4f} "
                f"plots={examples_saved_total} missing={missing_classes}"
            )
            print(f"    -> Saved to: {fold_out}")

            del test_loader, test_ds
            del model, best_ckpt, test_cm
            cleanup_gpu_cache()
            gc.collect()

    all_fold_summaries = sorted(
        output_dir.glob("*/fold_*_holdout_*/reports/fold_summary.json")
    )
    if len(all_fold_summaries) == 0:
        raise RuntimeError(
            "No fold summaries found under output directory. Nothing to aggregate."
        )

    all_rows: list[dict[str, Any]] = []
    for summary_path in all_fold_summaries:
        with summary_path.open("r", encoding="utf-8") as f:
            all_rows.append(json.load(f))

    df = pd.DataFrame(all_rows)
    df.to_csv(
        output_dir / "cross_volcano_eval_fold_metrics.csv",
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
        for col in ["test_mean_f1", "test_mean_iou", "test_loss"]:
            stats = compute_summary(grp[col].astype(float).tolist())
            row[f"{col}_mean"] = float(stats["mean"])
            row[f"{col}_std"] = float(stats["std"])
        for cls in CLASS_NAMES:
            for metric in ["test_f1", "test_iou"]:
                col = f"{metric}_{cls}"
                stats = compute_summary(grp[col].astype(float).tolist())
                row[f"{col}_mean"] = float(stats["mean"])
                row[f"{col}_std"] = float(stats["std"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["held_out_volcano", "test_mean_f1_mean"],
        ascending=[True, False],
    )
    summary_df.to_csv(
        output_dir / "cross_volcano_eval_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    print(f"Aggregated {len(all_rows)} fold summaries.")
    print("=" * 90)
    print("Cross-volcano leave-one-out EVAL ONLY run complete")
    print(f"Fold metrics: {output_dir / 'cross_volcano_eval_fold_metrics.csv'}")
    print(f"Summary: {output_dir / 'cross_volcano_eval_summary.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
