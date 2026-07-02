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

from utils.edge_features import compute_rsam
from utils.fold_io_utils import append_row_csv
from utils.model_registry import MODEL_SPECS

MODEL_SPECS = dict(reversed(list(MODEL_SPECS.items())))


from utils.script_common import parse_csv_selection, resolve_project_path
from utils.station_info import build_volcano_geometry_bank, infer_volcano_name_from_path
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


def _validate_volcano_idx(
    volcano_idx: torch.Tensor, held_out: str, volcano_name_to_idx: dict[str, int]
) -> None:
    """Verify that volcano_idx matches expected held-out volcano."""
    expected_idx = volcano_name_to_idx[held_out]
    if volcano_idx is not None:
        unique_indices = set(volcano_idx.cpu().numpy().flatten().tolist())
        if expected_idx not in unique_indices:
            raise ValueError(
                f"ERROR: Expected volcano index {expected_idx} ({held_out}) in batch, "
                f"but got: {unique_indices}"
            )


def _log_geometry_check(
    model: torch.nn.Module, held_out: str, volcano_idx: int
) -> None:
    """Log geometry bank info to verify per-sample indexing."""
    geom_bank = getattr(model, "volcano_geom_nodes", None)
    if geom_bank is not None and geom_bank.numel() > 0:
        geom_shape = geom_bank.shape
        sample_geom = geom_bank[volcano_idx : volcano_idx + 1]
        geom_sum = float(sample_geom.abs().sum().item())
        print(
            f"      [GEOM] Held_out={held_out} (idx={volcano_idx}) | "
            f"Bank shape={geom_shape} | Sample geometry L1 norm={geom_sum:.6f}"
        )


def _append_contract_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _log_req(log_path: Path, message: str) -> None:
    tagged = f"[REQCHK] {message}"
    print(tagged)
    _append_contract_log(log_path, tagged)


def _log_warn(log_path: Path, message: str) -> None:
    tagged = f"[REQWARN] {message}"
    print(tagged)
    _append_contract_log(log_path, tagged)


def _log_vram(log_path: Path, *, tag: str, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        _append_contract_log(
            log_path, f"[VRAM] {tag} device={device} (cuda_unavailable)"
        )
        return

    allocated_mb = torch.cuda.memory_allocated(device=device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device=device) / (1024**2)
    peak_allocated_mb = torch.cuda.max_memory_allocated(device=device) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device=device) / (1024**2)
    _append_contract_log(
        log_path,
        (
            f"[VRAM] {tag} device={device} "
            f"alloc_mb={allocated_mb:.2f} reserved_mb={reserved_mb:.2f} "
            f"peak_alloc_mb={peak_allocated_mb:.2f} peak_reserved_mb={peak_reserved_mb:.2f}"
        ),
    )


def _backward_with_diagnostics(
    loss: torch.Tensor,
    *,
    log_path: Path,
    device: torch.device,
    model_key: str,
    fold_idx: int,
    epoch_idx: int,
    batch_idx: int,
    family: str,
    batch_size: int,
) -> None:
    try:
        loss.backward()
    except RuntimeError as exc:
        err = str(exc)
        err_lower = err.lower()
        likely_vram = (
            "out of memory" in err_lower
            or "cudnn_status_execution_failed" in err_lower
            or "cuda" in err_lower
        )

        _log_vram(
            log_path,
            tag=(
                "failure "
                f"model={model_key} family={family} fold={fold_idx} "
                f"epoch={epoch_idx + 1} batch={batch_idx} "
                f"batch_size={batch_size}"
            ),
            device=device,
        )
        _log_warn(
            log_path,
            (
                "backward_exception "
                f"model={model_key} fold={fold_idx} epoch={epoch_idx + 1} "
                f"batch={batch_idx} family={family} "
                f"likely_vram={int(likely_vram)} error={err}"
            ),
        )

        if likely_vram:
            raise RuntimeError(
                f"{err}\n"
                "Likely CUDA/VRAM pressure during backward. "
                f"Try a smaller batch size (current={batch_size}), e.g. --batch-size {max(1, batch_size // 2)}."
            ) from exc
        raise


class GraphForwardWrapper(torch.nn.Module):
    """Inject RSAM kwargs on demand while preserving model signature expectations."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        needs_rsam: bool,
    ):
        super().__init__()
        self.base_model = base_model
        self.needs_rsam = bool(needs_rsam)

        self.use_envelope = getattr(base_model, "use_envelope", False)
        self.num_descriptors = int(getattr(base_model, "num_descriptors", 0))

    def forward(self, x: torch.Tensor, **kwargs):
        if self.needs_rsam:
            waveforms_np = x.detach().cpu().numpy()
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

    if descriptor_payload is not None and "edge_attr_dynamic" in descriptor_payload:
        edge_attr_dynamic = descriptor_payload["edge_attr_dynamic"]
        if not torch.is_tensor(edge_attr_dynamic):
            edge_attr_dynamic = torch.as_tensor(edge_attr_dynamic)
        forward_kwargs["edge_attr_dynamic"] = edge_attr_dynamic.to(
            device=xb.device,
            dtype=xb.dtype,
        )

    return xb, y_onehot, forward_kwargs


def _extract_volcano_idx_from_batch(batch):
    if len(batch) > 4 and torch.is_tensor(batch[4]):
        return batch[4]
    if len(batch) > 3 and torch.is_tensor(batch[3]):
        return batch[3]
    return None


def _canonicalize_dataset_volcano_indices(
    dataset,
    *,
    volcano_name_to_idx: dict[str, int],
    log_path: Path,
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
        _log_req(
            log_path,
            (
                f"model={model_key} split={split_name} canonicalized volcano_idx from filepaths "
                f"(no prior sample_volcano_idx)"
            ),
        )
        return

    current_arr = np.asarray(current, dtype=np.int64)
    if current_arr.shape != inferred.shape:
        raise ValueError(
            f"[{model_key}][{split_name}] sample_volcano_idx shape {current_arr.shape} "
            f"does not match inferred shape {inferred.shape}."
        )

    if not np.array_equal(current_arr, inferred):
        dataset.sample_volcano_idx = inferred
        mismatch_count = int(np.sum(current_arr != inferred))
        _log_warn(
            log_path,
            (
                f"model={model_key} split={split_name} remapped volcano_idx to canonical mapping "
                f"(mismatch_count={mismatch_count}/{len(inferred)})"
            ),
        )
    else:
        _log_req(
            log_path,
            f"model={model_key} split={split_name} volcano_idx already canonical",
        )


def _log_split_contract_check(
    *,
    log_path: Path,
    split_name: str,
    model_key: str,
    family: str,
    held_out: str,
    model: torch.nn.Module,
    wrapper: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    volcano_name_to_idx: dict[str, int],
    volcano_order: tuple[str, ...],
    volcano_geom_bank,
    needs_xcorr: bool,
) -> None:
    sample_batch = next(iter(loader))
    xb, _y_onehot, forward_kwargs = _extract_graph_batch(sample_batch, device, model)
    batch_volcano_idx = forward_kwargs.get("volcano_idx", None)
    held_out_idx = int(volcano_name_to_idx[held_out])

    _log_req(
        log_path,
        (
            f"model={model_key} family={family} split={split_name} "
            f"forward_keys={sorted(forward_kwargs.keys())} "
            f"needs_rsam={bool(getattr(wrapper, 'needs_rsam', False))} "
            f"needs_xcorr={bool(needs_xcorr)}"
        ),
    )

    if batch_volcano_idx is None:
        raise ValueError(
            f"[{model_key}][{split_name}] Missing volcano_idx in graph batch forward kwargs."
        )

    unique_indices = sorted(
        set(batch_volcano_idx.detach().cpu().numpy().flatten().tolist())
    )
    idx_to_name = {v: k for k, v in volcano_name_to_idx.items()}
    unique_names = sorted(
        [idx_to_name.get(int(i), f"UNK_{int(i)}") for i in unique_indices]
    )
    _log_req(
        log_path,
        (
            f"model={model_key} split={split_name} "
            f"batch_volcano_idx={unique_indices} batch_volcano_names={unique_names}"
        ),
    )

    if split_name == "train":
        if held_out in unique_names:
            raise ValueError(
                f"[{model_key}][train] held_out volcano '{held_out}' leaked into training batch: "
                f"idx={unique_indices}, names={unique_names}."
            )
    elif split_name == "test":
        if held_out not in unique_names:
            raise ValueError(
                f"[{model_key}][test] held_out volcano '{held_out}' not found in test batch: "
                f"idx={unique_indices}, names={unique_names}."
            )
    else:
        # Validation in this protocol may come from training volcanoes only.
        if held_out in unique_names:
            _log_warn(
                log_path,
                (
                    f"model={model_key} split=val includes held_out_idx={held_out_idx}; "
                    "this is allowed but should match your fold construction intent"
                ),
            )
        else:
            _log_req(
                log_path,
                f"model={model_key} split=val excludes held_out_idx={held_out_idx} (expected for train-volcano validation)",
            )

    if needs_xcorr:
        if "edge_attr_dynamic" not in forward_kwargs:
            raise ValueError(
                f"[{model_key}][{split_name}] edge_attr_dynamic missing for xcorr model."
            )
        ead = forward_kwargs["edge_attr_dynamic"]
        expected_shape = (
            int(xb.shape[0]),
            int(getattr(model, "n_station_pairs", -1)),
            int(getattr(model, "xcorr_feat_dim", -1)),
        )
        if tuple(ead.shape) != expected_shape:
            raise ValueError(
                f"[{model_key}][{split_name}] edge_attr_dynamic shape mismatch. "
                f"Expected {expected_shape}, got {tuple(ead.shape)}."
            )
        _log_req(
            log_path,
            f"model={model_key} split={split_name} edge_attr_dynamic_shape={tuple(ead.shape)}",
        )

    sample_indices_to_log = [0, min(1, int(batch_volcano_idx.shape[0]) - 1)]
    sample_indices_to_log = sorted(set(i for i in sample_indices_to_log if i >= 0))
    for sample_idx in sample_indices_to_log:
        vol_idx = int(batch_volcano_idx[sample_idx].item())
        vol_name = volcano_order[vol_idx]
        sample_geom = volcano_geom_bank[vol_idx]
        if isinstance(sample_geom, np.ndarray):
            geom_norm = float(np.abs(sample_geom).sum())
            first_node = sample_geom[0]
        else:
            geom_norm = float(sample_geom.abs().sum().item())
            first_node = sample_geom[0].detach().cpu().numpy()
        _log_req(
            log_path,
            (
                f"model={model_key} split={split_name} sample={sample_idx} "
                f"volcano_idx={vol_idx} volcano={vol_name} "
                f"geom_node0=[{first_node[0]:.3f},{first_node[1]:.3f},{first_node[2]:.3f}] "
                f"geom_l1={geom_norm:.6f}"
            ),
        )

    del xb, _y_onehot


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
    contract_log_path = output_dir / "input_contract_checks.log"
    with contract_log_path.open("w", encoding="utf-8") as f:
        f.write("# Input contract checks\n")
        f.write(f"created_at={datetime.now().isoformat(timespec='seconds')}\n")

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

    volcano_geom_bank, volcano_name_to_idx, volcano_order = build_volcano_geometry_bank(
        n_stations=8
    )

    # Validate volcano geometry bank structure
    print(f"[GEOMETRY] Volcano geometry bank shape: {volcano_geom_bank.shape}")
    print(f"[GEOMETRY] Volcanoes in order: {volcano_order}")
    print(f"[GEOMETRY] Volcano name -> index mapping: {volcano_name_to_idx}")
    if volcano_geom_bank.shape[0] != len(volcano_order):
        raise ValueError(
            f"Geometry bank size ({volcano_geom_bank.shape[0]}) does not match "
            f"volcano_order length ({len(volcano_order)})"
        )

    # Verify that geometries actually differ between volcanoes
    print(
        f"[GEOMETRY] OK Geometry bank validated: {volcano_geom_bank.shape[0]} volcanoes, "
        f"{volcano_geom_bank.shape[1]} nodes, {volcano_geom_bank.shape[2]} features each"
    )
    geom_checksums = []
    for v_idx, v_name in enumerate(volcano_order):
        geom_array = volcano_geom_bank[v_idx]
        if isinstance(geom_array, np.ndarray):
            geom_norm = float(np.abs(geom_array).sum())
        else:
            geom_norm = float(geom_array.abs().sum().item())
        geom_checksums.append(geom_norm)
        print(f"        {v_idx}: {v_name:8s} L1 norm = {geom_norm:.6f}")

    # Check that geometries are actually distinct
    if len(set(geom_checksums)) != len(geom_checksums):
        print("[GEOMETRY] WARNING: Some volcanoes have identical geometry norms!")
    else:
        print(
            f"[GEOMETRY] OK All {len(volcano_order)} volcanoes have distinct geometries"
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
    _log_req(
        contract_log_path,
        f"run_start data_root={data_root} output_dir={output_dir} device={device}",
    )

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
        _log_req(
            contract_log_path,
            f"model_start model={model_key} family={family} batch_size={batch_size}",
        )

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

            # Determine training volcanoes
            training_volcanoes = [v for v in volcano_order if v != held_out]
            print(
                f"  [FOLD {fold_idx}] Testing on: {held_out} | Training on: {', '.join(training_volcanoes)}"
            )
            print(
                f"    [VOLCANO_IDX] Held out volcano '{held_out}' = index {volcano_name_to_idx[held_out]}"
            )
            _log_req(
                contract_log_path,
                (
                    f"fold_start model={model_key} fold={fold_idx} held_out={held_out} "
                    f"held_out_idx={int(volcano_name_to_idx[held_out])}"
                ),
            )

            run_start = time.time()
            fold_out = output_dir / model_key / fold_dir.name
            ckpt_dir = fold_out / "checkpoints"
            reports_dir = fold_out / "reports"
            cm_dir = fold_out / "confusion_matrices"
            for p in [ckpt_dir, reports_dir, cm_dir]:
                p.mkdir(parents=True, exist_ok=True)

            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)
            _log_vram(
                contract_log_path,
                tag=f"model={model_key} fold={fold_idx} phase=fold_start",
                device=device,
            )

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
                needs_xcorr = model_kwargs.get("edge_feature_mode") == "delta_pos_xcorr"

                # Log graph model configuration
                log_parts = []

                # Family-specific backend info
                if model_kwargs.get("graph_backend"):
                    log_parts.append(
                        f"Graph backend: {model_kwargs.get('graph_backend')}"
                    )
                elif model_kwargs.get("graph_topology"):
                    log_parts.append(
                        f"Graph topology: {model_kwargs.get('graph_topology')}"
                    )

                if model_kwargs.get("use_message_passing"):
                    log_parts.append("Message passing: ON")
                if model_kwargs.get("use_bottleneck_attention"):
                    log_parts.append("Bottleneck attention: ON")
                if needs_xcorr:
                    log_parts.append(
                        f"Edge features: xcorr (dim={model_kwargs.get('xcorr_feat_dim', 'N/A')})"
                    )
                elif model_kwargs.get("edge_feature_mode"):
                    log_parts.append(
                        f"Edge features: {model_kwargs.get('edge_feature_mode')}"
                    )
                if model_num_desc > 0:
                    log_parts.append(
                        f"Descriptors: {model_num_desc} ({', '.join(descriptor_names) if descriptor_names else 'N/A'})"
                    )
                if model_kwargs.get("use_rsam_node_feat"):
                    log_parts.append("RSAM node features: ON")
                print(f"    [CONFIG] {' | '.join(log_parts)}")
                _log_req(
                    contract_log_path,
                    (
                        f"family={family} "
                        f"requirements model={model_key} "
                        f"needs_waveforms=True needs_geometry={bool((family == 'graphsage' and model_kwargs.get('node_feature_mode', 'geometry') in {'geometry', 'both'}) or (family == 'mpnn' and (model_kwargs.get('node_feature_mode', 'geometry') == 'geometry' or model_kwargs.get('edge_feature_mode', 'delta_pos') != 'none')))} "
                        f"needs_xcorr={bool(needs_xcorr)} needs_rsam={bool(model_kwargs.get('use_rsam_node_feat', False))}"
                    ),
                )

                def _edge_npz(npz_path: Path) -> Path:
                    return npz_path.parent / "edge_data" / npz_path.name

                train_ds = CrossVolcanoLOODataset(
                    train_npz,
                    descriptor_names=descriptor_names if model_num_desc > 0 else None,
                    edge_data_npz=_edge_npz(train_npz) if needs_xcorr else None,
                    return_volcano_idx=True,
                    volcano_name_to_idx=volcano_name_to_idx,
                )
                val_ds = CrossVolcanoLOODataset(
                    val_npz,
                    descriptor_names=descriptor_names if model_num_desc > 0 else None,
                    edge_data_npz=_edge_npz(val_npz) if needs_xcorr else None,
                    return_volcano_idx=True,
                    volcano_name_to_idx=volcano_name_to_idx,
                )
                test_ds = CrossVolcanoLOODataset(
                    test_npz,
                    descriptor_names=descriptor_names if model_num_desc > 0 else None,
                    edge_data_npz=_edge_npz(test_npz) if needs_xcorr else None,
                    return_volcano_idx=True,
                    volcano_name_to_idx=volcano_name_to_idx,
                )

                _canonicalize_dataset_volcano_indices(
                    train_ds,
                    volcano_name_to_idx=volcano_name_to_idx,
                    log_path=contract_log_path,
                    model_key=model_key,
                    split_name="train",
                )
                _canonicalize_dataset_volcano_indices(
                    val_ds,
                    volcano_name_to_idx=volcano_name_to_idx,
                    log_path=contract_log_path,
                    model_key=model_key,
                    split_name="val",
                )
                _canonicalize_dataset_volcano_indices(
                    test_ds,
                    volcano_name_to_idx=volcano_name_to_idx,
                    log_path=contract_log_path,
                    model_key=model_key,
                    split_name="test",
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
                    needs_rsam=bool(model_kwargs.get("use_rsam_node_feat", False)),
                ).to(device)

                # Validate geometry indexing
                expected_volcano_idx = volcano_name_to_idx[held_out]
                _log_geometry_check(model, held_out, expected_volcano_idx)

                # One-batch contract checks per split (single consolidated log file).
                _log_split_contract_check(
                    log_path=contract_log_path,
                    split_name="train",
                    model_key=model_key,
                    family=family,
                    held_out=held_out,
                    model=model,
                    wrapper=wrapper,
                    loader=train_loader,
                    device=device,
                    volcano_name_to_idx=volcano_name_to_idx,
                    volcano_order=volcano_order,
                    volcano_geom_bank=volcano_geom_bank,
                    needs_xcorr=needs_xcorr,
                )
                _log_split_contract_check(
                    log_path=contract_log_path,
                    split_name="val",
                    model_key=model_key,
                    family=family,
                    held_out=held_out,
                    model=model,
                    wrapper=wrapper,
                    loader=val_loader,
                    device=device,
                    volcano_name_to_idx=volcano_name_to_idx,
                    volcano_order=volcano_order,
                    volcano_geom_bank=volcano_geom_bank,
                    needs_xcorr=needs_xcorr,
                )
                _log_split_contract_check(
                    log_path=contract_log_path,
                    split_name="test",
                    model_key=model_key,
                    family=family,
                    held_out=held_out,
                    model=model,
                    wrapper=wrapper,
                    loader=test_loader,
                    device=device,
                    volcano_name_to_idx=volcano_name_to_idx,
                    volcano_order=volcano_order,
                    volcano_geom_bank=volcano_geom_bank,
                    needs_xcorr=needs_xcorr,
                )
                _log_vram(
                    contract_log_path,
                    tag=f"model={model_key} fold={fold_idx} phase=post_contract_checks",
                    device=device,
                )

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

            print(
                f"    [TRAINING] Starting {int(args.epochs)} epochs | lr={float(args.lr):.2e} -> {float(args.lr_final):.2e}"
            )

            for epoch in range(int(args.epochs)):
                wrapper.train()
                train_loss = 0.0

                # On first epoch, validate per-batch volcano indices for graph models
                if epoch == 0 and family != "unet":
                    sample_batch = next(iter(train_loader))
                    batch_volcano_idx = _extract_volcano_idx_from_batch(sample_batch)
                    if batch_volcano_idx is not None:
                        unique_in_batch = set(
                            batch_volcano_idx.cpu().numpy().flatten().tolist()
                        )
                        held_out_idx = volcano_name_to_idx[held_out]

                        # Training batches should NOT contain held_out volcano
                        if held_out_idx in unique_in_batch:
                            raise ValueError(
                                f"ERROR: Training batch should exclude held_out volcano {held_out} "
                                f"(idx={held_out_idx}), but found it in batch indices: {unique_in_batch}"
                            )

                        print(
                            f"      [BATCH_IDX] First training batch volcano indices: {sorted(unique_in_batch)} OK (excluded {held_out})"
                        )

                        # Log actual geometry values being used per sample - show samples from different volcanoes
                        print(
                            f"      [SAMPLE_GEOM] Verifying per-sample geometry usage (training volcanoes):"
                        )
                        batch_size = batch_volcano_idx.shape[0]
                        sample_indices_to_log = [
                            0,
                            min(1, batch_size - 1),
                            batch_size - 1,
                        ]
                        sample_indices_to_log = sorted(set(sample_indices_to_log))

                        for sample_idx in sample_indices_to_log:
                            vol_idx = int(batch_volcano_idx[sample_idx].item())
                            vol_name = volcano_order[vol_idx]
                            sample_geom = volcano_geom_bank[vol_idx]  # shape: [9, 3]
                            if isinstance(sample_geom, np.ndarray):
                                geom_norm = float(np.abs(sample_geom).sum())
                                first_node = sample_geom[0]
                            else:
                                geom_norm = float(sample_geom.abs().sum().item())
                                first_node = sample_geom[0].detach().cpu().numpy()
                            print(
                                f"          sample {sample_idx}: volcano_idx={vol_idx} ({vol_name:8s}) | "
                                f"node[0]=[{first_node[0]:7.3f}, {first_node[1]:7.3f}, {first_node[2]:7.3f}] | "
                                f"total_norm={geom_norm:.6f}"
                            )

                if family == "unet":
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
                        _backward_with_diagnostics(
                            loss,
                            log_path=contract_log_path,
                            device=device,
                            model_key=model_key,
                            fold_idx=fold_idx,
                            epoch_idx=epoch,
                            batch_idx=batch_idx,
                            family=family,
                            batch_size=batch_size,
                        )
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
                        _backward_with_diagnostics(
                            loss,
                            log_path=contract_log_path,
                            device=device,
                            model_key=model_key,
                            fold_idx=fold_idx,
                            epoch_idx=epoch,
                            batch_idx=batch_idx,
                            family=family,
                            batch_size=batch_size,
                        )
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
                    print(f"      ✓ epoch={epoch+1:3d} val_f1={val_mean_f1:.4f} [BEST]")
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
            print(f"    -> Saved to: {fold_out}")
            _log_vram(
                contract_log_path,
                tag=(
                    f"model={model_key} fold={fold_idx} phase=fold_end "
                    f"held_out={held_out}"
                ),
                device=device,
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
