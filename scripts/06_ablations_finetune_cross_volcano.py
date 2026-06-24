"""
Fine-tune ablation checkpoints on cross-volcano small-train protocols, then evaluate.

Scope:
- Retrains only ablation models from complete experiment checkpoints.
- Uses cross-volcano train artifacts: train_01pct/train_05pct/train_10pct/train_20pct.
- Splits each train artifact into 85% fine-tune train + 15% validation.
- Evaluates best fine-tuned checkpoint on test_80 of the same target volcano.

Protocols:
- protocol_a_all_weights: fine-tune all weights.
- protocol_b_decoder_only: freeze encoder + graph layers, train decoder only.

Outputs:
- <experiment_root>/finetune_cross_volcano/
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.UNet_GraphSAGE import UNet_GraphSAGE
from utils.finetune_utils import (
    apply_finetune_protocol,
    split_indices_stratified,
    trainable_parameter_count,
)
from utils.fold_io_utils import checkpoint_path_for_fold
from utils.script_common import discover_targets, parse_csv_selection, resolve_project_path
from utils.train_utils import (
    BalancedBatchSampler,
    GraphSAGEDataset,
    cleanup_gpu_cache,
    combined_dice_ce_loss,
    compute_event_f1_iou_graphsage,
    compute_summary,
    save_confusion_matrix_image,
)


CLASS_NAMES = ["VT", "LP", "TR", "AV", "IC"]
FOLDS = range(1, 6)
PERCENT_ARTIFACTS = ["train_01pct.npz", "train_05pct.npz", "train_10pct.npz", "train_20pct.npz"]

RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "complete_experiment"
DEFAULT_CROSS_DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data" / "cross_volcano"
DEFAULT_OUTPUT_NAME = "finetune_cross_volcano"

# Keep ablation-specific batch-size overrides used during original training.
BATCH_SIZE_OVERRIDES = {
    "ablation_2_mlp_backend": 20,
    "ablation_3_no_message_passing": 20,
    "ablation_4_no_bottleneck_attention": 14,
    "ablation_5_no_norm": 18,
    "ablation_6_batchnorm": 18,
    "ablation_7_mean_virtual_node_pool": 14,
    "ablation_8_graph_only_bottleneck": 24,
    "ablation_9_no_skip_graph": 18,
}

PROTOCOL_SPECS = {
    "protocol_a_all_weights": {
        "display_name": "Protocol A - Fine tune all weights",
    },
    "protocol_b_decoder_only": {
        "display_name": "Protocol B - Freeze encoder+graph, train decoder only",
    },
}

# Fallback ablation kwargs. Script 06 first loads kwargs from the base run manifest
# and then falls back to this map for ablations added after that manifest was created.
GRAPH_LEVELS = [3, 4]
V5_FULL_KWARGS = {
    "graph_levels": GRAPH_LEVELS,
    "attention_pool_mode": "bottleneck_only",
    "use_bottleneck_attention": True,
    "graph_norm_type": "graphnorm",
    "node_feature_mode": "geometry",
    "graph_backend": "graphsage",
    "use_message_passing": True,
    "virtual_node_pool_mode": "learned",
    "bottleneck_virtual_node_pool_mode": "learned",
    "use_skip_graph": True,
    "init_features": 16,
    "depth": 5,
}

ABLATION_MODEL_KWARGS = {
    "v5_full_with_level_2": {
        **V5_FULL_KWARGS,
        "graph_levels": [2, 3, 4],
    },
    "v5_full_bigger_model": {
        **V5_FULL_KWARGS,
        "init_features": 24,
    },
    "v5_full_all_levels": {
        **V5_FULL_KWARGS,
        "attention_pool_mode": "all_levels",
    },
    "v5_full": {
        **V5_FULL_KWARGS,
    },
    "ablation_2_mlp_backend": {
        **V5_FULL_KWARGS,
        "graph_backend": "mlp",
    },
    "ablation_3_no_message_passing": {
        **V5_FULL_KWARGS,
        "use_message_passing": False,
    },
    "ablation_4_no_bottleneck_attention": {
        **V5_FULL_KWARGS,
        "use_bottleneck_attention": False,
    },
    "ablation_5_no_norm": {
        **V5_FULL_KWARGS,
        "graph_norm_type": "none",
    },
    "ablation_6_batchnorm": {
        **V5_FULL_KWARGS,
        "graph_norm_type": "batchnorm",
    },
    "ablation_7_mean_virtual_node_pool": {
        **V5_FULL_KWARGS,
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
    },
    "ablation_8_graph_only_bottleneck": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
    },
    "ablation_9_no_skip_graph": {
        **V5_FULL_KWARGS,
        "use_skip_graph": False,
    },
    "ablation_10_learned_station_embedding_only": {
        **V5_FULL_KWARGS,
        "node_feature_mode": "learned_station_embedding",
        "station_embedding_dim": 3,
    },
    "only_bottleneck_attention": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "use_bottleneck_attention": True,
        "use_skip_graph": False,
        "use_message_passing": False,
    },
    "only_graph_no_attention": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "use_bottleneck_attention": False,
        "graph_norm_type": "none",
    },
    "leaner_model": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "graph_norm_type": "none",
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
        "use_skip_graph": False,
    },
    "leanest_model": {
        **V5_FULL_KWARGS,
        "graph_levels": [],
        "graph_norm_type": "none",
        "virtual_node_pool_mode": "mean",
        "bottleneck_virtual_node_pool_mode": "mean",
        "use_skip_graph": False,
        "node_feature_mode": "learned_station_embedding",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune all ablation checkpoints on cross-volcano percent-train artifacts "
            "under two protocols, then evaluate on test_80."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Ablation experiment root (default: complete_experiment).",
    )
    parser.add_argument(
        "--cross-data-root",
        type=Path,
        default=DEFAULT_CROSS_DATA_ROOT,
        help="Cross-volcano root with <VOLCANO>/train_*pct.npz and test_80.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output folder for fine-tune reports. Defaults to "
            "<experiment-root>/finetune_cross_volcano."
        ),
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=None,
        help="Comma-separated ablation names. Default: all discovered under ablations/.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated target volcano names. Default: all folders with required artifacts.",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default=None,
        help=(
            "Comma-separated protocol keys. Default: all protocols. "
            "Allowed: protocol_a_all_weights, protocol_b_decoder_only"
        ),
    )
    parser.add_argument(
        "--allow-missing-folds",
        action="store_true",
        help="Skip missing source fold checkpoint files instead of failing.",
    )
    parser.add_argument(
        "--save-confusion-matrices",
        action="store_true",
        help="Save validation/test confusion matrices for each fine-tune run.",
    )
    return parser.parse_args()


def load_base_manifest(experiment_root: Path) -> dict:
    manifest_path = experiment_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Base run manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_ablations(ablations_root: Path) -> list[str]:
    if not ablations_root.exists():
        raise FileNotFoundError(f"Ablations root not found: {ablations_root}")
    return sorted([p.name for p in ablations_root.iterdir() if p.is_dir()])


def evaluate_graphsage(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], float, list[float], float, list[float], float, float, np.ndarray]:
    (
        f1_per_class,
        mean_f1,
        iou_per_class,
        mean_iou,
        iou_all_classes,
        mean_iou_all,
        eval_loss,
        cm,
    ) = compute_event_f1_iou_graphsage(
        model,
        loader,
        device,
        return_cm=True,
        return_val_loss=True,
        return_event_plot_payloads=False,
        save_event_plots=False,
        max_event_plots=0,
        epoch=None,
    )

    return (
        [float(x) for x in f1_per_class],
        float(mean_f1),
        [float(x) for x in iou_per_class],
        float(mean_iou),
        [float(x) for x in iou_all_classes],
        float(mean_iou_all),
        float(eval_loss),
        cm,
    )


def write_reports(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(
        output_dir / "finetune_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    grouped = fold_df.groupby(
        ["ablation", "target_volcano", "train_pct", "protocol"],
        sort=True,
    )

    summary_rows = []
    per_class_f1_rows = []
    per_class_iou_rows = []

    for (ablation, target, train_pct, protocol), grp in grouped:
        summary_row = {
            "ablation": str(ablation),
            "target_volcano": str(target),
            "train_pct": str(train_pct),
            "protocol": str(protocol),
            "n_folds": int(len(grp)),
        }

        for metric_col in [
            "test_mean_f1",
            "test_mean_iou",
            "test_mean_iou_all",
            "test_loss",
            "best_val_mean_f1",
            "best_val_mean_iou",
            "best_epoch",
        ]:
            s = compute_summary(grp[metric_col].astype(float).tolist())
            summary_row[f"{metric_col}_mean"] = float(s["mean"])
            summary_row[f"{metric_col}_std"] = float(s["std"])

        per_class_f1_row = {
            "ablation": str(ablation),
            "target_volcano": str(target),
            "train_pct": str(train_pct),
            "protocol": str(protocol),
        }
        per_class_iou_row = {
            "ablation": str(ablation),
            "target_volcano": str(target),
            "train_pct": str(train_pct),
            "protocol": str(protocol),
        }

        for class_name in CLASS_NAMES:
            f1_col = f"test_f1_{class_name}"
            iou_col = f"test_iou_{class_name}"
            f1_s = compute_summary(grp[f1_col].astype(float).tolist())
            iou_s = compute_summary(grp[iou_col].astype(float).tolist())

            summary_row[f"{f1_col}_mean"] = float(f1_s["mean"])
            summary_row[f"{f1_col}_std"] = float(f1_s["std"])
            summary_row[f"{iou_col}_mean"] = float(iou_s["mean"])
            summary_row[f"{iou_col}_std"] = float(iou_s["std"])

            per_class_f1_row[f"{class_name}_mean"] = float(f1_s["mean"])
            per_class_f1_row[f"{class_name}_std"] = float(f1_s["std"])
            per_class_iou_row[f"{class_name}_mean"] = float(iou_s["mean"])
            per_class_iou_row[f"{class_name}_std"] = float(iou_s["std"])

        summary_rows.append(summary_row)
        per_class_f1_rows.append(per_class_f1_row)
        per_class_iou_rows.append(per_class_iou_row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["target_volcano", "train_pct", "protocol", "test_mean_f1_mean"],
        ascending=[True, True, True, False],
    )
    summary_df.to_csv(
        output_dir / "finetune_summary.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_f1_rows).sort_values(
        by=["target_volcano", "train_pct", "protocol", "ablation"],
        ascending=[True, True, True, True],
    ).to_csv(
        output_dir / "finetune_per_class_f1.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    pd.DataFrame(per_class_iou_rows).sort_values(
        by=["target_volcano", "train_pct", "protocol", "ablation"],
        ascending=[True, True, True, True],
    ).to_csv(
        output_dir / "finetune_per_class_iou.csv",
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    for target in sorted(summary_df["target_volcano"].unique().tolist()):
        target_df = summary_df[summary_df["target_volcano"] == target].copy()
        for train_pct in sorted(target_df["train_pct"].unique().tolist()):
            pct_df = target_df[target_df["train_pct"] == train_pct].copy()
            for protocol in sorted(pct_df["protocol"].unique().tolist()):
                block = pct_df[pct_df["protocol"] == protocol].copy()
                out_subdir = comparisons_dir / target / train_pct / protocol
                out_subdir.mkdir(parents=True, exist_ok=True)

                block.sort_values(by="test_mean_f1_mean", ascending=False).to_csv(
                    out_subdir / "rank_by_mean_f1.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )
                block.sort_values(by="test_mean_iou_mean", ascending=False).to_csv(
                    out_subdir / "rank_by_mean_iou.csv",
                    index=False,
                    encoding="utf-8-sig",
                    sep=";",
                    decimal=",",
                )


def main() -> None:
    args = parse_args()

    experiment_root = resolve_project_path(args.experiment_root, PROJECT_ROOT)
    cross_data_root = resolve_project_path(args.cross_data_root, PROJECT_ROOT)

    output_dir = (
        resolve_project_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir is not None
        else (experiment_root / DEFAULT_OUTPUT_NAME)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = load_base_manifest(experiment_root)
    base_config = dict(base_manifest.get("config", {}))
    manifest_kwargs_map = dict(base_manifest.get("ablation_model_kwargs", {}))

    # Priority: manifest kwargs (reproducibility) > local fallback kwargs.
    model_kwargs_map = dict(ABLATION_MODEL_KWARGS)
    model_kwargs_map.update(manifest_kwargs_map)

    required_cfg = ["epochs", "early_stop_patience", "lr", "lr_final", "dice_weight", "ce_weight", "seed", "batch_size"]
    missing_cfg = [k for k in required_cfg if k not in base_config]
    if len(missing_cfg) > 0:
        raise KeyError(f"Missing config keys in base manifest: {missing_cfg}")

    ablations_root = experiment_root / "ablations"
    discovered_ablations = discover_ablations(ablations_root)
    selected_ablations = parse_csv_selection(args.ablations, discovered_ablations, "ablations")

    missing_in_manifest = sorted(set(selected_ablations) - set(manifest_kwargs_map.keys()))
    if len(missing_in_manifest) > 0:
        print(
            "[WARN] Using local fallback kwargs for ablations missing in base manifest: "
            f"{missing_in_manifest}"
        )

    unknown_kwargs = sorted(set(selected_ablations) - set(model_kwargs_map.keys()))
    if len(unknown_kwargs) > 0:
        raise KeyError(
            "Selected ablations missing model kwargs in both base manifest and local fallback map: "
            f"{unknown_kwargs}"
        )

    discovered_targets = discover_targets(
        cross_data_root,
        required_files=PERCENT_ARTIFACTS + ["test_80.npz"],
    )
    selected_targets = parse_csv_selection(args.targets, discovered_targets, "targets")

    selected_protocols = parse_csv_selection(
        args.protocols,
        sorted(PROTOCOL_SPECS.keys()),
        "protocols",
    )

    seed = int(base_config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "experiment_root": str(experiment_root),
        "cross_data_root": str(cross_data_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "val_split_frac": 0.15,
        "percent_artifacts": PERCENT_ARTIFACTS,
        "protocols": {k: PROTOCOL_SPECS[k] for k in selected_protocols},
        "selected_ablations": selected_ablations,
        "selected_targets": selected_targets,
        "folds": [int(f) for f in FOLDS],
        "base_config": base_config,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 90)
    print("Fine-tune cross-volcano ablations")
    print(f"Experiment root: {experiment_root}")
    print(f"Cross-data root: {cross_data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Ablations ({len(selected_ablations)}): {selected_ablations}")
    print(f"Targets ({len(selected_targets)}): {selected_targets}")
    print(f"Protocols ({len(selected_protocols)}): {selected_protocols}")
    print(f"Device: {device}")
    print("=" * 90)

    rows: list[dict] = []

    for ablation_name in selected_ablations:
        print(f"\n[ABLATION] {ablation_name}")
        ablation_root = ablations_root / ablation_name
        model_kwargs = dict(model_kwargs_map[ablation_name])

        batch_size = int(BATCH_SIZE_OVERRIDES.get(ablation_name, int(base_config["batch_size"])))
        epochs = int(base_config["epochs"])
        early_stop_patience = int(base_config["early_stop_patience"])
        lr = float(base_config["lr"])
        lr_final = float(base_config["lr_final"])
        dice_weight = float(base_config["dice_weight"])
        ce_weight = float(base_config["ce_weight"])

        for fold_id in FOLDS:
            source_ckpt = checkpoint_path_for_fold(ablation_root, fold_id)
            if not source_ckpt.exists():
                msg = f"Missing source checkpoint: {source_ckpt}"
                if args.allow_missing_folds:
                    print(f"[WARN] {msg}")
                    continue
                raise FileNotFoundError(msg)

            for target_name in selected_targets:
                target_root = cross_data_root / target_name
                test_npz_path = target_root / "test_80.npz"
                if not test_npz_path.exists():
                    raise FileNotFoundError(f"Missing target test artifact: {test_npz_path}")

                test_ds = GraphSAGEDataset(test_npz_path)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                for pct_name in PERCENT_ARTIFACTS:
                    train_npz_path = target_root / pct_name
                    if not train_npz_path.exists():
                        raise FileNotFoundError(f"Missing fine-tune artifact: {train_npz_path}")

                    full_train_ds = GraphSAGEDataset(train_npz_path)
                    train_idx, val_idx = split_indices_stratified(
                        label_ids=full_train_ds.label_ids,
                        val_frac=0.15,
                        seed=seed + int(fold_id),
                    )

                    train_subset = Subset(full_train_ds, train_idx.tolist())
                    val_subset = Subset(full_train_ds, val_idx.tolist())

                    train_labels = np.asarray(full_train_ds.label_ids)[train_idx]
                    train_sampler = BalancedBatchSampler(train_labels, batch_size=batch_size)

                    train_loader = DataLoader(train_subset, batch_sampler=train_sampler)
                    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

                    for protocol_key in selected_protocols:
                        protocol_spec = PROTOCOL_SPECS[protocol_key]

                        run_out_dir = (
                            output_dir
                            / ablation_name
                            / f"fold_{fold_id:02d}"
                            / target_name
                            / pct_name.replace(".npz", "")
                            / protocol_key
                        )
                        ckpt_dir = run_out_dir / "checkpoints"
                        reports_dir = run_out_dir / "reports"
                        cm_dir = run_out_dir / "confusion_matrices"
                        for p in [ckpt_dir, reports_dir, cm_dir]:
                            p.mkdir(parents=True, exist_ok=True)

                        model = UNet_GraphSAGE(in_channels=1, out_channels=6, **model_kwargs).to(device)
                        source_state = torch.load(source_ckpt, map_location=device, weights_only=False)
                        model.load_state_dict(source_state["model_state_dict"])

                        apply_finetune_protocol(model, protocol_key)
                        trainable_count, total_count = trainable_parameter_count(model)

                        optimizer = optim.Adam(
                            [p for p in model.parameters() if p.requires_grad],
                            lr=lr,
                        )
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=max(1, int(epochs / 4)),
                            eta_min=lr_final,
                        )

                        best_val_mean_f1 = float("-inf")
                        best_val_mean_iou = float("-inf")
                        best_epoch = -1
                        epochs_without_improvement = 0

                        metrics_rows = []
                        run_start = time.time()

                        print(
                            "  "
                            f"fold={fold_id:02d} target={target_name} pct={pct_name} "
                            f"protocol={protocol_key} "
                            f"n_train={len(train_subset)} n_val={len(val_subset)} n_test={len(test_ds)} "
                            f"trainable={trainable_count}/{total_count}"
                        )

                        for epoch in range(epochs):
                            model.train()
                            train_loss = 0.0

                            for xb, y_onehot, _y_label in train_loader:
                                xb = xb.to(device)
                                y_onehot = y_onehot.to(device)

                                optimizer.zero_grad(set_to_none=True)
                                out = model(xb)
                                loss, dice_component, ce_component = combined_dice_ce_loss(
                                    out,
                                    y_onehot,
                                    class_weights=None,
                                    dice_weight=dice_weight,
                                    ce_weight=ce_weight,
                                )
                                loss.backward()
                                optimizer.step()

                                train_loss += float(loss.item())
                                del xb, y_onehot, out, loss, dice_component, ce_component

                            scheduler.step()

                            (
                                val_f1_per_class,
                                val_mean_f1,
                                val_iou_per_class,
                                val_mean_iou,
                                val_iou_all_classes,
                                val_mean_iou_all,
                                val_loss,
                                val_cm,
                            ) = evaluate_graphsage(model=model, loader=val_loader, device=device)

                            improved = float(val_mean_f1) > float(best_val_mean_f1)
                            if improved:
                                best_val_mean_f1 = float(val_mean_f1)
                                best_val_mean_iou = float(val_mean_iou)
                                best_epoch = int(epoch)
                                epochs_without_improvement = 0
                                torch.save(
                                    {
                                        "epoch": int(epoch),
                                        "model_state_dict": model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "val_loss": float(val_loss),
                                        "val_mean_f1": float(val_mean_f1),
                                        "protocol": protocol_key,
                                    },
                                    ckpt_dir / "best_finetune_f1.pt",
                                )
                            else:
                                epochs_without_improvement += 1

                            metrics_rows.append(
                                {
                                    "epoch": int(epoch),
                                    "lr": float(optimizer.param_groups[0]["lr"]),
                                    "train_loss": float(train_loss),
                                    "val_loss": float(val_loss),
                                    "val_mean_f1": float(val_mean_f1),
                                    "val_mean_iou": float(val_mean_iou),
                                    "val_mean_iou_all": float(val_mean_iou_all),
                                    "best_epoch": int(best_epoch),
                                    "no_improve": int(epochs_without_improvement),
                                    "val_f1_VT": float(val_f1_per_class[0]),
                                    "val_f1_LP": float(val_f1_per_class[1]),
                                    "val_f1_TR": float(val_f1_per_class[2]),
                                    "val_f1_AV": float(val_f1_per_class[3]),
                                    "val_f1_IC": float(val_f1_per_class[4]),
                                    "val_iou_VT": float(val_iou_per_class[0]),
                                    "val_iou_LP": float(val_iou_per_class[1]),
                                    "val_iou_TR": float(val_iou_per_class[2]),
                                    "val_iou_AV": float(val_iou_per_class[3]),
                                    "val_iou_IC": float(val_iou_per_class[4]),
                                    "val_iou_all_BG": float(val_iou_all_classes[0]),
                                    "val_iou_all_VT": float(val_iou_all_classes[1]),
                                    "val_iou_all_LP": float(val_iou_all_classes[2]),
                                    "val_iou_all_TR": float(val_iou_all_classes[3]),
                                    "val_iou_all_AV": float(val_iou_all_classes[4]),
                                    "val_iou_all_IC": float(val_iou_all_classes[5]),
                                }
                            )

                            if args.save_confusion_matrices:
                                save_confusion_matrix_image(
                                    cm=val_cm,
                                    labels=CLASS_NAMES,
                                    out_path=cm_dir / f"val_cm_epoch_{epoch:03d}.png",
                                    title=(
                                        f"Val CM | {ablation_name} | fold {fold_id:02d} | "
                                        f"{target_name} | {pct_name} | {protocol_key} | epoch {epoch:03d}"
                                    ),
                                )

                            del val_cm
                            cleanup_gpu_cache()

                            if epochs_without_improvement >= early_stop_patience:
                                break

                        pd.DataFrame(metrics_rows).to_csv(
                            reports_dir / "finetune_training_metrics.csv",
                            index=False,
                            encoding="utf-8-sig",
                            sep=";",
                            decimal=",",
                        )

                        best_ckpt_path = ckpt_dir / "best_finetune_f1.pt"
                        if not best_ckpt_path.exists():
                            raise RuntimeError(
                                "Fine-tune produced no best checkpoint: "
                                f"{best_ckpt_path}"
                            )

                        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                        model.load_state_dict(best_ckpt["model_state_dict"])
                        model.eval()

                        (
                            test_f1_per_class,
                            test_mean_f1,
                            test_iou_per_class,
                            test_mean_iou,
                            test_iou_all_classes,
                            test_mean_iou_all,
                            test_loss,
                            test_cm,
                        ) = evaluate_graphsage(model=model, loader=test_loader, device=device)

                        if args.save_confusion_matrices:
                            save_confusion_matrix_image(
                                cm=test_cm,
                                labels=CLASS_NAMES,
                                out_path=cm_dir / "test_cm_best_finetune_f1.png",
                                title=(
                                    f"Test CM | {ablation_name} | fold {fold_id:02d} | "
                                    f"{target_name} | {pct_name} | {protocol_key} | best_finetune_f1"
                                ),
                            )

                        elapsed = float(time.time() - run_start)

                        row = {
                            "ablation": ablation_name,
                            "fold": int(fold_id),
                            "target_volcano": target_name,
                            "train_pct": pct_name.replace(".npz", ""),
                            "protocol": protocol_key,
                            "protocol_display_name": protocol_spec["display_name"],
                            "source_checkpoint": str(source_ckpt),
                            "best_finetune_checkpoint": str(best_ckpt_path),
                            "n_train": int(len(train_subset)),
                            "n_val": int(len(val_subset)),
                            "n_test": int(len(test_ds)),
                            "trainable_params": int(trainable_count),
                            "total_params": int(total_count),
                            "best_epoch": int(best_epoch),
                            "best_val_mean_f1": float(best_val_mean_f1),
                            "best_val_mean_iou": float(best_val_mean_iou),
                            "test_loss": float(test_loss),
                            "test_mean_f1": float(test_mean_f1),
                            "test_mean_iou": float(test_mean_iou),
                            "test_mean_iou_all": float(test_mean_iou_all),
                            "elapsed_seconds": float(elapsed),
                        }

                        for idx, class_name in enumerate(CLASS_NAMES):
                            row[f"test_f1_{class_name}"] = float(test_f1_per_class[idx])
                            row[f"test_iou_{class_name}"] = float(test_iou_per_class[idx])

                        row["test_iou_all_BG"] = float(test_iou_all_classes[0])
                        row["test_iou_all_VT"] = float(test_iou_all_classes[1])
                        row["test_iou_all_LP"] = float(test_iou_all_classes[2])
                        row["test_iou_all_TR"] = float(test_iou_all_classes[3])
                        row["test_iou_all_AV"] = float(test_iou_all_classes[4])
                        row["test_iou_all_IC"] = float(test_iou_all_classes[5])

                        rows.append(row)

                        with (reports_dir / "finetune_fold_summary.json").open("w", encoding="utf-8") as f:
                            json.dump(row, f, indent=2)

                        print(
                            "    "
                            f"best_epoch={best_epoch} "
                            f"val_f1={best_val_mean_f1:.4f} "
                            f"test_f1={test_mean_f1:.4f} test_iou={test_mean_iou:.4f}"
                        )

                        del test_cm, model, source_state, best_ckpt, optimizer, scheduler
                        cleanup_gpu_cache()

                    del train_loader, val_loader, train_subset, val_subset, full_train_ds
                    cleanup_gpu_cache()

                del test_loader, test_ds
                cleanup_gpu_cache()

    if len(rows) == 0:
        raise RuntimeError("No fine-tune evaluations executed. Check checkpoints and inputs.")

    write_reports(output_dir=output_dir, rows=rows)

    latest_dir = RESULTS_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "pointer_finetune_cross_volcano.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(output_dir),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print("=" * 90)
    print("Fine-tune cross-volcano evaluation complete")
    print(f"Fold metrics: {output_dir / 'finetune_fold_metrics.csv'}")
    print(f"Summary: {output_dir / 'finetune_summary.csv'}")
    print(f"Comparisons: {output_dir / 'comparisons'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
