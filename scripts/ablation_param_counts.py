"""Instantiate all registry models and print/save parameter counts.

Usage:
    python scripts/ablation_param_counts.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_registry import build_model_from_spec, list_model_specs

RESULTS_ROOT = PROJECT_ROOT / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
DEFAULT_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "complete_experiment"
OUTPUT_FILENAME = "model_param_counts.csv"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    experiment_root = DEFAULT_EXPERIMENT_ROOT
    experiment_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    specs = list_model_specs(preserve_order=True)

    for model_key, spec in specs.items():
        model = build_model_from_spec(model_key)
        total, trainable = count_parameters(model)
        rows.append(
            {
                "model_key": model_key,
                "display_name": spec["display_name"],
                "family": spec["family"],
                "trainer_kind": spec["trainer_kind"],
                "total_params": total,
                "trainable_params": trainable,
            }
        )

    name_width = max(len("model_key"), *(len(str(row["model_key"])) for row in rows))
    total_width = len("total_params")
    trainable_width = len("trainable_params")

    header = (
        f"{'model_key':<{name_width}}  "
        f"{'total_params':>{total_width}}  "
        f"{'trainable_params':>{trainable_width}}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{str(row['model_key']):<{name_width}}  "
            f"{int(row['total_params']):>{total_width},}  "
            f"{int(row['trainable_params']):>{trainable_width},}"
        )

    output_path = experiment_root / OUTPUT_FILENAME
    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    print()
    print(f"Saved parameter counts to: {output_path}")


if __name__ == "__main__":
    main()
