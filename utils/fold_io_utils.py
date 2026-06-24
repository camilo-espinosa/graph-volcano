from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def checkpoint_path_for_fold(root: Path, fold_id: int, checkpoint_name: str = "best_f1.pt") -> Path:
    return root / f"fold_{fold_id:02d}" / "checkpoints" / checkpoint_name


def load_fold_summary(
    root: Path,
    fold_id: int,
    candidate_relative_paths: list[Path] | None = None,
) -> dict | None:
    fold_dir = root / f"fold_{fold_id:02d}"
    candidates = (
        list(candidate_relative_paths)
        if candidate_relative_paths is not None
        else [Path("reports") / "fold_summary.json", Path("fold_summary.json")]
    )

    for rel_path in candidates:
        summary_path = fold_dir / rel_path
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_training_fold_summary(root: Path, fold_id: int) -> dict | None:
    return load_fold_summary(
        root=root,
        fold_id=fold_id,
        candidate_relative_paths=[Path("reports") / "fold_summary.json"],
    )


def append_row_csv(csv_path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([[row.get(col, None) for col in fieldnames]], columns=fieldnames)
    row_df.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )


def load_completed_keys(csv_path: Path, id_columns: list[str]) -> set[tuple[Any, ...]]:
    if not csv_path.exists():
        return set()

    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    if not set(id_columns).issubset(df.columns):
        return set()

    keys: set[tuple[Any, ...]] = set()
    for _, row in df.iterrows():
        values: list[Any] = []
        valid = True
        for col in id_columns:
            value = row[col]
            if pd.isna(value):
                valid = False
                break
            if col == "fold":
                values.append(int(value))
            else:
                values.append(str(value))
        if valid:
            keys.add(tuple(values))

    return keys
