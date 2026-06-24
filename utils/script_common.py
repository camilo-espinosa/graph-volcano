from __future__ import annotations

from pathlib import Path


def resolve_project_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_csv_selection(raw_csv: str | None, available: list[str], name: str) -> list[str]:
    if raw_csv is None:
        return list(available)

    selected = [x.strip() for x in raw_csv.split(",") if x.strip()]
    unknown = sorted(set(selected) - set(available))
    if len(unknown) > 0:
        raise ValueError(f"Unknown {name}: {unknown}. Available: {available}")
    return selected


def discover_targets(
    cross_data_root: Path,
    required_files: list[str] | None = None,
) -> list[str]:
    if not cross_data_root.exists():
        raise FileNotFoundError(f"Cross-volcano root not found: {cross_data_root}")

    required = list(required_files) if required_files is not None else ["test_80.npz"]
    targets: list[str] = []

    for folder in sorted(cross_data_root.iterdir()):
        if not folder.is_dir():
            continue

        names = {p.name for p in folder.iterdir() if p.is_file()}
        if all(name in names for name in required):
            targets.append(folder.name)

    if len(targets) == 0:
        raise FileNotFoundError(
            f"No target folders with required files were found under: {cross_data_root}. "
            f"Required: {required}"
        )

    return targets
