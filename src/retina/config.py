"""Configuration loading and path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class Paths:
    """Resolved absolute paths used throughout the project."""

    project_root: Path
    dataset_root: Path
    full_df_csv: Path
    odir_xlsx: Path
    image_dir: Path
    reference_image_dirs: dict[str, Path]
    legacy_embeddings_dir: Path | None
    audit_dir: Path
    splits_dir: Path
    logs_dir: Path
    figures_dir: Path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML configuration file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def _under(root: Path, value: str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def resolve_paths(cfg: dict[str, Any]) -> Paths:
    """Turn configuration strings into absolute paths and create output folders."""
    data = cfg["data"]
    dataset_root = Path(os.environ.get("RETINA_DATA_ROOT", data["dataset_root"]))
    out = cfg["output"]

    paths = Paths(
        project_root=PROJECT_ROOT,
        dataset_root=dataset_root,
        full_df_csv=_under(dataset_root, data["full_df_csv"]),
        odir_xlsx=_under(dataset_root, data.get("odir_xlsx")),
        image_dir=_under(dataset_root, data["image_dir"]),
        reference_image_dirs={
            name: _under(dataset_root, rel)
            for name, rel in (data.get("reference_image_dirs") or {}).items()
        },
        legacy_embeddings_dir=_under(dataset_root, data.get("legacy_embeddings_dir")),
        audit_dir=_under(PROJECT_ROOT, out["audit_dir"]),
        splits_dir=_under(PROJECT_ROOT, out["splits_dir"]),
        logs_dir=_under(PROJECT_ROOT, out["logs_dir"]),
        figures_dir=_under(PROJECT_ROOT, out["figures_dir"]),
    )
    for folder in (paths.audit_dir, paths.splits_dir, paths.logs_dir, paths.figures_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return paths
