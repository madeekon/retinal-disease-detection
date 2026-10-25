"""Patient-grouped, label-stratified train/validation/test split.

Unit of splitting = split group (a patient, or several patients merged because they
share an identical image file). Both eyes of a patient therefore always land in the
same subset. Groups are stratified by their composition (number of images and number
of abnormal images) so that every subset gets a similar mix of normal, abnormal and
mixed-eye patients.

The split is created ONCE and saved. Every experiment must load it with
`load_split()`, which verifies the SHA-256 checksums recorded at creation time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import load_json, save_json, sha256_file

log = logging.getLogger("retina")

SPLIT_NAMES = ("train", "val", "test")


def group_table(manifest: pd.DataFrame, group_col: str, min_stratum_size: int) -> pd.DataFrame:
    groups = manifest.groupby(group_col).agg(
        n_images=("image_id", "size"),
        n_abnormal=("binary_target", "sum"),
        n_patients=("patient_id", "nunique"),
    )
    groups["stratum"] = groups["n_images"].astype(str) + "img_" + groups["n_abnormal"].astype(str) + "abn"
    counts = groups["stratum"].value_counts()
    rare = counts[counts < min_stratum_size].index
    groups.loc[groups["stratum"].isin(rare), "stratum"] = "pooled_rare"
    # A pooled stratum that is still too small cannot be stratified; fold it into the largest one.
    if 0 < (groups["stratum"] == "pooled_rare").sum() < min_stratum_size:
        largest = groups.loc[groups["stratum"] != "pooled_rare", "stratum"].value_counts().idxmax()
        groups.loc[groups["stratum"] == "pooled_rare", "stratum"] = largest
    return groups


def make_split(manifest: pd.DataFrame, ratios: dict[str, float], seed: int,
               group_col: str = "split_group", min_stratum_size: int = 6) -> pd.DataFrame:
    """Return the manifest with an added `split` column."""
    total = ratios["train"] + ratios["val"] + ratios["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    groups = group_table(manifest, group_col, min_stratum_size)
    g_trainval, g_test = train_test_split(
        groups.index.to_numpy(), test_size=ratios["test"], random_state=seed,
        stratify=groups["stratum"].to_numpy())
    val_share = ratios["val"] / (ratios["train"] + ratios["val"])
    g_train, g_val = train_test_split(
        g_trainval, test_size=val_share, random_state=seed,
        stratify=groups.loc[g_trainval, "stratum"].to_numpy())
    assignment = {**{g: "train" for g in g_train}, **{g: "val" for g in g_val}, **{g: "test" for g in g_test}}
    out = manifest.copy()
    out["split"] = out[group_col].map(assignment)
    return out


def verify_split(df: pd.DataFrame, group_col: str = "split_group",
                 linked_pairs: pd.DataFrame | None = None) -> dict[str, Any]:
    """Hard checks that must pass before the split may be used.

    linked_pairs: optional table with columns filename_a, filename_b (near duplicates that were merged);
    both images of every pair must be in the same subset.
    """
    checks = {
        "every_image_assigned": bool(df["split"].notna().all()),
        "image_ids_unique": bool(df["image_id"].is_unique),
        "filenames_unique": bool(df["filename"].is_unique),
        "groups_in_one_split": bool((df.groupby(group_col)["split"].nunique() == 1).all()),
        "patients_in_one_split": bool((df.groupby("patient_id")["split"].nunique() == 1).all()),
    }
    md5 = df[df["md5"].notna()] if "md5" in df.columns else df.iloc[0:0]
    checks["identical_files_in_one_split"] = bool((md5.groupby("md5")["split"].nunique() <= 1).all())
    if linked_pairs is not None and len(linked_pairs):
        where = df.set_index("filename")["split"]
        both = linked_pairs[linked_pairs["filename_a"].isin(where.index) & linked_pairs["filename_b"].isin(where.index)]
        separated = both[where.loc[both["filename_a"]].to_numpy() != where.loc[both["filename_b"]].to_numpy()]
        checks["near_duplicate_pairs_checked"] = int(len(both))
        checks["near_duplicate_pairs_in_one_split"] = bool(separated.empty)
    overlaps = {}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                pa = set(df.loc[df["split"] == a, "patient_id"])
                pb = set(df.loc[df["split"] == b, "patient_id"])
                overlaps[f"{a}&{b}"] = len(pa & pb)
    checks["patient_overlap_counts"] = overlaps
    checks["all_passed"] = all(v for k, v in checks.items() if isinstance(v, bool)) and not any(overlaps.values())
    return checks


def split_composition(df: pd.DataFrame, class_codes: list[str]) -> pd.DataFrame:
    rows = []
    n_all = len(df)
    p_all = df["patient_id"].nunique()
    for name in SPLIT_NAMES:
        s = df[df["split"] == name]
        rows.append({
            "split": name,
            "images": len(s),
            "image_share": len(s) / n_all,
            "patients": s["patient_id"].nunique(),
            "patient_share": s["patient_id"].nunique() / p_all,
            "split_groups": s["split_group"].nunique(),
            "normal_0": int((s["binary_target"] == 0).sum()),
            "abnormal_1": int((s["binary_target"] == 1).sum()),
            "abnormal_fraction": float(s["binary_target"].mean()),
            "left_eyes": int((s["eye"] == "left").sum()),
            "right_eyes": int((s["eye"] == "right").sum()),
            **{f"class_{c}": int((s["label_code"] == c).sum()) for c in class_codes},
        })
    return pd.DataFrame(rows)


def save_split(df: pd.DataFrame, splits_dir: Path, meta: dict[str, Any], force: bool = False) -> dict[str, Any]:
    summary_path = splits_dir / "split_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(
            f"{summary_path} already exists. Splits are created once and reused. "
            "Delete the folder content or pass --force only if you intend to replace them.")
    splits_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}.csv"
        df[df["split"] == name].drop(columns=["split"]).to_csv(path, index=False)
        files[name] = {"path": path.name, "sha256": sha256_file(path), "n_rows": int((df["split"] == name).sum())}
    all_path = splits_dir / "split_assignments.csv"
    df[["image_id", "filename", "patient_id", "split_group", "binary_target", "label_code", "split"]].to_csv(
        all_path, index=False)
    files["assignments"] = {"path": all_path.name, "sha256": sha256_file(all_path)}
    meta = {**meta, "files": files}
    save_json(meta, summary_path)
    return meta


def load_split(splits_dir: Path, name: str, verify: bool = True) -> pd.DataFrame:
    """Load one subset and confirm it has not changed since creation."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split '{name}'")
    path = splits_dir / f"{name}.csv"
    if verify:
        summary = load_json(splits_dir / "split_summary.json")
        expected = summary["files"][name]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{path} was modified after creation (sha256 mismatch).")
    return pd.read_csv(path)
