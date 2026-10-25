"""Create the patient-grouped 70/15/15 train/validation/test split (seed 42).

Reads outputs/data_audit/clean_manifest.csv (run 01_data_audit.py first), assigns
whole split groups (patients) to subsets, verifies that no patient or identical
image appears in two subsets, and saves the manifests to data/splits/.
An existing split is reused if it was built from the current clean manifest; if the
manifest changed, the script stops (stale split) unless --force is given.

Usage (from the project root):
    python scripts/02_make_splits.py [--config configs/config.yaml] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retina.config import load_config, resolve_paths  # noqa: E402
from retina.data_audit import CLASS_CODES  # noqa: E402
from retina.splits import load_split, make_split, save_split, split_composition, verify_split  # noqa: E402
from retina.utils import library_versions, load_json, sha256_file, setup_logging, timestamp  # noqa: E402

NORMAL_COLOR = "#2a78d6"
ABNORMAL_COLOR = "#eb6834"


def plot_composition(comp: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(comp))
    w = 0.38
    ax = axes[0]
    b1 = ax.bar(x - w / 2 - 0.01, comp["normal_0"], w, color=NORMAL_COLOR, label="Normal (0)")
    b2 = ax.bar(x + w / 2 + 0.01, comp["abnormal_1"], w, color=ABNORMAL_COLOR, label="Abnormal (1)")
    for bars in (b1, b2):
        ax.bar_label(bars, padding=2, fontsize=9, color="#333333")
    ax.set_xticks(x, [f"{s}\n({n} patients)" for s, n in zip(comp["split"], comp["patients"])])
    ax.set_ylabel("Number of images")
    ax.set_title("Binary label per split")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(comp["split"], comp["abnormal_fraction"], marker="o", linestyle="none", markersize=9,
            color=ABNORMAL_COLOR)
    for xi, v in zip(comp["split"], comp["abnormal_fraction"]):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Abnormal fraction")
    ax.set_title("Abnormal fraction per split")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.25)
    fig.suptitle("Patient-grouped train / validation / test split (seed 42)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="Replace existing split files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    log = setup_logging(paths.logs_dir / "02_make_splits.log")
    scfg = cfg["split"]
    manifest_path = paths.audit_dir / "clean_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run scripts/01_data_audit.py first.")
    manifest = pd.read_csv(manifest_path)
    log.info("Clean manifest: %d images, %d patients, %d split groups", len(manifest),
             manifest["patient_id"].nunique(), manifest[scfg["group_column"]].nunique())

    summary_path = paths.splits_dir / "split_summary.json"
    if summary_path.exists() and not args.force:
        existing = load_json(summary_path)
        if existing.get("source_manifest_sha256") == sha256_file(manifest_path):
            for name in ("train", "val", "test"):
                load_split(paths.splits_dir, name)  # raises if a split file was modified
            log.info("Split already exists, matches the current clean manifest and passed checksum verification. "
                     "Reusing it unchanged.")
            return
        raise RuntimeError(
            "clean_manifest.csv changed after the split was created (different SHA-256). The existing split is "
            "stale. Recreate it with `python scripts/02_make_splits.py --force` ONLY if no experiment has used it yet.")

    pairs_path = paths.audit_dir / "near_duplicate_pairs.csv"
    linked = None
    if pairs_path.exists():
        nd = pd.read_csv(pairs_path)
        if len(nd):
            linked = nd[nd["merge_into_one_split_group"].astype(bool)]

    ratios = {k: float(scfg[k]) for k in ("train", "val", "test")}
    df = make_split(manifest, ratios, int(scfg["seed"]), scfg["group_column"], int(scfg["min_stratum_size"]))
    checks = verify_split(df, scfg["group_column"], linked)
    log.info("Verification: %s", checks)
    if not checks["all_passed"]:
        raise RuntimeError(f"Split verification failed: {checks}")

    comp = split_composition(df, CLASS_CODES)
    meta = {
        "timestamp": timestamp(),
        "seed": int(scfg["seed"]),
        "ratios_requested": ratios,
        "method": "sklearn.train_test_split on split groups (patients), stratified by group composition "
                  "(n_images x n_abnormal); test first, then validation from the remainder",
        "group_column": scfg["group_column"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "verification": checks,
        "composition": comp.to_dict(orient="records"),
        "library_versions": library_versions(["numpy", "pandas", "sklearn"]),
    }
    save_split(df, paths.splits_dir, meta, force=args.force)
    comp.to_csv(paths.splits_dir / "split_composition.csv", index=False)
    plot_composition(comp, paths.figures_dir / "split_composition.png")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print()
        print(comp[["split", "images", "image_share", "patients", "patient_share", "normal_0",
                    "abnormal_1", "abnormal_fraction", "left_eyes", "right_eyes"]].round(4).to_string(index=False))
        print()
        print(comp[["split"] + [f"class_{c}" for c in CLASS_CODES]].to_string(index=False))
    log.info("Saved split manifests to %s", paths.splits_dir)


if __name__ == "__main__":
    main()
