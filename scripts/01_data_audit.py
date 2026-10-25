"""Audit the ODIR-5K dataset and write the clean image manifest.

No model is trained here. The script checks the schema of full_df.csv, builds
image-level labels from `labels`/`target`, matches every labeled row to an image
file by filename and patient ID, decodes and hashes the images, finds duplicates,
applies the exclusion policy and audits the legacy DenseNet121 embeddings.

Usage (from the project root):
    python scripts/01_data_audit.py [--config configs/config.yaml]

Outputs: outputs/data_audit/ (audit_report.json, audit_summary.txt, clean_manifest.csv, ...)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina.config import load_config, resolve_paths  # noqa: E402
from retina.data_audit import run_audit, summary_text  # noqa: E402
from retina.utils import library_versions, save_json, set_seed, setup_logging, timestamp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    log = setup_logging(paths.logs_dir / "01_data_audit.log")
    set_seed(int(cfg["seed"]))
    log.info("Config: %s", cfg["_config_path"])
    log.info("Python: %s", sys.executable)

    t0 = time.time()
    report = run_audit(cfg, paths)
    report["meta"] = {
        "timestamp": timestamp(),
        "runtime_seconds": round(time.time() - t0, 1),
        "seed": cfg["seed"],
        "config_path": cfg["_config_path"],
        "python_executable": sys.executable,
        "library_versions": library_versions(),
    }
    save_json(report, paths.audit_dir / "audit_report.json")
    text = summary_text(report)
    (paths.audit_dir / "audit_summary.txt").write_text(text, encoding="utf-8")
    print()
    print(text)
    log.info("Audit finished in %.1f s. Outputs in %s", time.time() - t0, paths.audit_dir)


if __name__ == "__main__":
    main()
