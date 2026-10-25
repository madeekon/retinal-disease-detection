"""Final project: collect the saved results of all experiments (no training) into one comparison.

Reads Assignment 03 (outputs/metrics), Assignment 04 (outputs/assignment04) and Final Project stages 1-2
(outputs/final_project/...) and writes:
  outputs/final_project/overall_model_comparison.csv   main models (selected configurations)
  outputs/final_project/all_classifier_runs_table.csv  every classifier configuration
  outputs/final_project/model_comparison_f1_auc.png    F1 and ROC-AUC chart
  outputs/final_project/final_project_summary.md       methods, results, findings, limitations, reproducibility

Usage (from the project root):  python scripts/10_final_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from retina.utils import library_versions, timestamp  # noqa: E402

FP = ROOT / "outputs" / "final_project"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)


def agg(df: pd.DataFrame, name: str, family: str, source: str, inputs: str) -> dict:
    """mean/std over runs of the test rows of one model."""
    t = df[df["split"] == "test"]
    row = {"model": name, "family": family, "source": source, "input": inputs, "n_runs": len(t)}
    for m in METRICS:
        row[f"{m}_mean"] = t[m].mean()
        row[f"{m}_std"] = t[m].std() if len(t) > 1 else 0.0
    for c in ("tn", "fp", "fn", "tp"):
        row[f"{c}_mean"] = t[c].mean()
    return row


def assignment03_rows() -> list[dict]:
    comp = load(ROOT / "outputs" / "metrics" / "comparison.csv")
    preds = load(ROOT / "outputs" / "metrics" / "test_predictions.csv")
    names = {"logreg": ("LogReg (DenseNet121 embeddings)", "Logistic Regression", "Classical ML",
                        "1024-d embeddings"),
             "cnn_no_aug": ("DenseNet121 frozen, no augmentation", "DenseNet121 CNN (frozen backbone)",
                            "Advanced CNN", "224x224 images"),
             "cnn_aug": ("DenseNet121 frozen, moderate augmentation", "DenseNet121 CNN + augmentation",
                         "Advanced CNN", "224x224 images")}
    rows = []
    for key, (orig, name, family, inputs) in names.items():
        t = comp[(comp["model"] == orig) & (comp["split"] == "test")].copy()
        t["roc_auc"] = roc_auc_score(preds["binary_target"], preds[f"prob_{key}"])  # from saved probabilities
        rows.append(agg(t, name, family, "Assignment 03", inputs))
    return rows


def main() -> None:
    # ---------------- collect ----------------
    a03 = assignment03_rows()
    base = load(FP / "baselines" / "metrics_runs.csv")
    mlp = load(FP / "mlp" / "metrics_runs.csv")
    mlp_sel = json.loads((FP / "mlp" / "results.json").read_text(encoding="utf-8"))["selected_config_by_mean_val_f1"]
    opt = load(FP / "optimizers" / "metrics_runs.csv")
    cpn = load(FP / "cpn" / "metrics_runs.csv")
    emb = "1024-d embeddings"
    main_rows = [a03[0],
                 agg(base[base.model == "SVM-RBF"], "SVM-RBF", "Classical ML", "Final stage 1", emb),
                 agg(base[base.model == "RandomForest"], "Random Forest", "Classical ML", "Final stage 1", emb),
                 agg(mlp[mlp.config == mlp_sel], f"MLP ({mlp_sel.split(': ')[1]})", "Neural network",
                     "Final stage 1", emb),
                 agg(cpn, "Counterpropagation Network (144 units)", "Neural network", "Final stage 2",
                     "PCA-32 of embeddings"),
                 a03[1], a03[2]]
    overall = pd.DataFrame(main_rows)
    all_rows = main_rows[:1] + [agg(base[base.model == m], m, "Classical ML", "Final stage 1", emb)
                                for m in base.model.unique()]
    all_rows += [agg(mlp[mlp.config == c], f"MLP {c}", "Neural network", "Final stage 1", emb) for c in mlp.config.unique()]
    all_rows += [agg(opt[opt.method == m], f"MLP optimizer: {m}", "Neural network", "Final stage 1", emb)
                 for m in opt.method.unique()]
    all_rows += [main_rows[4], a03[1], a03[2]]
    every = pd.DataFrame(all_rows)

    def fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["model", "family", "source", "input", "n_runs"]].copy()
        for m in METRICS:
            out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
        out["false_negatives"] = df["fn_mean"].round(1)
        out["false_positives"] = df["fp_mean"].round(1)
        return out

    FP.mkdir(parents=True, exist_ok=True)
    pd.concat([fmt(overall), overall[[f"{m}_mean" for m in METRICS]]], axis=1).to_csv(
        FP / "overall_model_comparison.csv", index=False)
    fmt(every).to_csv(FP / "all_classifier_runs_table.csv", index=False)

    # ---------------- chart ----------------
    o = overall.sort_values("f1_mean")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, m, title, color in ((axes[0], "f1", "F1 (Abnormal class)", "#2a78d6"),
                                (axes[1], "roc_auc", "ROC-AUC", "#eb6834")):
        ax.barh(o["model"], o[f"{m}_mean"], xerr=o[f"{m}_std"], color=color, capsize=3, height=0.6)
        for y, (v, s) in enumerate(zip(o[f"{m}_mean"], o[f"{m}_std"])):
            ax.text(v + s + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
        ax.set(xlim=(0.5, max(0.85, (o[f"{m}_mean"] + o[f"{m}_std"]).max() + 0.06)), title=title, xlabel="Test set (mean ± std over seeds where repeated)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Retinal disease detection (Normal vs Abnormal), ODIR-5K test set: model comparison")
    fig.tight_layout()
    fig.savefig(FP / "model_comparison_f1_auc.png", dpi=150)
    plt.close(fig)

    # ---------------- summary markdown ----------------
    audit = json.loads((ROOT / "outputs" / "data_audit" / "audit_report.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data" / "splits" / "split_summary.json").read_text(encoding="utf-8"))
    som = load(ROOT / "outputs" / "assignment04" / "metrics" / "som_comparison_table.csv")
    xai = json.loads((FP / "xai" / "results.json").read_text(encoding="utf-8"))
    cd = audit["clean_dataset"]
    best_f1 = overall.loc[overall.f1_mean.idxmax()]
    best_auc = overall.loc[overall.roc_auc_mean.idxmax()]
    emb_models = overall[overall.input == emb]
    lr, cnn, cnn_aug = overall.iloc[0], overall.iloc[5], overall.iloc[6]
    d_f1, d_auc = cnn.f1_mean - emb_models.f1_mean.max(), cnn.roc_auc_mean - emb_models.roc_auc_mean.max()
    cnn_vs = ("is on the same level as the best embedding-based models" if abs(d_f1) < 0.02 and abs(d_auc) < 0.02
              else "differs from the best embedding-based models")

    def md_table(df: pd.DataFrame) -> str:
        cols = ["model", "n_runs", *METRICS, "false_negatives"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in fmt(df).iterrows()]
        return "\n".join(lines)

    comp_rows = "\n".join(f"| {r['split']} | {r['images']} | {r['patients']} | {r['normal_0']} | {r['abnormal_1']} |"
                          for r in split["composition"])
    som_rows = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in som.itertuples(index=False))
    text = f"""# Final Project Summary

**Comparative Analysis of Machine Learning and Neural Network Architectures for Retinal Disease Detection**

Generated by `scripts/10_final_summary.py` on {timestamp()} from saved result files only (no training).

## 1. Methods
* **Task:** binary classification of single fundus images, Normal (0) vs any disease or abnormality (1),
  using image-level labels (`labels`/`target` of `full_df.csv`), not the patient-level N..O flags.
* **Data:** ODIR-5K export, {cd['n_images']} labeled images of {cd['n_patients']} patients after the audit
  (abnormal fraction {cd['abnormal_fraction']:.3f}). Repeated retinas under different patient IDs were kept in
  one subset; 3 contradictory records were excluded.
* **Split:** patient-grouped 70/15/15, seed 42, saved once with SHA-256 checksums and reused by every experiment.
  Hyperparameters, epochs and configurations were chosen on the validation subset only; the test subset was used
  only for the final evaluation.
* **Features:** 1024-d DenseNet121 (ImageNet) embeddings, StandardScaler fitted on the training subset
  (PCA-32 fitted on train for SOM and CPN).
* **Models:** Logistic Regression; SVM-RBF; Random Forest; PyTorch MLP (4 configurations; optimizer comparison of
  full-batch GD, SGD batch 1, mini-batch GD, SGD with momentum, Adam under the same 60-epoch budget);
  Counterpropagation Network (Kohonen + Grossberg layers); DenseNet121 CNN with frozen backbone and trainable head,
  without and with moderate augmentation (early stopping on validation loss).
* **Unsupervised analysis:** Kohonen SOM with 3x3, 5x5, 8x8, 10x10 lattices (5 seeds each).
* **Explainability:** Grad-CAM on the frozen DenseNet121 for 12 test images (TP, TN, FP, FN).
* **Metrics:** accuracy, precision, recall, F1 (Abnormal = positive class), ROC-AUC, confusion matrices,
  threshold 0.5; mean ± std over 3 seeds where repeated.

## 2. Data split
| split | images | patients | normal | abnormal |
|---|---|---|---|---|
{comp_rows}

## 3. Results: main models (test set)
{md_table(overall)}

Full table of all configurations: `all_classifier_runs_table.csv`. Chart: `model_comparison_f1_auc.png`.

## 4. Results: SOM lattice size (Assignment 04)
| {' | '.join(som.columns)} |
|{'---|' * len(som.columns)}
{som_rows}

## 5. Results: Grad-CAM
* Probabilities re-computed for Grad-CAM matched the saved Assignment 03 predictions (max difference
  {xai['check_max_abs_diff_vs_saved_probability']:.4f}).
* Mean share of Grad-CAM heat outside the retinal disc: {xai['mean_heat_share_outside_fundus']:.3f}.
* Details: `xai/xai_summary.md`, figure `xai/gradcam_examples.png`.

## 6. Main findings
* Highest test F1: **{best_f1['model']}** ({best_f1['f1_mean']:.3f}); highest ROC-AUC: **{best_auc['model']}**
  ({best_auc['roc_auc_mean']:.3f}).
* Models trained on the same DenseNet121 embeddings (LogReg, SVM, Random Forest, MLP) lie within
  F1 {emb_models.f1_mean.min():.3f}-{emb_models.f1_mean.max():.3f} and ROC-AUC
  {emb_models.roc_auc_mean.min():.3f}-{emb_models.roc_auc_mean.max():.3f}: nonlinear classifiers add little over
  Logistic Regression (F1 {lr['f1_mean']:.3f}). The frozen ImageNet representation, not the classifier, limits
  performance.
* The frozen DenseNet121 CNN reached F1 {cnn['f1_mean']:.3f} and ROC-AUC {cnn['roc_auc_mean']:.3f} and {cnn_vs}
  (difference to the best of them: F1 {d_f1:+.3f}, ROC-AUC {d_auc:+.3f}); with a frozen backbone it is essentially
  another classifier on the same kind of features. Moderate augmentation vs no augmentation:
  F1 {cnn_aug['f1_mean']:.3f} vs {cnn['f1_mean']:.3f}, ROC-AUC {cnn_aug['roc_auc_mean']:.3f} vs {cnn['roc_auc_mean']:.3f}.
* The optimizer comparison showed similar final quality for all methods but different dynamics: full-batch GD
  needs many epochs and overfits least, SGD with batch 1 overfits fastest and is the slowest in wall-clock time,
  momentum and Adam reach their best validation loss within about 7-10 epochs.
* The CPN and the SOM show that the unsupervised cluster structure of the embeddings follows general image
  appearance more than disease status (CPN F1 {main_rows[4]['f1_mean']:.3f}; SOM label agreement 0.56-0.59 against
  a 0.55 majority baseline). Larger SOM lattices lower quantization error; 8x8 was the best balance.
* Grad-CAM shows that correct abnormal predictions focus on large central lesions, while errors are linked to dark
  or low-contrast photographs, small lesions and artefacts. Grad-CAM does not confirm pathology.

## 7. Limitations
* Single dataset (ODIR-5K) and a single held-out test set of 957 images; no external validation.
* Labels come from keywords of the dataset; some label noise is likely (e.g. normal-labelled images with visible
  spots, same retina labelled differently under two patient IDs).
* The CNN backbone was frozen and inputs were downscaled to 224x224; fine-tuning and higher resolution were not
  explored. Small lesions (microaneurysms, cup-to-disc changes) are hard to detect at this scale.
* Binary task only; the eight disease categories were merged into "Abnormal".
* Near-duplicate detection relies on embedding similarity thresholds chosen by visual review.
* Training times depend on the computer; part of the experiments ran on CPU, the CNNs on an RTX 3050 laptop GPU.

## 8. Reproducibility
* Seeds: 42 (split, single runs); 42, 43, 44 (repeated runs); 42-46 (SOM).
* Split files and checksums: `data/splits/split_summary.json` (seed {split['seed']}); every script verifies the
  checksums before use.
* Configuration: `configs/config.yaml`; per-experiment settings in each `results.json`.
* Library versions when this summary was generated: {library_versions(['numpy', 'pandas', 'sklearn', 'torch', 'torchvision', 'matplotlib'])}.
* Order of scripts: 01_data_audit, 02_make_splits, 03_experiments, 04_som_experiments, 05_final_baselines,
  06_final_mlp, 07_final_optimizers, 08_final_cpn, 09_final_xai, 10_final_summary; inference: 11_predict.
"""
    (FP / "final_project_summary.md").write_text(text, encoding="utf-8")
    print(fmt(overall).to_string(index=False))
    print("\nWritten:", FP / "overall_model_comparison.csv", FP / "model_comparison_f1_auc.png",
          FP / "final_project_summary.md", sep="\n  ")


if __name__ == "__main__":
    main()
