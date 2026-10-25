"""Final project, stage 1a: nonlinear classical baselines on the DenseNet121 embeddings.

SVM with RBF kernel and Random Forest, trained on the Assignment 03 training split
(StandardScaler fitted on train only). Hyperparameters are selected on validation F1;
the test set is used only for the final evaluation. Random Forest is repeated with 3 seeds
(SVM is deterministic). Also provides shared helpers used by scripts 06 and 07.

Usage (from the project root):  python scripts/05_final_baselines.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from retina.config import load_config, resolve_paths  # noqa: E402
from retina.splits import load_split  # noqa: E402
from retina.utils import library_versions, load_json, save_json, timestamp  # noqa: E402

FINAL = ROOT / "outputs" / "final_project"
SEEDS = [42, 43, 44]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# ----------------------------------------------------------------------------- shared helpers
def load_embeddings():
    """Standardized 1024-d embeddings and binary labels for train/val/test (scaler fit on train only)."""
    paths = resolve_paths(load_config())
    if not load_json(paths.audit_dir / "audit_report.json")["legacy_embeddings"].get("verified"):
        raise RuntimeError("Embeddings not verified by the Assignment 03 audit.")
    emb = np.load(paths.legacy_embeddings_dir / "embeddings.npy")
    splits = {k: load_split(paths.splits_dir, k) for k in ("train", "val", "test")}  # checksum-verified split
    X = {k: emb[d["row_index"].to_numpy()].astype(np.float32) for k, d in splits.items()}
    y = {k: d["binary_target"].to_numpy() for k, d in splits.items()}
    scaler = StandardScaler().fit(X["train"])
    X = {k: scaler.transform(v).astype(np.float32) for k, v in X.items()}
    return X, y, splits, scaler


def evaluate(y_true, score, threshold=0.5) -> dict:
    """score = probability of Abnormal (or decision value with threshold 0 for SVM)."""
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0), "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, score), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(runs: pd.DataFrame, by: str) -> pd.DataFrame:
    """mean ± std of test metrics per model/config."""
    test = runs[runs["split"] == "test"]
    agg = test.groupby(by, sort=False)[METRICS].agg(["mean", "std"])
    out = pd.DataFrame({by: agg.index, "n_runs": test.groupby(by, sort=False).size().to_numpy()})
    for m in METRICS:
        out[m] = [f"{a:.3f} ± {0 if np.isnan(b) else b:.3f}" for a, b in zip(agg[(m, "mean")], agg[(m, "std")])]
    return out


def plot_confusions(cms: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.9), squeeze=False)
    for ax, (name, m) in zip(axes[0], cms.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i, j in itertools.product(range(2), range(2)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"],
               xlabel="Predicted", ylabel="True")
        ax.set_title(name, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- experiments
def main() -> None:
    out = FINAL / "baselines"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    X, y, splits, scaler = load_embeddings()
    print("Train/val/test:", {k: len(v) for k, v in y.items()})

    # SVM-RBF: grid search on validation F1 (deterministic, one run)
    svm_grid = [{"C": c, "gamma": g} for c in (0.1, 1.0, 10.0) for g in ("scale", 1e-4)]
    search = []
    for p in svm_grid:
        t0 = time.perf_counter()
        m = SVC(kernel="rbf", **p).fit(X["train"], y["train"])
        search.append({"model": "SVM-RBF", **p, "val_f1": evaluate(y["val"], m.decision_function(X["val"]), 0)["f1"],
                       "fit_seconds": time.perf_counter() - t0})
        print("SVM", p, f"val F1 {search[-1]['val_f1']:.4f}")
    svm_best = max((r for r in search if r["model"] == "SVM-RBF"), key=lambda r: r["val_f1"])
    svm_params = {"C": svm_best["C"], "gamma": svm_best["gamma"]}

    # Random Forest: grid search on validation F1 with seed 42
    rf_grid = [{"max_features": mf, "min_samples_leaf": leaf} for mf in ("sqrt", 0.1) for leaf in (1, 5)]
    for p in rf_grid:
        t0 = time.perf_counter()
        m = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42, **p).fit(X["train"], y["train"])
        search.append({"model": "RandomForest", **p, "val_f1": evaluate(y["val"], m.predict_proba(X["val"])[:, 1])["f1"],
                       "fit_seconds": time.perf_counter() - t0})
        print("RF", p, f"val F1 {search[-1]['val_f1']:.4f}")
    rf_best = max((r for r in search if r["model"] == "RandomForest"), key=lambda r: r["val_f1"])
    rf_params = {"max_features": rf_best["max_features"], "min_samples_leaf": rf_best["min_samples_leaf"]}
    pd.DataFrame(search).to_csv(out / "hyperparameter_search_val.csv", index=False)

    # final models: SVM once (deterministic), RF with 3 seeds
    runs, cms, test_scores = [], {}, {}
    t0 = time.perf_counter()
    svm = SVC(kernel="rbf", **svm_params).fit(X["train"], y["train"])
    secs = time.perf_counter() - t0
    joblib.dump({"scaler": scaler, "model": svm}, out / "checkpoints" / "svm_rbf.joblib")
    for part in ("val", "test"):
        runs.append({"model": "SVM-RBF", "seed": "deterministic", "split": part, "fit_seconds": secs,
                     **evaluate(y[part], svm.decision_function(X[part]), 0)})
    cms["SVM-RBF"] = runs[-1]
    test_scores["svm_rbf_decision"] = svm.decision_function(X["test"])
    for seed in SEEDS:
        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed, **rf_params).fit(X["train"], y["train"])
        secs = time.perf_counter() - t0
        if seed == SEEDS[0]:
            joblib.dump({"scaler": scaler, "model": rf}, out / "checkpoints" / "random_forest_seed42.joblib")
            test_scores["random_forest_prob_seed42"] = rf.predict_proba(X["test"])[:, 1]
        for part in ("val", "test"):
            runs.append({"model": "RandomForest", "seed": seed, "split": part, "fit_seconds": secs,
                         **evaluate(y[part], rf.predict_proba(X[part])[:, 1])})
        if seed == SEEDS[0]:
            cms["Random Forest (seed 42)"] = runs[-1]

    runs = pd.DataFrame(runs)
    runs.to_csv(out / "metrics_runs.csv", index=False)
    summary = summarize(runs, "model")
    summary.to_csv(out / "summary_test.csv", index=False)
    pd.DataFrame({"filename": splits["test"]["filename"], "y_true": y["test"], **test_scores}).to_csv(
        out / "test_scores.csv", index=False)
    plot_confusions(cms, out / "confusion_matrices_test.png", "Classical nonlinear baselines (test set)")
    save_json({"timestamp": timestamp(), "features": "1024-d DenseNet121 embeddings, StandardScaler fit on train",
               "selection": "hyperparameters chosen by validation F1; test used once",
               "svm": {"kernel": "rbf", **svm_params, "threshold": "decision value 0"},
               "random_forest": {"n_estimators": 500, **rf_params, "seeds": SEEDS},
               "search": search, "runs": runs.to_dict(orient="records"), "summary_test": summary.to_dict(orient="records"),
               "library_versions": library_versions()}, out / "results.json")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
