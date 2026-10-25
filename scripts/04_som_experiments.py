"""Assignment 04, Variant 9: SOM lattice size and mapping quality.

Kohonen SOMs of size 3x3, 5x5, 8x8 and 10x10 are trained (unsupervised) on the verified
1024-d DenseNet121 embeddings of the Assignment 03 training split, after StandardScaler and
PCA(32) fitted on the training split only. Every map gets the same budget of sample
presentations and is trained with 5 seeds. Labels (Normal / Abnormal) are used only afterwards
to interpret the maps. Assignment 03 files are only read, never changed.

Usage (from the project root):  python scripts/04_som_experiments.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from retina.config import load_config, resolve_paths  # noqa: E402
from retina.splits import load_split  # noqa: E402
from retina.utils import library_versions, load_json, save_json, timestamp  # noqa: E402

SIZES = [(3, 3), (5, 5), (8, 8), (10, 10)]
SEEDS = [42, 43, 44, 45, 46]
PCA_COMPONENTS = 32
EPOCHS = 20                  # presentation budget = EPOCHS x n_train, identical for every map
LR_START, LR_END = 0.5, 0.01
SIGMA_END = 0.5              # sigma starts at max(rows, cols) / 2
OUT = ROOT / "outputs" / "assignment04"


class SOM:
    """Online Kohonen SOM on a rectangular lattice with a Gaussian neighbourhood."""

    def __init__(self, rows: int, cols: int, dim: int, seed: int):
        self.rows, self.cols, self.rng = rows, cols, np.random.default_rng(seed)
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)
        self.grid_d2 = ((self.grid[:, None, :] - self.grid[None, :, :]) ** 2).sum(-1)
        self.W = np.zeros((rows * cols, dim))

    def train(self, X: np.ndarray, n_steps: int) -> None:
        self.W = X[self.rng.choice(len(X), len(self.W), replace=False)].copy()  # init from data samples
        sigma0 = max(self.rows, self.cols) / 2
        order = np.concatenate([self.rng.permutation(len(X)) for _ in range(int(np.ceil(n_steps / len(X))))])
        for t in range(n_steps):
            x = X[order[t]]
            frac = t / n_steps
            lr = LR_START * (LR_END / LR_START) ** frac          # exponential decay
            sigma = sigma0 * (SIGMA_END / sigma0) ** frac
            bmu = np.argmin(((self.W - x) ** 2).sum(1))
            h = np.exp(-self.grid_d2[bmu] / (2 * sigma ** 2))
            self.W += lr * h[:, None] * (x - self.W)

    def distances(self, X: np.ndarray) -> np.ndarray:
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ self.W.T + (self.W ** 2).sum(1)[None, :]
        return np.sqrt(np.maximum(d2, 0))

    def evaluate(self, X: np.ndarray) -> dict:
        D = self.distances(X)
        best2 = np.argsort(D, axis=1)[:, :2]
        qe = D[np.arange(len(X)), best2[:, 0]].mean()
        # topographic error: best and second-best units are not direct (4-)neighbours on the lattice
        te = (self.grid_d2[best2[:, 0], best2[:, 1]] > 1).mean()
        hits = np.bincount(best2[:, 0], minlength=len(self.W))
        return {"qe": float(qe), "te": float(te), "empty_pct": float((hits == 0).mean() * 100),
                "bmu": best2[:, 0], "hits": hits}

    def u_matrix(self) -> np.ndarray:
        U = np.zeros(len(self.W))
        for k in range(len(self.W)):
            nb = np.where(self.grid_d2[k] == 1)[0]
            U[k] = np.linalg.norm(self.W[nb] - self.W[k], axis=1).mean()
        return U.reshape(self.rows, self.cols)


def numerical_example() -> dict:
    """One BMU selection and one weight update on a 3-neuron line lattice (2-D inputs)."""
    x = np.array([0.8, 0.2])
    W = np.array([[0.1, 0.9], [0.6, 0.4], [0.9, 0.9]])  # neurons at lattice positions 0, 1, 2
    d = np.linalg.norm(W - x, axis=1)
    bmu = int(np.argmin(d))
    lr, sigma = 0.5, 1.0
    h = np.exp(-((np.arange(3) - bmu) ** 2) / (2 * sigma ** 2))
    W_new = W + lr * h[:, None] * (x - W)
    return {"x": x.tolist(), "W": W.tolist(), "distances": d.round(4).tolist(), "bmu_index": bmu,
            "learning_rate": lr, "sigma": sigma, "neighbourhood_h": h.round(4).tolist(),
            "W_updated": W_new.round(4).tolist(),
            "formula": "w_j(t+1) = w_j(t) + lr * h_j * (x - w_j(t)),  h_j = exp(-d_grid(j, bmu)^2 / (2 sigma^2))"}


def load_data(cfg, paths):
    audit = load_json(paths.audit_dir / "audit_report.json")
    if not audit["legacy_embeddings"].get("verified"):
        raise RuntimeError("Embeddings not verified by the Assignment 03 audit.")
    emb = np.load(paths.legacy_embeddings_dir / "embeddings.npy")
    splits = {k: load_split(paths.splits_dir, k) for k in ("train", "val", "test")}
    X = {k: emb[d["row_index"].to_numpy()].astype(float) for k, d in splits.items()}
    y = {k: d["binary_target"].to_numpy() for k, d in splits.items()}
    scaler = StandardScaler().fit(X["train"])                     # train only
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42).fit(scaler.transform(X["train"]))  # train only
    Z = {k: pca.transform(scaler.transform(v)) for k, v in X.items()}
    return Z, y, float(pca.explained_variance_ratio_.sum())


def grid_plot(maps: dict, title: str, path: Path, cmap: str, cbar: str, vmin=None, vmax=None, annotate=False):
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 4))
    for ax, (name, M) in zip(axes, maps.items()):
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax)
        im.cmap.set_bad("#d9d9d9")
        if annotate and M.shape[0] <= 10:
            for (r, c), v in np.ndenumerate(M):
                ax.text(c, r, "" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=6)
        ax.set(title=name, xlabel="Lattice column", ylabel="Lattice row", xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, fraction=0.046, label=cbar)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    paths = resolve_paths(cfg)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    Z, y, pca_var = load_data(cfg, paths)
    n_steps = EPOCHS * len(Z["train"])
    print(f"Train/val/test: {len(Z['train'])}/{len(Z['val'])}/{len(Z['test'])}, PCA variance kept {pca_var:.3f}, "
          f"presentations per map {n_steps}")

    runs, keep = [], {}
    for rows, cols in SIZES:
        for seed in SEEDS:
            som = SOM(rows, cols, PCA_COMPONENTS, seed)
            t0 = time.perf_counter()
            som.train(Z["train"], n_steps)
            secs = time.perf_counter() - t0
            ev = {k: som.evaluate(Z[k]) for k in ("train", "test")}
            # label interpretation (after training): majority label of each unit from TRAIN hits -> test agreement
            ab = np.bincount(ev["train"]["bmu"], weights=y["train"], minlength=len(som.W))
            unit_label = np.where(ev["train"]["hits"] > 0, (ab / np.maximum(ev["train"]["hits"], 1)) >= 0.5,
                                  y["train"].mean() >= 0.5).astype(int)
            agree = float((unit_label[ev["test"]["bmu"]] == y["test"]).mean())
            runs.append({"size": f"{rows}x{cols}", "units": rows * cols, "seed": seed, "train_seconds": secs,
                         "qe_train": ev["train"]["qe"], "qe_test": ev["test"]["qe"],
                         "te_train": ev["train"]["te"], "te_test": ev["test"]["te"],
                         "empty_pct_train": ev["train"]["empty_pct"], "empty_pct_test": ev["test"]["empty_pct"],
                         "label_agreement_test": agree})
            print(f"{rows}x{cols} seed {seed}: QE {ev['train']['qe']:.3f}  TE {ev['train']['te']:.3f}  "
                  f"empty {ev['train']['empty_pct']:.1f}%  {secs:.1f}s")
            if seed == SEEDS[0]:
                keep[f"{rows}x{cols}"] = (som, ev, ab)

    df = pd.DataFrame(runs)
    df.to_csv(OUT / "metrics" / "som_runs.csv", index=False)
    metrics = [c for c in df.columns if c not in ("size", "units", "seed")]
    summary = df.groupby(["size", "units"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "metrics" / "som_comparison.csv", index=False)
    fmt = pd.DataFrame({"Lattice": summary["size"]})
    for m, label, d in [("qe_train", "QE train", 3), ("qe_test", "QE test", 3), ("te_train", "TE train", 3),
                        ("te_test", "TE test", 3), ("empty_pct_train", "Empty units % (train)", 1),
                        ("train_seconds", "Train time s", 1), ("label_agreement_test", "Label agreement (test)", 3)]:
        fmt[label] = [f"{a:.{d}f} ± {b:.{d}f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    fmt.to_csv(OUT / "metrics" / "som_comparison_table.csv", index=False)

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    x = np.arange(len(SIZES))
    for ax, (m, label) in zip(axes, [("qe", "Quantization error"), ("te", "Topographic error"),
                                     ("empty_pct", "Empty units (%)"), ("train_seconds", "Training time (s)")]):
        if m == "train_seconds":
            ax.bar(x, summary[f"{m}_mean"], yerr=summary[f"{m}_std"], color="#2a78d6", capsize=4)
        else:
            w = 0.38
            ax.bar(x - w / 2 - 0.01, summary[f"{m}_train_mean"], w, yerr=summary[f"{m}_train_std"],
                   color="#2a78d6", capsize=4, label="train")
            ax.bar(x + w / 2 + 0.01, summary[f"{m}_test_mean"], w, yerr=summary[f"{m}_test_std"],
                   color="#eb6834", capsize=4, label="test")
            top = (summary[[f"{m}_train_mean", f"{m}_test_mean"]].to_numpy()
                   + summary[[f"{m}_train_std", f"{m}_test_std"]].fillna(0).to_numpy()).max()
            ax.set_ylim(0, max(top, 1e-6) * 1.3)
            ax.legend(frameon=False, loc="upper right", ncol=2)
        ax.set(xticks=x, xticklabels=summary["size"], xlabel="Lattice size", title=label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"SOM mapping quality vs lattice size (mean ± std over {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "som_metrics_comparison.png", dpi=150)
    plt.close(fig)

    grid_plot({k: s.u_matrix() for k, (s, _, _) in keep.items()}, "U-matrix (mean distance to neighbour units, seed 42)",
              OUT / "figures" / "som_u_matrix.png", "bone_r", "distance")
    grid_plot({k: np.where(e["train"]["hits"] > 0, e["train"]["hits"], np.nan).reshape(s.rows, s.cols).astype(float)
               for k, (s, e, _) in keep.items()}, "Hit maps: training samples per unit (grey = empty, seed 42)",
              OUT / "figures" / "som_hit_maps.png", "Blues", "hits", annotate=True)
    grid_plot({k: np.where(e["train"]["hits"] > 0, a / np.maximum(e["train"]["hits"], 1), np.nan).reshape(s.rows, s.cols)
               for k, (s, e, a) in keep.items()},
              "Share of Abnormal samples per unit (labels used only after training, grey = empty)",
              OUT / "figures" / "som_normal_vs_abnormal.png", "RdBu_r", "abnormal fraction", 0, 1)
    s8 = keep["8x8"][0]
    grid_plot({f"PC{i + 1}": s8.W[:, i].reshape(8, 8) for i in range(4)}, "Component planes of the 8x8 SOM (first 4 PCA components)",
              OUT / "figures" / "som_component_planes_8x8.png", "viridis", "weight value")

    save_json({"timestamp": timestamp(), "variant": "9 - SOM lattice size and mapping quality",
               "data": {"features": "1024-d DenseNet121 embeddings (verified), StandardScaler + PCA fitted on train only",
                        "pca_components": PCA_COMPONENTS, "pca_explained_variance": pca_var,
                        "n_train": len(Z["train"]), "n_val": len(Z["val"]), "n_test": len(Z["test"]),
                        "abnormal_fraction_train": float(y["train"].mean())},
               "settings": {"sizes": [f"{r}x{c}" for r, c in SIZES], "seeds": SEEDS, "epochs": EPOCHS,
                            "presentations_per_map": n_steps, "lattice": "rectangular",
                            "neighbourhood": "Gaussian", "init": "random training samples",
                            "learning_rate": f"{LR_START} -> {LR_END} exponential",
                            "sigma": f"max(rows, cols)/2 -> {SIGMA_END} exponential",
                            "topographic_error": "BMU and 2nd BMU not 4-neighbours",
                            "label_agreement": "unit majority label from train hits, applied to test (interpretation only)"},
               "summary": summary.to_dict(orient="records"), "runs": runs,
               "numerical_example": numerical_example(), "library_versions": library_versions()},
              OUT / "results.json")
    print("\n" + fmt.to_string(index=False))


if __name__ == "__main__":
    main()
