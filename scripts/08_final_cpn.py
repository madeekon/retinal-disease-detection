"""Final project, stage 2a: Counterpropagation Network (Hecht-Nielsen) for Normal vs Abnormal.

Input: verified 1024-d DenseNet121 embeddings -> StandardScaler + PCA(32) fitted on train only.
Layer 1 (Kohonen, unsupervised): K competing instars, winner-take-all, w_k <- w_k + alpha (x - w_k).
Layer 2 (Grossberg, supervised): one outstar weight per Kohonen unit, v_k <- v_k + beta (t - v_k)
for the winning unit k; v_k estimates P(Abnormal | cluster k) and is the network output.
Three Kohonen layer sizes are compared on validation F1 (seed 42); the selected size is repeated
with 3 seeds and evaluated once on the test set.

Usage (from the project root):  python scripts/08_final_cpn.py
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_spec = importlib.util.spec_from_file_location("baselines", Path(__file__).with_name("05_final_baselines.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)  # load_embeddings, evaluate, summarize, plot_confusions, SEEDS, FINAL

from retina.utils import library_versions, save_json, timestamp  # noqa: E402

PCA_COMPONENTS = 32
CONFIGS = {"CPN-16 (4x4 Kohonen units)": 16, "CPN-64 (8x8 Kohonen units)": 64, "CPN-144 (12x12 Kohonen units)": 144}
KOHONEN_EPOCHS, ALPHA_START, ALPHA_END = 20, 0.5, 0.01
GROSSBERG_EPOCHS, BETA = 10, 0.05


class CPN:
    def __init__(self, n_units: int, seed: int):
        self.k, self.rng = n_units, np.random.default_rng(seed)

    def winners(self, X):
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ self.W.T + (self.W ** 2).sum(1)[None, :]
        return d2.argmin(1), np.sqrt(np.maximum(d2.min(1), 0))

    def fit(self, X, t):
        # --- stage 1: Kohonen layer (unsupervised, only the winner learns)
        self.W = X[self.rng.choice(len(X), self.k, replace=False)].copy()  # init from data: no dead units at start
        n_steps, self.qe_curve = KOHONEN_EPOCHS * len(X), []
        step = 0
        for _ in range(KOHONEN_EPOCHS):
            for i in self.rng.permutation(len(X)):
                alpha = ALPHA_START * (ALPHA_END / ALPHA_START) ** (step / n_steps)
                k = np.argmin(((self.W - X[i]) ** 2).sum(1))
                self.W[k] += alpha * (X[i] - self.W[k])          # instar rule
                step += 1
            self.qe_curve.append(float(self.winners(X)[1].mean()))
        # --- stage 2: Grossberg layer (supervised outstar rule for the winner)
        self.prior = float(t.mean())
        self.v = np.full(self.k, self.prior)
        win = self.winners(X)[0]
        for _ in range(GROSSBERG_EPOCHS):
            for i in self.rng.permutation(len(X)):
                self.v[win[i]] += BETA * (t[i] - self.v[win[i]])   # outstar rule
        self.hits = np.bincount(win, minlength=self.k)
        self.v[self.hits == 0] = self.prior                       # units never winning: class prior
        return self

    def predict_proba(self, X):
        return self.v[self.winners(X)[0]]


def main() -> None:
    out = base.FINAL / "cpn"
    (out / "parameters").mkdir(parents=True, exist_ok=True)
    X, y, _, scaler = base.load_embeddings()                       # scaler fitted on train only
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42).fit(X["train"])   # train only
    Z = {k: pca.transform(v) for k, v in X.items()}

    search = []
    for name, k in CONFIGS.items():
        t0 = time.perf_counter()
        m = CPN(k, base.SEEDS[0]).fit(Z["train"], y["train"])
        search.append({"config": name, "kohonen_units": k, "val_f1": base.evaluate(y["val"], m.predict_proba(Z["val"]))["f1"],
                       "val_roc_auc": base.evaluate(y["val"], m.predict_proba(Z["val"]))["roc_auc"],
                       "dead_units": int((m.hits == 0).sum()), "seconds": time.perf_counter() - t0})
        print(f"{name}: val F1 {search[-1]['val_f1']:.4f}, dead units {search[-1]['dead_units']}")
    pd.DataFrame(search).to_csv(out / "config_search_val.csv", index=False)
    selected = max(search, key=lambda r: r["val_f1"])["config"]
    k_sel = CONFIGS[selected]

    runs, models = [], {}
    for seed in base.SEEDS:
        t0 = time.perf_counter()
        m = CPN(k_sel, seed).fit(Z["train"], y["train"])
        secs = time.perf_counter() - t0
        models[seed] = m
        np.savez(out / "parameters" / f"cpn_k{k_sel}_seed{seed}.npz", kohonen_W=m.W, grossberg_v=m.v, hits=m.hits,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, pca_components=pca.components_, pca_mean=pca.mean_)
        for part in ("val", "test"):
            runs.append({"config": selected, "seed": seed, "split": part, "train_seconds": secs,
                         "dead_units": int((m.hits == 0).sum()), **base.evaluate(y[part], m.predict_proba(Z[part]))})
        print(f"{selected} seed {seed}: test F1 {runs[-1]['f1']:.4f}, ROC-AUC {runs[-1]['roc_auc']:.4f}")
    runs = pd.DataFrame(runs)
    runs.to_csv(out / "metrics_runs.csv", index=False)
    summary = base.summarize(runs, "config")
    summary.to_csv(out / "summary_test.csv", index=False)

    # figures
    m42 = models[base.SEEDS[0]]
    base.plot_confusions({f"{selected}, seed 42": runs[(runs.seed == 42) & (runs.split == "test")].iloc[0]},
                         out / "confusion_matrix_test.png", "Counterpropagation Network (test set)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    s = pd.DataFrame(search)
    axes[0].bar(range(len(s)), s["val_f1"], color="#2a78d6")
    for i, v in enumerate(s["val_f1"]):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set(xticks=range(len(s)), xticklabels=[f"K = {k}" for k in s["kohonen_units"]], ylim=(0, 1),
                title="Validation F1 per Kohonen layer size (seed 42)", ylabel="F1")
    axes[1].plot(range(1, KOHONEN_EPOCHS + 1), m42.qe_curve, color="#2a78d6", linewidth=2, marker="o", markersize=4)
    axes[1].set(title=f"Kohonen layer training ({selected}, seed 42)", xlabel="Epoch",
                ylabel="Mean distance to winning unit")
    side = int(np.sqrt(k_sel))
    im = axes[2].imshow(m42.v.reshape(side, side), cmap="RdBu_r", vmin=0, vmax=1)
    for (r, c), h in np.ndenumerate(m42.hits.reshape(side, side)):
        axes[2].text(c, r, h, ha="center", va="center", fontsize=6)
    axes[2].set(title="Grossberg outputs v_k = P(Abnormal | unit), grid layout for display only\n(numbers = training samples won)",
                xticks=[], yticks=[])
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="P(Abnormal)")
    for ax in axes[:2]:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "cpn_overview.png", dpi=150)
    plt.close(fig)

    save_json({"timestamp": timestamp(),
               "architecture": {"input": f"1024-d DenseNet121 embeddings -> StandardScaler -> PCA({PCA_COMPONENTS}), fit on train",
                                "kohonen": "winner-take-all, Euclidean distance, instar rule, init from training samples",
                                "grossberg": "one outstar weight per unit, output = P(Abnormal | winning unit)",
                                "decision": "Abnormal if output >= 0.5"},
               "training": {"kohonen_epochs": KOHONEN_EPOCHS, "alpha": f"{ALPHA_START} -> {ALPHA_END} exponential",
                            "grossberg_epochs": GROSSBERG_EPOCHS, "beta": BETA, "seeds": base.SEEDS,
                            "pca_explained_variance": float(pca.explained_variance_ratio_.sum())},
               "config_search_val": search, "selected_config": selected, "summary_test": summary.to_dict(orient="records"),
               "runs": runs.to_dict(orient="records"), "library_versions": library_versions()}, out / "results.json")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
