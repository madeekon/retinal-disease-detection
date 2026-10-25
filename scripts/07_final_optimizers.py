"""Final project, stage 1c: optimizer comparison on a fixed MLP (1024-256-1, ReLU, dropout 0.3).

Methods: full-batch gradient descent, stochastic gradient descent (batch 1), mini-batch gradient
descent (batch 64), mini-batch SGD with momentum 0.9 and Adam (batch 64).
Every method gets the same budget of 60 epochs (= same number of sample presentations).
The learning rate of each method is chosen from a small grid on validation loss (seed 42),
then the chosen setting is repeated with 3 seeds. The weights with the lowest validation loss
are evaluated on the test set. Finally a combined stage-1 summary table is written.

Usage (from the project root):  python scripts/07_final_optimizers.py   (run 05 and 06 first)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("baselines", "05_final_baselines.py")
mlp = _load("final_mlp", "06_final_mlp.py")
from retina.utils import library_versions, load_json, save_json, timestamp  # noqa: E402

EPOCHS = 60
ARCH = {"hidden": [256], "act": "relu", "dropout": 0.3}
METHODS = {  # name: (batch size or None = full batch, optimizer factory, learning-rate grid)
    "Full-batch GD": (None, lambda p, lr: torch.optim.SGD(p, lr=lr), [0.3, 0.1, 0.03]),
    "SGD (batch 1)": (1, lambda p, lr: torch.optim.SGD(p, lr=lr), [0.003, 0.001, 0.0003]),
    "Mini-batch GD (batch 64)": (64, lambda p, lr: torch.optim.SGD(p, lr=lr), [0.3, 0.1, 0.03]),
    "SGD + momentum 0.9 (batch 64)": (64, lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9), [0.03, 0.01, 0.003]),
    "Adam (batch 64)": (64, lambda p, lr: torch.optim.Adam(p, lr=lr), [1e-3, 3e-4, 1e-4]),
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]


def run(X, y, method, lr, seed):
    batch, factory, _ = METHODS[method]
    bs = len(X["train"]) if batch is None else batch
    torch.manual_seed(seed)
    model = mlp.MLP(X["train"].shape[1], **ARCH)
    res = mlp.train_model(model, X, y, lambda p: factory(p, lr), bs, EPOCHS, seed, patience=None)
    res["updates_per_epoch"] = int(np.ceil(len(X["train"]) / bs))
    return model, res


def main() -> None:
    out = base.FINAL / "optimizers"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    X, y, _, _ = base.load_embeddings()

    search, chosen = [], {}
    for method, (_, _, grid) in METHODS.items():
        for lr in grid:
            _, res = run(X, y, method, lr, base.SEEDS[0])
            search.append({"method": method, "lr": lr, "best_val_loss": res["best_val_loss"],
                           "best_epoch": res["best_epoch"], "seconds": res["train_seconds"]})
            print(f"{method:32s} lr {lr:<7g} best val loss {res['best_val_loss']:.4f} (epoch {res['best_epoch']})")
        chosen[method] = min((r for r in search if r["method"] == method), key=lambda r: r["best_val_loss"])["lr"]
    pd.DataFrame(search).to_csv(out / "lr_search_val.csv", index=False)

    runs, curves = [], {}
    for method in METHODS:
        for seed in base.SEEDS:
            model, res = run(X, y, method, chosen[method], seed)
            if seed == base.SEEDS[0]:
                curves[method] = res
                torch.save({"state_dict": model.state_dict(), "arch": ARCH, "method": method, "lr": chosen[method]},
                           out / "checkpoints" / f"{method.split(' (')[0].replace(' ', '_').replace('+', 'plus')}_seed42.pt")
            h = res["history"]
            for part in ("val", "test"):
                runs.append({"method": method, "lr": chosen[method], "seed": seed, "split": part,
                             "best_epoch": res["best_epoch"], "best_val_loss": res["best_val_loss"],
                             "final_train_loss": h[-1]["train_loss"], "final_val_loss": h[-1]["val_loss"],
                             "updates_total": res["updates_per_epoch"] * EPOCHS, "train_seconds": res["train_seconds"],
                             **base.evaluate(y[part], mlp.predict(model, X[part]))})
            print(f"{method} seed {seed}: best val loss {res['best_val_loss']:.4f}, {res['train_seconds']:.1f}s")

    runs = pd.DataFrame(runs)
    runs.to_csv(out / "metrics_runs.csv", index=False)
    summary = base.summarize(runs, "method")
    v = runs[runs["split"] == "val"].groupby("method", sort=False)
    for col, d in (("best_val_loss", 4), ("final_train_loss", 4), ("best_epoch", 1), ("train_seconds", 1)):
        summary[col] = [f"{a:.{d}f} ± {b:.{d}f}" for a, b in zip(v[col].mean(), v[col].std())]
    summary.insert(1, "lr", summary["method"].map(chosen))
    summary.insert(2, "updates_total", summary["method"].map(v["updates_total"].first()))
    summary.to_csv(out / "summary_test.csv", index=False)

    # learning curves (seed 42): train loss, validation loss per epoch, validation loss per parameter update
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for c, (method, res) in zip(COLORS, curves.items()):
        h = pd.DataFrame(res["history"])
        axes[0].plot(h["epoch"], h["train_loss"], color=c, linewidth=2, label=method)
        axes[1].plot(h["epoch"], h["val_loss"], color=c, linewidth=2, label=method)
        axes[2].plot(h["epoch"] * res["updates_per_epoch"], h["val_loss"], color=c, linewidth=2, label=method)
    axes[0].set(title="Training loss vs epoch", xlabel="Epoch (same sample budget)", ylabel="BCE loss")
    axes[1].set(title="Validation loss vs epoch", xlabel="Epoch (same sample budget)", ylabel="BCE loss")
    axes[2].set(title="Validation loss vs parameter updates", xlabel="Number of updates (log scale)",
                ylabel="BCE loss", xscale="log")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Optimizer comparison on MLP 1024-256-1 (ReLU, dropout 0.3), seed 42")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)
    base.plot_confusions({m: runs[(runs["method"] == m) & (runs["seed"] == 42) & (runs["split"] == "test")].iloc[0]
                          for m in METHODS}, out / "confusion_matrices_test.png", "Optimizers (test set, seed 42)")
    save_json({"timestamp": timestamp(), "architecture": ARCH, "epochs_budget": EPOCHS,
               "loss": "BCEWithLogitsLoss (1 logit)", "weight_decay": 0, "seeds": base.SEEDS,
               "lr_selection": "lowest validation loss over the budget, seed 42", "chosen_lr": chosen,
               "lr_search": search, "summary_test": summary.to_dict(orient="records"),
               "runs": runs.to_dict(orient="records"),
               "curves_seed42": {m: r["history"] for m, r in curves.items()}, "library_versions": library_versions()},
              out / "results.json")
    print("\n" + summary.to_string(index=False))
    write_combined_table()


def write_combined_table() -> None:
    """Combined stage-1 table (test set, mean ± std) from the three summary files."""
    summary = pd.read_csv(base.FINAL / "optimizers" / "summary_test.csv")
    rows = [pd.read_csv(base.FINAL / "baselines" / "summary_test.csv").rename(columns={"model": "model_or_config"})]
    m = pd.read_csv(base.FINAL / "mlp" / "summary_test.csv")
    sel = load_json(base.FINAL / "mlp" / "results.json")["selected_config_by_mean_val_f1"]
    m["model_or_config"] = "MLP " + m["config"] + np.where(m["config"] == sel, " [selected on val]", "")
    rows.append(m)
    o = summary.copy()
    o["model_or_config"] = "MLP optimizer: " + o["method"]
    rows.append(o)
    cols = ["model_or_config", "n_runs", *base.METRICS]
    pd.concat([r[cols] for r in rows], ignore_index=True).to_csv(base.FINAL / "stage1_summary_test.csv", index=False)
    print("\nCombined table:", base.FINAL / "stage1_summary_test.csv")


if __name__ == "__main__":
    main()
