"""Final project, stage 1b: PyTorch MLP on the DenseNet121 embeddings.

Four configurations (hidden sizes, depth, activation, dropout), each trained with 3 seeds.
Adam, BCEWithLogitsLoss (one logit), mini-batch 64, early stopping on validation loss.
The configuration is selected by mean validation F1; the test set is only evaluated.
Also provides the MLP and training helpers used by script 07.

Usage (from the project root):  python scripts/06_final_mlp.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_spec = importlib.util.spec_from_file_location("baselines", Path(__file__).with_name("05_final_baselines.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)  # shared helpers: load_embeddings, evaluate, summarize, plot_confusions

from retina.utils import library_versions, save_json, timestamp  # noqa: E402

DEVICE = torch.device("cpu")  # tiny models on 1024-d vectors: CPU avoids GPU launch overhead
CONFIGS = {
    "A: 1024-256-1, ReLU, no dropout": {"hidden": [256], "act": "relu", "dropout": 0.0},
    "B: 1024-256-1, ReLU, dropout 0.5": {"hidden": [256], "act": "relu", "dropout": 0.5},
    "C: 1024-512-128-1, ReLU, dropout 0.3": {"hidden": [512, 128], "act": "relu", "dropout": 0.3},
    "D: 1024-256-1, Tanh, dropout 0.3": {"hidden": [256], "act": "tanh", "dropout": 0.3},
}
# lr 1e-4: with 1e-3 the validation loss was lowest after 1-5 epochs (fast overfitting); 1e-4 was also the
# best Adam learning rate in the validation search of script 07.
MAX_EPOCHS, PATIENCE, BATCH, LR, WEIGHT_DECAY = 200, 15, 64, 1e-4, 1e-4


class MLP(nn.Module):
    """x -> [Linear -> activation -> Dropout] x L -> Linear -> one logit."""

    def __init__(self, in_dim: int, hidden: list[int], act: str = "relu", dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU() if act == "relu" else nn.Tanh(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def full_loss_and_prob(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return loss_fn(logits, y).item(), torch.sigmoid(logits).numpy()


def train_model(model, X, y, make_opt, batch_size, max_epochs, seed, patience=None):
    """Mini-batch training; records full train/val loss after every epoch.
    patience=None -> fixed budget; the weights with the lowest validation loss are returned either way."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt, loss_fn = make_opt(model.parameters()), nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X["train"]), torch.from_numpy(y["train"].astype(np.float32))
    Xv, yv = torch.from_numpy(X["val"]), torch.from_numpy(y["val"].astype(np.float32))
    hist, best, best_state, bad, t0 = [], float("inf"), None, 0, time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()   # backpropagation: gradients of the loss w.r.t. all weights
            opt.step()
        tr, _ = full_loss_and_prob(model, Xt, yt, loss_fn)
        va, _ = full_loss_and_prob(model, Xv, yv, loss_fn)
        hist.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best - 1e-5:
            best, bad, best_state = va, 0, {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"history": hist, "best_epoch": best_epoch, "best_val_loss": best,
            "epochs_run": len(hist), "train_seconds": time.perf_counter() - t0}


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def plot_curves(curves: dict, path: Path, title: str, ncols: int = 2) -> None:
    n = len(curves)
    rows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(6 * ncols, 3.8 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (name, (hist, best_epoch)) in zip(axes.flat, curves.items()):
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["train_loss"], color="#2a78d6", linewidth=2, label="train")
        ax.plot(h["epoch"], h["val_loss"], color="#eb6834", linewidth=2, label="validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best validation epoch")
        ax.set(title=name, xlabel="Epoch", ylabel="BCE loss")
        ax.title.set_fontsize(10)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    out = base.FINAL / "mlp"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    X, y, _, _ = base.load_embeddings()
    runs, curves, cms, info = [], {}, {}, {}
    for name, cfg in CONFIGS.items():
        for seed in base.SEEDS:
            torch.manual_seed(seed)
            model = MLP(X["train"].shape[1], **cfg).to(DEVICE)
            res = train_model(model, X, y, lambda p: torch.optim.Adam(p, lr=LR, weight_decay=WEIGHT_DECAY),
                              BATCH, MAX_EPOCHS, seed, PATIENCE)
            tag = name.split(":")[0]
            torch.save({"state_dict": model.state_dict(), "config": cfg, "seed": seed},
                       out / "checkpoints" / f"mlp_{tag}_seed{seed}.pt")
            for part in ("val", "test"):
                runs.append({"config": name, "seed": seed, "split": part, "best_epoch": res["best_epoch"],
                             "train_seconds": res["train_seconds"], **base.evaluate(y[part], predict(model, X[part]))})
            print(f"{name} seed {seed}: best epoch {res['best_epoch']}, val F1 {runs[-2]['f1']:.4f}")
            if seed == base.SEEDS[0]:
                curves[name] = (res["history"], res["best_epoch"])
                cms[name] = runs[-1]
                info[name] = {"history_seed42": res["history"], "n_parameters": sum(p.numel() for p in model.parameters())}

    runs = pd.DataFrame(runs)
    runs.to_csv(out / "metrics_runs.csv", index=False)
    summary = base.summarize(runs, "config")
    val_f1 = runs[runs["split"] == "val"].groupby("config", sort=False)["f1"].mean()
    summary.insert(1, "mean_val_f1", summary["config"].map(val_f1).round(4))
    summary.insert(2, "n_parameters", summary["config"].map(lambda c: info[c]["n_parameters"]))
    summary.to_csv(out / "summary_test.csv", index=False)
    selected = val_f1.idxmax()
    plot_curves(curves, out / "loss_curves.png", "MLP configurations: training and validation loss (seed 42)")
    base.plot_confusions(cms, out / "confusion_matrices_test.png", "MLP configurations (test set, seed 42)")
    save_json({"timestamp": timestamp(), "configs": CONFIGS,
               "training": {"optimizer": "Adam", "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH,
                            "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
                            "loss": "BCEWithLogitsLoss (1 logit)", "threshold": 0.5, "seeds": base.SEEDS,
                            "device": str(DEVICE)},
               "selected_config_by_mean_val_f1": selected, "summary_test": summary.to_dict(orient="records"),
               "runs": runs.to_dict(orient="records"), "details": info, "library_versions": library_versions()},
              out / "results.json")
    print("\n" + summary.to_string(index=False))
    print("Selected on validation:", selected)


if __name__ == "__main__":
    main()
