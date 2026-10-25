"""Assignment 03 experiments (binary: Normal 0 vs Abnormal 1).

1. Logistic Regression on the verified DenseNet121 embeddings (C selected on validation F1).
2. DenseNet121 (ImageNet weights, frozen backbone, trainable head) without augmentation.
3. The same DenseNet121 with moderate augmentation.

The CNN epoch is selected by early stopping on validation loss. All three final models
are evaluated once on the same test set.

Usage (from the project root):
    python scripts/03_experiments.py              # full run
    python scripts/03_experiments.py --limit 64   # quick smoke test on a few images
"""

from __future__ import annotations

import argparse
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import models, transforms  # noqa: E402

from retina.config import load_config, resolve_paths  # noqa: E402
from retina.splits import load_split  # noqa: E402
from retina.utils import library_versions, load_json, save_json, set_seed, setup_logging, timestamp  # noqa: E402

SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
LR_C_GRID = [0.001, 0.01, 0.1, 1.0]
MODEL_NAMES = {"logreg": "LogReg (DenseNet121 embeddings)",
               "cnn_no_aug": "DenseNet121 frozen, no augmentation",
               "cnn_aug": "DenseNet121 frozen, moderate augmentation"}

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# Moderate, anatomy-preserving augmentation: small crop/zoom, mirror, small rotation, mild colour change.
AUG_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def binary_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# ----------------------------------------------------------------------------- Logistic Regression
def run_logreg(splits: dict, paths, log) -> tuple[dict, dict]:
    audit = load_json(paths.audit_dir / "audit_report.json")
    if not audit["legacy_embeddings"].get("verified"):
        raise RuntimeError("Legacy embeddings were not verified by the audit; re-extract them first.")
    emb = np.load(paths.legacy_embeddings_dir / "embeddings.npy")
    X = {k: emb[d["row_index"].to_numpy()] for k, d in splits.items()}  # row_index = full_df row = embedding row
    y = {k: d["binary_target"].to_numpy() for k, d in splits.items()}
    scaler = StandardScaler().fit(X["train"])  # fitted on training data only
    Xs = {k: scaler.transform(v) for k, v in X.items()}
    search = []
    for c in LR_C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000, random_state=SEED).fit(Xs["train"], y["train"])
        search.append({"C": c, "val_f1": f1_score(y["val"], clf.predict(Xs["val"]))})
        log.info("LogReg C=%-6g val F1=%.4f", c, search[-1]["val_f1"])
    best_c = max(search, key=lambda r: r["val_f1"])["C"]
    clf = LogisticRegression(C=best_c, max_iter=3000, random_state=SEED).fit(Xs["train"], y["train"])
    joblib.dump({"scaler": scaler, "model": clf, "C": best_c}, paths.project_root / "outputs/checkpoints/logreg.joblib")
    probs = {k: clf.predict_proba(Xs[k])[:, 1] for k in ("val", "test")}
    info = {"selected_C": best_c, "C_search_on_val_f1": search,
            "features": "1024-d DenseNet121 embeddings (legacy, verified alignment), StandardScaler fit on train"}
    return info, probs


# ----------------------------------------------------------------------------- CNN
class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.y, self.tf = df["image_path"].tolist(), df["binary_target"].to_numpy(np.float32), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.y[i]


def build_cnn() -> nn.Module:
    net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in net.features.parameters():
        p.requires_grad = False  # frozen backbone
    net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.classifier.in_features, 1))  # one raw logit
    return net


def run_epoch(net, loader, device, loss_fn, opt=None, scaler=None):
    training = opt is not None
    net.train(training)
    net.features.eval()  # frozen BatchNorm statistics stay fixed
    total, n, probs = 0.0, 0, []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = net(x).squeeze(1).float()
            loss = loss_fn(logits, y)
            if training:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total += loss.item() * len(y)
            n += len(y)
            probs.append(torch.sigmoid(logits).detach().cpu())
    return total / n, torch.cat(probs).numpy()


def train_cnn(name: str, augment: bool, splits: dict, args, device, paths, log) -> tuple[dict, dict]:
    set_seed(SEED)
    net = build_cnn().to(device)
    dl = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(FundusDataset(splits["train"], AUG_TF if augment else EVAL_TF), shuffle=True, **dl)
    eval_dl = {k: DataLoader(FundusDataset(splits[k], EVAL_TF), shuffle=False, **dl) for k in ("val", "test")}
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    ckpt = paths.project_root / f"outputs/checkpoints/{name}.pt"
    history, best_loss, best_epoch, bad = [], float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        tr_loss, _ = run_epoch(net, train_dl, device, loss_fn, opt, scaler)
        va_loss, va_prob = run_epoch(net, eval_dl["val"], device, loss_fn)
        va_f1 = f1_score(splits["val"]["binary_target"], va_prob >= 0.5)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_f1": va_f1})
        log.info("%s epoch %2d  train_loss %.4f  val_loss %.4f  val_F1 %.4f  (%.0f s)",
                 name, epoch, tr_loss, va_loss, va_f1, time.time() - te)
        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            torch.save({"head_state_dict": net.classifier.state_dict(), "epoch": epoch, "augment": augment,
                        "backbone": "torchvision densenet121 IMAGENET1K_V1 (frozen)"}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                log.info("%s early stopping at epoch %d (best epoch %d)", name, epoch, best_epoch)
                break
    net.classifier.load_state_dict(torch.load(ckpt, map_location=device)["head_state_dict"])
    probs = {k: run_epoch(net, eval_dl[k], device, loss_fn)[1] for k in ("val", "test")}
    info = {"augmentation": str(AUG_TF) if augment else "none", "best_epoch": best_epoch,
            "best_val_loss": best_loss, "epochs_run": len(history), "train_minutes": (time.time() - t0) / 60,
            "history": history}
    return info, probs


# ----------------------------------------------------------------------------- figures
def plot_losses(results: dict, path: Path) -> None:
    names = [n for n in ("cnn_no_aug", "cnn_aug") if n in results]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4), squeeze=False)
    for ax, n in zip(axes[0], names):
        h = pd.DataFrame(results[n]["history"])
        ax.plot(h["epoch"], h["train_loss"], color="#2a78d6", linewidth=2, label="train")
        ax.plot(h["epoch"], h["val_loss"], color="#eb6834", linewidth=2, label="validation")
        ax.axvline(results[n]["best_epoch"], color="gray", linestyle="--", linewidth=1, label="selected epoch")
        ax.set(title=MODEL_NAMES[n], xlabel="Epoch", ylabel="BCE loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusions(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8), squeeze=False)
    for ax, (n, r) in zip(axes[0], results.items()):
        m = r["test"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"],
               xlabel="Predicted", ylabel="True", title=f"{MODEL_NAMES[n]}\n(test set)")
        ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(test_df: pd.DataFrame, prob: np.ndarray, model: str, path: Path, n: int = 5) -> pd.DataFrame:
    df = test_df.assign(prob=prob, pred=(prob >= 0.5).astype(int))
    wrong = df[df["pred"] != df["binary_target"]].copy()
    wrong["confidence"] = (wrong["prob"] - 0.5).abs()
    fn = wrong[wrong["binary_target"] == 1].nlargest(3, "confidence")
    fp = wrong[wrong["binary_target"] == 0].nlargest(n - len(fn), "confidence")
    sel = pd.concat([fn, fp]).head(n)
    fig, axes = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 3.6), squeeze=False)
    for ax, (_, r) in zip(axes[0], sel.iterrows()):
        with Image.open(r["image_path"]) as im:
            ax.imshow(im.convert("RGB"))
        kind = "False negative" if r["binary_target"] == 1 else "False positive"
        ax.set_title(f"{kind}\n{r['filename']} (true {r['label_code']})\nP(abnormal) = {r['prob']:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Test-set errors of {MODEL_NAMES[model]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return sel[["filename", "label_code", "binary_target", "pred", "prob"]]


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images per split (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    for sub in ("outputs/metrics", "outputs/checkpoints"):
        (paths.project_root / sub).mkdir(parents=True, exist_ok=True)
    log = setup_logging(paths.logs_dir / "03_experiments.log")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log.info("Device: %s %s", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    splits = {k: load_split(paths.splits_dir, k) for k in ("train", "val", "test")}  # checksum-verified
    if args.limit:
        splits = {k: d.head(args.limit).reset_index(drop=True) for k, d in splits.items()}
    log.info("Images: train %d, val %d, test %d", *(len(splits[k]) for k in ("train", "val", "test")))

    results, probs = {}, {}
    results["logreg"], probs["logreg"] = run_logreg(splits, paths, log)
    for name, aug in (("cnn_no_aug", False), ("cnn_aug", True)):
        results[name], probs[name] = train_cnn(name, aug, splits, args, device, paths, log)

    rows, preds = [], splits["test"][["filename", "label_code", "binary_target"]].copy()
    for name in results:
        for part in ("val", "test"):
            y = splits[part]["binary_target"].to_numpy()
            results[name][part] = binary_metrics(y, (probs[name][part] >= 0.5).astype(int))
            rows.append({"model": MODEL_NAMES[name], "split": part, **results[name][part]})
        preds[f"prob_{name}"] = probs[name]["test"]
    table = pd.DataFrame(rows)
    table.to_csv(paths.project_root / "outputs/metrics/comparison.csv", index=False)
    preds.to_csv(paths.project_root / "outputs/metrics/test_predictions.csv", index=False)

    best = max(results, key=lambda n: results[n]["val"]["f1"])  # chosen on validation, not test
    errors = plot_errors(splits["test"], probs[best]["test"], best, paths.figures_dir / "misclassified_examples.png")
    plot_losses(results, paths.figures_dir / "cnn_loss_curves.png")
    plot_confusions(results, paths.figures_dir / "confusion_matrices_test.png")

    save_json({"timestamp": timestamp(), "seed": SEED, "device": str(device),
               "settings": {**vars(args), "image_size": 224, "normalization": "ImageNet mean/std",
                            "loss": "BCEWithLogitsLoss (1 logit)", "optimizer": "AdamW (weight_decay 1e-4)",
                            "threshold": 0.5, "early_stopping": "validation loss",
                            "mixed_precision": device.type == "cuda"},
               "best_model_by_val_f1": best, "error_examples": errors.to_dict(orient="records"),
               "results": results, "library_versions": library_versions()},
              paths.project_root / "outputs/metrics/results.json")

    print("\n" + table.round(4).to_string(index=False))
    print(f"\nBest model on validation F1: {MODEL_NAMES[best]}")
    print(errors.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
