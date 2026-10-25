"""Final project, stage 2b: Grad-CAM for the Assignment 03 DenseNet121 classifier (no retraining).

Loads outputs/checkpoints/cnn_no_aug.pt (ImageNet DenseNet121 with frozen backbone + trained head),
picks 12 real test images from Assignment 03 predictions (3 true positives, 3 true negatives,
3 false positives, 3 false negatives; the most confident of each group) and computes Grad-CAM on the
last dense block (7x7 feature maps). The map explains the evidence for the PREDICTED class:
score = logit if predicted Abnormal, -logit if predicted Normal. Grad-CAM shows where the model looks;
it does not confirm any medical finding.

Usage (from the project root):  python scripts/09_final_xai.py
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

_spec = importlib.util.spec_from_file_location("a03", Path(__file__).with_name("03_experiments.py"))
a03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a03)  # build_cnn, EVAL_TF, MEAN, STD (same model and preprocessing as Assignment 03)

from retina.config import load_config, resolve_paths  # noqa: E402
from retina.splits import load_split  # noqa: E402
from retina.utils import library_versions, save_json, timestamp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_project" / "xai"
PER_GROUP = 3
GROUPS = ["True positive", "True negative", "False positive", "False negative"]


def grad_cam(net, x):
    """Return (probability of Abnormal, Grad-CAM map 7x7 normalised to [0, 1]) for one image tensor."""
    store = {}

    def hook(_m, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = net.features.register_forward_hook(hook)
    x = x.clone().requires_grad_(True)  # backbone is frozen: input grad makes autograd track the feature maps
    logit = net(x).squeeze()
    prob = torch.sigmoid(logit).item()
    score = logit if prob >= 0.5 else -logit   # evidence for the predicted class
    net.zero_grad()
    score.backward()
    h.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)         # alpha_k: averaged gradients
    cam = F.relu((weights * store["act"]).sum(1)).squeeze(0)       # ReLU(sum_k alpha_k A_k)
    cam = cam / (cam.max() + 1e-8)
    return prob, cam.detach().cpu().numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(load_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = a03.build_cnn()
    ckpt = torch.load(ROOT / "outputs" / "checkpoints" / "cnn_no_aug.pt", map_location="cpu")
    net.classifier.load_state_dict(ckpt["head_state_dict"])
    net = net.to(device).eval()

    test = load_split(paths.splits_dir, "test")
    preds = pd.read_csv(ROOT / "outputs" / "metrics" / "test_predictions.csv")[["filename", "prob_cnn_no_aug"]]
    df = test.merge(preds, on="filename")
    df["pred"] = (df["prob_cnn_no_aug"] >= 0.5).astype(int)
    df["group"] = np.select([(df.binary_target == 1) & (df.pred == 1), (df.binary_target == 0) & (df.pred == 0),
                             (df.binary_target == 0) & (df.pred == 1)],
                            GROUPS[:3], GROUPS[3])
    df["confidence"] = (df["prob_cnn_no_aug"] - 0.5).abs()
    cases = pd.concat([df[df.group == g].nlargest(PER_GROUP, "confidence") for g in GROUPS], ignore_index=True)
    cases["col"] = cases.groupby("group").cumcount()

    rows = []
    fig, axes = plt.subplots(len(GROUPS), 2 * PER_GROUP, figsize=(3.0 * 2 * PER_GROUP, 3.4 * len(GROUPS)))
    for _, r in cases.iterrows():
        with Image.open(r["image_path"]) as im:
            rgb = im.convert("RGB")
        x = a03.EVAL_TF(rgb).unsqueeze(0).to(device)
        prob, cam = grad_cam(net, x)
        shown = np.asarray(rgb.resize((224, 224)), dtype=np.float32) / 255
        heat = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255
        fundus = shown.mean(2) > 0.04                                  # circular retina area (not black border)
        cy, cx = np.unravel_index(heat.argmax(), heat.shape)
        rows.append({"group": r["group"], "filename": r["filename"], "label_code": r["label_code"],
                     "true": int(r["binary_target"]), "pred": int(prob >= 0.5), "prob_abnormal": prob,
                     "prob_saved_in_assignment03": float(r["prob_cnn_no_aug"]),
                     "heat_share_outside_fundus": float(heat[~fundus].sum() / (heat.sum() + 1e-8)),
                     "heat_share_central_third": float(heat[75:149, 75:149].sum() / (heat.sum() + 1e-8)),
                     "peak_xy_224": [int(cx), int(cy)]})
        g, j = GROUPS.index(r["group"]), int(r["col"])
        a0, a1 = axes[g, 2 * j], axes[g, 2 * j + 1]
        a0.imshow(shown)
        a0.set_title(f"{r['group']}\n{r['filename']} (true {r['label_code']})", fontsize=8)
        a1.imshow(shown)
        a1.imshow(heat, cmap="jet", alpha=0.45)
        a1.set_title(f"Grad-CAM, predicted {'Abnormal' if prob >= 0.5 else 'Normal'}\nP(abnormal) = {prob:.2f}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Grad-CAM for DenseNet121 (frozen backbone, Assignment 03), test images; "
                 "heat = evidence for the predicted class", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gradcam_examples.png", dpi=130)
    plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gradcam_cases.csv", index=False)
    max_diff = float((res["prob_abnormal"] - res["prob_saved_in_assignment03"]).abs().max())
    save_json({"timestamp": timestamp(), "model": "Assignment 03 DenseNet121, frozen backbone, no augmentation (cnn_no_aug.pt)",
               "target_layer": "net.features (last dense block + norm5, 7x7x1024)",
               "explained_score": "logit for predicted Abnormal, -logit for predicted Normal",
               "selection": f"{PER_GROUP} most confident test images per group from Assignment 03 predictions",
               "check_max_abs_diff_vs_saved_probability": max_diff,
               "mean_heat_share_outside_fundus": float(res["heat_share_outside_fundus"].mean()),
               "cases": rows, "library_versions": library_versions()}, OUT / "results.json")
    print(res[["group", "filename", "label_code", "prob_abnormal", "heat_share_outside_fundus",
               "heat_share_central_third"]].round(3).to_string(index=False))
    print(f"\nMax difference to Assignment 03 saved probabilities: {max_diff:.5f} (should be close to 0)")
    print("Saved:", OUT / "gradcam_examples.png")


if __name__ == "__main__":
    main()
