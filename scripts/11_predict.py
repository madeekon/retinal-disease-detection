"""Inference with the saved Assignment 03 DenseNet121 classifier (frozen ImageNet backbone + trained head).

Usage (from the project root):
    python scripts/11_predict.py path\\to\\fundus_image.jpg [more images ...]

Prints P(Abnormal) and the predicted class (threshold 0.5) for each image.
Research prototype only - not a medical device.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image

_spec = importlib.util.spec_from_file_location("a03", Path(__file__).with_name("03_experiments.py"))
a03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a03)  # same model definition and preprocessing as in training

CHECKPOINT = Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / "cnn_no_aug.pt"


def load_model(device: torch.device) -> torch.nn.Module:
    net = a03.build_cnn()  # ImageNet DenseNet121 (downloaded once by torchvision) + new head
    net.classifier.load_state_dict(torch.load(CHECKPOINT, map_location="cpu")["head_state_dict"])
    return net.to(device).eval()


def predict(net: torch.nn.Module, image_path: str, device: torch.device) -> float:
    with Image.open(image_path) as im:
        x = a03.EVAL_TF(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(net(x)).item()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(device)
    for path in sys.argv[1:]:
        p = predict(net, path, device)
        print(f"{Path(path).name}: P(Abnormal) = {p:.3f} -> {'Abnormal' if p >= 0.5 else 'Normal'}")


if __name__ == "__main__":
    main()
