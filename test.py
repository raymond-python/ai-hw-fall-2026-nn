"""Evaluate a trained model on the MNIST TEST split (10,000 images).

Reports:
  * clean test accuracy (the official number)
  * per-class accuracy and a confusion matrix
  * robustness: accuracy on the same test images randomly rotated/shifted/scaled

Example:
    python test.py --model cnn --model-path saved_models/cnn_aug.pth
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import MNIST_MEAN, MNIST_STD, get_device, set_seed, test_loader
from models import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds, images = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu())
        ys.append(y)
        images.append(x)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy(), torch.cat(images)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    off = cm.copy().astype(float)
    np.fill_diagonal(off, np.nan)  # colour the mistakes, not the (huge) diagonal
    im = ax.imshow(off, cmap="Reds")
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="black" if i != j else "#1a5f2a",
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="errors")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_mistakes(images, ys, preds, title, path, n=24):
    wrong = np.where(ys != preds)[0][:n]
    cols = 8
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.45))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, wrong):
        img = images[i, 0].numpy() * MNIST_STD + MNIST_MEAN
        ax.imshow(img, cmap="gray_r")
        ax.set_title(f"{ys[i]}→{preds[i]}", fontsize=9, color="#b00020")
    fig.suptitle(title + "  (true→predicted)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mlp", "cnn", "transformer"])
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.model_path, map_location=device)
    model = build_model(args.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    tag = os.path.splitext(os.path.basename(args.model_path))[0]

    ys, preds, images = predict(model, test_loader(args.data_dir, workers=args.workers), device)
    acc = 100.0 * (ys == preds).mean()

    cm = np.zeros((10, 10), dtype=int)
    for t, pr in zip(ys, preds):
        cm[t, pr] += 1
    per_class = {str(c): round(100.0 * cm[c, c] / cm[c].sum(), 2) for c in range(10)}

    set_seed(0)  # same random perturbations for every model -> fair comparison
    ys_p, preds_p, _ = predict(model, test_loader(args.data_dir, perturbed=True,
                                                  workers=0), device)
    robust_acc = 100.0 * (ys_p == preds_p).mean()

    os.makedirs("results", exist_ok=True)
    plot_confusion(cm, f"{tag}: test acc {acc:.2f}%", f"results/confusion_{tag}.png")
    plot_mistakes(images, ys, preds, f"{tag}: misclassified test digits",
                  f"results/mistakes_{tag}.png")

    result = {"tag": tag, "model": args.model, "augment": bool(ckpt.get("augment")),
              "test_acc": round(acc, 2), "test_errors": int((ys != preds).sum()),
              "perturbed_test_acc": round(robust_acc, 2),
              "best_epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc"),
              "per_class_acc": per_class, "confusion_matrix": cm.tolist()}
    with open(f"results/test_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] TEST accuracy: {acc:.2f}%  ({result['test_errors']} errors / 10000)"
          f"   perturbed-test accuracy: {robust_acc:.2f}%")


if __name__ == "__main__":
    main()
