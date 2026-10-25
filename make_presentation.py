"""Build the PDF slide deck (presentation/MNIST_NN_Presentation.pdf) from the
real numbers in results/. Run after run_all.py.

    python make_presentation.py
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

W, H = 13.333, 7.5          # 16:9 slide in inches
INK, MUTED, ACCENT, BG = "#1b1f24", "#5b6470", "#1f4e8c", "#ffffff"
NAMES = {"mlp": "MLP", "cnn": "CNN", "transformer": "Transformer"}
OUT = "presentation/MNIST_NN_Presentation.pdf"


def slide(title=None, subtitle=None):
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    if title:
        fig.text(0.05, 0.90, title, fontsize=28, fontweight="bold", color=INK)
        fig.add_artist(plt.Line2D([0.05, 0.12], [0.875, 0.875], color=ACCENT, lw=4))
    if subtitle:
        fig.text(0.05, 0.82, subtitle, fontsize=14, color=MUTED)
    return fig


def bullets(fig, items, x=0.06, y=0.72, size=17, width=60, gap=None):
    line_h = size * 1.4 / 72 / H            # one text line in figure units
    for it in items:
        lines = textwrap.wrap(it, width)
        fig.text(x, y, "•", fontsize=size, color=ACCENT, fontweight="bold", va="top")
        fig.text(x + 0.022, y, "\n".join(lines), fontsize=size, color=INK, va="top",
                 linespacing=1.4)
        y -= line_h * len(lines) + line_h * 0.7
    return y


def image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def footer(fig, n):
    fig.text(0.95, 0.03, str(n), fontsize=11, color=MUTED, ha="right")
    fig.text(0.05, 0.03, "ai-hw-fall-2026-nn · MNIST", fontsize=11, color=MUTED)


def make_aug_examples(path):
    """Show a few training digits next to random augmented versions."""
    import torch
    from torchvision import datasets
    from data import MNIST_MEAN, MNIST_STD, augment_tf
    torch.manual_seed(1)
    ds = datasets.MNIST("data", train=True, download=True)
    idx = [0, 1, 2, 3, 5]
    fig, axes = plt.subplots(len(idx), 6, figsize=(6, 5.2))
    for r, i in enumerate(idx):
        img, label = ds[i]
        axes[r, 0].imshow(img, cmap="gray_r")
        for c in range(1, 6):
            t = augment_tf(img)[0] * MNIST_STD + MNIST_MEAN
            axes[r, c].imshow(t.numpy(), cmap="gray_r")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("original", fontsize=10)
    for c in range(1, 6):
        axes[0, c].set_title(f"aug {c}", fontsize=10, color=ACCENT)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    rows = json.load(open("results/summary.json"))
    by = {r["tag"]: r for r in rows}
    best = max(rows, key=lambda r: r["test_acc"])
    os.makedirs("presentation", exist_ok=True)
    aug_png = "presentation/augmentation_examples.png"
    make_aug_examples(aug_png)

    with PdfPages(OUT) as pdf:
        n = 1
        # 1. Title
        fig = slide()
        fig.text(0.06, 0.60, "Small Neural Networks\nfor Handwritten Digits",
                 fontsize=44, fontweight="bold", color=INK, linespacing=1.15)
        fig.add_artist(plt.Line2D([0.06, 0.16], [0.47, 0.47], color=ACCENT, lw=5))
        fig.text(0.06, 0.38, "MLP vs CNN vs Transformer on MNIST — and what image "
                 "augmentation adds", fontsize=18, color=MUTED)
        fig.text(0.06, 0.30, f"Best clean test accuracy: {best['test_acc']:.2f}% "
                 f"({NAMES[best['model']]}{', augmented' if best['aug'] == 'aug' else ''})",
                 fontsize=18, color=ACCENT, fontweight="bold")
        fig.text(0.06, 0.12, "AI homework · Fall 2026", fontsize=14, color=MUTED)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 2. Problem & data
        fig = slide("Problem and data", "Classify 28×28 grayscale digits into 10 classes")
        bullets(fig, [
            "MNIST: 60,000 training images, 10,000 test images.",
            "Strict split: train on the train part only; the test part is touched "
            "once, at the end.",
            "Last 5,000 training images held out as validation to pick the best "
            "epoch, so no decision ever looks at the test set.",
            "Same optimizer for all models: AdamW + one-cycle learning rate, "
            "label smoothing 0.05, batch 128.",
        ], width=52)
        try:
            from torchvision import datasets
            ds = datasets.MNIST("data", train=False, download=True)
            for k in range(12):
                ax = fig.add_axes([0.66 + (k % 4) * 0.075, 0.55 - (k // 4) * 0.14,
                                   0.065, 0.12])
                ax.imshow(ds[k][0], cmap="gray_r"); ax.axis("off")
            fig.text(0.66, 0.18, "sample test images", fontsize=12, color=MUTED)
        except Exception:
            pass
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 3. Models
        fig = slide("Three architectures", "Same data, same training recipe")
        p = {m: by.get(f"{m}_no_aug", {}).get("params", 0) for m in NAMES}
        cols = [
            ("MLP", f"{p['mlp']:,} params",
             ["Flatten 784 pixels", "1 hidden layer (256, ReLU)", "Dropout → 10 classes",
              "No notion of neighbours"]),
            ("CNN", f"{p['cnn']:,} params",
             ["2 conv blocks (+BN, max-pool)", "28×28 → 14×14 → 7×7",
              "Dense 128 → 10 classes", "Built-in locality + weight sharing"]),
            ("Transformer", f"{p['transformer']:,} params",
             ["16 patches of 7×7 → 64-d tokens", "+ [CLS] token, position emb.",
              "4 encoder layers, 4 heads", "Attention between patches"]),
        ]
        for i, (name, sub, pts) in enumerate(cols):
            x = 0.06 + i * 0.31
            fig.add_artist(plt.Rectangle((x, 0.16), 0.28, 0.58, transform=fig.transFigure,
                                         facecolor="#f3f6fa", edgecolor="#d6dde6"))
            fig.text(x + 0.02, 0.67, name, fontsize=22, fontweight="bold", color=ACCENT)
            fig.text(x + 0.02, 0.62, sub, fontsize=13, color=MUTED)
            for j, t in enumerate(pts):
                fig.text(x + 0.02, 0.53 - j * 0.085, "• " + t, fontsize=13.5, color=INK)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 4. Augmentation
        fig = slide("Image augmentation", "Random, label-preserving changes to each "
                    "training image, every epoch")
        bullets(fig, [
            "Rotation up to ±12°, shift up to 12%, scale 0.88–1.12, shear ±8°.",
            "Random erasing of a small patch (25% of images).",
            "No flips — a mirrored 2 or 7 is no longer the same digit.",
            "Validation and test images are never augmented.",
            "Extra robustness test: the same test images rotated ±20°, shifted "
            "15%, scaled 0.85–1.15.",
        ], width=44, size=16)
        image(fig, aug_png, [0.55, 0.10, 0.42, 0.70])
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 5. Results table
        fig = slide("Results on the 10,000-image test set")
        ax = fig.add_axes([0.05, 0.14, 0.90, 0.66]); ax.axis("off")
        header = ["Model", "Augment", "Params", "Epochs", "Train time",
                  "Test acc", "Errors", "Perturbed test"]
        cells = [[NAMES[r["model"]], "yes" if r["aug"] == "aug" else "no",
                  f"{r['params']:,}", str(r["epochs"]), f"{r['train_time_sec']:.0f}s",
                  f"{r['test_acc']:.2f}%", str(r["test_errors"]),
                  f"{r['perturbed_test_acc']:.2f}%"] for r in rows]
        tb = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(15); tb.scale(1, 2.6)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor("#d6dde6")
            if i == 0:
                c.set_facecolor(ACCENT); c.get_text().set_color("white")
                c.get_text().set_fontweight("bold")
            elif cells[i - 1][0] == NAMES[best["model"]] and \
                    cells[i - 1][1] == ("yes" if best["aug"] == "aug" else "no"):
                c.set_facecolor("#e3ecf8"); c.get_text().set_fontweight("bold")
        fig.text(0.05, 0.08, f"Device: {rows[0]['device']}. Best epoch chosen by "
                 "validation accuracy.", fontsize=12, color=MUTED)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 6. Accuracy chart
        fig = slide("Clean vs perturbed accuracy")
        image(fig, "results/accuracy_comparison.png", [0.04, 0.10, 0.92, 0.72])
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 7. Training curves
        fig = slide("Training curves", "Solid = validation, dashed = training")
        image(fig, "results/training_curves.png", [0.03, 0.12, 0.94, 0.66])
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 8. Best model errors
        fig = slide(f"Where the best model still fails",
                    f"{NAMES[best['model']]}{' + augmentation' if best['aug'] == 'aug' else ''}"
                    f": {best['test_errors']} mistakes out of 10,000")
        image(fig, f"results/confusion_{best['tag']}.png", [0.03, 0.08, 0.44, 0.70])
        image(fig, f"results/mistakes_{best['tag']}.png", [0.48, 0.18, 0.50, 0.55])
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 9. Takeaways (computed from the numbers)
        fig = slide("Takeaways")
        pts = []
        clean = {m: by[f"{m}_no_aug"]["test_acc"] for m in NAMES if f"{m}_no_aug" in by}
        if clean:
            order = sorted(clean, key=clean.get, reverse=True)
            pts.append("Without augmentation: " + " > ".join(
                f"{NAMES[m]} {clean[m]:.2f}%" for m in order) + ".")
        for m in NAMES:
            a, b = by.get(f"{m}_no_aug"), by.get(f"{m}_aug")
            if a and b:
                pts.append(f"{NAMES[m]}: augmentation changed clean test accuracy by "
                           f"{b['test_acc'] - a['test_acc']:+.2f} pts and perturbed "
                           f"accuracy by {b['perturbed_test_acc'] - a['perturbed_test_acc']:+.2f} pts.")
        pts.append(f"Best overall: {NAMES[best['model']]}"
                   f"{' with augmentation' if best['aug'] == 'aug' else ''} at "
                   f"{best['test_acc']:.2f}% ({best['test_errors']} errors).")
        bullets(fig, pts, size=17, width=78)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

    print(f"Saved {OUT} ({n - 1} slides)")


if __name__ == "__main__":
    main()
