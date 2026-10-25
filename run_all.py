"""Train + test all 6 experiments (3 models x {no augmentation, augmentation}),
then write the summary table and comparison charts into results/.

    python run_all.py              # everything
    python run_all.py --quick      # 1 epoch each, just to check the pipeline
    python run_all.py --summary    # only rebuild the summary from existing results
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["mlp", "cnn", "transformer"]
NAMES = {"mlp": "MLP", "cnn": "CNN", "transformer": "Transformer"}
COLORS = {"no_aug": "#8aa4c8", "aug": "#1f4e8c"}


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load(path):
    with open(path) as f:
        return json.load(f)


def summarize():
    rows = []
    for m in MODELS:
        for aug in ["no_aug", "aug"]:
            tag = f"{m}_{aug}"
            h, t = f"results/history_{tag}.json", f"results/test_{tag}.json"
            if not (os.path.exists(h) and os.path.exists(t)):
                continue
            h, t = load(h), load(t)
            rows.append({"tag": tag, "model": m, "aug": aug, "params": h["params"],
                         "epochs": h["epochs"], "train_time_sec": h["train_time_sec"],
                         "best_val_acc": round(h["best_val_acc"], 2),
                         "test_acc": t["test_acc"], "test_errors": t["test_errors"],
                         "perturbed_test_acc": t["perturbed_test_acc"],
                         "history": h["history"], "device": h["device"]})
    if not rows:
        print("No results yet."); return

    # CSV + markdown table
    cols = ["tag", "params", "epochs", "train_time_sec", "best_val_acc",
            "test_acc", "test_errors", "perturbed_test_acc"]
    with open("results/summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    with open("results/summary.md", "w") as f:
        f.write("| Model | Augmentation | Params | Epochs | Train time | Val acc | "
                "**Test acc** | Test errors | Perturbed-test acc |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {NAMES[r['model']]} | {'yes' if r['aug'] == 'aug' else 'no'} | "
                    f"{r['params']:,} | {r['epochs']} | {r['train_time_sec']:.0f}s | "
                    f"{r['best_val_acc']:.2f}% | **{r['test_acc']:.2f}%** | "
                    f"{r['test_errors']} | {r['perturbed_test_acc']:.2f}% |\n")
        f.write(f"\nDevice: {rows[0]['device']}\n")
    with open("results/summary.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "history"} for r in rows], f, indent=2)

    by = {r["tag"]: r for r in rows}

    # Chart 1: clean vs perturbed test accuracy, grouped by model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [(axes[0], "test_acc", "Clean test accuracy"),
                           (axes[1], "perturbed_test_acc", "Perturbed test accuracy\n"
                            "(test images rotated ±20°, shifted, scaled)")]:
        w = 0.38
        for k, aug in enumerate(["no_aug", "aug"]):
            xs, vals = [], []
            for i, m in enumerate(MODELS):
                r = by.get(f"{m}_{aug}")
                if r:
                    xs.append(i + (k - 0.5) * w); vals.append(r[key])
            bars = ax.bar(xs, vals, w, color=COLORS[aug],
                          label="with augmentation" if aug == "aug" else "no augmentation")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        vals_all = [r[key] for r in rows]
        ax.set_ylim(max(0, min(vals_all) - 3), 100.4)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([NAMES[m] for m in MODELS])
        ax.set_ylabel("accuracy (%)"); ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("results/accuracy_comparison.png", dpi=150); plt.close(fig)

    # Chart 2: validation accuracy per epoch
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, m in zip(axes, MODELS):
        for aug in ["no_aug", "aug"]:
            r = by.get(f"{m}_{aug}")
            if not r:
                continue
            ep = [e["epoch"] for e in r["history"]]
            ax.plot(ep, [e["val_acc"] for e in r["history"]], "-o", ms=3,
                    color=COLORS[aug], label=("aug" if aug == "aug" else "no aug") + " · val")
            ax.plot(ep, [e["train_acc"] for e in r["history"]], "--", lw=1,
                    color=COLORS[aug], alpha=0.7,
                    label=("aug" if aug == "aug" else "no aug") + " · train")
        ax.set_title(NAMES[m]); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    lo = min(min(e["val_acc"] for e in r["history"]) for r in rows)
    axes[0].set_ylim(max(90, lo - 0.5), 100); axes[0].set_ylabel("accuracy (%)")
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("results/training_curves.png", dpi=150); plt.close(fig)

    # Put the table into README.md between the markers
    table = open("results/summary.md").read()
    if os.path.exists("README.md"):
        readme = open("README.md").read()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in readme and b in readme:
            readme = readme[:readme.index(a) + len(a)] + "\n" + table + readme[readme.index(b):]
            open("README.md", "w").write(readme)

    print("\n" + table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--workers", default="2")
    args = p.parse_args()
    os.makedirs("results", exist_ok=True)

    if not args.summary:
        for m in MODELS:
            for aug in [False, True]:
                tag = f"{m}_{'aug' if aug else 'no_aug'}"
                cmd = [sys.executable, "train.py", "--model", m, "--workers", args.workers]
                if aug:
                    cmd.append("--augment")
                if args.quick:
                    cmd += ["--epochs", "1"]
                run(cmd)
                run([sys.executable, "test.py", "--model", m,
                     "--model-path", f"saved_models/{tag}.pth", "--workers", args.workers])
    summarize()


if __name__ == "__main__":
    main()
