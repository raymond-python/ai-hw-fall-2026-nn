"""Train one model on the MNIST TRAIN split.

Examples:
    python train.py --model mlp --epochs 10
    python train.py --model cnn --epochs 8 --augment
    python train.py --model transformer --epochs 15 --augment
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from data import get_device, set_seed, train_val_loaders
from models import build_model, count_params

DEFAULTS = {  # sensible per-model settings
    "mlp":         {"epochs": 10, "lr": 1e-3},
    "cnn":         {"epochs": 8,  "lr": 2e-3},
    "transformer": {"epochs": 15, "lr": 1e-3},
}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(DEFAULTS))
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--augment", action="store_true", help="use image augmentation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="saved_models")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    epochs = args.epochs or DEFAULTS[args.model]["epochs"]
    lr = args.lr or DEFAULTS[args.model]["lr"]
    tag = f"{args.model}_{'aug' if args.augment else 'no_aug'}"

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader = train_val_loaders(args.data_dir, args.augment,
                                                 args.batch_size, args.workers)
    model = build_model(args.model).to(device)
    n_params = count_params(model)
    print(f"[{tag}] device={device} params={n_params:,} epochs={epochs} lr={lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pth")
    history, best_val = [], -1.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss, train_acc = loss_sum / total, 100.0 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        flag = ""
        if val_acc > best_val:  # keep best epoch by VALIDATION accuracy only
            best_val = val_acc
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "augment": args.augment, "epoch": epoch, "val_acc": val_acc}, ckpt_path)
            flag = "  *saved"
        print(f"  epoch {epoch:2d}/{epochs}  train {train_acc:6.2f}%  "
              f"val {val_acc:6.2f}%  loss {train_loss:.4f}{flag}")

    train_time = time.time() - t0
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"history_{tag}.json"), "w") as f:
        json.dump({"tag": tag, "model": args.model, "augment": args.augment,
                   "params": n_params, "epochs": epochs, "lr": lr,
                   "train_time_sec": round(train_time, 1), "best_val_acc": best_val,
                   "device": str(device), "history": history}, f, indent=2)
    print(f"[{tag}] done in {train_time:.0f}s, best val {best_val:.2f}% -> {ckpt_path}")


if __name__ == "__main__":
    main()
