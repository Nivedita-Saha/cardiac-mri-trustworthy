"""Training loop for the ACDC segmentation U-Net."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose

from src.data.dataset import make_loader
from src.models.unet import NUM_CLASSES, build_unet, count_parameters, get_device

STRUCTURES = ["RV", "myocardium", "LV"]
CHECKPOINT_DIR = "checkpoints"
HISTORY_PATH = "results/metrics/training_history.csv"


def run_epoch(model, loader, loss_fn, optimiser, device, scaler=None):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        optimiser.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimiser.step()

        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, metric):
    """Return mean loss and per-structure Dice, excluding background."""
    model.eval()
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])
    metric.reset()

    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)
        logits = model(images)
        total += loss_fn(logits, labels).item() * images.size(0)
        n += images.size(0)

        preds = [post_pred(i) for i in decollate_batch(logits)]
        targets = [post_label(i) for i in decollate_batch(labels)]
        metric(y_pred=preds, y=targets)

    per_class = metric.aggregate().cpu().numpy()
    return total / max(n, 1), per_class


def main(epochs=40, batch_size=16, lr=1e-3, dropout=0.1, workers=0, limit=None):
    device = get_device()
    print(f"device: {device}")

    train_loader = make_loader("train", batch_size=batch_size, augment=True,
                               num_workers=workers)
    val_loader = make_loader("val", batch_size=batch_size, augment=False,
                             num_workers=workers)

    if limit is not None:  # smoke test mode
        from torch.utils.data import DataLoader, Subset
        train_loader = DataLoader(Subset(train_loader.dataset, range(limit)),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(val_loader.dataset, range(limit)),
                                batch_size=batch_size, shuffle=False)

    model = build_unet(dropout=dropout).to(device)
    print(f"parameters: {count_parameters(model):,}")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    metric = DiceMetric(include_background=False, reduction="mean_batch")
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)

    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, optimiser, device, scaler)
        val_loss, dice = evaluate(model, val_loader, loss_fn, device, metric)
        scheduler.step()
        mean_dice = float(np.mean(dice))

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "dice_mean": round(mean_dice, 4),
            **{f"dice_{s}": round(float(d), 4) for s, d in zip(STRUCTURES, dice)},
            "lr": round(scheduler.get_last_lr()[0], 6),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)

        marker = ""
        if mean_dice > best:
            best = mean_dice
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "dice": mean_dice, "dropout": dropout},
                       Path(CHECKPOINT_DIR) / "best.pt")
            marker = "  <- best"

        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"dice {mean_dice:.4f}  (RV {dice[0]:.3f} MYO {dice[1]:.3f} "
              f"LV {dice[2]:.3f})  {row['seconds']:.0f}s{marker}")

    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"\nbest mean Dice: {best:.4f}")
    print(f"checkpoint: {CHECKPOINT_DIR}/best.pt")
    print(f"history: {HISTORY_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N slices per split, for a quick smoke test")
    args = ap.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.dropout, args.workers, args.limit)
