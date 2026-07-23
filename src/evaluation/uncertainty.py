"""Monte Carlo dropout uncertainty for cardiac segmentation.

Dropout was enabled during training, so uncertainty can be estimated by
keeping dropout active at inference time and running several stochastic
forward passes. The spread across passes indicates where the model is
unsure. No retraining is required.

Two measures are computed: the predictive entropy of the mean softmax
distribution, and the variance of predicted probabilities across passes.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation

from src.data.cache import load_cache
from src.data.dataset import ACDCSliceDataset
from src.models.unet import NUM_CLASSES, build_unet, get_device

N_PASSES = 20
EPS = 1e-8


def enable_dropout(model):
    """Put only the dropout layers back into training mode."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_predict(split, checkpoint="checkpoints/best.pt", n_passes=N_PASSES,
               batch_size=16):
    """Run several stochastic passes, returning mean prediction and uncertainty."""
    device = get_device()
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_unet(dropout=ck.get("dropout", 0.1)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_layers = enable_dropout(model)

    ds = ACDCSliceDataset(split, augment=False)
    n = len(ds)
    mean_prob = np.zeros((n, NUM_CLASSES, 256, 256), dtype=np.float32)
    sq_prob = np.zeros((n, NUM_CLASSES, 256, 256), dtype=np.float32)

    for start in range(0, n, batch_size):
        idx = range(start, min(start + batch_size, n))
        images = torch.stack([ds[i]["image"] for i in idx]).to(device)
        acc = torch.zeros(len(images), NUM_CLASSES, 256, 256, device=device)
        acc_sq = torch.zeros_like(acc)
        for _ in range(n_passes):
            p = torch.softmax(model(images), dim=1)
            acc += p
            acc_sq += p ** 2
        mean_prob[start:start + len(images)] = (acc / n_passes).cpu().numpy()
        sq_prob[start:start + len(images)] = (acc_sq / n_passes).cpu().numpy()

    var = np.clip(sq_prob - mean_prob ** 2, 0, None)
    entropy = -(mean_prob * np.log(mean_prob + EPS)).sum(axis=1)
    return {
        "pred": mean_prob.argmax(axis=1).astype(np.uint8),
        "entropy": entropy.astype(np.float32),
        "variance": var.sum(axis=1).astype(np.float32),
        "n_dropout_layers": n_layers,
    }


def roi_mask(pred_slice, margin=12):
    """Region where the segmentation decision is actually made.

    Mean uncertainty over a whole slice is dominated by confident background,
    and apical slices are mostly background, so a slice-wide average measures
    structure size rather than confidence. Restricting to the predicted
    foreground plus a margin removes that confound.
    """
    fg = pred_slice > 0
    if not fg.any():
        return np.ones_like(fg, dtype=bool)
    return binary_dilation(fg, iterations=margin)


def slice_dice(pred, true, classes=(1, 2, 3)):
    """Mean Dice over foreground classes present in either mask."""
    vals = []
    for c in classes:
        p, t = pred == c, true == c
        denom = p.sum() + t.sum()
        if denom > 0:
            vals.append(2.0 * (p & t).sum() / denom)
    return float(np.mean(vals)) if vals else np.nan


def main(split="test", n_passes=N_PASSES):
    out = mc_predict(split, n_passes=n_passes)
    print(f"dropout layers reactivated: {out['n_dropout_layers']}")
    print(f"{split}: {len(out['pred'])} slices, {n_passes} stochastic passes")

    meta = load_cache(split)
    true = meta["masks"]

    rows = []
    for i in range(len(out["pred"])):
        rows.append({
            "patient": str(meta["patient"][i]),
            "phase": str(meta["phase"][i]),
            "slice_index": int(meta["slice_index"][i]),
            "dice": slice_dice(out["pred"][i], true[i]),
            "mean_entropy": float(out["entropy"][i].mean()),
            "max_entropy": float(out["entropy"][i].max()),
            "mean_variance": float(out["variance"][i].mean()),
            "roi_entropy": float(out["entropy"][i][roi_mask(out["pred"][i])].mean()),
            "roi_variance": float(out["variance"][i][roi_mask(out["pred"][i])].mean()),
        })

    counts = defaultdict(int)
    for r in rows:
        counts[(r["patient"], r["phase"])] += 1
    for r in rows:
        n_sl = counts[(r["patient"], r["phase"])]
        r["position"] = round(r["slice_index"] / max(n_sl - 1, 1), 4)

    valid = [r for r in rows if np.isfinite(r["dice"])]
    dice = np.array([r["dice"] for r in valid])
    ent = np.array([r["mean_entropy"] for r in valid])
    var = np.array([r["mean_variance"] for r in valid])
    roi_ent = np.array([r["roi_entropy"] for r in valid])
    roi_var = np.array([r["roi_variance"] for r in valid])
    pos = np.array([r["position"] for r in valid])

    print("\nDoes uncertainty predict error?")
    print(f"  correlation, entropy vs Dice : {np.corrcoef(ent, dice)[0,1]:+.3f}")
    print(f"  correlation, variance vs Dice: {np.corrcoef(var, dice)[0,1]:+.3f}")
    print("  restricted to the decision region:")
    print(f"  correlation, ROI entropy vs Dice : {np.corrcoef(roi_ent, dice)[0,1]:+.3f}")
    print(f"  correlation, ROI variance vs Dice: {np.corrcoef(roi_var, dice)[0,1]:+.3f}")

    print("\nDice by uncertainty quartile (entropy)")
    ent = roi_ent   # use the region-restricted measure from here on
    q = np.quantile(ent, [0.25, 0.5, 0.75])
    bounds = [(-np.inf, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], np.inf)]
    for name, (lo, hi) in zip(["lowest", "second", "third", "highest"], bounds):
        sel = (ent > lo) & (ent <= hi)
        print(f"  {name:8s} uncertainty  Dice {dice[sel].mean():.3f}  (n={sel.sum()})")

    print("\nUncertainty by slice position")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        sel = (pos >= lo) & (pos <= hi)
        if sel.any():
            print(f"  {lo:.1f}-{hi:.1f}  entropy {ent[sel].mean():.4f}  "
                  f"Dice {dice[sel].mean():.3f}")

    print("\nReferral simulation")
    worst = dice < np.quantile(dice, 0.10)
    for frac in (0.05, 0.10, 0.20):
        k = int(len(ent) * frac)
        flagged = np.zeros(len(ent), bool)
        flagged[np.argsort(ent)[::-1][:k]] = True
        caught = (flagged & worst).sum() / max(worst.sum(), 1)
        print(f"  reviewing the most uncertain {frac*100:4.0f}% catches "
              f"{caught*100:5.1f}% of the worst 10% of slices")

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    with open("results/metrics/uncertainty.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nSaved results/metrics/uncertainty.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--passes", type=int, default=N_PASSES)
    args = ap.parse_args()
    main(args.split, args.passes)
