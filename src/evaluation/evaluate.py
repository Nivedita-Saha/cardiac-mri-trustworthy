"""Evaluate the trained U-Net on the held out test split.

Metrics are computed per volume rather than per slice. Slices are
reassembled into patient and phase volumes first, so that Dice and
Hausdorff distance are directly comparable with published ACDC results.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

from src.data.cache import load_cache
from src.data.dataset import ACDCSliceDataset
from src.evaluation.postprocess import keep_largest_component
from src.models.unet import NUM_CLASSES, build_unet, get_device

STRUCTURES = ["RV", "myocardium", "LV"]
TARGET_SPACING = 1.5
RESULTS_PATH = "results/metrics/test_results.csv"
PER_VOLUME_PATH = "results/metrics/test_per_volume.csv"


@torch.no_grad()
def predict_split(split, checkpoint="checkpoints/best.pt", batch_size=16):
    device = get_device()
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_unet(dropout=ck.get("dropout", 0.1)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = ACDCSliceDataset(split, augment=False)
    meta = load_cache(split)
    preds = np.zeros((len(ds), 256, 256), dtype=np.uint8)

    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        images = torch.stack([ds[i]["image"] for i in idx]).to(device)
        logits = model(images)
        preds[start:start + len(images)] = (
            logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        )

    return {
        "pred": preds,
        "true": meta["masks"],
        "patient": meta["patient"],
        "group": meta["group"],
        "phase": meta["phase"],
        "slice_index": meta["slice_index"],
        "spacing_z": meta["spacing_z"],
        "epoch": ck.get("epoch"),
    }


def group_into_volumes(out):
    """Reassemble slices into per patient, per phase volumes."""
    buckets = defaultdict(list)
    for i in range(len(out["pred"])):
        key = (str(out["patient"][i]), str(out["phase"][i]), str(out["group"][i]))
        buckets[key].append((int(out["slice_index"][i]), i))

    volumes = []
    for (patient, phase, group), items in sorted(buckets.items()):
        order = [i for _, i in sorted(items)]
        volumes.append({
            "patient": patient,
            "phase": phase,
            "group": group,
            "spacing_z": float(out["spacing_z"][order[0]]),
            "pred": np.stack([out["pred"][i] for i in order]),
            "true": np.stack([out["true"][i] for i in order]),
        })
    return volumes


def volume_metrics(volumes):
    """Per volume Dice and 95th percentile Hausdorff distance in mm."""
    to_onehot = AsDiscrete(to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="none"
    )

    rows = []
    for v in volumes:
        pred = to_onehot(torch.as_tensor(v["pred"])[None].float()).unsqueeze(0)
        true = to_onehot(torch.as_tensor(v["true"])[None].float()).unsqueeze(0)

        dice = dice_metric(y_pred=pred, y=true)[0].numpy()
        hd = hd_metric(
            y_pred=pred, y=true,
            spacing=[v["spacing_z"], TARGET_SPACING, TARGET_SPACING],
        )[0].numpy()
        dice_metric.reset()
        hd_metric.reset()

        row = {"patient": v["patient"], "phase": v["phase"], "group": v["group"]}
        for name, d, h in zip(STRUCTURES, dice, hd):
            row[f"dice_{name}"] = round(float(d), 4)
            row[f"hd95_{name}"] = round(float(h), 3)
        row["dice_mean"] = round(float(np.mean(dice)), 4)
        rows.append(row)
    return rows


def summarise(rows):
    def stat(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return vals.mean(), vals.std()

    summary = []
    for name in STRUCTURES:
        dm, dsd = stat(f"dice_{name}")
        hm, hsd = stat(f"hd95_{name}")
        summary.append({
            "structure": name,
            "dice_mean": round(dm, 4), "dice_std": round(dsd, 4),
            "hd95_mean_mm": round(hm, 3), "hd95_std_mm": round(hsd, 3),
        })
    dm, dsd = stat("dice_mean")
    summary.append({
        "structure": "mean", "dice_mean": round(dm, 4), "dice_std": round(dsd, 4),
        "hd95_mean_mm": "", "hd95_std_mm": "",
    })
    return summary


def main(split="test", checkpoint="checkpoints/best.pt"):
    out = predict_split(split, checkpoint)
    print(f"checkpoint from epoch {out['epoch']}")
    print(f"{split}: {len(out['pred'])} slices")

    volumes = group_into_volumes(out)
    print(f"reassembled into {len(volumes)} volumes "
          f"({len(set(v['patient'] for v in volumes))} patients x 2 phases)\n")

    raw_rows = volume_metrics(volumes)
    post_volumes = [dict(v, pred=keep_largest_component(v["pred"])) for v in volumes]
    post_rows = volume_metrics(post_volumes)
    raw, post = summarise(raw_rows), summarise(post_rows)

    print("Effect of keeping the largest connected component")
    print(f"{'structure':12s} {'Dice raw':>9s} {'Dice post':>10s} "
          f"{'HD95 raw':>10s} {'HD95 post':>10s}   (mm)")
    for r, q in zip(raw, post):
        if r["structure"] == "mean":
            print(f"{'mean':12s} {r['dice_mean']:9.4f} {q['dice_mean']:10.4f}")
        else:
            print(f"{r['structure']:12s} {r['dice_mean']:9.4f} {q['dice_mean']:10.4f} "
                  f"{r['hd95_mean_mm']:10.2f} {q['hd95_mean_mm']:10.2f}")

    Path(PER_VOLUME_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PER_VOLUME_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(post_rows[0].keys()))
        w.writeheader(); w.writerows(post_rows)
    with open(RESULTS_PATH, "w", newline="") as f:
        fields = ["structure", "dice_mean", "dice_std", "hd95_mean_mm", "hd95_std_mm",
                  "dice_mean_raw", "hd95_mean_mm_raw"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, q in zip(raw, post):
            row = dict(q)
            row["dice_mean_raw"] = r["dice_mean"]
            row["hd95_mean_mm_raw"] = r["hd95_mean_mm"]
            w.writerow(row)

    print("\nFinal test results (post-processed)")
    for q in post:
        if q["structure"] == "mean":
            print(f"  {'mean':12s} Dice {q['dice_mean']:.4f} +/- {q['dice_std']:.4f}")
        else:
            print(f"  {q['structure']:12s} Dice {q['dice_mean']:.4f} +/- {q['dice_std']:.4f}"
                  f"   HD95 {q['hd95_mean_mm']:6.2f} +/- {q['hd95_std_mm']:.2f} mm")

    print("\nDice by diagnosis group (post-processed)")
    by_group = defaultdict(list)
    for r in post_rows:
        by_group[r["group"]].append(r["dice_mean"])
    for g in sorted(by_group):
        vals = np.array(by_group[g])
        print(f"  {g:5s} {vals.mean():.4f}  (n={len(vals)})")

    print(f"\nSaved {RESULTS_PATH}")
    print(f"Saved {PER_VOLUME_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = ap.parse_args()
    main(args.split, args.checkpoint)
