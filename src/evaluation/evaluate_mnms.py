"""Zero-shot evaluation of the ACDC-trained model on M&Ms.

The model is applied to multi-vendor data from different hospitals with no
retraining and no fine tuning, to quantify the performance drop under domain
shift. Preprocessing is identical to the ACDC pipeline.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

from src.data.mnms import (MNMS_ROOT, detect_lv_label, load_metadata,
                           load_patient)
from src.data.preprocess import TARGET_SPACING, preprocess_slice
from src.evaluation.postprocess import keep_largest_component
from src.models.unet import NUM_CLASSES, build_unet, get_device

STRUCTURES = ["RV", "myocardium", "LV"]
OUT_PATH = "results/metrics/mnms_results.csv"
PER_VOLUME_PATH = "results/metrics/mnms_per_volume.csv"


def preprocess_patient(p, phase):
    """Preprocess one phase of one patient into stacked slices."""
    img, msk = p[f"{phase}_image"], p[f"{phase}_mask"]
    sx, sy = float(p["spacing"][0]), float(p["spacing"][1])
    images, masks = [], []
    for i in range(img.shape[2]):
        a, b = preprocess_slice(img[:, :, i], msk[:, :, i], (sx, sy))
        images.append(a)
        masks.append(b)
    return np.stack(images), np.stack(masks)


@torch.no_grad()
def predict(model, device, images, batch_size=16):
    preds = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        x = torch.as_tensor(chunk[:, None], dtype=torch.float32).to(device)
        preds.append(model(x).argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(preds)


def volume_metrics(pred, true, spacing_z):
    to_onehot = AsDiscrete(to_onehot=NUM_CLASSES)
    dm = DiceMetric(include_background=False, reduction="none")
    hm = HausdorffDistanceMetric(include_background=False, percentile=95,
                                 reduction="none")
    p = to_onehot(torch.as_tensor(pred)[None].float()).unsqueeze(0)
    t = to_onehot(torch.as_tensor(true)[None].float()).unsqueeze(0)
    dice = dm(y_pred=p, y=t)[0].numpy()
    hd = hm(y_pred=p, y=t,
            spacing=[spacing_z, TARGET_SPACING, TARGET_SPACING])[0].numpy()
    return dice, hd


def main(checkpoint="checkpoints/best.pt", postprocess=True):
    device = get_device()
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_unet(dropout=ck.get("dropout", 0.1)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"device {device}, checkpoint from epoch {ck.get('epoch')}")

    df = load_metadata()
    print(f"M&Ms patients: {len(df)}")

    # determine label convention once, empirically
    raw = []
    for _, row in df.head(6).iterrows():
        raw.append(load_patient(row["External code"], row, lv_label=None)["ed_mask"])
    lv_label, m1, m3, _ = detect_lv_label(raw)
    print(f"label convention: LV cavity is label {lv_label} "
          f"(enclosure {m1:.3f} vs {m3:.3f})")

    rows = []
    for n, (_, row) in enumerate(df.iterrows(), 1):
        code = row["External code"]
        p = load_patient(code, row, lv_label=lv_label)
        spacing_z = float(p["spacing"][2])
        for phase in ("ed", "es"):
            images, masks = preprocess_patient(p, phase)
            pred = predict(model, device, images)
            if postprocess:
                pred = keep_largest_component(pred)
            dice, hd = volume_metrics(pred, masks, spacing_z)
            rec = {
                "patient": code,
                "vendor": row["Vendor"],
                "vendor_name": row["VendorName"],
                "centre": row["Centre"],
                "pathology": row["Pathology"],
                "phase": phase,
            }
            for name, d, h in zip(STRUCTURES, dice, hd):
                rec[f"dice_{name}"] = round(float(d), 4)
                rec[f"hd95_{name}"] = round(float(h), 3)
            rec["dice_mean"] = round(float(np.nanmean(dice)), 4)
            rows.append(rec)
        if n % 10 == 0:
            print(f"  processed {n}/{len(df)} patients")

    Path(PER_VOLUME_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PER_VOLUME_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def agg(sel):
        out = {}
        for name in STRUCTURES:
            v = np.array([r[f"dice_{name}"] for r in sel], float)
            v = v[np.isfinite(v)]
            out[name] = v.mean() if len(v) else np.nan
        v = np.array([r["dice_mean"] for r in sel], float)
        out["mean"] = np.nanmean(v)
        return out

    overall = agg(rows)
    print("\nZero-shot on M&Ms, overall")
    print(f"  {'RV':11s} {overall['RV']:.4f}")
    print(f"  {'myocardium':11s} {overall['myocardium']:.4f}")
    print(f"  {'LV':11s} {overall['LV']:.4f}")
    print(f"  {'mean':11s} {overall['mean']:.4f}")

    print("\nBy vendor")
    print(f"{'vendor':16s} {'n':>4s} {'RV':>8s} {'MYO':>8s} {'LV':>8s} {'mean':>8s}")
    summary = []
    by_vendor = defaultdict(list)
    for r in rows:
        by_vendor[(r["vendor"], r["vendor_name"])].append(r)
    for (v, vname), sel in sorted(by_vendor.items()):
        a = agg(sel)
        label = f"{v} {vname}"
        print(f"{label:16s} {len(sel):4d} {a['RV']:8.4f} {a['myocardium']:8.4f} "
              f"{a['LV']:8.4f} {a['mean']:8.4f}")
        summary.append({"vendor": v, "vendor_name": vname, "n_volumes": len(sel),
                        **{k: round(float(x), 4) for k, x in a.items()}})

    summary.append({"vendor": "ALL", "vendor_name": "all vendors",
                    "n_volumes": len(rows),
                    **{k: round(float(x), 4) for k, x in overall.items()}})
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    print("\nReference, ACDC held out test set: mean Dice 0.8751")
    drop = 0.8751 - overall["mean"]
    print(f"Zero-shot drop on M&Ms: {drop:.4f} "
          f"({100 * drop / 0.8751:.1f}% relative)")
    print(f"\nSaved {OUT_PATH}")
    print(f"Saved {PER_VOLUME_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--no-postprocess", action="store_true")
    args = ap.parse_args()
    main(args.checkpoint, postprocess=not args.no_postprocess)
