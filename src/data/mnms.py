"""Loading utilities for the M&Ms multi-vendor cardiac dataset.

M&Ms differs from ACDC in three ways that matter:
  - images are 4D (height, width, slices, frames), with only the ED and ES
    frames annotated
  - ED and ES indices live in a shared CSV, not a per patient file, and are
    zero based rather than one based
  - the label convention is LV=1, myocardium=2, RV=3, which swaps LV and RV
    relative to ACDC

The label mapping is verified geometrically rather than assumed: the LV
cavity is enclosed by myocardium and the RV is not.
"""

import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

MNMS_ROOT = "data/raw/mnms"


def load_metadata(root=MNMS_ROOT):
    """Read the dataset CSV, restricted to patients present on disk."""
    csv_path = glob.glob(os.path.join(root, "*.csv"))
    if not csv_path:
        raise FileNotFoundError(f"no metadata CSV found in {root}")
    df = pd.read_csv(csv_path[0])
    have = {os.path.basename(p).replace("_sa.nii.gz", "")
            for p in glob.glob(os.path.join(root, "*_sa.nii.gz"))}
    df = df[df["External code"].isin(have)].reset_index(drop=True)
    return df


def enclosure_score(mask_2d, label, myo_label=2):
    """Fraction of a structure's immediate surroundings that is myocardium."""
    m = mask_2d == label
    if not m.any():
        return np.nan
    ring = binary_dilation(m, iterations=2) & ~m
    if not ring.any():
        return np.nan
    return float((mask_2d[ring] == myo_label).mean())


def detect_lv_label(masks, n_check=40):
    """Determine which label is the LV cavity, geometrically.

    Returns the label value that behaves like the LV cavity, that is the one
    enclosed by myocardium. Averaged over several slices for robustness.
    """
    s1, s3 = [], []
    checked = 0
    for m in masks:
        for i in range(m.shape[2] if m.ndim == 3 else 1):
            sl = m[:, :, i] if m.ndim == 3 else m
            if (sl == 2).sum() < 50:
                continue
            a, b = enclosure_score(sl, 1), enclosure_score(sl, 3)
            if np.isfinite(a):
                s1.append(a)
            if np.isfinite(b):
                s3.append(b)
            checked += 1
            if checked >= n_check:
                break
        if checked >= n_check:
            break
    mean1 = float(np.mean(s1)) if s1 else 0.0
    mean3 = float(np.mean(s3)) if s3 else 0.0
    return (1 if mean1 > mean3 else 3), mean1, mean3, checked


def remap_to_acdc(mask, lv_label):
    """Convert a mask to the ACDC convention: 1=RV, 2=myocardium, 3=LV."""
    if lv_label == 3:
        return mask.copy()          # already ACDC convention
    out = mask.copy()
    out[mask == 1] = 3
    out[mask == 3] = 1
    return out


def load_patient(code, row, root=MNMS_ROOT, lv_label=None):
    """Load one M&Ms patient's ED and ES frames with masks."""
    root = Path(root)
    img = nib.load(str(root / f"{code}_sa.nii.gz"))
    gt = nib.load(str(root / f"{code}_sa_gt.nii.gz"))
    spacing = img.header.get_zooms()[:3]

    data = img.get_fdata()
    labels = gt.get_fdata().astype(np.uint8)
    ed, es = int(row["ED"]), int(row["ES"])     # zero based

    out = {
        "patient": code,
        "vendor": row["Vendor"],
        "vendor_name": row["VendorName"],
        "centre": row["Centre"],
        "pathology": row["Pathology"],
        "spacing": spacing,
        "ed_frame": ed,
        "es_frame": es,
        "ed_image": data[..., ed],
        "es_image": data[..., es],
        "ed_mask": labels[..., ed],
        "es_mask": labels[..., es],
    }
    if lv_label is not None:
        out["ed_mask"] = remap_to_acdc(out["ed_mask"], lv_label)
        out["es_mask"] = remap_to_acdc(out["es_mask"], lv_label)
    return out


if __name__ == "__main__":
    df = load_metadata()
    print(f"patients on disk: {len(df)}")
    print(df.groupby(["Vendor", "VendorName"]).size().to_string())

    # load a handful of raw masks and determine the convention empirically
    raw = []
    for _, row in df.head(6).iterrows():
        p = load_patient(row["External code"], row, lv_label=None)
        raw.append(p["ed_mask"])

    lv_label, m1, m3, n = detect_lv_label(raw)
    print(f"\nConvention check over {n} slices")
    print(f"  label 1 enclosed by myocardium: {m1:.3f}")
    print(f"  label 3 enclosed by myocardium: {m3:.3f}")
    print(f"  -> LV cavity is label {lv_label}")
    print(f"  -> {'remapping required' if lv_label == 1 else 'already ACDC convention'}")

    p = load_patient(df.iloc[0]["External code"], df.iloc[0], lv_label=lv_label)
    print(f"\nExample: {p['patient']} ({p['vendor_name']}, {p['pathology']})")
    print(f"  image shape {p['ed_image'].shape}, spacing {tuple(round(float(s),3) for s in p['spacing'])}")
    print(f"  ED frame {p['ed_frame']}, ES frame {p['es_frame']}")
    print(f"  mask labels {sorted(set(p['ed_mask'].ravel().tolist()))}")
    for name, lab in [("RV", 1), ("myocardium", 2), ("LV", 3)]:
        print(f"  {name:11s} voxels {(p['ed_mask'] == lab).sum():6d}")
