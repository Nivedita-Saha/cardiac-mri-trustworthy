"""Extract biomarkers for every patient, from ground truth and predicted masks.

Both sources are computed so that the agreement between them can be measured
directly. This tests whether the segmentation model is accurate enough for
the clinical measurements derived from it to be trustworthy.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.acdc import read_info
from src.evaluation.biomarkers import compute_biomarkers
from src.evaluation.evaluate import group_into_volumes, predict_split
from src.evaluation.postprocess import keep_largest_component

OUT_PATH = "results/metrics/biomarkers.csv"
RAW_ROOT = "data/raw/training"


def biomarkers_for_split(split, volumes, source="pred"):
    """Return one row per patient for the given split and mask source."""
    by_patient = defaultdict(dict)
    for v in volumes:
        by_patient[v["patient"]][v["phase"]] = v

    rows = []
    for patient, phases in sorted(by_patient.items()):
        if "ed" not in phases or "es" not in phases:
            continue
        ed, es = phases["ed"], phases["es"]
        info = read_info(Path(RAW_ROOT) / patient)
        key = "pred" if source == "pred" else "true"

        b = compute_biomarkers(
            ed[key], es[key], ed["spacing_z"], info["height"], info["weight"]
        )
        rows.append({
            "patient": patient,
            "group": ed["group"],
            "split": split,
            "source": source,
            **{k: (round(v, 3) if np.isfinite(v) else "") for k, v in b.items()},
        })
    return rows


def main():
    all_rows = []
    for split in ("train", "val", "test"):
        # one inference pass per split, reused for both mask sources
        volumes = group_into_volumes(predict_split(split))
        for v in volumes:
            v["pred"] = keep_largest_component(v["pred"])
        for source in ("true", "pred"):
            rows = biomarkers_for_split(split, volumes, source)
            all_rows += rows
            print(f"{split:6s} {source:5s} {len(rows):3d} patients")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {OUT_PATH}  ({len(all_rows)} rows)")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    gt = df[df["source"] == "true"]

    print("\nGround truth biomarkers by diagnosis group")
    cols = ["lv_edv_ml", "lv_ef_pct", "rv_ef_pct", "lv_mass_g",
            "wall_thickness_max_mm"]
    header = " ".join(f"{c.replace('_', ' '):>21s}" for c in cols)
    print(f"{'group':6s} {header}")
    for g in ["NOR", "DCM", "HCM", "MINF", "RV"]:
        sub = gt[gt["group"] == g]
        if sub.empty:
            continue
        line = f"{g:6s} "
        for c in cols:
            line += f"{pd.to_numeric(sub[c], errors='coerce').mean():21.1f} "
        print(line)

    print("\nAgreement of predicted biomarkers against ground truth")
    merged = df[df["source"] == "pred"].merge(
        df[df["source"] == "true"], on=["patient", "group", "split"],
        suffixes=("_p", "_t"))
    print(f"{'biomarker':26s} {'bias':>9s} {'95% limits of agreement':>26s} {'r':>8s}")
    for c in ["lv_edv_ml", "lv_esv_ml", "lv_ef_pct", "rv_edv_ml", "rv_ef_pct",
              "lv_mass_g", "wall_thickness_max_mm"]:
        a = pd.to_numeric(merged[f"{c}_p"], errors="coerce")
        b = pd.to_numeric(merged[f"{c}_t"], errors="coerce")
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        diff = a - b
        bias, sd = diff.mean(), diff.std()
        r = np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan
        print(f"{c:26s} {bias:9.2f}   [{bias - 1.96 * sd:8.2f}, "
              f"{bias + 1.96 * sd:8.2f}] {r:8.3f}")


if __name__ == "__main__":
    main()
