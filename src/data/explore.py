"""Dataset level exploration of ACDC."""

import numpy as np
import pandas as pd
from src.data.acdc import list_patients, load_patient


def build_summary(root="data/raw/training"):
    """Return a dataframe with one row per patient."""
    rows = []
    for path in list_patients(root):
        p = load_patient(path)
        h, w, n_slices = p["ed_image"].shape
        rows.append({
            "patient": p["patient"],
            "group": p["group"],
            "height_cm": p["height"],
            "weight_kg": p["weight"],
            "n_frames": p["n_frames"],
            "ed_frame": p["ed_frame"],
            "es_frame": p["es_frame"],
            "rows": h,
            "cols": w,
            "n_slices": n_slices,
            "spacing_x": round(float(p["spacing"][0]), 4),
            "spacing_y": round(float(p["spacing"][1]), 4),
            "spacing_z": round(float(p["spacing"][2]), 4),
            "intensity_min": float(p["ed_image"].min()),
            "intensity_max": float(p["ed_image"].max()),
            "intensity_mean": round(float(p["ed_image"].mean()), 2),
            "unlabelled_ed_slices": int(sum(
                p["ed_mask"][:, :, i].sum() == 0 for i in range(n_slices)
            )),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_summary()
    df.to_csv("results/metrics/dataset_summary.csv", index=False)

    print("PATIENTS:", len(df))
    print("\nDIAGNOSIS GROUPS")
    print(df["group"].value_counts().to_string())
    print("\nIMAGE DIMENSIONS")
    print("  rows      :", df["rows"].min(), "to", df["rows"].max())
    print("  cols      :", df["cols"].min(), "to", df["cols"].max())
    print("  slices    :", df["n_slices"].min(), "to", df["n_slices"].max())
    print("\nVOXEL SPACING (mm)")
    print("  in-plane  :", df["spacing_x"].min(), "to", df["spacing_x"].max())
    print("  slice gap :", df["spacing_z"].min(), "to", df["spacing_z"].max())
    print("\nINTENSITY")
    print("  max value :", df["intensity_max"].min(), "to", df["intensity_max"].max())
    print("\nUNLABELLED ED SLICES PER PATIENT")
    print(df["unlabelled_ed_slices"].value_counts().sort_index().to_string())
    print("\nSaved to results/metrics/dataset_summary.csv")
