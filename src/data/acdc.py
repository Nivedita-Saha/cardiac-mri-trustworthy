"""Loading utilities for the ACDC cardiac MRI dataset."""

from pathlib import Path
import nibabel as nib
import numpy as np

# Label values used in the ACDC ground truth masks
LABELS = {0: "background", 1: "RV", 2: "myocardium", 3: "LV"}

# The five diagnosis groups
GROUPS = ["NOR", "MINF", "DCM", "HCM", "RV"]


def read_info(patient_dir):
    """Read Info.cfg into a dictionary with sensible types."""
    patient_dir = Path(patient_dir)
    info = {}
    with open(patient_dir / "Info.cfg") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

    return {
        "patient": patient_dir.name,
        "ed_frame": int(info["ED"]),
        "es_frame": int(info["ES"]),
        "group": info["Group"],
        "height": float(info["Height"]),
        "weight": float(info["Weight"]),
        "n_frames": int(info["NbFrame"]),
    }


def load_volume(path):
    """Load a NIfTI file. Returns the array and the voxel spacing in mm."""
    img = nib.load(str(path))
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    return data, spacing


def load_patient(patient_dir):
    """Load one patient's ED and ES images with their ground truth masks.

    Returns a dictionary containing the metadata from Info.cfg plus four
    arrays (ed_image, ed_mask, es_image, es_mask) and the voxel spacing.
    """
    patient_dir = Path(patient_dir)
    info = read_info(patient_dir)
    name = patient_dir.name

    ed = f"{name}_frame{info['ed_frame']:02d}"
    es = f"{name}_frame{info['es_frame']:02d}"

    ed_image, spacing = load_volume(patient_dir / f"{ed}.nii.gz")
    ed_mask, _ = load_volume(patient_dir / f"{ed}_gt.nii.gz")
    es_image, _ = load_volume(patient_dir / f"{es}.nii.gz")
    es_mask, _ = load_volume(patient_dir / f"{es}_gt.nii.gz")

    return {
        **info,
        "spacing": spacing,
        "ed_image": ed_image,
        "ed_mask": ed_mask.astype(np.uint8),
        "es_image": es_image,
        "es_mask": es_mask.astype(np.uint8),
    }


def list_patients(root):
    """Return a sorted list of patient directories under the given root."""
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("patient"))
