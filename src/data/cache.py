"""Build preprocessed slice caches, one compressed archive per split.

Slices are preprocessed once and stored as float16 images with uint8 masks.
This keeps training fast (no repeated NIfTI decompression) and produces a
compact artefact that can be uploaded to a GPU environment without moving
the full raw dataset.
"""

import json
import time
from pathlib import Path

import numpy as np

from src.data.acdc import load_patient
from src.data.preprocess import preprocess_slice

CACHE_DIR = "data/processed"


def build_cache(split_name, patients, root="data/raw/training", out_dir=CACHE_DIR):
    """Preprocess every slice for the given patients and save one archive."""
    root = Path(root)
    images, masks, meta = [], [], []

    for name in patients:
        p = load_patient(root / name)
        sx, sy, _ = p["spacing"]
        for phase in ("ed", "es"):
            img, msk = p[f"{phase}_image"], p[f"{phase}_mask"]
            for i in range(img.shape[2]):
                a, b = preprocess_slice(img[:, :, i], msk[:, :, i], (sx, sy))
                images.append(a.astype(np.float16))
                masks.append(b)
                meta.append((p["patient"], p["group"], phase, i))

    if not images:
        raise ValueError(f"split '{split_name}' contains no slices")

    images = np.stack(images)
    masks = np.stack(masks)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        images=images,
        masks=masks,
        patient=np.array([m[0] for m in meta]),
        group=np.array([m[1] for m in meta]),
        phase=np.array([m[2] for m in meta]),
        slice_index=np.array([m[3] for m in meta], dtype=np.int16),
    )
    return out_path, images.shape


def load_cache(split_name, cache_dir=CACHE_DIR):
    """Load a preprocessed split into a dictionary of arrays."""
    data = np.load(Path(cache_dir) / f"{split_name}.npz", allow_pickle=False)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    with open("results/metrics/splits.json") as f:
        splits = json.load(f)

    total = 0.0
    for name in ("train", "val", "test"):
        t0 = time.time()
        path, shape = build_cache(name, splits[name])
        mb = path.stat().st_size / 1e6
        total += mb
        print(f"{name:6s} {shape[0]:5d} slices  {shape[1]}x{shape[2]}  "
              f"{mb:7.1f} MB  ({time.time() - t0:.0f}s)")
    print(f"{'total':6s} {'':5s}                    {total:7.1f} MB")

    d = load_cache("val")
    print("\nSanity check on the validation cache")
    print("  images        ", d["images"].shape, d["images"].dtype)
    print("  masks         ", d["masks"].shape, d["masks"].dtype)
    print("  labels present", sorted(set(d["masks"].ravel().tolist())))
    print("  patients      ", len(set(d["patient"].tolist())))
    print("  image mean    ", round(float(d["images"].astype(np.float32).mean()), 4))
    print("  image std     ", round(float(d["images"].astype(np.float32).std()), 4))
    empty = int(sum(d["masks"][i].sum() == 0 for i in range(len(d["masks"]))))
    print("  all-background slices", empty)
