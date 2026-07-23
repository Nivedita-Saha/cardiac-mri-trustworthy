"""PyTorch datasets and loaders for the preprocessed ACDC slices."""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
from torch.utils.data import DataLoader, Dataset

from src.data.cache import load_cache

KEYS = ["image", "label"]


def train_transforms():
    """Augmentation pipeline for training.

    Geometric transforms are applied identically to image and label, with the
    label using nearest neighbour interpolation so class values stay integers.
    Intensity transforms are applied to the image only, and are motivated by
    the 20-fold variation in intensity maxima measured across the dataset.
    """
    return Compose([
        RandRotated(
            keys=KEYS,
            range_x=0.26,               # about 15 degrees
            prob=0.5,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
            keep_size=True,
        ),
        RandZoomd(
            keys=KEYS,
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.3,
            mode=["bilinear", "nearest"],
            keep_size=True,
        ),
        RandFlipd(keys=KEYS, spatial_axis=0, prob=0.5),
        RandFlipd(keys=KEYS, spatial_axis=1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])


class ACDCSliceDataset(Dataset):
    """Slice level dataset backed by a preprocessed cache archive."""

    def __init__(self, split, cache_dir="data/processed", augment=False):
        data = load_cache(split, cache_dir)
        self.images = data["images"]
        self.masks = data["masks"]
        self.patient = data["patient"]
        self.group = data["group"]
        self.phase = data["phase"]
        self.transform = train_transforms() if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)[None]   # (1, H, W)
        label = self.masks[idx].astype(np.float32)[None]    # (1, H, W)

        if self.transform is not None:
            out = self.transform({"image": image, "label": label})
            image, label = out["image"], out["label"]

        image = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
        label = torch.as_tensor(np.ascontiguousarray(label), dtype=torch.long)[0]
        return {"image": image, "label": label}


def make_loader(split, batch_size=16, augment=False, shuffle=None, num_workers=0):
    """Build a DataLoader. Shuffling defaults to on for augmented splits."""
    dataset = ACDCSliceDataset(split, augment=augment)
    if shuffle is None:
        shuffle = augment
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    for split in ("train", "val", "test"):
        ds = ACDCSliceDataset(split, augment=(split == "train"))
        print(f"{split:6s} {len(ds):5d} slices")

    print("\nBatch check")
    loader = make_loader("train", batch_size=8, augment=True)
    batch = next(iter(loader))
    img, lab = batch["image"], batch["label"]
    print("  image", tuple(img.shape), img.dtype,
          f"range [{img.min():.2f}, {img.max():.2f}]")
    print("  label", tuple(lab.shape), lab.dtype,
          "classes", sorted(set(lab.unique().tolist())))

    print("\nAugmentation integrity over 200 samples")
    ds = ACDCSliceDataset("train", augment=True)
    bad, shapes = set(), set()
    for i in range(200):
        s = ds[i % len(ds)]
        bad |= (set(s["label"].unique().tolist()) - {0, 1, 2, 3})
        shapes.add(tuple(s["image"].shape))
    print("  unexpected label values:", bad or "none")
    print("  image shapes seen:", shapes)

    ds_val = ACDCSliceDataset("val", augment=False)
    print("\nValidation is deterministic:",
          torch.equal(ds_val[0]["image"], ds_val[0]["image"]))
