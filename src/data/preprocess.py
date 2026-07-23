"""Preprocessing for ACDC: resample, normalise, crop or pad.

Design choices follow directly from the dataset exploration in Phase 1:
  - in-plane spacing varies 0.70 to 1.92 mm, so slices are resampled to 1.5 mm
  - intensity maxima vary 184 to 4025, so slices are normalised individually
  - image sizes vary 154-428 x 154-512, so slices are cropped or padded to 256
"""

import numpy as np
from scipy.ndimage import zoom

TARGET_SPACING = 1.5   # mm, in plane
TARGET_SIZE = 256      # pixels, square


def resample_slice(slice_2d, spacing_xy, target_spacing=TARGET_SPACING, is_mask=False):
    """Resample a 2D slice to a common in-plane pixel spacing.

    Masks use nearest neighbour so that label values stay integers.
    """
    factors = (spacing_xy[0] / target_spacing, spacing_xy[1] / target_spacing)
    if np.allclose(factors, 1.0, atol=1e-3):
        return slice_2d
    order = 0 if is_mask else 1
    out = zoom(slice_2d, factors, order=order, prefilter=not is_mask)
    return out.astype(np.uint8) if is_mask else out


def normalise_slice(slice_2d, low=1.0, high=99.0):
    """Clip to percentiles then z-score.

    Applied per slice because MRI intensities have no absolute meaning,
    unlike CT Hounsfield units. Percentile clipping removes outlier voxels
    that would otherwise dominate the standard deviation.
    """
    lo, hi = np.percentile(slice_2d, [low, high])
    clipped = np.clip(slice_2d, lo, hi)
    mean, std = clipped.mean(), clipped.std()
    if std < 1e-8:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - mean) / std).astype(np.float32)


def crop_or_pad(array_2d, size=TARGET_SIZE, pad_value=0):
    """Centre crop or zero pad a 2D array to size x size."""
    out = array_2d
    for axis in (0, 1):
        current = out.shape[axis]
        if current > size:
            start = (current - size) // 2
            out = out.take(range(start, start + size), axis=axis)
        elif current < size:
            total = size - current
            before = total // 2
            pad = [(0, 0), (0, 0)]
            pad[axis] = (before, total - before)
            out = np.pad(out, pad, mode="constant", constant_values=pad_value)
    return out


def preprocess_slice(image_2d, mask_2d, spacing_xy):
    """Run the full pipeline on one image and mask pair."""
    img = resample_slice(image_2d, spacing_xy, is_mask=False)
    msk = resample_slice(mask_2d, spacing_xy, is_mask=True)
    img = normalise_slice(img)
    img = crop_or_pad(img, pad_value=0)
    msk = crop_or_pad(msk, pad_value=0)
    return img.astype(np.float32), msk.astype(np.uint8)


def crop_retention(mask_2d, spacing_xy, size=TARGET_SIZE):
    """Fraction of labelled pixels surviving the crop or pad step alone.

    Resampling legitimately changes the pixel count, so this compares the
    mask after resampling against the mask after cropping. That isolates
    the only real risk here: centre cropping cutting off anatomy in
    patients with an unusually large field of view.
    """
    resampled = resample_slice(mask_2d, spacing_xy, is_mask=True)
    before = (resampled > 0).sum()
    if before == 0:
        return 1.0
    cropped = crop_or_pad(resampled, size=size, pad_value=0)
    return float((cropped > 0).sum()) / float(before)
