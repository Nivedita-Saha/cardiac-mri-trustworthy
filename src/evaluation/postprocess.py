"""Post-processing for segmentation predictions."""

import numpy as np
from scipy.ndimage import label


def keep_largest_component(volume, classes=(1, 2, 3)):
    """Keep only the largest connected component for each foreground class.

    Anatomically motivated: a heart has exactly one right ventricle, one
    myocardium and one left ventricle cavity, so disconnected fragments are
    necessarily false positives. Components are computed in 3D across the
    reassembled volume rather than slice by slice, so genuine structures
    that thin out across slices are not fragmented.

    Dice is largely insensitive to small spurious components, but Hausdorff
    distance is not, so this step targets the latter.
    """
    out = volume.copy()
    for c in classes:
        mask = volume == c
        if not mask.any():
            continue
        labelled, n = label(mask)
        if n <= 1:
            continue
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        keep = sizes.argmax()
        out[mask & (labelled != keep)] = 0
    return out
