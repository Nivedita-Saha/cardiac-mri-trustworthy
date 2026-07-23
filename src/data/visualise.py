"""Visualisation helpers for ACDC cardiac MRI."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Transparent background, then red RV, green myocardium, blue LV
MASK_CMAP = ListedColormap([
    (0, 0, 0, 0),
    (0.90, 0.20, 0.20, 0.55),
    (0.20, 0.75, 0.30, 0.55),
    (0.20, 0.45, 0.95, 0.55),
])


def show_slices(patient, phase="ed", out_path=None):
    """Plot every slice for one cardiac phase, image on top, overlay below."""
    image = patient[f"{phase}_image"]
    mask = patient[f"{phase}_mask"]
    n_slices = image.shape[2]

    fig, axes = plt.subplots(2, n_slices, figsize=(2 * n_slices, 4.5))
    if n_slices == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_slices):
        axes[0, i].imshow(image[:, :, i], cmap="gray")
        axes[0, i].set_title(f"slice {i}", fontsize=9)

        axes[1, i].imshow(image[:, :, i], cmap="gray")
        axes[1, i].imshow(mask[:, :, i], cmap=MASK_CMAP, vmin=0, vmax=3)

        for row in (0, 1):
            axes[row, i].axis("off")

    title = (
        f"{patient['patient']}  |  group {patient['group']}  |  "
        f"{phase.upper()} phase  |  red RV, green myocardium, blue LV"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved to {out_path}")

    return fig
