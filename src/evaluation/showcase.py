"""Showcase figures: 3D heart reconstruction and accuracy by slice position."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from src.evaluation.evaluate import group_into_volumes, predict_split
from src.evaluation.postprocess import keep_largest_component

STRUCTURES = {1: ("RV", "#E63333"), 2: ("myocardium", "#33BF4D"), 3: ("LV", "#3373F2")}
# The myocardium encloses the LV cavity, so it is drawn semi-transparent and
# last, otherwise the LV is completely occluded in the 3D view.
ALPHA_3D = {1: 0.45, 2: 0.22, 3: 0.95}
TARGET_SPACING = 1.5


def _add_surface(ax, volume, label_value, colour, spacing_z, alpha=0.55):
    """Extract and draw an isosurface for one label using marching cubes."""
    mask = (volume == label_value).astype(np.float32)
    if mask.sum() < 10:
        return None
    mask = np.pad(mask, 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = marching_cubes(
            mask, level=0.5, spacing=(spacing_z, TARGET_SPACING, TARGET_SPACING)
        )
    except (RuntimeError, ValueError):
        return None
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(colour)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    return verts


def render_3d(out_path="results/figures/heart_3d.png", patient=None):
    """Side by side 3D reconstruction of ground truth and predicted anatomy."""
    out = predict_split("test")
    volumes = group_into_volumes(out)
    for v in volumes:
        v["pred"] = keep_largest_component(v["pred"])

    ed = [v for v in volumes if v["phase"] == "ed"]
    v = (next(x for x in ed if x["patient"] == patient) if patient
         else max(ed, key=lambda x: (x["true"] > 0).sum()))

    fig = plt.figure(figsize=(12, 6))
    for col, (key, title) in enumerate([("true", "Ground truth"),
                                        ("pred", "Prediction")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        allv = []
        for value in (3, 1, 2):
            name, colour = STRUCTURES[value]
            got = _add_surface(ax, v[key], value, colour, v["spacing_z"],
                               alpha=ALPHA_3D[value])
            if got is not None:
                allv.append(got)
        if allv:
            pts = np.concatenate(allv)
            for setter, axis in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], range(3)):
                lo, hi = pts[:, axis].min(), pts[:, axis].max()
                pad = 0.1 * (hi - lo) + 1
                setter(lo - pad, hi + pad)
        ax.set_box_aspect((1, 1.4, 1.4))
        ax.view_init(elev=18, azim=-62)
        ax.set_title(title, fontsize=13, pad=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.fill = False
            pane.pane.set_edgecolor("#DDDDDD")

    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=11,
                      markerfacecolor=c, markeredgecolor="none", label=n)
               for n, c in STRUCTURES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"3D reconstruction from predicted segmentations\n"
                 f"{v['patient']}, {v['group']}, end diastole", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def slice_dice(pred, true, value):
    """Dice for one class on one slice. NaN when the class is absent in both."""
    p, t = pred == value, true == value
    denom = p.sum() + t.sum()
    if denom == 0:
        return np.nan
    return 2.0 * (p & t).sum() / denom


def slice_position_figure(out_path="results/figures/dice_by_slice_position.png"):
    """Accuracy from base to apex, alongside structure size to explain it."""
    out = predict_split("test")
    volumes = group_into_volumes(out)
    for v in volumes:
        v["pred"] = keep_largest_component(v["pred"])

    rows = []
    for v in volumes:
        n = len(v["true"])
        for i in range(n):
            p = i / max(n - 1, 1)   # 0 = base, 1 = apex
            for value, (name, _) in STRUCTURES.items():
                d = slice_dice(v["pred"][i], v["true"][i], value)
                if not np.isnan(d):
                    rows.append((p, name, d))

    pos = np.array([r[0] for r in rows])
    names = np.array([r[1] for r in rows])
    dice = np.array([r[2] for r in rows])

    bins = np.linspace(0, 1, 6)
    centres = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for name, colour in STRUCTURES.values():
        m = names == name
        means, stds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = m & (pos >= lo) & (pos <= hi)
            means.append(dice[sel].mean() if sel.any() else np.nan)
            stds.append(dice[sel].std() if sel.any() else np.nan)
        means, stds = np.array(means), np.array(stds)
        axes[0].plot(centres, means, "-o", color=colour, label=name, linewidth=2)
        axes[0].fill_between(centres, means - stds, means + stds,
                             color=colour, alpha=0.15)

    axes[0].set_xlabel("relative slice position   (0 = base, 1 = apex)")
    axes[0].set_ylabel("Dice")
    axes[0].set_title("Segmentation accuracy from base to apex")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    sizes = {n: [] for n, _ in STRUCTURES.values()}
    positions = {n: [] for n, _ in STRUCTURES.values()}
    for v in volumes:
        n_sl = len(v["true"])
        for i in range(n_sl):
            for value, (name, _) in STRUCTURES.items():
                area = (v["true"][i] == value).sum() * TARGET_SPACING ** 2 / 100.0
                if area > 0:
                    sizes[name].append(area)
                    positions[name].append(i / max(n_sl - 1, 1))

    for name, colour in STRUCTURES.values():
        p = np.array(positions[name]); s = np.array(sizes[name])
        means = [s[(p >= lo) & (p <= hi)].mean() if ((p >= lo) & (p <= hi)).any()
                 else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
        axes[1].plot(centres, means, "-o", color=colour, label=name, linewidth=2)

    axes[1].set_xlabel("relative slice position   (0 = base, 1 = apex)")
    axes[1].set_ylabel("mean structure area (cm$^2$)")
    axes[1].set_title("Structure size from base to apex")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Where the model struggles, and why", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\nDice by slice position")
    print(f"{'position':>10s} " + " ".join(f"{n:>12s}" for n, _ in STRUCTURES.values()))
    for lo, hi in zip(bins[:-1], bins[1:]):
        line = f"{lo:.1f}-{hi:.1f}   "
        for name, _ in STRUCTURES.values():
            sel = (names == name) & (pos >= lo) & (pos <= hi)
            line += f"{dice[sel].mean():12.3f} " if sel.any() else f"{'-':>12s} "
        print(line)


if __name__ == "__main__":
    render_3d()
    slice_position_figure()
