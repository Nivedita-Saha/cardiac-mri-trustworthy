"""Qualitative figures: best, median and worst test predictions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.visualise import MASK_CMAP
from src.evaluation.evaluate import group_into_volumes, predict_split
from src.evaluation.postprocess import keep_largest_component


def qualitative_figure(out_path="results/figures/test_predictions.png"):
    out = predict_split("test")
    volumes = group_into_volumes(out)
    for v in volumes:
        v["pred"] = keep_largest_component(v["pred"])

    df = pd.read_csv("results/metrics/test_per_volume.csv")
    df = df.sort_values("dice_mean").reset_index(drop=True)
    picks = [("worst", df.iloc[0]),
             ("median", df.iloc[len(df) // 2]),
             ("best", df.iloc[-1])]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10.5))
    for row, (tag, r) in enumerate(picks):
        v = next(x for x in volumes
                 if x["patient"] == r["patient"] and x["phase"] == r["phase"])
        # pick the mid slice that actually contains structures
        sums = [(v["true"][i] > 0).sum() for i in range(len(v["true"]))]
        s = int(np.argmax(sums))

        img = out["true"]  # placeholder, replaced below
        idx = [i for i in range(len(out["patient"]))
               if str(out["patient"][i]) == r["patient"]
               and str(out["phase"][i]) == r["phase"]
               and int(out["slice_index"][i]) == s]
        from src.data.cache import load_cache
        images = load_cache("test")["images"]
        image = images[idx[0]].astype(np.float32)

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_ylabel(
            f"{tag}\n{r['patient']} {r['phase'].upper()} ({r['group']})\n"
            f"Dice {r['dice_mean']:.3f}", fontsize=9)

        axes[row, 1].imshow(image, cmap="gray")
        axes[row, 1].imshow(v["true"][s], cmap=MASK_CMAP, vmin=0, vmax=3)

        axes[row, 2].imshow(image, cmap="gray")
        axes[row, 2].imshow(v["pred"][s], cmap=MASK_CMAP, vmin=0, vmax=3)

        for col in range(3):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    for col, title in enumerate(["image", "ground truth", "prediction"]):
        axes[0, col].set_title(title, fontsize=11)

    fig.suptitle("Test set predictions: red RV, green myocardium, blue LV",
                 fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


def training_curve(out_path="results/figures/training_curve.png"):
    df = pd.read_csv("results/metrics/training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Dice + cross entropy loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    for name in ["RV", "myocardium", "LV"]:
        axes[1].plot(df["epoch"], df[f"dice_{name}"], label=name)
    axes[1].plot(df["epoch"], df["dice_mean"], "k--", label="mean", linewidth=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    training_curve()
    qualitative_figure()
