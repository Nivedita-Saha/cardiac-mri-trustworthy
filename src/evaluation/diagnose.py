"""Explainable diagnosis from cardiac biomarkers.

A Random Forest predicts the five ACDC diagnosis groups from the biomarkers
derived in the previous phase, and SHAP explains which measurements drive
each prediction.

Two evaluations are reported:
  1. Stratified cross validation on the train and validation patients using
     ground truth derived biomarkers, measuring how much diagnostic signal
     the biomarkers themselves carry.
  2. A fully end to end test: trained on ground truth derived biomarkers and
     evaluated on held out test patients using biomarkers derived from the
     model's own predicted segmentations, with no ground truth anywhere in
     the test path.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BIOMARKERS_PATH = "results/metrics/biomarkers.csv"
GROUPS = ["NOR", "MINF", "DCM", "HCM", "RV"]
SEED = 42

FEATURES = [
    "lv_edv_ml", "lv_esv_ml", "lv_sv_ml", "lv_ef_pct",
    "rv_edv_ml", "rv_esv_ml", "rv_sv_ml", "rv_ef_pct",
    "lv_mass_g", "wall_thickness_mean_mm", "wall_thickness_max_mm",
    "lv_edv_index", "lv_esv_index", "rv_edv_index", "lv_mass_index",
    "lv_rv_edv_ratio", "mass_to_volume",
]


def load_features(source, splits):
    df = pd.read_csv(BIOMARKERS_PATH)
    df = df[(df["source"] == source) & (df["split"].isin(splits))]
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return X.reset_index(drop=True), df["group"].reset_index(drop=True)


def build_model():
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def print_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=GROUPS)
    print(f"\n{title}")
    print(f"{'true|pred':>10s} " + " ".join(f"{g:>5s}" for g in GROUPS))
    for i, g in enumerate(GROUPS):
        print(f"{g:>10s} " + " ".join(f"{v:5d}" for v in cm[i]))
    return cm


def main():
    # ---------- 1. cross validation on ground truth biomarkers ----------
    X, y = load_features("true", ["train", "val"])
    print(f"cross validation set: {len(X)} patients, {len(FEATURES)} features")

    min_class = int(y.value_counts().min())
    n_splits = min(5, max(2, min_class))
    if n_splits < 5:
        print(f"  note: using {n_splits}-fold, smallest class has {min_class}")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_cv = cross_val_predict(build_model(), X, y, cv=cv)
    acc_cv = accuracy_score(y, y_cv)
    print(f"\n{n_splits}-fold cross validated accuracy: {acc_cv:.3f}")
    print_confusion(y, y_cv, "Cross validation confusion matrix")
    print("\n" + classification_report(y, y_cv, labels=GROUPS, zero_division=0))

    # ---------- 2. end to end test on predicted biomarkers ----------
    model = build_model().fit(X, y)
    X_test, y_test = load_features("pred", ["test"])
    y_hat = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_hat)
    print(f"End to end test accuracy (predicted segmentations): {acc_test:.3f} "
          f"({int(round(acc_test * len(y_test)))} of {len(y_test)})")
    print_confusion(y_test, y_hat, "End to end test confusion matrix")

    X_test_gt, y_test_gt = load_features("true", ["test"])
    acc_gt = accuracy_score(y_test_gt, model.predict(X_test_gt))
    print(f"\nSame patients with ground truth biomarkers: {acc_gt:.3f}")

    # ---------- 3. SHAP ----------
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    classes = list(model.classes_)

    mean_abs = np.abs(sv).mean(axis=0)          # (features, classes)
    overall = mean_abs.mean(axis=1)
    order = np.argsort(overall)[::-1]

    print("\nMost important biomarkers overall (mean |SHAP|)")
    for i in order[:8]:
        print(f"  {FEATURES[i]:26s} {overall[i]:.4f}")

    print("\nTop biomarkers per diagnosis group")
    for ci, cls in enumerate(classes):
        top = np.argsort(mean_abs[:, ci])[::-1][:3]
        print(f"  {cls:5s} {', '.join(FEATURES[i] for i in top)}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    k = 10
    idx = order[:k][::-1]
    axes[0].barh([FEATURES[i] for i in idx], overall[idx], color="#3373F2")
    axes[0].set_xlabel("mean |SHAP value|")
    axes[0].set_title("Overall biomarker importance")
    axes[0].grid(alpha=0.3, axis="x")

    bottom = np.zeros(k)
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for ci, cls in enumerate(classes):
        vals = mean_abs[idx, ci]
        axes[1].barh([FEATURES[i] for i in idx], vals, left=bottom,
                     label=cls, color=colours[ci % len(colours)])
        bottom += vals
    axes[1].set_xlabel("mean |SHAP value|, stacked by class")
    axes[1].set_title("Which biomarker explains which diagnosis")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle("Explaining the diagnosis: SHAP on cardiac biomarkers", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/shap_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved results/figures/shap_importance.png")

    summary = {
        "cv_accuracy": round(float(acc_cv), 4),
        "cv_folds": n_splits,
        "test_accuracy_predicted_masks": round(float(acc_test), 4),
        "test_accuracy_ground_truth_masks": round(float(acc_gt), 4),
        "n_cv_patients": int(len(X)),
        "n_test_patients": int(len(X_test)),
        "top_features": [FEATURES[i] for i in order[:8]],
    }
    with open("results/metrics/diagnosis_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results/metrics/diagnosis_results.json")


if __name__ == "__main__":
    main()
