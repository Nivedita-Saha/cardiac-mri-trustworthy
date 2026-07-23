"""Patient level train, validation and test splits for ACDC.

Splits are made at patient level, never slice level: slices from the same
heart are highly correlated, so mixing them across splits would inflate
test performance. Splits are stratified by diagnosis group so that all
five conditions appear in each split in equal proportion.
"""

import json
from collections import defaultdict
from pathlib import Path

from src.data.acdc import list_patients, read_info

SPLIT_FILE = "results/metrics/splits.json"
SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def build_splits(root="data/raw/training", seed=SEED):
    """Return a dict mapping split name to a sorted list of patient names."""
    import random

    by_group = defaultdict(list)
    for path in list_patients(root):
        info = read_info(path)
        by_group[info["group"]].append(info["patient"])

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for group in sorted(by_group):
        patients = sorted(by_group[group])
        rng.shuffle(patients)
        n = len(patients)
        n_train = round(n * FRACTIONS["train"])
        n_val = round(n * FRACTIONS["val"])
        splits["train"] += patients[:n_train]
        splits["val"] += patients[n_train:n_train + n_val]
        splits["test"] += patients[n_train + n_val:]

    return {k: sorted(v) for k, v in splits.items()}


def save_splits(splits, path=SPLIT_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path=SPLIT_FILE):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    from src.data.acdc import list_patients, read_info

    splits = build_splits()
    save_splits(splits)

    groups = {read_info(p)["patient"]: read_info(p)["group"]
              for p in list_patients("data/raw/training")}

    all_assigned = [p for v in splits.values() for p in v]
    assert len(all_assigned) == 100, f"expected 100 patients, got {len(all_assigned)}"
    assert len(set(all_assigned)) == 100, "a patient appears in more than one split"

    print(f"{'split':6s} {'n':>4s}   group breakdown")
    for name in ("train", "val", "test"):
        counts = defaultdict(int)
        for p in splits[name]:
            counts[groups[p]] += 1
        breakdown = "  ".join(f"{g} {counts[g]}" for g in sorted(counts))
        print(f"{name:6s} {len(splits[name]):>4d}   {breakdown}")

    print("\nNo patient overlap between splits.")
    print(f"Saved to {SPLIT_FILE}")
