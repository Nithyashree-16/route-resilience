import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.road_dataset import RoadDataset


MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "spacenet_paris_tiles.csv"
)

STATS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "spacenet_paris_band_stats.json"
)


def find_positive_index(dataset):
    for index in range(len(dataset)):
        sample = dataset[index]

        if sample["has_road"]:
            return index

    raise RuntimeError("No positive road tile found.")


def main():
    # Force occlusion probability to 1.0.
    # Training should therefore always get an occlusion.
    train_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="train",
        stats_path=STATS,
        occlusion_enabled=True,
        occlusion_probability=1.0,
    )

    # Even with occlusion_enabled=True, validation must remain clean.
    val_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="val",
        stats_path=STATS,
        occlusion_enabled=True,
        occlusion_probability=1.0,
    )

    train_index = find_positive_index(train_dataset)
    val_index = find_positive_index(val_dataset)

    train_sample = train_dataset[train_index]
    val_sample = val_dataset[val_index]

    print("Split behavior test")
    print("=" * 60)

    print("TRAIN")
    print("Scene:", train_sample["scene_id"])
    print("Tile:", train_sample["tile"])
    print("Has road:", train_sample["has_road"])
    print("Occlusion:", train_sample["occlusion_effects"])

    print()

    print("VALIDATION")
    print("Scene:", val_sample["scene_id"])
    print("Tile:", val_sample["tile"])
    print("Has road:", val_sample["has_road"])
    print("Occlusion:", val_sample["occlusion_effects"])

    print()

    train_mask_values = train_sample["mask"].unique().tolist()
    val_mask_values = val_sample["mask"].unique().tolist()

    print("Train mask values:", train_mask_values)
    print("Val mask values:", val_mask_values)

    if not train_sample["occlusion_effects"]:
        raise RuntimeError(
            "Training sample did not receive occlusion "
            "despite probability=1.0."
        )

    if val_sample["occlusion_effects"]:
        raise RuntimeError(
            "Validation sample received synthetic occlusion."
        )

    if train_mask_values not in ([0.0, 1.0], [0.0]):
        raise RuntimeError(
            f"Unexpected training mask values: {train_mask_values}"
        )

    if val_mask_values not in ([0.0, 1.0], [0.0]):
        raise RuntimeError(
            f"Unexpected validation mask values: {val_mask_values}"
        )

    print()
    print("PASS: training/validation occlusion separation is correct.")


if __name__ == "__main__":
    main()