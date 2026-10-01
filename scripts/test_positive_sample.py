import sys
from pathlib import Path

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


def main():
    dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="train",
        stats_path=STATS,
    )

    positive_index = None

    for index in range(len(dataset)):
        sample = dataset[index]

        if sample["has_road"]:
            positive_index = index
            break

    if positive_index is None:
        raise RuntimeError(
            "No positive road tile found in training dataset."
        )

    sample = dataset[positive_index]

    road_pixels = int(sample["mask"].sum())
    total_pixels = sample["mask"].numel()

    print("Positive sample found")
    print("=" * 50)
    print("Dataset size:", len(dataset))
    print("Index:", positive_index)
    print("Scene:", sample["scene_id"])
    print("Tile:", sample["tile"])
    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Road pixels:", road_pixels)
    print("Road fraction:", f"{road_pixels / total_pixels:.4%}")
    print("Mask values:", sample["mask"].unique().tolist())


if __name__ == "__main__":
    main()