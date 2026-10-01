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

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Image shape:", sample["image"].shape)
    print("Image dtype:", sample["image"].dtype)

    print("Mask shape:", sample["mask"].shape)
    print("Mask dtype:", sample["mask"].dtype)

    print("Image min:", float(sample["image"].min()))
    print("Image max:", float(sample["image"].max()))

    print(
        "Mask values:",
        sample["mask"].unique().tolist(),
    )

    print("Scene:", sample["scene_id"])
    print("Tile:", sample["tile"])
    print("Has road:", sample["has_road"])


if __name__ == "__main__":
    main()