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
        occlusion_enabled=True,
        occlusion_probability=1.0,
    )

    for index in range(len(dataset)):
        sample = dataset[index]

        if sample["has_road"] and sample["occlusion_effects"]:
            print("Occluded training sample found")
            print("=" * 50)
            print("Index:", index)
            print("Scene:", sample["scene_id"])
            print("Tile:", sample["tile"])
            print("Image:", sample["image"].shape)
            print("Mask:", sample["mask"].shape)
            print(
                "Mask values:",
                sample["mask"].unique().tolist(),
            )
            print(
                "Occlusion effects:",
                sample["occlusion_effects"],
            )
            return

    raise RuntimeError(
        "No occluded positive training sample was generated."
    )


if __name__ == "__main__":
    main()