import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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


def road_collate_fn(batch):
    """
    Stack tensor fields while preserving variable-length metadata.
    """
    return {
        "image": torch.stack(
            [item["image"] for item in batch]
        ),
        "mask": torch.stack(
            [item["mask"] for item in batch]
        ),
        "scene_id": [
            item["scene_id"] for item in batch
        ],
        "tile": [
            item["tile"] for item in batch
        ],
        "has_road": torch.tensor(
            [item["has_road"] for item in batch],
            dtype=torch.bool,
        ),
        "occlusion_effects": [
            item["occlusion_effects"] for item in batch
        ],
    }


def main():
    dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="train",
        stats_path=STATS,
        occlusion_enabled=True,
        occlusion_probability=0.5,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=road_collate_fn,
    )

    batch = next(iter(loader))

    print("DataLoader test successful")
    print("=" * 50)
    print("Dataset size:", len(dataset))
    print("Batch image shape:", batch["image"].shape)
    print("Batch mask shape:", batch["mask"].shape)
    print("Image dtype:", batch["image"].dtype)
    print("Mask dtype:", batch["mask"].dtype)
    print("Has road:", batch["has_road"])
    print("Scene IDs:", batch["scene_id"])
    print("Tiles:", batch["tile"])
    print("Occlusion effects:", batch["occlusion_effects"])
    print("=" * 50)


if __name__ == "__main__":
    main()