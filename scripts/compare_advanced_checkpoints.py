from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.road_dataset import RoadDataset
from src.models.occlusion_unet import OcclusionAwareUNet


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

OLD_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "spacenet_occlusion_aware_best.pt"
)

NEW_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "spacenet_occlusion_aware_positive_sampled.pt"
)


def collate_fn(batch):
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
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> OcclusionAwareUNet:

    model = OcclusionAwareUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        transformer_dim=256,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="val",
        stats_path=STATS,
        occlusion_enabled=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    image = batch["image"].to(device)

    old_model = load_model(
        OLD_CHECKPOINT,
        device,
    )

    new_model = load_model(
        NEW_CHECKPOINT,
        device,
    )

    with torch.no_grad():
        old_probability = torch.sigmoid(
            old_model(image)
        )

        new_probability = torch.sigmoid(
            new_model(image)
        )

    difference = (
        old_probability
        - new_probability
    ).abs()

    print("=" * 60)
    print("ADVANCED CHECKPOINT PREDICTION COMPARISON")
    print("=" * 60)

    print("Device:", device)
    print("Scene:", batch["scene_id"][0])
    print("Tile:", batch["tile"][0])

    print()

    print("OLD CHECKPOINT")
    print(
        "Min :",
        old_probability.min().item(),
    )
    print(
        "Max :",
        old_probability.max().item(),
    )
    print(
        "Mean:",
        old_probability.mean().item(),
    )
    print(
        "Std :",
        old_probability.std().item(),
    )

    print()

    print("NEW POSITIVE-SAMPLED CHECKPOINT")
    print(
        "Min :",
        new_probability.min().item(),
    )
    print(
        "Max :",
        new_probability.max().item(),
    )
    print(
        "Mean:",
        new_probability.mean().item(),
    )
    print(
        "Std :",
        new_probability.std().item(),
    )

    print()

    print("DIFFERENCE")
    print(
        "Max absolute difference:",
        difference.max().item(),
    )

    print(
        "Mean absolute difference:",
        difference.mean().item(),
    )

    print("=" * 60)


if __name__ == "__main__":
    main()