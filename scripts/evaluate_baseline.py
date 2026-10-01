import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.road_dataset import RoadDataset
from src.models.unet import UNet


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

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "spacenet_unet_smoketest.pt"
)


def collate_fn(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch]
        ),
        "mask": torch.stack(
            [item["mask"] for item in batch]
        ),
    }


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
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    threshold = 0.5

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= threshold
            ).float()

            true_positive += int(
                ((predictions == 1) & (masks == 1)).sum()
            )

            false_positive += int(
                ((predictions == 1) & (masks == 0)).sum()
            )

            false_negative += int(
                ((predictions == 0) & (masks == 1)).sum()
            )

            true_negative += int(
                ((predictions == 0) & (masks == 0)).sum()
            )

    eps = 1e-8

    iou = true_positive / (
        true_positive
        + false_positive
        + false_negative
        + eps
    )

    dice = (2 * true_positive) / (
        2 * true_positive
        + false_positive
        + false_negative
        + eps
    )

    precision = true_positive / (
        true_positive
        + false_positive
        + eps
    )

    recall = true_positive / (
        true_positive
        + false_negative
        + eps
    )

    print("SpaceNet Baseline U-Net — Validation Evaluation")
    print("=" * 60)
    print("Device:", device)
    print("Validation tiles:", len(dataset))
    print("Threshold:", threshold)
    print()
    print(f"TP:        {true_positive:,}")
    print(f"FP:        {false_positive:,}")
    print(f"FN:        {false_negative:,}")
    print(f"TN:        {true_negative:,}")
    print()
    print(f"IoU:       {iou:.6f}")
    print(f"Dice:      {dice:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()