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
    / "spacenet_occlusion_aware_positive_sampled.pt"
)


def collate_fn(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            probabilities.append(probs.cpu())
            targets.append(masks.cpu())

    probabilities = torch.cat(probabilities)
    targets = torch.cat(targets)

    print("Baseline threshold diagnosis")
    print("=" * 60)
    print("Probability min :", probabilities.min().item())
    print("Probability max :", probabilities.max().item())
    print("Probability mean:", probabilities.mean().item())

    print()
    print("Threshold    IoU       Dice      Precision   Recall")
    print("-" * 60)

    eps = 1e-8

    for threshold in [
        0.50,
        0.40,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.05,
    ]:
        predictions = (
            probabilities >= threshold
        )

        tp = ((predictions == 1) & (targets == 1)).sum().item()
        fp = ((predictions == 1) & (targets == 0)).sum().item()
        fn = ((predictions == 0) & (targets == 1)).sum().item()

        iou = tp / (tp + fp + fn + eps)

        dice = (2 * tp) / (
            2 * tp + fp + fn + eps
        )

        precision = tp / (
            tp + fp + eps
        )

        recall = tp / (
            tp + fn + eps
        )

        print(
            f"{threshold:>8.2f}   "
            f"{iou:>8.4f}  "
            f"{dice:>8.4f}  "
            f"{precision:>10.4f}  "
            f"{recall:>8.4f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()