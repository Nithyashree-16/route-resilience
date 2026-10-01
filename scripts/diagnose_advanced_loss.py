from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.road_dataset import RoadDataset
from src.models.losses import (
    DiceLoss,
    SoftIoULoss,
    BoundaryLoss,
)
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


def collate_fn(batch):
    return {
        "image": torch.stack(
            [x["image"] for x in batch]
        ),
        "mask": torch.stack(
            [x["mask"] for x in batch]
        ),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="train",
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

    model = OcclusionAwareUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        transformer_dim=256,
    ).to(device)

    model.train()

    dice_loss = DiceLoss().to(device)
    iou_loss = SoftIoULoss().to(device)
    boundary_loss = BoundaryLoss().to(device)

    print("Advanced loss component diagnosis")
    print("=" * 60)

    for batch_index, batch in enumerate(loader):
        if batch_index >= 10:
            break

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        model.zero_grad(set_to_none=True)

        logits = model(images)

        dice = dice_loss(logits, masks)
        iou = iou_loss(logits, masks)
        boundary = boundary_loss(logits, masks)

        total = (
            0.4 * dice
            + 0.3 * iou
            + 0.3 * boundary
        )

        total.backward()

        grad_values = []

        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_values.append(
                    parameter.grad.detach().abs().mean().item()
                )

        mean_gradient = (
            sum(grad_values) / len(grad_values)
            if grad_values
            else 0.0
        )

        probability = torch.sigmoid(logits)

        print(
            f"Batch {batch_index + 1:02d} | "
            f"Dice={dice.item():.6f} | "
            f"IoU={iou.item():.6f} | "
            f"Boundary={boundary.item():.6f} | "
            f"Total={total.item():.6f} | "
            f"MeanGrad={mean_gradient:.8f} | "
            f"ProbMean={probability.mean().item():.6f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()