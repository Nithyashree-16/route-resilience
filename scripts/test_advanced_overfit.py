import sys
from pathlib import Path

import torch
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.road_dataset import RoadDataset
from src.models.losses import AdvancedRoadLoss
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


def dice_iou(logits, targets, threshold=0.5):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    return iou, dice


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

    # Find a known positive tile.
    index = next(
        i
        for i in range(len(dataset))
        if dataset[i]["has_road"]
    )

    sample = dataset[index]

    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)

    print("Advanced-model one-tile overfit test")
    print("=" * 60)
    print("Device:", device)
    print("Index:", index)
    print("Scene:", sample["scene_id"])
    print("Tile:", sample["tile"])
    print("Image:", image.shape)
    print("Mask:", mask.shape)

    model = OcclusionAwareUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        transformer_dim=256,
    ).to(device)

    criterion = AdvancedRoadLoss(
        dice_weight=0.4,
        iou_weight=0.3,
        boundary_weight=0.3,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)

        logits = model(image)

        loss = criterion(
            logits,
            mask,
        )

        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0:
            iou, dice = dice_iou(
                logits.detach(),
                mask,
            )

            print(
                f"Step {step:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"IoU: {iou:.6f} | "
                f"Dice: {dice:.6f}"
            )

    print("=" * 60)
    print("Overfit test complete.")


if __name__ == "__main__":
    main()