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
        "has_road": [
            item["has_road"] for item in batch
        ],
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

    all_logits = []
    all_probabilities = []
    road_tile_probabilities = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_logits.append(logits.cpu())
            all_probabilities.append(
                probabilities.cpu()
            )

            for i, has_road in enumerate(
                batch["has_road"]
            ):
                if has_road:
                    road_tile_probabilities.append(
                        probabilities[i].cpu()
                    )

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    )

    print("Baseline prediction diagnosis")
    print("=" * 60)

    print("Logit min:", logits.min().item())
    print("Logit max:", logits.max().item())
    print("Logit mean:", logits.mean().item())

    print()

    print(
        "Probability min:",
        probabilities.min().item(),
    )

    print(
        "Probability max:",
        probabilities.max().item(),
    )

    print(
        "Probability mean:",
        probabilities.mean().item(),
    )

    print()

    for threshold in [
        0.50,
        0.30,
        0.20,
        0.10,
        0.05,
    ]:
        predicted_fraction = (
            probabilities >= threshold
        ).float().mean().item()

        print(
            f"Threshold {threshold:0.2f} "
            f"-> predicted road pixels: "
            f"{predicted_fraction:.6%}"
        )

    print()

    if road_tile_probabilities:
        road_probs = torch.cat(
            road_tile_probabilities,
            dim=0,
        )

        print(
            "Validation road-containing tiles:",
            len(road_tile_probabilities),
        )

        print(
            "Positive-tile probability min:",
            road_probs.min().item(),
        )

        print(
            "Positive-tile probability max:",
            road_probs.max().item(),
        )

        print(
            "Positive-tile probability mean:",
            road_probs.mean().item(),
        )

    print("=" * 60)


if __name__ == "__main__":
    main()