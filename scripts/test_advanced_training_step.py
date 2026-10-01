import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

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

    use_amp = device.type == "cuda"

    print("Advanced model training-step test")
    print("=" * 60)
    print("Device:", device)

    if use_amp:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="train",
        stats_path=STATS,
        occlusion_enabled=True,
        occlusion_probability=1.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    model = OcclusionAwareUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        transformer_dim=256,
    ).to(device)

    criterion = AdvancedRoadLoss().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )

    batch = next(iter(loader))

    images = batch["image"].to(
        device,
        non_blocking=use_amp,
    )

    masks = batch["mask"].to(
        device,
        non_blocking=use_amp,
    )

    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)

    optimizer.zero_grad(
        set_to_none=True
    )

    with autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp,
    ):
        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

    print("Forward loss:", loss.item())

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            "GPU allocated:",
            f"{allocated:.3f} GB",
        )

        print(
            "GPU reserved:",
            f"{reserved:.3f} GB",
        )

    print("=" * 60)
    print(
        "PASS: advanced model training step "
        "completed successfully."
    )


if __name__ == "__main__":
    main()