import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.road_dataset import RoadDataset
from src.models.unet import UNet
from src.models.losses import BaselineRoadLoss


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

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def collate_fn(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
        "has_road": torch.tensor(
            [x["has_road"] for x in batch],
            dtype=torch.bool,
        ),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="train",
        stats_path=STATS,
        occlusion_enabled=True,
        occlusion_probability=0.5,
    )

    val_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="val",
        stats_path=STATS,
        occlusion_enabled=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
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

    criterion = BaselineRoadLoss().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scaler = GradScaler(
        enabled=torch.cuda.is_available()
    )

    epochs = 1

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                enabled=torch.cuda.is_available()
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_index % 25 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                with autocast(
                    enabled=torch.cuda.is_available()
                ):
                    logits = model(images)
                    loss = criterion(logits, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print("=" * 60)
        print(
            f"Epoch {epoch + 1} complete | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
        print("=" * 60)

    checkpoint = CHECKPOINT_DIR / "spacenet_unet_smoketest.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        checkpoint,
    )

    print("Training smoke test successful.")
    print("Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()