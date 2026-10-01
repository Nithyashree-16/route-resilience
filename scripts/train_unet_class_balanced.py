import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.road_dataset import RoadDataset
from src.models.losses import ClassBalancedRoadLoss
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

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "spacenet_unet_classbalanced_best.pt"
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


def validate(
    model,
    loader,
    criterion,
    device,
    use_amp,
):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=use_amp,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=use_amp,
            )

            with autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"

    print("SpaceNet Class-Balanced U-Net")
    print("=" * 60)
    print("Device:", device)

    if use_amp:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    train_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="train",
        stats_path=STATS,
        occlusion_enabled=False,
        occlusion_probability=0.0,
    )

    val_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="val",
        stats_path=STATS,
        occlusion_enabled=False,
        occlusion_probability=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
    ).to(device)

    criterion = ClassBalancedRoadLoss(
        pos_weight=30.335043,
        dice_weight=0.5,
        bce_weight=0.5,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )

    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(
                device,
                non_blocking=use_amp,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=use_amp,
            )

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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_loss": val_loss,
                    "pos_weight": 30.335043,
                },
                BEST_CHECKPOINT,
            )

            print(
                f"  -> Best checkpoint saved "
                f"(val_loss={val_loss:.4f})"
            )

    print("=" * 60)
    print("Class-balanced training complete.")
    print(
        "Best validation loss:",
        best_val_loss,
    )
    print(
        "Checkpoint:",
        BEST_CHECKPOINT,
    )
    print("=" * 60)


if __name__ == "__main__":
    main()