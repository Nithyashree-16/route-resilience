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

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "spacenet_occlusion_aware_best.pt"
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
                loss = criterion(
                    logits,
                    masks,
                )

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"

    print("SpaceNet Occlusion-Aware Training")
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
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
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

    criterion = AdvancedRoadLoss(
        dice_weight=0.4,
        iou_weight=0.3,
        boundary_weight=0.3,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )

    epochs = 10
    accumulation_steps = 4

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

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

                raw_loss = criterion(
                    logits,
                    masks,
                )

                loss = (
                    raw_loss
                    / accumulation_steps
                )

            scaler.scale(loss).backward()

            if (
                step % accumulation_steps == 0
                or step == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += raw_loss.item()

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
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_loss": val_loss,
                },
                BEST_CHECKPOINT,
            )

            print(
                f"  -> Best checkpoint saved "
                f"(val_loss={val_loss:.4f})"
            )

    print("=" * 60)
    print("Advanced occlusion-aware training complete.")
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