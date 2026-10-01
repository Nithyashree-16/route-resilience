from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

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

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "spacenet_occlusion_aware_positive_sampled.pt"
)

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

EPOCHS = 5
BATCH_SIZE = 1
ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BASE_CHANNELS = 16
TRANSFORMER_DIM = 256

# Positive-tile sampling weight.
# Road-containing tiles receive 3x the sampling weight
# of empty tiles.
POSITIVE_WEIGHT = 3.0
NEGATIVE_WEIGHT = 1.0


def collate_fn(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch]
        ),
        "mask": torch.stack(
            [item["mask"] for item in batch]
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


def build_sampler(dataset):
    """
    Give road-containing tiles higher probability while
    retaining background-only tiles.
    """

    weights = []

    for index in range(len(dataset)):
        has_road = bool(
            dataset.data.iloc[index]["has_road"]
        )

        if has_road:
            weights.append(POSITIVE_WEIGHT)
        else:
            weights.append(NEGATIVE_WEIGHT)

    weights = torch.tensor(
        weights,
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )

    return sampler


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("SpaceNet Advanced Model — POSITIVE-SAMPLED DIAGNOSTIC")
    print("=" * 70)

    print("Device:", device)

    if use_amp:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Synthetic occlusion: False")
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print(
        "Gradient accumulation:",
        ACCUMULATION_STEPS,
    )
    print("Learning rate:", LEARNING_RATE)
    print("Positive tile weight:", POSITIVE_WEIGHT)
    print("Negative tile weight:", NEGATIVE_WEIGHT)

    print("=" * 70)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

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

    positive_count = int(
        train_dataset.data["has_road"].sum()
    )

    negative_count = (
        len(train_dataset)
        - positive_count
    )

    print(
        "Training positive tiles:",
        positive_count,
    )

    print(
        "Training empty tiles:",
        negative_count,
    )

    # --------------------------------------------------------
    # SAMPLER
    # --------------------------------------------------------

    sampler = build_sampler(
        train_dataset
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = OcclusionAwareUNet(
        in_channels=3,
        out_channels=1,
        base_channels=BASE_CHANNELS,
        transformer_dim=TRANSFORMER_DIM,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Model parameters:",
        f"{parameter_count:,}",
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = AdvancedRoadLoss(
        dice_weight=0.4,
        iou_weight=0.3,
        boundary_weight=0.3,
    ).to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

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
                    / ACCUMULATION_STEPS
                )

            scaler.scale(
                loss
            ).backward()

            should_step = (
                step % ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            )

            if should_step:
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

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
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
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": LEARNING_RATE,
                    "positive_weight": POSITIVE_WEIGHT,
                    "negative_weight": NEGATIVE_WEIGHT,
                    "synthetic_occlusion": False,
                },
                BEST_CHECKPOINT,
            )

            print(
                "  -> Best checkpoint saved:"
                f" {BEST_CHECKPOINT.name}"
            )

    print("=" * 70)
    print("Positive-sampled diagnostic complete.")
    print(
        "Best validation loss:",
        best_val_loss,
    )
    print(
        "Checkpoint:",
        BEST_CHECKPOINT,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()