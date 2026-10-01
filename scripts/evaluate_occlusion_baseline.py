from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.occlusion import apply_occlusions
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
    / "spacenet_unet_baseline_best.pt"
)


def collate_fn(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch]
        ),
        "mask": torch.stack(
            [item["mask"] for item in batch]
        ),
        "scene_id": [
            item["scene_id"] for item in batch
        ],
        "tile": [
            item["tile"] for item in batch
        ],
    }


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
):
    tp = (
        (predictions == 1)
        & (targets == 1)
    ).sum().item()

    fp = (
        (predictions == 1)
        & (targets == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        & (targets == 1)
    ).sum().item()

    eps = 1e-8

    iou = tp / (
        tp + fp + fn + eps
    )

    dice = (2 * tp) / (
        2 * tp + fp + fn + eps
    )

    recall = tp / (
        tp + fn + eps
    )

    return tp, fp, fn, iou, dice, recall


def load_raw_image(
    relative_path: str,
):
    path = PROJECT_ROOT / relative_path

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    return image


def normalize_image(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:

    image = (
        image - mean[:, None, None]
    ) / std[:, None, None]

    return torch.from_numpy(
        image.copy()
    ).float()


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("SpaceNet baseline occlusion evaluation")
    print("=" * 60)
    print("Device:", device)

    dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="val",
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

    mean = dataset.mean.astype(
        np.float32
    )

    std = dataset.std.astype(
        np.float32
    )

    clean_predictions = []
    clean_targets = []

    occluded_predictions = []
    occluded_targets = []
    occlusion_recall_predictions = []
    occlusion_recall_targets = []

    threshold = 0.25

    evaluated_tiles = 0

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader
        ):

            image_path = batch["scene_id"][0]
            tile_name = batch["tile"][0]

            relative_image_path = (
                dataset.data.iloc[batch_index][
                    "image_path"
                ]
            )

            raw_image = load_raw_image(
                relative_image_path
            )

            mask = batch["mask"][0].numpy()[0]

            # --------------------------------------------------
            # CLEAN IMAGE
            # --------------------------------------------------

            clean_image = normalize_image(
                raw_image,
                mean,
                std,
            ).unsqueeze(0).to(device)

            clean_logits = model(
                clean_image
            )

            clean_probabilities = torch.sigmoid(
                clean_logits
            )

            clean_pred = (
                clean_probabilities >= threshold
            ).float().cpu()[0, 0]

            clean_target = torch.from_numpy(
                mask
            ).float()

            clean_predictions.append(
                clean_pred
            )

            clean_targets.append(
                clean_target
            )

            # --------------------------------------------------
            # OCCLUDED IMAGE
            # --------------------------------------------------

            display_image = (
                raw_image.transpose(
                    1,
                    2,
                    0,
                )
            )

            display_min = display_image.min()
            display_max = display_image.max()

            display_image = (
                display_image - display_min
            ) / (
                display_max
                - display_min
                + 1e-8
            )

            # Retry until we get an actual
            # visual occlusion effect rather
            # than illumination-only augmentation.
            for attempt in range(10):

                occluded_display, effects = (
                    apply_occlusions(
                        display_image,
                        min_effects=1,
                        max_effects=3,
                    )
                )

                visual_effects = [
                    x
                    for x in effects
                    if x != "illumination"
                ]

                if visual_effects:
                    break

            # Estimate the actual modified
            # regions from the image difference.
            difference = np.abs(
                occluded_display
                - display_image
            )

            occlusion_mask = (
                difference.max(axis=2)
                > 0.015
            )

            # Keep only actual occlusion effects.
            if not visual_effects:
                continue

            occluded_raw = (
                occluded_display
                * (
                    display_max
                    - display_min
                )
                + display_min
            )

            occluded_raw = (
                occluded_raw.transpose(
                    2,
                    0,
                    1,
                ).astype(np.float32)
            )

            occluded_image = normalize_image(
                occluded_raw,
                mean,
                std,
            ).unsqueeze(0).to(device)

            occluded_logits = model(
                occluded_image
            )

            occluded_probabilities = torch.sigmoid(
                occluded_logits
            )

            occluded_pred = (
                occluded_probabilities >= threshold
            ).float().cpu()[0, 0]

            occluded_predictions.append(
                occluded_pred
            )

            occluded_targets.append(
                clean_target
            )

            # --------------------------------------------------
            # OCCLUDED ROAD PIXELS ONLY
            # --------------------------------------------------

            occluded_road_region = (
                torch.from_numpy(
                    occlusion_mask
                ).bool()
                & (clean_target == 1)
            )

            occlusion_recall_predictions.append(
                occluded_pred[
                    occluded_road_region
                ]
            )

            occlusion_recall_targets.append(
                clean_target[
                    occluded_road_region
                ]
            )

            evaluated_tiles += 1

            if evaluated_tiles % 10 == 0:
                print(
                    f"Evaluated occluded tiles: "
                    f"{evaluated_tiles}"
                )

    # ------------------------------------------------------
    # CLEAN METRICS
    # ------------------------------------------------------

    clean_predictions = torch.stack(
        clean_predictions
    )

    clean_targets = torch.stack(
        clean_targets
    )

    (
        clean_tp,
        clean_fp,
        clean_fn,
        clean_iou,
        clean_dice,
        clean_recall,
    ) = compute_metrics(
        clean_predictions,
        clean_targets,
    )

    # ------------------------------------------------------
    # OCCLUDED METRICS
    # ------------------------------------------------------

    occluded_predictions = torch.stack(
        occluded_predictions
    )

    occluded_targets = torch.stack(
        occluded_targets
    )

    (
        occ_tp,
        occ_fp,
        occ_fn,
        occ_iou,
        occ_dice,
        occ_recall,
    ) = compute_metrics(
        occluded_predictions,
        occluded_targets,
    )

    # ------------------------------------------------------
    # OCCLUSION RECALL
    # ------------------------------------------------------

    if occlusion_recall_predictions:

        occlusion_predictions = torch.cat(
            occlusion_recall_predictions
        )

        occlusion_targets = torch.cat(
            occlusion_recall_targets
        )

        occlusion_tp = (
            (
                (occlusion_predictions == 1)
                & (occlusion_targets == 1)
            )
            .sum()
            .item()
        )

        occlusion_fn = (
            (
                (occlusion_predictions == 0)
                & (occlusion_targets == 1)
            )
            .sum()
            .item()
        )

        occlusion_recall = (
            occlusion_tp
            / (
                occlusion_tp
                + occlusion_fn
                + 1e-8
            )
        )

    else:
        occlusion_recall = 0.0

    print()
    print("=" * 60)
    print("SPACE NET BASELINE — OCCLUSION EVALUATION")
    print("=" * 60)

    print("Threshold:", threshold)
    print("Occluded tiles:", evaluated_tiles)

    print()
    print("CLEAN")
    print(f"TP:        {clean_tp:,}")
    print(f"FP:        {clean_fp:,}")
    print(f"FN:        {clean_fn:,}")
    print(f"IoU:       {clean_iou:.6f}")
    print(f"Dice:      {clean_dice:.6f}")
    print(f"Recall:    {clean_recall:.6f}")

    print()
    print("OCCLUDED")
    print(f"TP:        {occ_tp:,}")
    print(f"FP:        {occ_fp:,}")
    print(f"FN:        {occ_fn:,}")
    print(f"IoU:       {occ_iou:.6f}")
    print(f"Dice:      {occ_dice:.6f}")
    print(f"Recall:    {occ_recall:.6f}")

    print()
    print(
        "OCCLUSION RECALL:",
        f"{occlusion_recall:.6f}",
    )

    print("=" * 60)


if __name__ == "__main__":
    main()