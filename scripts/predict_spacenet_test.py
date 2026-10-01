from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
import torch

from src.data.road_dataset import RoadDataset
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

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "spacenet_occlusion_aware_positive_sampled.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_predictions"
)

THRESHOLD = 0.25

BASE_CHANNELS = 16
TRANSFORMER_DIM = 256


def main() -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("SpaceNet Test Prediction")
    print("=" * 72)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="test",
        stats_path=STATS,
        occlusion_enabled=False,
        occlusion_probability=0.0,
    )

    print(
        "Test tiles:",
        len(dataset),
    )

    model = OcclusionAwareUNet(
        in_channels=3,
        out_channels=1,
        base_channels=BASE_CHANNELS,
        transformer_dim=TRANSFORMER_DIM,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    total_positive_pixels = 0
    total_tiles = 0

    with torch.no_grad():

        for index in range(len(dataset)):

            sample = dataset[index]

            image = (
                sample["image"]
                .unsqueeze(0)
                .to(device)
            )

            scene_id = str(
                sample["scene_id"]
            )

            tile_name = str(
                sample["tile"]
            )

            logits = model(image)

            probability = (
                torch.sigmoid(logits)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            road_mask = (
                probability >= THRESHOLD
            ).astype(np.uint8)

            image_path = Path(
                dataset._resolve_path(
                    str(
                        dataset.data.iloc[index][
                            "image_path"
                        ]
                    )
                )
            )

            with rasterio.open(
                image_path
            ) as src:

                profile = src.profile.copy()

                profile.update(
                    count=1,
                    dtype="float32",
                    compress="lzw",
                    nodata=None,
                )

                probability_path = (
                    OUTPUT_DIR
                    / (
                        f"{Path(tile_name).stem}"
                        "_probability.tif"
                    )
                )

                with rasterio.open(
                    probability_path,
                    "w",
                    **profile,
                ) as dst:

                    dst.write(
                        probability,
                        1,
                    )

                mask_profile = (
                    profile.copy()
                )

                mask_profile.update(
                    dtype="uint8",
                    compress="lzw",
                )

                mask_path = (
                    OUTPUT_DIR
                    / (
                        f"{Path(tile_name).stem}"
                        "_road_mask.tif"
                    )
                )

                with rasterio.open(
                    mask_path,
                    "w",
                    **mask_profile,
                ) as dst:

                    dst.write(
                        road_mask,
                        1,
                    )

            total_positive_pixels += int(
                road_mask.sum()
            )

            total_tiles += 1

            print(
                f"[{total_tiles:02d}/{len(dataset):02d}] "
                f"{scene_id} | {tile_name} | "
                f"predicted road pixels: "
                f"{int(road_mask.sum()):,}"
            )

    print("=" * 72)
    print("Prediction complete.")
    print("Test tiles:", total_tiles)
    print(
        "Predicted road pixels:",
        f"{total_positive_pixels:,}",
    )
    print(
        "Output directory:",
        OUTPUT_DIR,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()