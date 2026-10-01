from pathlib import Path
import json

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris"
    / "train"
    / "images"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "spacenet_paris_band_stats.json"
)


def main():
    image_files = sorted(IMAGE_ROOT.rglob("*.tif"))

    if not image_files:
        raise FileNotFoundError(
            f"No training TIFFs found in {IMAGE_ROOT}"
        )

    # Running statistics for each RGB band.
    pixel_count = np.zeros(3, dtype=np.int64)
    sum_values = np.zeros(3, dtype=np.float64)
    sum_squared = np.zeros(3, dtype=np.float64)

    global_min = np.full(3, np.inf)
    global_max = np.full(3, -np.inf)

    for index, image_path in enumerate(image_files, start=1):
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float64)

        if image.shape[0] != 3:
            raise ValueError(
                f"Expected 3 bands, got {image.shape[0]} "
                f"in {image_path}"
            )

        for band in range(3):
            values = image[band].ravel()

            pixel_count[band] += values.size
            sum_values[band] += values.sum()
            sum_squared[band] += np.square(values).sum()

            global_min[band] = min(
                global_min[band],
                values.min(),
            )
            global_max[band] = max(
                global_max[band],
                values.max(),
            )

        if index % 25 == 0 or index == len(image_files):
            print(
                f"Processed {index}/{len(image_files)} "
                f"training images"
            )

    mean = sum_values / pixel_count

    variance = (
        sum_squared / pixel_count
        - np.square(mean)
    )

    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    stats = {
        "dataset": "SpaceNet 3 Paris sample",
        "split": "train",
        "num_images": len(image_files),
        "num_bands": 3,
        "bands": ["red", "green", "blue"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": global_min.tolist(),
        "max": global_max.tolist(),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("Training-set band statistics")
    print("=" * 60)

    for band, name in enumerate(
        ["Red", "Green", "Blue"]
    ):
        print(
            f"{name:5} | "
            f"mean={mean[band]:.6f} | "
            f"std={std[band]:.6f} | "
            f"min={global_min[band]:.0f} | "
            f"max={global_max[band]:.0f}"
        )

    print("=" * 60)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()