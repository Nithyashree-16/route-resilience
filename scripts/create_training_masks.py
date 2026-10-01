from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data/interim/masks/spacenet_paris"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/interim/masks/spacenet_paris_training"
)

DILATION_RADIUS_PIXELS = 5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(INPUT_DIR.glob("road_centerline_*.tif"))

    if not mask_files:
        raise FileNotFoundError(
            f"No centerline masks found in {INPUT_DIR}"
        )

    print("Creating training road-corridor masks")
    print(f"Dilation radius: {DILATION_RADIUS_PIXELS} pixels")
    print("=" * 60)

    for mask_path in mask_files:
        output_path = (
            OUTPUT_DIR
            / mask_path.name.replace(
                "road_centerline_",
                "road_training_",
            )
        )

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            profile = src.profile.copy()

        centerline = mask > 0

        # Square structuring element.
        kernel_size = 2 * DILATION_RADIUS_PIXELS + 1

        dilated = binary_dilation(
            centerline,
            structure=np.ones(
                (kernel_size, kernel_size),
                dtype=bool,
            ),
        )

        training_mask = dilated.astype(np.uint8)

        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(training_mask, 1)

        centerline_pixels = int(np.count_nonzero(centerline))
        training_pixels = int(np.count_nonzero(training_mask))
        coverage = training_pixels / training_mask.size

        print(
            f"[OK] {mask_path.stem} | "
            f"centerline={centerline_pixels} | "
            f"training={training_pixels} | "
            f"coverage={coverage:.4%}"
        )

    print("=" * 60)
    print(f"Created: {len(mask_files)} training masks")
    print(f"Output:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()