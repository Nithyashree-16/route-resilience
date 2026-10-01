from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from skimage.morphology import skeletonize


INPUT_MASK = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_predictions"
    / "AOI_3_Paris_img84_road_mask.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_graph"
)

OUTPUT_SKELETON = (
    OUTPUT_DIR
    / "AOI_3_Paris_img84_skeleton.tif"
)


def main() -> None:
    if not INPUT_MASK.exists():
        raise FileNotFoundError(
            f"Road mask not found:\n{INPUT_MASK}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SpaceNet Road Skeletonization")
    print("=" * 72)
    print("Input:", INPUT_MASK)

    with rasterio.open(INPUT_MASK) as src:
        road_mask = src.read(1)
        profile = src.profile.copy()

    binary_mask = road_mask > 0

    print(
        "Road pixels before skeletonization:",
        f"{int(binary_mask.sum()):,}",
    )

    skeleton = skeletonize(
        binary_mask
    )

    skeleton_uint8 = skeleton.astype(
        np.uint8
    )

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="lzw",
        nodata=0,
    )

    with rasterio.open(
        OUTPUT_SKELETON,
        "w",
        **profile,
    ) as dst:
        dst.write(
            skeleton_uint8,
            1,
        )

    print(
        "Skeleton pixels:",
        f"{int(skeleton.sum()):,}",
    )

    print(
        "Output:",
        OUTPUT_SKELETON,
    )

    print("=" * 72)
    print("Skeletonization complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
    