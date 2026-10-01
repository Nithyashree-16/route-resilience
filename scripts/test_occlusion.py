import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.occlusion import apply_occlusions


IMAGE_PATH = (
    PROJECT_ROOT
    / "data/processed/spacenet_paris/train/images/"
    "AOI_3_Paris_img235/"
    "AOI_3_Paris_img235_tile_023.tif"
)

MASK_PATH = (
    PROJECT_ROOT
    / "data/processed/spacenet_paris/train/masks/"
    "AOI_3_Paris_img235/"
    "AOI_3_Paris_img235_tile_023.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/interim/occluded/"
    "spaceNet_paris_occlusion_test.png"
)


def main():
    with rasterio.open(IMAGE_PATH) as src:
        image = src.read().transpose(1, 2, 0).astype(np.float32)

    with rasterio.open(MASK_PATH) as src:
        mask = src.read(1)

    # Min-max only for visualization/testing.
    image = image - image.min()
    image = image / (image.max() + 1e-8)

    occluded, effects = apply_occlusions(
        image,
        min_effects=2,
        max_effects=3,
        seed=42,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(image)
    axes[0].set_title("Original RGB")

    axes[1].imshow(occluded)
    axes[1].set_title(
        "Occluded: " + ", ".join(effects)
    )

    axes[2].imshow(occluded)
    axes[2].imshow(
        mask,
        alpha=0.35,
    )
    axes[2].set_title(
        "Occluded + Original Road Mask"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("Occlusion test successful.")
    print("Effects:", effects)
    print("Mask unchanged.")
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()