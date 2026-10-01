from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SOURCE_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spacenet"
    / "SpaceNet_Roads_Sample"
    / "AOI_3_Paris_Roads_Sample"
    / "RGB-PanSharpen"
    / "RGB-PanSharpen_AOI_3_Paris_img84.tif"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_predictions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_predictions"
)

OUTPUT_PROBABILITY = (
    OUTPUT_DIR
    / "AOI_3_Paris_img84_probability.tif"
)

OUTPUT_MASK = (
    OUTPUT_DIR
    / "AOI_3_Paris_img84_road_mask.tif"
)

OUTPUT_VALID_MASK = (
    OUTPUT_DIR
    / "AOI_3_Paris_img84_valid_data_mask.tif"
)


# ---------------------------------------------------------------------
# Tiling configuration used to create the existing test tiles
# ---------------------------------------------------------------------

TILE_SIZE = 256
STRIDE = 192

TILES_PER_ROW = 6
TILES_PER_COLUMN = 6
TOTAL_TILES = 36

THRESHOLD = 0.25


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def tile_index_to_window(
    tile_index: int,
) -> tuple[int, int]:

    row = tile_index // TILES_PER_ROW
    col = tile_index % TILES_PER_ROW

    top = row * STRIDE
    left = col * STRIDE

    return top, left


def read_prediction(
    tile_index: int,
) -> np.ndarray:

    filename = (
        f"AOI_3_Paris_img84_tile_"
        f"{tile_index:03d}_probability.tif"
    )

    path = (
        PREDICTION_DIR
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction tile:\n{path}"
        )

    with rasterio.open(path) as src:
        prediction = src.read(1)

    if prediction.shape != (
        TILE_SIZE,
        TILE_SIZE,
    ):
        raise ValueError(
            f"Unexpected prediction shape "
            f"for tile {tile_index}: "
            f"{prediction.shape}"
        )

    return prediction.astype(
        np.float32,
        copy=False,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "SpaceNet Test Prediction Stitching"
    )
    print("=" * 72)

    print(
        "Source image:",
        SOURCE_IMAGE,
    )

    print(
        "Prediction directory:",
        PREDICTION_DIR,
    )

    # ---------------------------------------------------------------
    # Open original SpaceNet image
    # ---------------------------------------------------------------

    with rasterio.open(
        SOURCE_IMAGE
    ) as src:

        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        print(
            "Scene size:",
            f"{width} x {height}",
        )

        print(
            "CRS:",
            crs,
        )

        # -----------------------------------------------------------
        # Read original RGB scene
        # -----------------------------------------------------------

        source_rgb = src.read(
            out_dtype=np.float32
        )

        if source_rgb.shape[0] != 3:
            raise ValueError(
                "Expected exactly 3 source image bands."
            )

        # A pixel is considered valid if at least one RGB band
        # contains non-zero data.
        valid_data_mask = np.any(
            source_rgb != 0,
            axis=0,
        )

        print(
            "Valid source pixels:",
            f"{int(valid_data_mask.sum()):,}",
        )

        print(
            "No-data pixels:",
            f"{int((~valid_data_mask).sum()):,}",
        )

        # -----------------------------------------------------------
        # Allocate mosaic accumulators
        # -----------------------------------------------------------

        probability_sum = np.zeros(
            (height, width),
            dtype=np.float64,
        )

        contribution_count = np.zeros(
            (height, width),
            dtype=np.uint16,
        )

        # -----------------------------------------------------------
        # Stitch prediction tiles
        # -----------------------------------------------------------

        for tile_index in range(
            TOTAL_TILES
        ):

            top, left = (
                tile_index_to_window(
                    tile_index
                )
            )

            bottom = (
                top + TILE_SIZE
            )

            right = (
                left + TILE_SIZE
            )

            if bottom > height:
                raise ValueError(
                    f"Tile {tile_index} exceeds "
                    "scene height."
                )

            if right > width:
                raise ValueError(
                    f"Tile {tile_index} exceeds "
                    "scene width."
                )

            prediction = read_prediction(
                tile_index
            )

            probability_sum[
                top:bottom,
                left:right,
            ] += prediction

            contribution_count[
                top:bottom,
                left:right,
            ] += 1

            print(
                f"[{tile_index + 1:02d}/{TOTAL_TILES:02d}] "
                f"tile={tile_index:03d} "
                f"top={top:04d} "
                f"left={left:04d}"
            )

        # -----------------------------------------------------------
        # Average overlapping predictions
        # -----------------------------------------------------------

        valid_prediction_area = (
            contribution_count > 0
        )

        probability = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        probability[
            valid_prediction_area
        ] = (
            probability_sum[
                valid_prediction_area
            ]
            / contribution_count[
                valid_prediction_area
            ]
        )

        # -----------------------------------------------------------
        # Remove all model predictions outside original imagery
        # -----------------------------------------------------------

        probability[
            ~valid_data_mask
        ] = 0.0

        # Pixels not covered by any prediction tile are also invalid.
        probability[
            ~valid_prediction_area
        ] = 0.0

        # -----------------------------------------------------------
        # Final binary road mask
        # -----------------------------------------------------------

        road_mask = (
            probability >= THRESHOLD
        ).astype(
            np.uint8
        )

        # Explicitly ensure no-data can never become road.
        road_mask[
            ~valid_data_mask
        ] = 0

        road_pixels = int(
            road_mask.sum()
        )

        print()
        print(
            "Final valid road pixels:",
            f"{road_pixels:,}",
        )

        # -----------------------------------------------------------
        # Output raster profile
        # -----------------------------------------------------------

        probability_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
            "nodata": 0.0,
        }

        mask_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
            "nodata": 0,
        }

        # -----------------------------------------------------------
        # Write probability raster
        # -----------------------------------------------------------

        with rasterio.open(
            OUTPUT_PROBABILITY,
            "w",
            **probability_profile,
        ) as dst:

            dst.write(
                probability,
                1,
            )

        # -----------------------------------------------------------
        # Write binary road mask
        # -----------------------------------------------------------

        with rasterio.open(
            OUTPUT_MASK,
            "w",
            **mask_profile,
        ) as dst:

            dst.write(
                road_mask,
                1,
            )

        # -----------------------------------------------------------
        # Write valid-data mask
        # -----------------------------------------------------------

        with rasterio.open(
            OUTPUT_VALID_MASK,
            "w",
            **mask_profile,
        ) as dst:

            dst.write(
                valid_data_mask.astype(
                    np.uint8
                ),
                1,
            )

    print()
    print("=" * 72)
    print(
        "Stitching complete."
    )
    print("=" * 72)

    print(
        "Probability raster:",
        OUTPUT_PROBABILITY,
    )

    print(
        "Road mask:",
        OUTPUT_MASK,
    )

    print(
        "Valid-data mask:",
        OUTPUT_VALID_MASK,
    )

    print(
        "Threshold:",
        THRESHOLD,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
    