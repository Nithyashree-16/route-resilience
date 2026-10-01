from pathlib import Path

import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data/raw/spacenet/SpaceNet_Roads_Sample/"
    "AOI_3_Paris_Roads_Sample/RGB-PanSharpen"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data/interim/masks/spacenet_paris_training"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data/processed/spacenet_paris"
)

TILE_SIZE = 256
STRIDE = 192


def scene_split(scene_index: int, total_scenes: int) -> str:
    """
    Deterministic scene-level split.

    70% train
    20% validation
    10% test
    """
    train_end = round(total_scenes * 0.70)
    val_end = round(total_scenes * 0.90)

    if scene_index < train_end:
        return "train"
    elif scene_index < val_end:
        return "val"
    return "test"


def generate_windows(width: int, height: int):
    """
    Generate fixed-size windows with overlap.
    Only complete TILE_SIZE x TILE_SIZE windows are retained.
    """
    for top in range(0, height - TILE_SIZE + 1, STRIDE):
        for left in range(0, width - TILE_SIZE + 1, STRIDE):
            yield Window(
                col_off=left,
                row_off=top,
                width=TILE_SIZE,
                height=TILE_SIZE,
            )


def main():
    image_files = sorted(IMAGE_DIR.glob("*.tif"))

    if not image_files:
        raise FileNotFoundError(
            f"No images found in {IMAGE_DIR}"
        )

    print("Route Resilience — Tile Generation")
    print("=" * 60)
    print(f"Scenes:     {len(image_files)}")
    print(f"Tile size:  {TILE_SIZE} x {TILE_SIZE}")
    print(f"Stride:     {STRIDE}")

    total_tiles = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for scene_index, image_path in enumerate(image_files):
        scene_id = image_path.stem.replace(
            "RGB-PanSharpen_",
            "",
        )

        mask_path = MASK_DIR / f"road_training_{scene_id}.tif"

        if not mask_path.exists():
            print(f"[SKIP] Missing mask: {scene_id}")
            continue

        split = scene_split(
            scene_index,
            len(image_files),
        )

        image_output = (
            OUTPUT_ROOT
            / split
            / "images"
            / scene_id
        )

        mask_output = (
            OUTPUT_ROOT
            / split
            / "masks"
            / scene_id
        )

        image_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(image_path) as image_src, \
                rasterio.open(mask_path) as mask_src:

            if (
                image_src.width != mask_src.width
                or image_src.height != mask_src.height
            ):
                raise ValueError(
                    f"Image/mask dimension mismatch: {scene_id}"
                )

            windows = list(
                generate_windows(
                    image_src.width,
                    image_src.height,
                )
            )

            scene_tiles = 0

            for tile_index, window in enumerate(windows):
                image = image_src.read(window=window)
                mask = mask_src.read(1, window=window)

                if image.shape[1:] != (
                    TILE_SIZE,
                    TILE_SIZE,
                ):
                    continue

                image_name = (
                    image_output
                    / f"{scene_id}_tile_{tile_index:03d}.tif"
                )

                mask_name = (
                    mask_output
                    / f"{scene_id}_tile_{tile_index:03d}.tif"
                )

                image_profile = image_src.profile.copy()
                image_profile.update(
                    {
                        "height": TILE_SIZE,
                        "width": TILE_SIZE,
                        "count": image.shape[0],
                        "compress": "lzw",
                    }
                )

                mask_profile = mask_src.profile.copy()
                mask_profile.update(
                    {
                        "height": TILE_SIZE,
                        "width": TILE_SIZE,
                        "count": 1,
                        "dtype": "uint8",
                        "compress": "lzw",
                        "nodata": 0,
                    }
                )

                with rasterio.open(
                    image_name,
                    "w",
                    **image_profile,
                ) as dst:
                    dst.write(image)

                with rasterio.open(
                    mask_name,
                    "w",
                    **mask_profile,
                ) as dst:
                    dst.write(mask, 1)

                scene_tiles += 1
                total_tiles[split] += 1

        print(
            f"[OK] {scene_id:25} "
            f"split={split:5} "
            f"tiles={scene_tiles}"
        )

    print("=" * 60)
    print("Tile generation complete.")
    print(f"Train tiles: {total_tiles['train']}")
    print(f"Val tiles:   {total_tiles['val']}")
    print(f"Test tiles:  {total_tiles['test']}")


if __name__ == "__main__":
    main()