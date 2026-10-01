from pathlib import Path
import csv

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "spacenet_paris_tiles.csv"
)


def analyze_split(split: str):
    image_root = DATA_ROOT / split / "images"
    mask_root = DATA_ROOT / split / "masks"

    rows = []

    image_files = sorted(image_root.rglob("*.tif"))

    for image_path in image_files:
        scene_id = image_path.parent.name
        tile_name = image_path.name

        mask_path = mask_root / scene_id / tile_name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for {image_path}"
            )

        with rasterio.open(image_path) as image_src:
            image_shape = (
                image_src.count,
                image_src.height,
                image_src.width,
            )

        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)

        if mask.shape != (
            image_shape[1],
            image_shape[2],
        ):
            raise ValueError(
                f"Image/mask shape mismatch: {image_path}"
            )

        road_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.size)
        road_fraction = road_pixels / total_pixels

        rows.append(
            {
                "split": split,
                "scene_id": scene_id,
                "tile": tile_name,
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "mask_path": str(
                    mask_path.relative_to(PROJECT_ROOT)
                ),
                "channels": image_shape[0],
                "height": image_shape[1],
                "width": image_shape[2],
                "road_pixels": road_pixels,
                "road_fraction": road_fraction,
                "has_road": road_pixels > 0,
            }
        )

    return rows


def main():
    all_rows = []

    for split in ["train", "val", "test"]:
        rows = analyze_split(split)
        all_rows.extend(rows)

        road_tiles = sum(
            1 for row in rows if row["has_road"]
        )

        empty_tiles = len(rows) - road_tiles

        mean_fraction = (
            sum(row["road_fraction"] for row in rows)
            / len(rows)
            if rows
            else 0
        )

        print(
            f"{split.upper():5} | "
            f"tiles={len(rows):3d} | "
            f"with_roads={road_tiles:3d} | "
            f"empty={empty_tiles:3d} | "
            f"mean_road_fraction={mean_fraction:.4%}"
        )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(all_rows[0].keys())

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 60)
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Total tiles: {len(all_rows)}")
    print("=" * 60)


if __name__ == "__main__":
    main()