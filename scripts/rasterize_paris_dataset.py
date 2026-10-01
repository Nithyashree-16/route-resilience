from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data/raw/spacenet/SpaceNet_Roads_Sample/"
    "AOI_3_Paris_Roads_Sample/RGB-PanSharpen"
)

VECTOR_DIR = (
    PROJECT_ROOT
    / "data/raw/spacenet/SpaceNet_Roads_Sample/"
    "AOI_3_Paris_Roads_Sample/geojson/spacenetroads"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/interim/masks/spacenet_paris"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.tif"))

    if not image_files:
        raise FileNotFoundError(f"No TIFF images found in {IMAGE_DIR}")

    processed = 0
    skipped = 0

    for image_path in image_files:
        scene_id = image_path.stem.replace("RGB-PanSharpen_", "")
        vector_path = VECTOR_DIR / f"spacenetroads_{scene_id}.geojson"

        if not vector_path.exists():
            print(f"[SKIP] Missing annotation: {scene_id}")
            skipped += 1
            continue

        output_path = OUTPUT_DIR / f"road_centerline_{scene_id}.tif"

        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()

        roads = gpd.read_file(vector_path)

        if roads.empty:
            print(f"[SKIP] Empty annotation: {scene_id}")
            skipped += 1
            continue

        if roads.crs != crs:
            roads = roads.to_crs(crs)

        shapes = [
            (geometry, 1)
            for geometry in roads.geometry
            if geometry is not None and not geometry.is_empty
        ]

        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

        road_pixels = int(np.count_nonzero(mask))
        coverage = road_pixels / mask.size

        print(
            f"[OK] {scene_id} | "
            f"roads={len(roads)} | "
            f"pixels={road_pixels} | "
            f"coverage={coverage:.6%}"
        )

        processed += 1

    print("\n" + "=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Output:    {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()