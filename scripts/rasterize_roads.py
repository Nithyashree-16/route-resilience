from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spacenet"
    / "SpaceNet_Roads_Sample"
    / "AOI_3_Paris_Roads_Sample"
    / "RGB-PanSharpen"
)

VECTOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spacenet"
    / "SpaceNet_Roads_Sample"
    / "AOI_3_Paris_Roads_Sample"
    / "geojson"
    / "spacenetroads"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "masks"
    / "spacenet_paris"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = IMAGE_DIR / "RGB-PanSharpen_AOI_3_Paris_img235.tif"
    vector_path = VECTOR_DIR / "spacenetroads_AOI_3_Paris_img235.geojson"

    output_path = OUTPUT_DIR / "road_centerline_AOI_3_Paris_img235.tif"

    with rasterio.open(image_path) as src:
        profile = src.profile.copy()

        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

    roads = gpd.read_file(vector_path)

    if roads.empty:
        raise ValueError("Road GeoJSON contains no features.")

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
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw",
        }
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    road_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    coverage = road_pixels / total_pixels

    print("Road mask generated successfully.")
    print(f"Image:        {image_path.name}")
    print(f"Road vectors: {len(roads)}")
    print(f"Mask:         {output_path}")
    print(f"Mask shape:   {mask.shape}")
    print(f"Road pixels:  {road_pixels}")
    print(f"Coverage:     {coverage:.6%}")


if __name__ == "__main__":
    main()