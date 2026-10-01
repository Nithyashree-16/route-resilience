from pathlib import Path

import matplotlib.pyplot as plt
import rasterio


ROOT = Path(__file__).resolve().parents[1]

image_path = (
    ROOT
    / "data/raw/spacenet/SpaceNet_Roads_Sample/"
    "AOI_3_Paris_Roads_Sample/RGB-PanSharpen/"
    "RGB-PanSharpen_AOI_3_Paris_img235.tif"
)

mask_path = (
    ROOT
    / "data/interim/masks/spacenet_paris/"
    "road_centerline_AOI_3_Paris_img235.tif"
)

output_path = ROOT / "data/interim/masks/spacenet_paris/preview_img235.png"


with rasterio.open(image_path) as src:
    image = src.read([1, 2, 3])

with rasterio.open(mask_path) as src:
    mask = src.read(1)


# Convert RGB to display-friendly values.
rgb = image.astype("float32")
rgb /= rgb.max()

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(rgb.transpose(1, 2, 0))
ax.imshow(mask, alpha=0.8)

ax.set_title("SpaceNet Paris img235 — RGB + Road Centerline")
ax.axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print("Preview created:")
print(output_path)
