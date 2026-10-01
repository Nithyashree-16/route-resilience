import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "spacenet_paris_tiles.csv"
)


def main():
    positive_pixels = 0
    total_pixels = 0
    tile_count = 0

    with MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] != "train":
                continue

            height = int(row["height"])
            width = int(row["width"])
            road_pixels = int(row["road_pixels"])

            pixels = height * width

            total_pixels += pixels
            positive_pixels += road_pixels
            tile_count += 1

    negative_pixels = total_pixels - positive_pixels

    positive_fraction = (
        positive_pixels / total_pixels
    )

    negative_fraction = (
        negative_pixels / total_pixels
    )

    pos_weight = (
        negative_pixels / positive_pixels
        if positive_pixels > 0
        else 0.0
    )

    print("SpaceNet training class balance")
    print("=" * 60)
    print("Training tiles:", tile_count)
    print("Total pixels:", f"{total_pixels:,}")
    print("Road pixels:", f"{positive_pixels:,}")
    print("Background pixels:", f"{negative_pixels:,}")
    print()
    print(
        "Road fraction:",
        f"{positive_fraction:.6%}",
    )
    print(
        "Background fraction:",
        f"{negative_fraction:.6%}",
    )
    print()
    print(
        "Recommended BCE pos_weight:",
        f"{pos_weight:.6f}",
    )
    print("=" * 60)


if __name__ == "__main__":
    main()