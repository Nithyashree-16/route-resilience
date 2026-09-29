from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


REQUIRED_DIRECTORIES = [
    "raw/sentinel2",
    "raw/resourcesat",
    "raw/cartosat",
    "raw/spacenet",
    "raw/deepglobe",
    "raw/opensatmap",
    "raw/osm",
    "interim/tiles",
    "interim/masks",
    "interim/occluded",
    "processed/train",
    "processed/val",
    "processed/test",
    "metadata",
]


def main():
    print("Route Resilience — Data Structure Check")
    print("=" * 50)

    missing = []

    for relative_path in REQUIRED_DIRECTORIES:
        path = DATA_ROOT / relative_path

        if path.exists():
            print(f"[OK]      {relative_path}")
        else:
            print(f"[MISSING] {relative_path}")
            missing.append(relative_path)

    print("=" * 50)

    if missing:
        print(f"Missing directories: {len(missing)}")
        raise SystemExit(1)

    print("All required data directories exist.")


if __name__ == "__main__":
    main()