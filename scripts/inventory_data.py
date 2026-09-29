from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def main():
    print("Route Resilience — Dataset Inventory")
    print("=" * 60)

    files = list(DATA_ROOT.rglob("*"))

    file_count = 0
    total_size = 0

    for path in files:
        if path.is_file():
            size = path.stat().st_size
            file_count += 1
            total_size += size

    print(f"Files: {file_count}")
    print(f"Total size: {human_size(total_size)}")

    print("\nDataset directories:")

    raw_dir = DATA_ROOT / "raw"

    for dataset_dir in sorted(raw_dir.iterdir()):
        if dataset_dir.is_dir():
            count = sum(1 for p in dataset_dir.rglob("*") if p.is_file())
            print(f"  {dataset_dir.name:15} {count:6} files")

    print("=" * 60)


if __name__ == "__main__":
    main()