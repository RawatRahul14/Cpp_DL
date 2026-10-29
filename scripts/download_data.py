# === Modules ===
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

# === Resolve project root and target data directory ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_CSV = PROJECT_ROOT / "data" / "winequality-red.csv"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def main() -> None:
    """
    Downloads the UCI archive in-memory and extracts winequality-red.csv.
    """
    TARGET_CSV.parent.mkdir(parents = True, exist_ok = True)

    if TARGET_CSV.exists():
        print(f"Dataset already exists: {TARGET_CSV}")
        return

    print("Downloading official UCI Wine Quality archive...")
    with urlopen(UCI_ARCHIVE_URL, timeout = 30) as response:
        archive_bytes = BytesIO(response.read())
        zip_file = ZipFile(archive_bytes)

    # === Locate the red wine dataset inside the zip archive ===
    member_name = next(
        (
            name
            for name in zip_file.namelist()
            if name.lower().endswith("winequality-red.csv")
        ),
        None,
    )

    if not member_name:
        raise RuntimeError("winequality-red.csv not found inside the UCI archive.")

    TARGET_CSV.write_bytes(zip_file.read(member_name))
    print(f"Successfully saved dataset to: {TARGET_CSV}")

if __name__ == "__main__":
    main()