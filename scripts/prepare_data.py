"""Download and extract the HMDB51 frame-folder dataset used by the notebook."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import gdown


DEFAULT_FILE_ID = "141MgG4CC7XffVH32hQy7lQ0PKCafskji"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download and prepare HMDB51 dataset.")
    parser.add_argument(
        "--file-id",
        type=str,
        default=DEFAULT_FILE_ID,
        help="Google Drive file ID of the dataset zip.",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="data/raw/HMDB51.zip",
        help="Local path to save the downloaded zip file.",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data/processed/hmdb51",
        help="Directory where the dataset will be extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing zip or extraction directory.",
    )
    return parser.parse_args()


def download_zip(file_id: str, output_path: Path, force: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_file() and not force:
        print(f"Zip already exists: {output_path}")
        return output_path

    if output_path.is_file() and force:
        output_path.unlink()

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading dataset from: {url}")
    gdown.download(url=url, output=str(output_path), quiet=False)
    return output_path


def extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> Path:
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    temp_extract_dir = extract_dir.parent / f"{extract_dir.name}_tmp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {temp_extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    inner_items = list(temp_extract_dir.iterdir())

    if len(inner_items) == 1 and inner_items[0].is_dir():
        source_dir = inner_items[0]
    else:
        source_dir = temp_extract_dir

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    shutil.move(str(source_dir), str(extract_dir))

    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir, ignore_errors=True)

    print(f"Dataset prepared at: {extract_dir}")
    return extract_dir


def main() -> None:
    """Main entry point."""
    args = parse_args()
    zip_path = Path(args.zip_path)
    extract_dir = Path(args.extract_dir)

    zip_path = download_zip(
        file_id=args.file_id,
        output_path=zip_path,
        force=args.force,
    )
    extract_zip(
        zip_path=zip_path,
        extract_dir=extract_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()