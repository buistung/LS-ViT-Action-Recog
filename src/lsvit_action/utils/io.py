"""I/O and filesystem helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path | None) -> Path | None:
    """Create a directory if it does not exist.

    Args:
        path: Directory path. If None, the function returns None.

    Returns:
        The resolved Path object, or None if the input is None.
    """
    if path is None:
        return None

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """Save a dictionary as a JSON file.

    Args:
        data: Data to save.
        path: Destination JSON path.
        indent: Indentation level for pretty printing.

    Returns:
        The saved file path.
    """
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    with path_obj.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    return path_obj


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"JSON file not found: {path_obj}")

    with path_obj.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_files(directory: str | Path, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    files = [path for path in directory_path.iterdir() if path.is_file()]
    if suffixes is not None:
        files = [path for path in files if path.suffix.lower() in suffixes]

    return sorted(files)