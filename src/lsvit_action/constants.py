"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

HMDB51_NUM_CLASSES: int = 51

DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_NUM_FRAMES: int = 16
DEFAULT_FRAME_STRIDE: int = 2

DEFAULT_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
DEFAULT_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

DEFAULT_PROJECT_ROOT: Path = Path(".")
DEFAULT_DATA_DIRNAME: str = "data"
DEFAULT_RAW_DIRNAME: str = "raw"
DEFAULT_PROCESSED_DIRNAME: str = "processed"
DEFAULT_OUTPUTS_DIRNAME: str = "outputs"
DEFAULT_CHECKPOINTS_DIRNAME: str = "checkpoints"
DEFAULT_LOGS_DIRNAME: str = "logs"
DEFAULT_FIGURES_DIRNAME: str = "figures"
DEFAULT_WEIGHTS_DIRNAME: str = "weights"