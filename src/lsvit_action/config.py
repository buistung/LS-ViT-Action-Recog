"""Configuration objects for the LSViT action recognition project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from lsvit_action.constants import (
    DEFAULT_CHECKPOINTS_DIRNAME,
    DEFAULT_DATA_DIRNAME,
    DEFAULT_FIGURES_DIRNAME,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LOGS_DIRNAME,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUTS_DIRNAME,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_PROCESSED_DIRNAME,
    DEFAULT_RAW_DIRNAME,
    DEFAULT_WEIGHTS_DIRNAME,
    HMDB51_NUM_CLASSES,
)


@dataclass(slots=True)
class PathConfig:
    project_root: Path = DEFAULT_PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_DATA_DIRNAME)
    raw_data_dir: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_DATA_DIRNAME / DEFAULT_RAW_DIRNAME
    )
    processed_data_dir: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_DATA_DIRNAME / DEFAULT_PROCESSED_DIRNAME
    )
    checkpoints_dir: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_CHECKPOINTS_DIRNAME
    )
    outputs_dir: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_OUTPUTS_DIRNAME
    )
    logs_dir: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_OUTPUTS_DIRNAME / DEFAULT_LOGS_DIRNAME
    )
    figures_dir: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_OUTPUTS_DIRNAME / DEFAULT_FIGURES_DIRNAME
    )
    weights_dir: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_WEIGHTS_DIRNAME
    )

    def create_directories(self) -> None:
        for path in (
            self.project_root,
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.checkpoints_dir,
            self.outputs_dir,
            self.logs_dir,
            self.figures_dir,
            self.weights_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DataConfig:
    data_root: Path = field(
        default_factory=lambda: DEFAULT_PROJECT_ROOT / DEFAULT_DATA_DIRNAME / DEFAULT_PROCESSED_DIRNAME / "hmdb51"
    )
    image_size: int = DEFAULT_IMAGE_SIZE
    num_frames: int = DEFAULT_NUM_FRAMES
    frame_stride: int = DEFAULT_FRAME_STRIDE
    batch_size: int = 4
    num_workers: int = 2
    val_ratio: float = 0.1
    seed: int = 42
    pin_memory: bool = True
    drop_last: bool = False


@dataclass(slots=True)
class ModelConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1
    qkv_bias: bool = True
    num_classes: int = HMDB51_NUM_CLASSES
    smif_window: int = 5


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 10
    base_lr: float = 5e-5
    head_lr: float = 2.5e-4
    weight_decay: float = 0.05
    grad_accum_steps: int = 16
    pretrained_name: str = "vit_base_patch16_224"
    mixed_precision: bool = True
    device: str | None = None
    save_best_only: bool = True

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment_name: str = "lsvit_hmdb51"

    def prepare(self) -> None:

        self.paths.create_directories()

    def to_dict(self) -> dict[str, Any]:
 
        return asdict(self)