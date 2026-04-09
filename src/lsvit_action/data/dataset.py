"""Dataset definitions for HMDB51-style frame folder datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lsvit_action.constants import IMAGE_EXTENSIONS
from lsvit_action.data.transforms import VideoTransform


class HMDB51Dataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        num_frames: int,
        frame_stride: int,
        image_size: int = 224,
        transform: Optional[VideoTransform] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = max(1, frame_stride)
        self.transform = transform or VideoTransform(
            image_size=image_size,
            is_train=(split == "train"),
        )
        self.to_tensor = transforms.ToTensor()

        if not self.root.is_dir():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        self.classes = sorted([item.name for item in self.root.iterdir() if item.is_dir()])
        if not self.classes:
            raise RuntimeError(f"No class folders found in {self.root}")

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = self._build_split_samples(val_ratio=val_ratio, seed=seed)

        if not self.samples:
            raise RuntimeError(
                "Selected split has no samples. Adjust val_ratio or verify dataset structure."
            )

    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frame_paths, label = self.samples[index]
        indices = self._select_indices(total_frames=len(frame_paths))

        frames: list[torch.Tensor] = []
        for frame_index in indices:
            frame_path = frame_paths[int(frame_index.item())]
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                frames.append(self.to_tensor(image))

        video = torch.stack(frames, dim=0)
        video = self.transform(video)
        return video, label

    def _build_split_samples(
        self,
        val_ratio: float,
        seed: int,
    ) -> list[tuple[list[Path], int]]:
        grouped_samples: dict[tuple[str, str], list[tuple[list[Path], int]]] = {}

        for class_name in self.classes:
            class_dir = self.root / class_name
            video_dirs = sorted([path for path in class_dir.iterdir() if path.is_dir()])

            for video_dir in video_dirs:
                frame_paths = sorted(
                    [
                        path
                        for path in video_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                    ]
                )
                if not frame_paths:
                    continue

                group_key = (class_name, self._base_video_name(video_dir.name))
                sample = (frame_paths, self.class_to_idx[class_name])
                grouped_samples.setdefault(group_key, []).append(sample)

        if not grouped_samples:
            raise RuntimeError(f"No frame folders found inside {self.root}")

        group_values = list(grouped_samples.values())
        rng = np.random.RandomState(seed)
        group_indices = np.arange(len(group_values))
        rng.shuffle(group_indices)

        split_point = int(len(group_indices) * (1.0 - val_ratio))

        if self.split == "train":
            selected_groups = group_indices[:split_point]
        elif self.split in {"val", "test"}:
            selected_groups = group_indices[split_point:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        samples: list[tuple[list[Path], int]] = []
        for group_index in selected_groups:
            samples.extend(group_values[int(group_index)])

        return samples

    def _select_indices(self, total_frames: int) -> torch.Tensor:
        if total_frames <= 0:
            raise ValueError("Video folder has no frames")

        if total_frames == 1:
            return torch.zeros(self.num_frames, dtype=torch.long)

        num_grid_steps = max(self.num_frames * self.frame_stride, self.num_frames)
        grid = torch.linspace(0, total_frames - 1, steps=num_grid_steps)
        indices = grid[:: self.frame_stride].long()

        if indices.numel() < self.num_frames:
            pad_value = indices[-1].item()
            padding = indices.new_full((self.num_frames - indices.numel(),), pad_value)
            indices = torch.cat([indices, padding], dim=0)

        return indices[: self.num_frames]

    @staticmethod
    def _base_video_name(name: str) -> str:
        match = re.match(r"(.+)_\d+$", name)
        return match.group(1) if match else name