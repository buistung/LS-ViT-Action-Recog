"""Video transform utilities."""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class VideoTransform:
    def __init__(
        self,
        image_size: int,
        is_train: bool = True,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:

        self.image_size = image_size
        self.is_train = is_train
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(
                f"`frames` must have shape (T, C, H, W), but got {tuple(frames.shape)}"
            )

        frames = frames.float()

        if self.is_train:
            frames = self._apply_train_transforms(frames)
        else:
            frames = self._apply_eval_transforms(frames)

        normalized_frames = [TF.normalize(frame, self.mean, self.std) for frame in frames]
        return torch.stack(normalized_frames, dim=0)

    def _apply_train_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape

        scale = random.uniform(0.8, 1.0)
        new_height = max(1, int(height * scale))
        new_width = max(1, int(width * scale))

        frames = TF.resize(
            frames,
            [new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
        )

        top = random.randint(0, max(0, new_height - self.image_size))
        left = random.randint(0, max(0, new_width - self.image_size))

        crop_height = min(self.image_size, new_height)
        crop_width = min(self.image_size, new_width)
        frames = TF.crop(frames, top, left, crop_height, crop_width)

        frames = TF.resize(
            frames,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        if random.random() < 0.5:
            frames = TF.hflip(frames)

        if random.random() < 0.3:
            brightness_factor = random.uniform(0.9, 1.1)
            frames = TF.adjust_brightness(frames, brightness_factor)

        if random.random() < 0.3:
            contrast_factor = random.uniform(0.9, 1.1)
            frames = TF.adjust_contrast(frames, contrast_factor)

        if random.random() < 0.3:
            saturation_factor = random.uniform(0.9, 1.1)
            frames = TF.adjust_saturation(frames, saturation_factor)

        return frames

    def _apply_eval_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        return TF.resize(
            frames,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )