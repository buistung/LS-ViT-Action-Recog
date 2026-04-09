"""Dataloader construction utilities."""

from __future__ import annotations

from torch.utils.data import DataLoader

from lsvit_action.config import DataConfig
from lsvit_action.data.dataset import HMDB51Dataset


def collate_fn(
    batch: list[tuple],
) -> tuple:
    import torch

    videos = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return videos, labels


def build_datasets(
    config: DataConfig,
) -> tuple[HMDB51Dataset, HMDB51Dataset]:
    train_dataset = HMDB51Dataset(
        root=config.data_root,
        split="train",
        num_frames=config.num_frames,
        frame_stride=config.frame_stride,
        image_size=config.image_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    val_dataset = HMDB51Dataset(
        root=config.data_root,
        split="val",
        num_frames=config.num_frames,
        frame_stride=config.frame_stride,
        image_size=config.image_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    return train_dataset, val_dataset


def build_dataloaders(
    config: DataConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader