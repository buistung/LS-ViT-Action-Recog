"""Visualization utilities for clips, predictions, and training history."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def denormalize(
    video: torch.Tensor,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=video.dtype, device=video.device)
    std_tensor = torch.tensor(std, dtype=video.dtype, device=video.device)

    if video.ndim == 4:
        mean_tensor = mean_tensor.view(1, 3, 1, 1)
        std_tensor = std_tensor.view(1, 3, 1, 1)
    elif video.ndim == 5:
        mean_tensor = mean_tensor.view(1, 1, 3, 1, 1)
        std_tensor = std_tensor.view(1, 1, 3, 1, 1)
    else:
        raise ValueError(
            f"`video` must have shape (T, C, H, W) or (B, T, C, H, W), got {tuple(video.shape)}"
        )

    video = video * std_tensor + mean_tensor
    return video.clamp(0.0, 1.0)


def plot_clip_grid(
    video: torch.Tensor,
    title: str | None = None,
    max_frames: int = 12,
    cols: int = 4,
    save_path: str | Path | None = None,
) -> None:
    if video.ndim != 4:
        raise ValueError(f"`video` must have shape (T, C, H, W), got {tuple(video.shape)}")

    frames_to_show = min(video.shape[0], max_frames)
    rows = math.ceil(frames_to_show / cols)

    plt.figure(figsize=(12, 3 * rows))
    for index in range(frames_to_show):
        plt.subplot(rows, cols, index + 1)
        frame = video[index].detach().cpu().permute(1, 2, 0).numpy()
        plt.imshow(frame)
        plt.axis("off")
        plt.title(f"Frame {index + 1}")

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()