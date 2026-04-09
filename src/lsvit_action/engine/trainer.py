"""Evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(slots=True)
class EvalResult:

    loss: float
    accuracy: float
    total_samples: int


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    show_progress: bool = True,
) -> EvalResult:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = tqdm(loader, desc="Val", leave=False) if show_progress else loader

    for videos, labels in iterator:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(videos)
        loss = F.cross_entropy(logits, labels)

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        if show_progress:
            iterator.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{total_correct / max(total_samples, 1):.4f}",
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)

    return EvalResult(
        loss=avg_loss,
        accuracy=avg_acc,
        total_samples=total_samples,
    )