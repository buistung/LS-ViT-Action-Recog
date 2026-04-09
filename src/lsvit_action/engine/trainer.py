from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lsvit_action.config import ExperimentConfig, TrainConfig
from lsvit_action.engine.checkpoint import save_checkpoint
from lsvit_action.engine.evaluator import EvalResult, evaluate


@dataclass(slots=True)
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }


def build_optimizer(
    model: torch.nn.Module,
    train_config: TrainConfig,
) -> torch.optim.Optimizer:
    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("backbone"):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": train_config.base_lr},
            {"params": head_params, "lr": train_config.head_lr},
        ],
        weight_decay=train_config.weight_decay,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    show_progress: bool = True,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    grad_accum_steps = max(1, grad_accum_steps)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    num_batches = len(loader)

    optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(loader, desc="Train", leave=False) if show_progress else loader

    for batch_index, (videos, labels) in enumerate(iterator):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device_type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(videos)
            loss = F.cross_entropy(logits, labels)

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        should_step = (
            (batch_index + 1) % grad_accum_steps == 0
            or (batch_index + 1) == num_batches
        )

        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if show_progress:
            iterator.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{total_correct / max(total_samples, 1):.4f}",
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    experiment_dir: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[torch.nn.Module, History]:
    device = config.train.resolve_device()
    model = model.to(device)

    scaler = torch.amp.GradScaler(
        enabled=(config.train.mixed_precision and device.type == "cuda")
    )

    history = History()
    best_acc = 0.0

    checkpoint_dir = (
        Path(experiment_dir)
        if experiment_dir is not None
        else config.paths.checkpoints_dir
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / f"{config.experiment_name}_best.pt"
    last_checkpoint_path = checkpoint_dir / f"{config.experiment_name}_last.pt"

    for epoch in range(config.train.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=config.train.grad_accum_steps,
            show_progress=show_progress,
        )

        val_result: EvalResult = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            show_progress=show_progress,
        )

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_result.loss)
        history.val_acc.append(val_result.accuracy)

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            metric=val_result.accuracy,
            history=history.to_dict(),
            extra={"experiment_name": config.experiment_name},
        )

        if val_result.accuracy > best_acc:
            best_acc = val_result.accuracy
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                metric=best_acc,
                history=history.to_dict(),
                extra={"experiment_name": config.experiment_name},
            )

        print(
            f"Epoch {epoch + 1}/{config.train.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_result.loss:.4f} | val_acc={val_result.accuracy:.4f} | "
            f"best={best_acc:.4f}"
        )

    return model, history