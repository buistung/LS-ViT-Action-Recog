"""Checkpoint and pretrained weight utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm

from lsvit_action.models.lsvit import LSViTBackbone


def load_vit_checkpoint(
    backbone: LSViTBackbone,
    pretrained_name: str,
    weights_dir: str | Path,
) -> tuple[list[str], list[str]]:
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)
    cache_path = weights_path / f"{pretrained_name}_timm.pth"

    if cache_path.is_file():
        state_dict = torch.load(cache_path, map_location="cpu")
    else:
        pretrained_model = timm.create_model(pretrained_name, pretrained=True)
        state_dict = pretrained_model.state_dict()
        torch.save(state_dict, cache_path)

    filtered_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("head"):
            continue

        normalized_key = key
        for prefix in ("module.", "backbone."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]

        filtered_state[normalized_key] = value

    incompatible = backbone.load_state_dict(filtered_state, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
    metric: float | None = None,
    history: dict[str, list[float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if metric is not None:
        payload["metric"] = metric
    if history is not None:
        payload["history"] = history
    if extra is not None:
        payload.update(extra)

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model"])

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])

    return payload