"""Motion-aware modules used by LSViT."""

from __future__ import annotations

import torch
import torch.nn as nn


class SMIFModule(nn.Module):

    def __init__(
        self,
        channels: int,
        window_size: int = 5,
        alpha: float = 0.5,
        threshold: float = 0.05,
    ) -> None:
        super().__init__()

        if window_size % 2 == 0:
            raise ValueError("`window_size` must be odd")

        self.channels = channels
        self.window_size = window_size
        self.half_window = window_size // 2
        self.threshold = threshold

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.conv_fuse = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"`video` must have shape (B, T, C, H, W), but got {tuple(video.shape)}"
            )

        batch_size, num_frames, channels, height, width = video.shape
        motion_accum = torch.zeros_like(video)

        for offset in range(1, self.half_window + 1):
            prev_frames = torch.roll(video, shifts=offset, dims=1)
            next_frames = torch.roll(video, shifts=-offset, dims=1)

            prev_frames[:, :offset] = video[:, :offset]
            next_frames[:, -offset:] = video[:, -offset:]

            diff_forward = next_frames - video
            diff_backward = video - prev_frames

            motion_accum = motion_accum + diff_forward.abs() + diff_backward.abs()

        motion_map = motion_accum / max(self.half_window, 1)
        motion_mask = (motion_map > self.threshold).float()
        motion_map = motion_map * motion_mask

        base = video.reshape(batch_size * num_frames, channels, height, width)
        motion_flat = motion_map.reshape(batch_size * num_frames, channels, height, width)

        fused = torch.cat([base, motion_flat], dim=1)
        fused = self.conv_fuse(fused)

        out = base + self.alpha.tanh() * fused
        out = out.clamp(min=-1.0, max=1.0)

        return out.view(batch_size, num_frames, channels, height, width)


class LMIModule(nn.Module):

    def __init__(self, dim: int, reduction: int = 4, delta: float = 0.1) -> None:
        super().__init__()

        reduced_dim = max(1, dim // reduction)

        self.reduce = nn.Linear(dim, reduced_dim)
        self.expand = nn.Linear(reduced_dim, dim)

        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(reduced_dim),
            nn.Linear(reduced_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, reduced_dim),
        )

        self.delta = nn.Parameter(torch.tensor(delta, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"`x` must have shape (B, T, N, C), but got {tuple(x.shape)}"
            )

        batch_size, num_frames, num_tokens, _ = x.shape
        reduced = self.reduce(x)

        if num_frames > 1:
            diff_forward = reduced[:, 1:] - reduced[:, :-1]
            diff_forward = torch.cat([diff_forward, diff_forward[:, -1:]], dim=1)

            diff_backward = reduced[:, :-1] - reduced[:, 1:]
            diff_backward = torch.cat([diff_backward[:, :1], diff_backward], dim=1)
        else:
            diff_forward = torch.zeros_like(reduced)
            diff_backward = torch.zeros_like(reduced)

        motion = (diff_forward.abs() + diff_backward.abs()).mean(dim=2)
        motion = self.temporal_mlp(motion)

        attn = torch.sigmoid(motion).unsqueeze(2)
        attn = self.expand(attn)
        attn = attn.expand(batch_size, num_frames, num_tokens, -1)

        enhanced = x * attn
        return x + self.delta.tanh() * enhanced