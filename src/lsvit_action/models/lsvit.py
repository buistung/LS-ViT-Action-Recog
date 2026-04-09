"""LSViT architecture for video action recognition."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsvit_action.config import ModelConfig
from lsvit_action.models.layers import Attention, DropPath, Mlp, PatchEmbed
from lsvit_action.models.motion import LMIModule, SMIFModule


class LSViTBlock(nn.Module):
    """One LSViT transformer block with temporal motion interaction."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        """Initialize one LSViT block.

        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            drop_rate: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop_rate,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop_rate)
        self.drop_path2 = DropPath(drop_path)

        self.lmim = LMIModule(dim)

    def forward(self, x: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        """Run one LSViT block.

        Args:
            x: Token tensor of shape `(B*T, N, C)`.
            batch_size: Original batch size `B`.
            num_frames: Number of frames `T`.

        Returns:
            Output tensor of shape `(B*T, N, C)`.

        Raises:
            ValueError: If the leading dimension is inconsistent with `B*T`.
        """
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        batch_time, num_tokens, dim = x.shape
        if batch_time != batch_size * num_frames:
            raise ValueError(
                f"Expected first dim to be B*T = {batch_size * num_frames}, "
                f"but got {batch_time}"
            )

        x = x.view(batch_size, num_frames, num_tokens, dim)
        x = self.lmim(x)
        x = x.view(batch_size * num_frames, num_tokens, dim)
        return x


class LSViTBackbone(nn.Module):
    """Backbone that extracts frame-wise ViT features with temporal interaction."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the LSViT backbone.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.drop_rate)

        drop_path_rates = torch.linspace(
            0.0,
            config.drop_path_rate,
            steps=config.depth,
        ).tolist()

        self.blocks = nn.ModuleList(
            [
                LSViTBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_rate=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=drop_path_rates[i],
                )
                for i in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate positional embeddings to match current patch count.

        Args:
            x: Token tensor of shape `(B, N, C)` including CLS token.

        Returns:
            Positional embedding tensor of shape `(1, N, C)`.
        """
        _, num_tokens, _ = x.shape
        num_patches = num_tokens - 1

        if num_patches == self.patch_embed.num_patches:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = patch_pos.shape[-1]

        grid_old = int(math.sqrt(patch_pos.shape[1]))
        grid_new = int(math.sqrt(num_patches))

        patch_pos = patch_pos.reshape(1, grid_old, grid_old, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(grid_new, grid_new),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_new * grid_new, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"`video` must have shape (B, T, C, H, W), but got {tuple(video.shape)}"
            )

        batch_size, num_frames, channels, height, width = video.shape

        x = video.reshape(batch_size * num_frames, channels, height, width)
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        pos_embed = self.interpolate_pos_encoding(x)
        x = x + pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, batch_size=batch_size, num_frames=num_frames)

        x = self.norm(x)
        x = x.view(batch_size, num_frames, x.shape[1], x.shape[2])
        return x


class LSViTForAction(nn.Module):
    """End-to-end LSViT model for video action recognition."""

    def __init__(
        self,
        config: ModelConfig,
        num_classes: int | None = None,
        smif_window: int | None = None,
    ) -> None:
        super().__init__()

        output_classes = num_classes if num_classes is not None else config.num_classes
        window_size = smif_window if smif_window is not None else config.smif_window

        self.smif = SMIFModule(config.in_chans, window_size=window_size)
        self.backbone = LSViTBackbone(config)
        self.head = nn.Linear(config.embed_dim, output_classes)

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        x = self.smif(video)
        feats = self.backbone(x)
        cls_tokens = feats[:, :, 0]
        pooled = cls_tokens.mean(dim=1)
        return pooled

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_features(video)
        logits = self.head(pooled)
        return logits