"""Small UNet epsilon-predictor for the toy DDPM.

Spec: ~150-400k params, unconditional, operates on 1x16x16 images with a
diffusion-timestep embedding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """16x16 -> 8x8 -> 16x16 UNet with one skip connection, epsilon-prediction."""

    def __init__(self, base_ch: int = 32, temb_raw_dim: int = 32, temb_dim: int = 128):
        super().__init__()
        C, C2 = base_ch, base_ch * 2
        self.temb_raw_dim = temb_raw_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_raw_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.in_conv = nn.Conv2d(1, C, 3, padding=1)
        self.down1 = ResBlock(C, C, temb_dim)
        self.downsample = nn.Conv2d(C, C2, 4, stride=2, padding=1)
        self.down2 = ResBlock(C2, C2, temb_dim)
        self.mid = ResBlock(C2, C2, temb_dim)
        self.upsample = nn.ConvTranspose2d(C2, C, 4, stride=2, padding=1)
        self.up1 = ResBlock(C + C, C, temb_dim)
        self.out_norm = nn.GroupNorm(min(8, C), C)
        self.out_conv = nn.Conv2d(C, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: [B,1,16,16], t: [B] int64 timesteps. Returns predicted epsilon [B,1,16,16]."""
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_raw_dim))
        h0 = self.in_conv(x)
        h1 = self.down1(h0, temb)  # [B,C,16,16]
        h2 = self.downsample(h1)  # [B,C2,8,8]
        h2 = self.down2(h2, temb)
        h2 = self.mid(h2, temb)
        h = self.upsample(h2)  # [B,C,16,16]
        h = torch.cat([h, h1], dim=1)  # [B,2C,16,16]
        h = self.up1(h, temb)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
