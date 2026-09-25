"""Boundary-Aware Soft Transition Pre-Filter.

Applies task-guided frequency suppression:
- Foreground objects (W ≈ 1.0): Protected with 100% bit-exact fidelity.
- Background regions (W ≈ 0.0): Strong high-frequency attenuation to slash DCT/DST bits.
- Boundary transition zone: Smooth non-linear blending to eliminate edge blocking artifacts.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareFilter(nn.Module):
    def __init__(self, default_sigma: float = 6.0):
        super().__init__()
        self.default_sigma = default_sigma

    def _gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply separable Gaussian blur to [B, C, T, H, W] tensor."""
        if sigma <= 0.1:
            return x
        b, c, t, h, w = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        k = int(2 * round(2 * sigma) + 1)
        k = max(3, k if k % 2 == 1 else k + 1)
        pad = k // 2

        coords = torch.arange(k, dtype=torch.float32, device=x.device) - pad
        g = torch.exp(-0.5 * (coords / sigma).square())
        g = g / g.sum()

        v_kernel = g.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
        h_kernel = g.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()

        # Reflect padding if dimension allows, else replicate
        pad_mode_v = "reflect" if pad < h else "replicate"
        pad_mode_h = "reflect" if pad < w else "replicate"

        out = F.conv2d(F.pad(flat, (0, 0, pad, pad), mode=pad_mode_v), v_kernel, groups=c)
        out = F.conv2d(F.pad(out, (pad, pad, 0, 0), mode=pad_mode_h), h_kernel, groups=c)
        return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        x: torch.Tensor,
        weight_map: torch.Tensor,
        sigma: float | torch.Tensor | None = None,
        edge_power: float = 1.5,
    ) -> torch.Tensor:
        """Filter unprotected regions while strictly preserving foreground pixels.

        Args:
            x: Input video [B, C, T, H, W] in [0, 1].
            weight_map: Task importance map [B, 1, T, H, W] in [0, 1].
            sigma: Blur intensity (float or scalar tensor).
            edge_power: Controls smoothness curve across boundaries.
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if weight_map.ndim == 4:
            weight_map = weight_map.unsqueeze(2)

        s = float(sigma) if sigma is not None else self.default_sigma
        if s <= 0.0:
            return x

        blurred = self._gaussian_blur(x, s)
        # Non-linear boundary smoothing
        w_smooth = torch.clamp(weight_map, 0.0, 1.0)
        if edge_power != 1.0:
            w_smooth = torch.pow(w_smooth, edge_power)

        # Bit-exact pass through at w == 1
        filtered = torch.where(
            weight_map >= 0.999,
            x,
            w_smooth * x + (1.0 - w_smooth) * blurred
        )
        return torch.clamp(filtered, 0.0, 1.0)
