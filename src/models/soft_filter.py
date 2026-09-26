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
    def __init__(self, default_sigma: float = 3.0):
        super().__init__()
        self.default_sigma = default_sigma

    def _gaussian_blur(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """Apply separable Gaussian blur differentiably to [B, C, T, H, W] tensor."""
        if not isinstance(sigma, torch.Tensor):
            sigma_t = torch.tensor(float(sigma), dtype=torch.float32, device=x.device)
        else:
            sigma_t = sigma.mean().to(device=x.device, dtype=torch.float32)

        sigma_clamped = torch.clamp(sigma_t, min=0.5, max=5.0)

        b, c, t, h, w = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # Dynamic padding radius scaled to 2.5 * sigma to eliminate kernel truncation
        pad = max(3, min(15, int(math.ceil(2.5 * float(sigma_clamped.detach().item())))))
        k = 2 * pad + 1

        coords = torch.arange(-pad, pad + 1, dtype=torch.float32, device=x.device)
        g = torch.exp(-0.5 * (coords / sigma_clamped).square())
        g = g / (g.sum() + 1e-8)

        v_kernel = g.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
        h_kernel = g.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()

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

        if sigma is None:
            sigma = self.default_sigma

        blurred = self._gaussian_blur(x, sigma)
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
