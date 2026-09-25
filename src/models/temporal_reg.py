"""Temporal Background Regularizer (TBR).

Stabilizes background pixels across consecutive frames to minimize
inter-frame motion residuals (P/B frames in H.264/H.265/VVC):
- Background (W ≈ 0.0): Recursive temporal smoothing reduces temporal noise and residual entropy.
- Foreground (W ≈ 1.0): Instantaneous response (no temporal lag or ghosting).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBackgroundRegularizer(nn.Module):
    def __init__(self, default_alpha: float = 0.85):
        super().__init__()
        self.default_alpha = default_alpha

    def forward(
        self,
        x: torch.Tensor,
        weight_map: torch.Tensor,
        alpha: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply temporal stabilization on low-salience regions.

        Args:
            x: Input video [B, C, T, H, W].
            weight_map: Task importance map [B, 1, T, H, W] in [0, 1].
            alpha: Temporal persistence factor in [0, 1). Higher = lower inter-frame bitrate.
        """
        if x.ndim == 4 or x.shape[2] <= 1:
            return x  # Single frame / image regime: no temporal regularization

        b, c, t, h, w = x.shape
        a = float(alpha) if alpha is not None else self.default_alpha
        a = max(0.0, min(0.98, a))

        out_frames = []
        prev_frame = x[:, :, 0]
        out_frames.append(prev_frame)

        for step in range(1, t):
            curr_frame = x[:, :, step]
            w_curr = weight_map[:, :, step]
            # Effective alpha is zero for foreground (w=1) and `a` for background (w=0)
            eff_alpha = (1.0 - w_curr) * a
            # Stabilized frame: blend previous stabilized state into current background
            stabilized = eff_alpha * prev_frame + (1.0 - eff_alpha) * curr_frame
            prev_frame = stabilized
            out_frames.append(stabilized)

        return torch.stack(out_frames, dim=2)
