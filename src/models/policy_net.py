"""Adaptive Policy Network and Unified AdaVCM Framework.

Learns the optimal Rate-Accuracy trade-off by predicting meta-parameters
(sigma, alpha, scale) based on frame complexity and target QP:
- Never corrupts foreground pixels directly.
- Solves the Rate-Accuracy Lagrangian optimization problem.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_filter import BoundaryAwareFilter
from .saliency import SpatioTemporalSalience
from .temporal_reg import TemporalBackgroundRegularizer


class AdaptivePolicyNet(nn.Module):
    """Predicts optimal preprocessing parameters conditioned on video stats and target QP."""

    def __init__(self, in_features: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # [sigma_norm, alpha_norm, scale_logits]
        )
        # Initialize near center of parameter ranges
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extract_stats(self, x: torch.Tensor, weight_map: torch.Tensor, qp: float) -> torch.Tensor:
        """Extract lightweight spatial, temporal, and semantic statistics."""
        b, c, t, h, w = x.shape
        # 1. Mean spatial variance
        var_spatial = torch.var(x, dim=(-2, -1)).mean(dim=(1, 2))  # [B]
        # 2. Foreground coverage ratio
        fg_ratio = weight_map.mean(dim=(-3, -2, -1)).squeeze(1)  # [B]
        # 3. Motion energy
        if t > 1:
            motion_energy = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean(dim=(1, 2, 3, 4))  # [B]
        else:
            motion_energy = torch.zeros(b, device=x.device)
        # 4. Normalized QP (typically in [20, 50])
        qp_norm = torch.full((b,), (qp - 20.0) / 30.0, device=x.device, dtype=torch.float32)
        # 5. Aspect ratio / resolution proxy
        res_proxy = torch.full((b,), (h * w) / (1920.0 * 1080.0), device=x.device, dtype=torch.float32)
        # 6. Mean luminance
        mean_luma = x.mean(dim=(1, 2, 3, 4))  # [B]

        stats = torch.stack([var_spatial, fg_ratio, motion_energy, qp_norm, res_proxy, mean_luma], dim=1)
        return stats

    def forward(
        self, x: torch.Tensor, weight_map: torch.Tensor, qp: float = 35.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts (sigma, alpha_temporal, scale_weight)."""
        stats = self.extract_stats(x, weight_map, qp)
        raw = self.net(stats)

        # Sigma mapped to [2.0, 16.0]
        sigma = 2.0 + 14.0 * torch.sigmoid(raw[:, 0])
        # Alpha mapped to [0.0, 0.95]
        alpha = 0.95 * torch.sigmoid(raw[:, 1])
        # Scale choice or weight
        scale_weight = torch.sigmoid(raw[:, 2])

        return sigma, alpha, scale_weight


class AdaVCM(nn.Module):
    """Unified Adaptive Preprocessing System for Video Coding for Machines."""

    def __init__(
        self,
        dilate_radius: int = 5,
        default_sigma: float = 6.0,
        default_alpha: float = 0.85,
        learnable_policy: bool = True,
    ):
        super().__init__()
        self.salience_estimator = SpatioTemporalSalience(dilate_radius=dilate_radius)
        self.spatial_filter = BoundaryAwareFilter(default_sigma=default_sigma)
        self.temporal_reg = TemporalBackgroundRegularizer(default_alpha=default_alpha)
        self.policy_net = AdaptivePolicyNet() if learnable_policy else None

    def forward(
        self,
        x: torch.Tensor,
        prior_mask: torch.Tensor | None = None,
        boxes: list | None = None,
        scores: list | None = None,
        qp: float = 35.0,
    ) -> dict[str, torch.Tensor]:
        """Full adaptive preprocessing pass.

        Returns:
            dict containing:
                'preprocessed': Processed video clip [B, C, T, H, W]
                'weight_map': Task salience map [B, 1, T, H, W]
                'sigma': Predicted or default blur sigma
                'alpha': Predicted or default temporal factor
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)

        # 1. Estimate Spatio-Temporal Salience Map
        weight_map = self.salience_estimator(x, prior_mask=prior_mask, boxes=boxes, scores=scores)

        # 2. Query Adaptive Policy Net (if enabled)
        if self.policy_net is not None:
            sigma, alpha, _ = self.policy_net(x, weight_map, qp=qp)
        else:
            sigma = torch.tensor(self.spatial_filter.default_sigma, device=x.device, dtype=torch.float32)
            alpha = torch.tensor(self.temporal_reg.default_alpha, device=x.device, dtype=torch.float32)

        # 3. Apply Temporal Background Regularization
        x_temporal = self.temporal_reg(x, weight_map, alpha=alpha)

        # 4. Apply Boundary-Aware Soft Transition Spatial Filtering
        x_out = self.spatial_filter(x_temporal, weight_map, sigma=sigma)

        return {
            "preprocessed": x_out,
            "weight_map": weight_map,
            "sigma": sigma,
            "alpha": alpha,
        }
