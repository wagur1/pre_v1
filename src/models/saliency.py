"""Spatio-Temporal Salience and Motion Estimator (ST-SME).

Identifies machine-critical regions in video frames:
1. Spatial object/semantic cues (from detections or lightweight visual saliency).
2. Temporal inter-frame motion gradients (temporal dynamics).
3. Soft morphological dilation to create smooth boundary transition buffers.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalSalience(nn.Module):
    def __init__(self, dilate_radius: int = 5, motion_weight: float = 0.4):
        super().__init__()
        self.dilate_radius = dilate_radius
        self.motion_weight = motion_weight
        # Gaussian smoothing kernel for soft boundary mask
        k_size = 2 * dilate_radius + 1
        sigma = dilate_radius / 2.0
        coords = torch.arange(k_size, dtype=torch.float32) - dilate_radius
        g1d = torch.exp(-0.5 * (coords / sigma).square())
        g2d = (g1d.view(-1, 1) @ g1d.view(1, -1))
        g2d = g2d / g2d.sum()
        self.register_buffer("smooth_kernel", g2d.view(1, 1, k_size, k_size))

    def compute_motion_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute inter-frame motion magnitude for [B, C, T, H, W]."""
        b, c, t, h, w = x.shape
        if t <= 1:
            return torch.zeros(b, 1, t, h, w, device=x.device, dtype=x.dtype)
        
        # Frame difference: |X_t - X_{t-1}|
        diff = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean(dim=1, keepdim=True)
        # Pad t=0 with the difference of t=1
        first_diff = diff[:, :, :1]
        motion = torch.cat([first_diff, diff], dim=2)
        # Normalize motion map per video clip to [0, 1]
        m_min = motion.amin(dim=(-2, -1), keepdim=True)
        m_max = motion.amax(dim=(-2, -1), keepdim=True)
        motion_norm = (motion - m_min) / (m_max - m_min + 1e-6)
        return motion_norm

    def build_detection_mask(
        self,
        boxes_list: list | torch.Tensor,
        scores_list: list | torch.Tensor | None,
        shape: tuple[int, int, int, int, int],
        score_thresh: float = 0.4,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Construct bounding-box saliency mask for batch [B, 1, T, H, W]."""
        b, _, t, h, w = shape
        mask = torch.zeros(b, 1, t, h, w, device=device, dtype=torch.float32)
        if boxes_list is None:
            return mask

        # Handle tensor or list of items
        if isinstance(boxes_list, torch.Tensor):
            boxes_flat = boxes_list.to(device).reshape(-1, 4)
            for box in boxes_flat:
                x1, y1, x2, y2 = [float(v) for v in box]
                x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
                x2, y2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
                if x2 > x1 and y2 > y1:
                    mask[:, 0, :, y1:y2, x1:x2] = 1.0
            return mask

        # Handle list format
        for batch_idx in range(min(b, len(boxes_list))):
            item = boxes_list[batch_idx]
            if item is None:
                continue
            try:
                boxes_t = torch.as_tensor(item, device=device, dtype=torch.float32).reshape(-1, 4)
            except Exception:
                continue
            for box in boxes_t:
                x1, y1, x2, y2 = [float(v) for v in box]
                x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
                x2, y2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
                if x2 > x1 and y2 > y1:
                    mask[batch_idx, 0, :, y1:y2, x1:x2] = 1.0
        return mask

    def smooth_dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply smooth continuous expansion to prevent DCT block artifacts."""
        b, c, t, h, w = mask.shape
        flat = mask.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        pad = self.dilate_radius
        # First max-pool to dilate
        dilated = F.max_pool2d(flat, kernel_size=2 * pad + 1, stride=1, padding=pad)
        # Then Gaussian filter to smooth boundaries continuously
        smoothed = F.conv2d(dilated, self.smooth_kernel, padding=pad)
        # Reshape back to [B, 1, T, H, W]
        out = smoothed.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4)
        return torch.clamp(out, 0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        prior_mask: torch.Tensor | None = None,
        boxes: list | None = None,
        scores: list | None = None,
    ) -> torch.Tensor:
        """Return combined task importance map W_task in [0, 1] [B, 1, T, H, W]."""
        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
        b, c, t, h, w = x.shape

        if prior_mask is not None:
            base_mask = prior_mask
        elif boxes is not None:
            base_mask = self.build_detection_mask(boxes, scores, (b, 1, t, h, w), device=x.device)
        else:
            # Fallback to high-gradient saliency when no prior detections are given
            gx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
            gy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            grad_mag = (gx + gy).mean(dim=1, keepdim=True)
            g_min = grad_mag.amin(dim=(-2, -1), keepdim=True)
            g_max = grad_mag.amax(dim=(-2, -1), keepdim=True)
            base_mask = (grad_mag - g_min) / (g_max - g_min + 1e-6)
            base_mask = (base_mask > 0.3).float()

        if t > 1:
            motion_map = self.compute_motion_map(x)
            fused_mask = torch.maximum(base_mask, self.motion_weight * motion_map)
        else:
            fused_mask = base_mask

        return self.smooth_dilate_mask(fused_mask)
