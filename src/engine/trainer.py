"""Policy Trainer for AdaVCM.

Optimizes the Adaptive Policy Network to balance:
1. High-fidelity task feature preservation on foreground (zero mAP drop).
2. Maximum background entropy/residual reduction (minimum bitrate).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import AdaVCM


class PolicyTrainer:
    def __init__(
        self,
        model: AdaVCM,
        lr: float = 3e-4,
        lambda_rate: float = 0.5,
        device: torch.device = torch.device("cpu"),
        out_dir: str | Path = "outputs/train",
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_rate = lambda_rate
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.model.policy_net is not None:
            self.optimizer = optim.AdamW(self.model.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = None

    def compute_rate_proxy(self, x: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        """Estimate bitrate cost via spatial gradient and temporal difference on background."""
        bg_weight = 1.0 - weight_map
        # Spatial gradient magnitude
        gx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        gy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        grad_energy = (gx * bg_weight[:, :, :, :, 1:]).mean() + (gy * bg_weight[:, :, :, 1:, :]).mean()

        # Temporal residual energy (for video)
        if x.shape[2] > 1:
            diff_t = torch.abs(x[:, :, 1:] - x[:, :, :-1])
            temp_energy = (diff_t * bg_weight[:, :, 1:]).mean()
        else:
            temp_energy = torch.tensor(0.0, device=x.device)

        return grad_energy + 2.0 * temp_energy

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_fg_loss = 0.0
        total_rate_loss = 0.0
        steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            clips = batch["clip"].to(self.device)
            boxes_batch = batch.get("boxes", None)
            scores_batch = batch.get("scores", None)
            qp = float(batch.get("target_qp", 35.0)[0]) if isinstance(batch.get("target_qp"), torch.Tensor) else 35.0

            if self.optimizer is not None:
                self.optimizer.zero_grad()

            # Forward pass through AdaVCM
            out = self.model(clips, boxes=boxes_batch, scores=scores_batch, qp=qp)
            x_prep = out["preprocessed"]
            w_map = out["weight_map"]

            # 1. Foreground Task Preservation Loss (Mean Absolute Error on protected region)
            fg_mask = (w_map >= 0.95).float()
            fg_diff = torch.abs(x_prep - clips) * fg_mask
            fg_loss = fg_diff.sum() / (fg_mask.sum() + 1e-6)

            # 2. Rate Minimization Loss (Proxy entropy reduction on background scaled by QP)
            rate_proxy = self.compute_rate_proxy(x_prep, w_map)
            qp_scale = float(qp) / 35.0

            # 3. Overall Objective (Rate-Accuracy Lagrangian)
            loss = fg_loss + (self.lambda_rate * qp_scale) * rate_proxy

            if self.optimizer is not None and loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_fg_loss += fg_loss.item()
            total_rate_loss += rate_proxy.item()
            steps += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "fg_diff": f"{fg_loss.item():.5f}",
                "rate_p": f"{rate_proxy.item():.4f}",
                "sigma": f"{out['sigma'].mean().item():.2f}",
                "alpha": f"{out['alpha'].mean().item():.2f}",
            })

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, steps),
            "fg_loss": total_fg_loss / max(1, steps),
            "rate_loss": total_rate_loss / max(1, steps),
        }
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict):
        ckpt_path = self.out_dir / f"adavcm_epoch_{epoch}.pth"
        best_path = self.out_dir / "adavcm_best.pth"
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        torch.save(state, ckpt_path)
        torch.save(state, best_path)
        print(f"[AdaVCM] Checkpoint saved: {ckpt_path}")
