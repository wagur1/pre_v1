"""Main training script for AdaVCM."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.models import AdaVCM
from src.data import VideoTaskDataset, SyntheticVCMDataset
from src.engine import PolicyTrainer


def main():
    p = argparse.ArgumentParser(description="Train AdaVCM Adaptive Preprocessing Policy")
    p.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data generator for smoke testing")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

    # Overrides
    epochs = args.epochs or cfg.get("train", {}).get("epochs", 10)
    lr = args.lr or cfg.get("train", {}).get("lr", 3e-4)
    batch_size = cfg.get("train", {}).get("batch_size", 4)
    lambda_rate = cfg.get("loss", {}).get("lambda_rate", 0.5)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[AdaVCM] Using device: {device}")

    # Dataset
    data_cfg = cfg.get("data", {})
    img_dir = data_cfg.get("img_dir", "")
    ann_file = data_cfg.get("ann_file", "")

    if args.synthetic or not Path(img_dir).exists():
        print("[AdaVCM] Initializing Synthetic Video VCM Dataset...")
        dataset = SyntheticVCMDataset(num_samples=100, num_frames=8, size=256)
    else:
        print(f"[AdaVCM] Loading dataset from: {img_dir}")
        dataset = VideoTaskDataset(
            img_dir=img_dir,
            ann_file=ann_file,
            image_size=data_cfg.get("image_size", 320),
            max_samples=data_cfg.get("max_samples", None),
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = AdaVCM(
        dilate_radius=cfg.get("model", {}).get("dilate_radius", 5),
        default_sigma=cfg.get("model", {}).get("default_sigma", 6.0),
        default_alpha=cfg.get("model", {}).get("default_alpha", 0.85),
        learnable_policy=True,
    )

    trainer = PolicyTrainer(
        model=model,
        lr=lr,
        lambda_rate=lambda_rate,
        device=device,
        out_dir=cfg.get("out_dir", "outputs/train"),
    )

    print(f"[AdaVCM] Starting training for {epochs} epochs...")
    for ep in range(1, epochs + 1):
        metrics = trainer.train_epoch(dataloader, ep)
        print(f"[Epoch {ep}] Loss: {metrics['train_loss']:.4f} | FG Diff: {metrics['fg_loss']:.6f} | Rate Proxy: {metrics['rate_loss']:.4f}")
        trainer.save_checkpoint(ep, metrics)

    print("[AdaVCM] Training completed successfully!")


if __name__ == "__main__":
    main()
