"""Evaluation and BD-Rate benchmark script for AdaVCM."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import numpy as np

from src.models import AdaVCM
from src.codecs import StandardVideoCodec
from src.metrics import bd_rate
from src.data import SyntheticVCMDataset, VideoTaskDataset


def main():
    p = argparse.ArgumentParser(description="Evaluate AdaVCM against Anchor Codec")
    p.add_argument("--checkpoint", default=None, help="Trained AdaVCM checkpoint")
    p.add_argument("--codec", default="h264", choices=["h264", "h265"])
    p.add_argument("--qps", default="27,32,38,43", help="Comma-separated QPs")
    p.add_argument("--synthetic", action="store_true", help="Run on synthetic sequences")
    args = p.parse_args()

    qps = [int(q.strip()) for q in args.qps.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AdaVCM Eval] Using device: {device} | Codec: {args.codec} | QPs: {qps}")

    model = AdaVCM(learnable_policy=True)
    ckpt_path = args.checkpoint or "outputs/train/adavcm_best.pth"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[AdaVCM Eval] Loaded checkpoint: {ckpt_path}")
    model.to(device).eval()

    dataset = SyntheticVCMDataset(num_samples=10, num_frames=8, size=256)
    codec = StandardVideoCodec(codec_name=args.codec)

    anchor_rates = []
    anchor_accuracies = []
    test_rates = []
    test_accuracies = []

    print("[AdaVCM Eval] Running Rate-Accuracy sweeps...")
    for qp in qps:
        bpp_anchor_list = []
        bpp_test_list = []
        acc_anchor_list = []
        acc_test_list = []

        for idx in range(len(dataset)):
            sample = dataset[idx]
            clip = sample["clip"].unsqueeze(0).to(device)  # [1, C, T, H, W]
            boxes = sample["boxes"]

            # Anchor run (unfiltered)
            rec_anchor, bpp_a = codec.encode_decode_clip(clip.squeeze(0), qp=qp)
            # Simulated task accuracy (foreground IoU or MSE fidelity)
            acc_a = 1.0 - float(torch.abs(rec_anchor - clip.squeeze(0)).mean().item())
            bpp_anchor_list.append(bpp_a)
            acc_anchor_list.append(acc_a)

            # AdaVCM Preprocessed run
            with torch.no_grad():
                out = model(clip, boxes=[boxes], qp=float(qp))
                prep_clip = out["preprocessed"].squeeze(0)

            rec_test, bpp_t = codec.encode_decode_clip(prep_clip, qp=qp)
            acc_t = 1.0 - float(torch.abs(rec_test - clip.squeeze(0)).mean().item())
            bpp_test_list.append(bpp_t)
            acc_test_list.append(acc_t)

        mean_bpp_a = float(np.mean(bpp_anchor_list))
        mean_acc_a = float(np.mean(acc_anchor_list))
        mean_bpp_t = float(np.mean(bpp_test_list))
        mean_acc_t = float(np.mean(acc_test_list))

        anchor_rates.append(mean_bpp_a)
        anchor_accuracies.append(mean_acc_a)
        test_rates.append(mean_bpp_t)
        test_accuracies.append(mean_acc_t)

        saving_pct = (1.0 - mean_bpp_t / (mean_bpp_a + 1e-8)) * 100.0
        print(f"QP {qp:2d} | Anchor: {mean_bpp_a:.4f} bpp, Acc: {mean_acc_a:.4f} | AdaVCM: {mean_bpp_t:.4f} bpp, Acc: {mean_acc_t:.4f} | Bit Saving: {saving_pct:+.2f}%")

    # Compute BD-Rate
    bd = bd_rate(anchor_rates, anchor_accuracies, test_rates, test_accuracies)
    print("=" * 60)
    print(f"[RESULT] Overall BD-Rate: {bd:.2f}% (Negative = bitrate saving)")
    print("=" * 60)


if __name__ == "__main__":
    main()
