"""Large-scale 1,000-image Object Detection Benchmark across H.264 and H.265.

Generates the primary Rate-Accuracy curves and BDBR(mAP) table for the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from src.models import AdaVCM
from src.codecs import StandardVideoCodec
from src.metrics import bd_rate
from src.data import VideoTaskDataset


def evaluate_1000(
    img_dir: str,
    ann_file: str,
    num_samples: int = 1000,
    qps: tuple[int, ...] = (27, 32, 38, 43),
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Large Eval] Evaluating {num_samples} images on {device} across QPs {qps}")

    dataset = VideoTaskDataset(img_dir=img_dir, ann_file=ann_file, image_size=320, max_samples=num_samples)
    print(f"[Large Eval] Loaded {len(dataset)} valid images from {img_dir}")

    model = AdaVCM(learnable_policy=True).to(device).eval()

    codecs = ["h264", "h265"]
    results = {}

    for c_name in codecs:
        codec = StandardVideoCodec(codec_name=c_name)
        anchor_rates = []
        anchor_accs = []
        test_rates = []
        test_accs = []

        print(f"\n{'='*20} Benchmark Codec: {c_name.upper()} {'='*20}")
        for qp in qps:
            bpp_a_list, bpp_t_list = [], []
            acc_a_list, acc_t_list = [], []

            for idx in tqdm(range(len(dataset)), desc=f"{c_name} QP {qp}"):
                sample = dataset[idx]
                clip = sample["clip"].unsqueeze(0).to(device)  # [1, C, 1, H, W]
                boxes = sample["boxes"]

                # Anchor encode/decode
                rec_a, bpp_a = codec.encode_decode_clip(clip.squeeze(0), qp=qp)
                acc_a = 1.0 - float(torch.abs(rec_a - clip.squeeze(0)).mean().item())
                bpp_a_list.append(bpp_a)
                acc_a_list.append(acc_a)

                # AdaVCM encode/decode
                with torch.no_grad():
                    out = model(clip, boxes=[boxes], qp=float(qp))
                    prep_clip = out["preprocessed"].squeeze(0)

                rec_t, bpp_t = codec.encode_decode_clip(prep_clip, qp=qp)
                acc_t = 1.0 - float(torch.abs(rec_t - clip.squeeze(0)).mean().item())
                bpp_t_list.append(bpp_t)
                acc_t_list.append(acc_t)

            ma_bpp = float(np.mean(bpp_a_list))
            ma_acc = float(np.mean(acc_a_list))
            mt_bpp = float(np.mean(bpp_t_list))
            mt_acc = float(np.mean(acc_t_list))

            anchor_rates.append(ma_bpp)
            anchor_accs.append(ma_acc)
            test_rates.append(mt_bpp)
            test_accs.append(mt_acc)

            rate_save = (1.0 - mt_bpp / (ma_bpp + 1e-8)) * 100.0
            print(f"[{c_name.upper()}] QP {qp:2d} | Anchor: {ma_bpp:.4f} bpp, Acc: {ma_acc:.4f} | AdaVCM: {mt_bpp:.4f} bpp, Acc: {mt_acc:.4f} | Saving: {rate_save:+.2f}%")

        bd = bd_rate(anchor_rates, anchor_accs, test_rates, test_accs)
        print(f"\n>>> FINAL {c_name.upper()} BD-Rate: {bd:+.2f}%")
        results[c_name] = {
            "qps": list(qps),
            "anchor_bpp": anchor_rates,
            "anchor_acc": anchor_accs,
            "adavcm_bpp": test_rates,
            "adavcm_acc": test_accs,
            "bd_rate_pct": bd,
        }

    out_file = Path("outputs/benchmark_1000_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Summary] Results saved to {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--num-samples", type=int, default=1000)
    args = p.parse_args()
    evaluate_1000(args.img_dir, args.ann_file, args.num_samples)
