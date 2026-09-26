"""Ablation Study Runner for AdaVCM.

Evaluates 4 configurations to isolate each contribution:
1. Full AdaVCM (Proposed)
2. No-TBR (Removes temporal background regularization)
3. Hard-Mask (Removes smooth sigmoid boundary, applies step cutoff)
4. Fixed-Params (Removes learned policy network, uses static sigma/alpha)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from src.models import AdaVCM
from src.codecs import StandardVideoCodec
from src.metrics import bd_rate
from src.data import VideoTaskDataset


def run_ablation(img_dir: str, ann_file: str, num_samples: int = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ablation] Running ablation study on {num_samples} samples...")

    dataset = VideoTaskDataset(img_dir=img_dir, ann_file=ann_file, image_size=320, max_samples=num_samples)
    codec = StandardVideoCodec(codec_name="h264")
    qps = [27, 32, 38, 43]

    model_full = AdaVCM(learnable_policy=True).to(device)
    for p in [REPO_ROOT / "outputs/train/adavcm_best.pth", Path("outputs/train/adavcm_best.pth")]:
        if p.exists():
            state = torch.load(p, map_location=device, weights_only=False)
            model_full.load_state_dict(state.get("model_state_dict", state))
            print(f"[Ablation] Successfully loaded trained checkpoint: {p}")
            break
    model_full.eval()

    variants = ["anchor", "full_adavcm", "no_tbr", "hard_mask", "fixed_params"]
    rates = {v: [] for v in variants}
    accs = {v: [] for v in variants}

    for qp in qps:
        temp_rates = {v: [] for v in variants}
        temp_accs = {v: [] for v in variants}

        for idx in tqdm(range(len(dataset)), desc=f"Ablation QP {qp}"):
            sample = dataset[idx]
            clip = sample["clip"].unsqueeze(0).to(device)
            boxes = sample["boxes"]
            boxes_arg = [boxes] if (boxes is not None and len(boxes) > 0) else None

            # 1. Anchor
            rec_a, bpp_a = codec.encode_decode_clip(clip.squeeze(0), qp=qp)
            acc_a = 1.0 - float(torch.abs(rec_a - clip.squeeze(0)).mean().item())
            temp_rates["anchor"].append(bpp_a)
            temp_accs["anchor"].append(acc_a)

            # 2. Full AdaVCM
            with torch.no_grad():
                out_full = model_full(clip, boxes=boxes_arg, qp=float(qp))
                prep_full = out_full["preprocessed"].squeeze(0)
            rec_full, bpp_full = codec.encode_decode_clip(prep_full, qp=qp)
            acc_full = 1.0 - float(torch.abs(rec_full - clip.squeeze(0)).mean().item())
            temp_rates["full_adavcm"].append(bpp_full)
            temp_accs["full_adavcm"].append(acc_full)

            # 3. No TBR (temporal alpha = 0)
            with torch.no_grad():
                w_map = model_full.salience_estimator(clip, boxes=boxes_arg)
                sigma = model_full.spatial_filter.default_sigma
                prep_notbr = model_full.spatial_filter(clip, w_map, sigma=sigma).squeeze(0)
            rec_notbr, bpp_notbr = codec.encode_decode_clip(prep_notbr, qp=qp)
            acc_notbr = 1.0 - float(torch.abs(rec_notbr - clip.squeeze(0)).mean().item())
            temp_rates["no_tbr"].append(bpp_notbr)
            temp_accs["no_tbr"].append(acc_notbr)

            # 4. Hard Mask (Step cutoff)
            with torch.no_grad():
                hard_w = (w_map >= 0.5).float()
                prep_hard = (clip * hard_w + 0.5 * (1.0 - hard_w)).squeeze(0)
            rec_hard, bpp_hard = codec.encode_decode_clip(prep_hard, qp=qp)
            acc_hard = 1.0 - float(torch.abs(rec_hard - clip.squeeze(0)).mean().item())
            temp_rates["hard_mask"].append(bpp_hard)
            temp_accs["hard_mask"].append(acc_hard)

            # 5. Fixed params (No policy)
            with torch.no_grad():
                prep_fixed = model_full.spatial_filter(clip, w_map, sigma=6.0).squeeze(0)
            rec_fixed, bpp_fixed = codec.encode_decode_clip(prep_fixed, qp=qp)
            acc_fixed = 1.0 - float(torch.abs(rec_fixed - clip.squeeze(0)).mean().item())
            temp_rates["fixed_params"].append(bpp_fixed)
            temp_accs["fixed_params"].append(acc_fixed)

        for v in variants:
            rates[v].append(float(np.mean(temp_rates[v])))
            accs[v].append(float(np.mean(temp_accs[v])))

    # Compute BD-Rate against Anchor for each variant
    ablation_summary = {}
    print("\n" + "=" * 50)
    print("ABLATION STUDY RESULTS (BD-Rate vs Anchor)")
    print("=" * 50)
    for v in ["full_adavcm", "no_tbr", "hard_mask", "fixed_params"]:
        bd = bd_rate(rates["anchor"], accs["anchor"], rates[v], accs[v])
        ablation_summary[v] = bd
        print(f"Variant: {v:15s} | BD-Rate: {bd:+.2f}%")
    print("=" * 50)

    out_file = Path("outputs/ablation_study_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(ablation_summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--num-samples", type=int, default=300)
    args = p.parse_args()
    run_ablation(args.img_dir, args.ann_file, args.num_samples)
