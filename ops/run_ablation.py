"""Ablation Study Runner for AdaVCM.

Evaluates 4 configurations to isolate each module's contribution:
1. Full AdaVCM (Proposed): Adaptive PolicyNet + Soft Sigmoid Filter + TBR
2. No-TBR: Adaptive PolicyNet + Soft Sigmoid Filter (temporal alpha = 0.0)
3. Hard-Mask: Binary step boundary (W in {0, 1}) without smooth sigmoid transition
4. Fixed-Params: Static sigma=5.0 and static TBR alpha=0.85 (No PolicyNet)
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
from src.data import VideoTaskDataset, SyntheticVCMDataset


def run_ablation(img_dir: str | None = None, ann_file: str | None = None, num_samples: int = 50, synthetic: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ablation] Running ablation study on {num_samples} samples (seed={seed}) on {device}...")

    if synthetic or not img_dir or not Path(img_dir).exists():
        print("[Ablation] Using SyntheticVCMDataset (256x256, 8 frames per clip, seeded)...")
        dataset = SyntheticVCMDataset(num_samples=num_samples, num_frames=8, size=256, seed=seed)
    else:
        print(f"[Ablation] Loading real dataset from {img_dir}...")
        dataset = VideoTaskDataset(img_dir=img_dir, ann_file=ann_file, image_size=320, max_samples=num_samples)

    codec = StandardVideoCodec(codec_name="h264")
    qps = [27, 32, 38, 43]

    model_full = AdaVCM(learnable_policy=True).to(device)
    for p in [REPO_ROOT / "checkpoints/adavcm_best.pth", REPO_ROOT / "outputs/train/adavcm_best.pth"]:
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
            acc_a = 1.0 - float(torch.abs(rec_a - clip.squeeze(0)).mean().item()) * 0.2
            temp_rates["anchor"].append(bpp_a)
            temp_accs["anchor"].append(acc_a)

            # Common salience estimation
            with torch.no_grad():
                w_map = model_full.salience_estimator(clip, boxes=boxes_arg)
                sigma_dyn, alpha_dyn, _ = model_full.policy_net(clip, w_map, qp=float(qp))

            # 2. Full AdaVCM (Adaptive Policy + Soft Sigmoid + TBR)
            with torch.no_grad():
                x_temp_full = model_full.temporal_reg(clip, w_map, alpha=alpha_dyn)
                prep_full = model_full.spatial_filter(x_temp_full, w_map, sigma=sigma_dyn).squeeze(0)
            rec_full, bpp_full = codec.encode_decode_clip(prep_full, qp=qp)
            acc_full = 1.0 - float(torch.abs(rec_full - clip.squeeze(0)).mean().item()) * 0.2
            temp_rates["full_adavcm"].append(bpp_full)
            temp_accs["full_adavcm"].append(acc_full)

            # 3. No TBR (Adaptive Policy + Soft Sigmoid, but alpha = 0.0)
            with torch.no_grad():
                prep_notbr = model_full.spatial_filter(clip, w_map, sigma=sigma_dyn).squeeze(0)
            rec_notbr, bpp_notbr = codec.encode_decode_clip(prep_notbr, qp=qp)
            acc_notbr = 1.0 - float(torch.abs(rec_notbr - clip.squeeze(0)).mean().item()) * 0.2
            temp_rates["no_tbr"].append(bpp_notbr)
            temp_accs["no_tbr"].append(acc_notbr)

            # 4. Hard Mask (Binary step cutoff W in {0, 1} without soft sigmoid boundary)
            with torch.no_grad():
                hard_w = (w_map >= 0.5).float()
                x_temp_hard = model_full.temporal_reg(clip, hard_w, alpha=alpha_dyn)
                prep_hard = model_full.spatial_filter(x_temp_hard, hard_w, sigma=sigma_dyn).squeeze(0)
            rec_hard, bpp_hard = codec.encode_decode_clip(prep_hard, qp=qp)
            acc_hard = 1.0 - float(torch.abs(rec_hard - clip.squeeze(0)).mean().item()) * 0.2
            temp_rates["hard_mask"].append(bpp_hard)
            temp_accs["hard_mask"].append(acc_hard)

            # 5. Fixed params (Static sigma=5.0 + Static TBR alpha=0.85, without policy adaptation)
            with torch.no_grad():
                x_fixed_spatial = model_full.spatial_filter(clip, w_map, sigma=5.0)
                prep_fixed = model_full.temporal_reg(x_fixed_spatial, w_map, alpha=0.85).squeeze(0)
            rec_fixed, bpp_fixed = codec.encode_decode_clip(prep_fixed, qp=qp)
            acc_fixed = 1.0 - float(torch.abs(rec_fixed - clip.squeeze(0)).mean().item()) * 0.2
            temp_rates["fixed_params"].append(bpp_fixed)
            temp_accs["fixed_params"].append(acc_fixed)

        for v in variants:
            rates[v].append(float(np.mean(temp_rates[v])))
            accs[v].append(float(np.mean(temp_accs[v])))

    # Compute BD-Rate and direct Bitrate Savings against Anchor
    is_synth = bool(synthetic or not img_dir or not Path(img_dir).exists())
    ablation_summary = {
        "metadata": {
            "num_samples": num_samples,
            "seed": seed,
            "dataset": "SyntheticVCMDataset (256x256, 8 frames)" if is_synth else "COCO-2017",
            "primary_metric": "avg_bitrate_saving_pct",
            "codec": "H.264 / libx264",
            "note": "Ablation isolates module contributions under controlled temporal synthetic sequences. Bitrate savings (Delta R) is the stable primary metric. Whole-frame pixel BD-rate is provided for continuity but is mathematically sensitive to polynomial interpolation over narrow fidelity intervals (1 - 0.2*MAE)."
        },
        "qps": qps,
        "anchor_rates": rates["anchor"],
        "anchor_accs": accs["anchor"],
        "variants": {}
    }

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS (Empirical Evaluation vs Anchor)")
    print("=" * 70)
    for v in ["full_adavcm", "no_tbr", "hard_mask", "fixed_params"]:
        bd = bd_rate(rates["anchor"], accs["anchor"], rates[v], accs[v])
        rate_savings = [(1.0 - rv / (ra + 1e-8)) * 100.0 for ra, rv in zip(rates["anchor"], rates[v])]
        mean_saving = float(np.mean(rate_savings))
        ablation_summary["variants"][v] = {
            "rates_bpp": rates[v],
            "accs": accs[v],
            "bitrate_savings_pct": [round(s, 2) for s in rate_savings],
            "avg_bitrate_saving_pct": round(mean_saving, 2),
            "pixel_proxy_bd_rate_pct": round(bd, 2),
        }
        print(f"Variant: {v:15s} | Avg Bit Saving: {mean_saving:+6.2f}% | Pixel BD-Rate: {bd:+6.2f}%")
    print("=" * 70)

    for out_path in [REPO_ROOT / "results" / "ablation_study_results.json", REPO_ROOT / "outputs" / "ablation_study_results.json"]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(ablation_summary, f, indent=2)
        print(f"[Ablation] Saved results to {out_path}")

    return ablation_summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default=None)
    p.add_argument("--ann-file", default=None)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_ablation(args.img_dir, args.ann_file, args.num_samples, synthetic=args.synthetic, seed=args.seed)
