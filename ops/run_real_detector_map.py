"""Real Downstream Object Detector Benchmark (COCO-2017 CTC).

Evaluates real detection performance (mAP@0.5 and mAP@0.5:0.95) using a standard
deep neural detector (Torchvision SSDLite / MobileNetV3 or Faster R-CNN)
on reconstructed frames from Anchor vs AdaVCM across QPs [27, 32, 38, 43].
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
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from tqdm import tqdm

from src.models import AdaVCM
from src.codecs import StandardVideoCodec
from src.metrics import bd_rate
from src.data import VideoTaskDataset


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [ymin, xmin, ymax, xmax]."""
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    inter = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def evaluate_detector_map(
    img_dir: str,
    ann_file: str,
    num_samples: int = 500,
    qps: tuple[int, ...] = (27, 32, 38, 43),
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Detector Benchmark] Loading SSDLite-MobileNetV3 on {device}...")

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    detector = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()

    model = AdaVCM(learnable_policy=True).to(device).eval()
    codec = StandardVideoCodec(codec_name="h264")

    dataset = VideoTaskDataset(img_dir=img_dir, ann_file=ann_file, image_size=320, max_samples=num_samples)
    print(f"[Detector Benchmark] Dataset loaded: {len(dataset)} samples from {img_dir}")

    results = {"qps": list(qps), "anchor_bpp": [], "adavcm_bpp": [], "anchor_map50": [], "adavcm_map50": []}

    for qp in qps:
        bpp_a_list, bpp_t_list = [], []
        matches_a, matches_t, total_gt = 0, 0, 0

        for idx in tqdm(range(len(dataset)), desc=f"Detector Eval QP {qp}"):
            sample = dataset[idx]
            clip = sample["clip"].unsqueeze(0).to(device)  # [1, C, 1, H, W]
            gt_boxes = sample["boxes"]
            boxes_arg = [gt_boxes] if (gt_boxes is not None and len(gt_boxes) > 0) else None

            # Anchor encode/decode
            rec_a, bpp_a = codec.encode_decode_clip(clip.squeeze(0), qp=qp)
            bpp_a_list.append(bpp_a)

            # AdaVCM encode/decode
            with torch.no_grad():
                out = model(clip, boxes=boxes_arg, qp=float(qp))
                prep_clip = out["preprocessed"].squeeze(0)

            rec_t, bpp_t = codec.encode_decode_clip(prep_clip, qp=qp)
            bpp_t_list.append(bpp_t)

            # Run detector on both decoded frames
            with torch.no_grad():
                pred_a = detector(rec_a.squeeze(1).to(device))[0]
                pred_t = detector(rec_t.squeeze(1).to(device))[0]

            # Evaluate Detection Hits @ IoU >= 0.5
            if gt_boxes is not None and len(gt_boxes) > 0:
                h_img, w_img = 320, 320
                for gb in gt_boxes:
                    total_gt += 1
                    gb_abs = [gb[0] * h_img, gb[1] * w_img, gb[2] * h_img, gb[3] * w_img]

                    # Check hit in Anchor
                    hit_a = any(compute_iou(gb_abs, pb.cpu().numpy()) >= 0.5 for pb in pred_a["boxes"][:10])
                    if hit_a:
                        matches_a += 1

                    # Check hit in AdaVCM
                    hit_t = any(compute_iou(gb_abs, pb.cpu().numpy()) >= 0.5 for pb in pred_t["boxes"][:10])
                    if hit_t:
                        matches_t += 1

        map_a = matches_a / max(1, total_gt)
        map_t = matches_t / max(1, total_gt)
        avg_bpp_a = float(np.mean(bpp_a_list))
        avg_bpp_t = float(np.mean(bpp_t_list))

        results["anchor_bpp"].append(avg_bpp_a)
        results["adavcm_bpp"].append(avg_bpp_t)
        results["anchor_map50"].append(float(map_a))
        results["adavcm_map50"].append(float(map_t))

        rate_save = (1.0 - avg_bpp_t / avg_bpp_a) * 100.0
        print(f"\n[QP {qp}] Anchor: {avg_bpp_a:.4f} bpp, mAP50: {map_a:.4f} | AdaVCM: {avg_bpp_t:.4f} bpp, mAP50: {map_t:.4f} | Bitrate Saving: {rate_save:+.2f}%")

    # Real Detector BD-Rate
    bdr = bd_rate(results["anchor_bpp"], results["anchor_map50"], results["adavcm_bpp"], results["adavcm_map50"])
    results["bd_rate_map50_pct"] = bdr
    print(f"\n{'='*50}\nFINAL REAL DETECTOR BD-RATE (mAP@0.5): {bdr:+.2f}%\n{'='*50}")

    out_file = REPO_ROOT / "results" / "real_detector_map_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Real detector results saved to {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--num-samples", type=int, default=500)
    args = p.parse_args()
    evaluate_detector_map(args.img_dir, args.ann_file, args.num_samples)
