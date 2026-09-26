"""Real Downstream Object Detector Benchmark (COCO-2017 CTC).

Evaluates real detection performance (true mAP@0.5 and mAP@0.5:0.95) using a standard
deep neural detector (Torchvision SSDLite-MobileNetV3) on reconstructed frames from
Anchor vs AdaVCM across standard MPEG-VCM QPs [27, 32, 38, 43].
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
from src.metrics import bd_rate, DetectionEvaluator
from src.data import VideoTaskDataset


def evaluate_detector_map(
    img_dir: str,
    ann_file: str,
    num_samples: int = 500,
    qps: tuple[int, ...] = (27, 32, 38, 43),
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Detector Benchmark] Initializing on {device}...")

    # 1. Load Real Object Detector
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    detector = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()

    # 2. Load AdaVCM Model and Trained Checkpoint
    model = AdaVCM(learnable_policy=True).to(device).eval()
    ckpt_candidates = [
        REPO_ROOT / "checkpoints" / "adavcm_best.pth",
        REPO_ROOT / "outputs" / "train" / "adavcm_best.pth",
        REPO_ROOT / "outputs" / "adavcm_best.pth",
    ]
    loaded_ckpt = False
    for ckpt_path in ckpt_candidates:
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            print(f"[Detector Benchmark] Successfully loaded AdaVCM checkpoint: {ckpt_path}")
            loaded_ckpt = True
            break
    if not loaded_ckpt:
        print("[Detector Benchmark] Warning: Checkpoint not found; running with default policy weights.")

    codec = StandardVideoCodec(codec_name="h264")
    dataset = VideoTaskDataset(img_dir=img_dir, ann_file=ann_file, image_size=320, max_samples=num_samples)
    print(f"[Detector Benchmark] Dataset loaded: {len(dataset)} valid samples from {img_dir}")

    results = {
        "qps": list(qps),
        "anchor_bpp": [],
        "adavcm_bpp": [],
        "bitrate_savings_pct": [],
        "anchor_map50": [],
        "adavcm_map50": [],
        "anchor_map50_95": [],
        "adavcm_map50_95": [],
    }

    for qp in qps:
        bpp_a_list, bpp_t_list = [], []
        evaluator_anchor = DetectionEvaluator()
        evaluator_adavcm = DetectionEvaluator()

        for idx in tqdm(range(len(dataset)), desc=f"Detector Eval QP {qp}"):
            sample = dataset[idx]
            clip = sample["clip"].unsqueeze(0).to(device)  # [1, C, 1, H, W]
            gt_boxes = sample["boxes"]  # List of [x1, y1, x2, y2] in pixel coords [0, 320]
            boxes_arg = [gt_boxes] if (gt_boxes is not None and len(gt_boxes) > 0) else None

            # 1. Anchor encode/decode
            rec_a, bpp_a = codec.encode_decode_clip(clip.squeeze(0), qp=qp)
            bpp_a_list.append(bpp_a)

            # 2. AdaVCM encode/decode
            with torch.no_grad():
                out = model(clip, boxes=boxes_arg, qp=float(qp))
                prep_clip = out["preprocessed"].squeeze(0)

            rec_t, bpp_t = codec.encode_decode_clip(prep_clip, qp=qp)
            bpp_t_list.append(bpp_t)

            # 3. Feed decoded frames to detector as List[Tensor[C, H, W]]
            with torch.no_grad():
                frame_a = rec_a.squeeze(1).squeeze(0).to(device)  # [C, H, W]
                frame_t = rec_t.squeeze(1).squeeze(0).to(device)  # [C, H, W]

                pred_a = detector([frame_a])[0]
                pred_t = detector([frame_t])[0]

            # 4. Accumulate for standard COCO/Pascal VOC mAP calculation
            gt_arr = np.array(gt_boxes, dtype=np.float32) if (gt_boxes and len(gt_boxes) > 0) else np.empty((0, 4), dtype=np.float32)

            evaluator_anchor.add_image_eval(
                gt_boxes=gt_arr,
                pred_boxes=pred_a["boxes"].cpu().numpy(),
                pred_scores=pred_a["scores"].cpu().numpy(),
            )
            evaluator_adavcm.add_image_eval(
                gt_boxes=gt_arr,
                pred_boxes=pred_t["boxes"].cpu().numpy(),
                pred_scores=pred_t["scores"].cpu().numpy(),
            )

        # Compute true AP@50 and AP@50:95
        res_a = evaluator_anchor.evaluate()
        res_t = evaluator_adavcm.evaluate()

        avg_bpp_a = float(np.mean(bpp_a_list))
        avg_bpp_t = float(np.mean(bpp_t_list))
        rate_save = (1.0 - avg_bpp_t / avg_bpp_a) * 100.0

        results["anchor_bpp"].append(avg_bpp_a)
        results["adavcm_bpp"].append(avg_bpp_t)
        results["bitrate_savings_pct"].append(rate_save)
        results["anchor_map50"].append(res_a["map50"])
        results["adavcm_map50"].append(res_t["map50"])
        results["anchor_map50_95"].append(res_a["map50_95"])
        results["adavcm_map50_95"].append(res_t["map50_95"])

        print(f"\n[QP {qp:2d}] Anchor: {avg_bpp_a:.4f} bpp, mAP50: {res_a['map50']:.4f}, mAP50:95: {res_a['map50_95']:.4f}")
        print(f"        AdaVCM: {avg_bpp_t:.4f} bpp, mAP50: {res_t['map50']:.4f}, mAP50:95: {res_t['map50_95']:.4f}")
        print(f"        Bitrate Saving: {rate_save:+.2f}%\n")

    # True Detector BD-Rates
    bdr_50 = bd_rate(results["anchor_bpp"], results["anchor_map50"], results["adavcm_bpp"], results["adavcm_map50"])
    bdr_50_95 = bd_rate(results["anchor_bpp"], results["anchor_map50_95"], results["adavcm_bpp"], results["adavcm_map50_95"])

    results["bd_rate_map50_pct"] = bdr_50
    results["bd_rate_map50_95_pct"] = bdr_50_95

    print("=" * 65)
    print(f"REAL DETECTOR EVALUATION COMPLETE")
    print(f"  BD-Rate (mAP@0.5)     : {bdr_50:+.2f}%")
    print(f"  BD-Rate (mAP@0.5:0.95): {bdr_50_95:+.2f}%")
    print(f"  Average Bitrate Saving: {np.mean(results['bitrate_savings_pct']):+.2f}%")
    print("=" * 65)

    out_file = REPO_ROOT / "results" / "real_detector_map_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Real detector results saved to {out_file}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--num-samples", type=int, default=500)
    args = p.parse_args()
    evaluate_detector_map(args.img_dir, args.ann_file, args.num_samples)
