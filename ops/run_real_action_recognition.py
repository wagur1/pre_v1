"""Real Downstream Neural Action Recognition Benchmark for AdaVCM.

Evaluates Anchor Codec vs AdaVCM across standard MPEG-VCM QPs [27, 32, 38, 43]
using a real deep 3D-CNN Action Recognition model (Torchvision R3D-18 pretrained
on Kinetics-400) on real action recognition video sequences (UCF-101, HMDB-51, Kinetics).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision.models.video as vmodels
from torchvision.models.video import R3D_18_Weights
from tqdm import tqdm

from src.models import AdaVCM
from src.codecs import StandardVideoCodec
from src.metrics import bd_rate


def extract_video_clips(
    video_dir: Path,
    clip_len: int = 8,
    max_total_clips: int = 30,
) -> list[dict]:
    """Extract multi-frame clips from downloaded action recognition video files."""
    video_files = sorted(video_dir.glob("*.avi")) + sorted(video_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    clips_per_video = max(2, max_total_clips // len(video_files))
    collected_clips = []

    for vfile in video_files:
        # Probe video to get total frame count, width, height
        cmd_probe = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames",
            "-of", "json", str(vfile),
        ]
        res = subprocess.run(cmd_probe, capture_output=True, text=True)
        try:
            info = json.loads(res.stdout).get("streams", [{}])[0]
            w = int(info.get("width", 320))
            h = int(info.get("height", 240))
            nb_frames = int(info.get("nb_frames", 60))
        except Exception:
            w, h, nb_frames = 320, 240, 60

        if nb_frames < clip_len:
            continue

        # Choose start frames distributed across video
        max_start = nb_frames - clip_len
        starts = np.linspace(0, max_start, min(clips_per_video, max_start // clip_len + 1), dtype=int)

        for s_idx, start_f in enumerate(starts):
            if len(collected_clips) >= max_total_clips:
                break
            cmd_read = [
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-ss", f"{start_f / 30.0:.3f}",
                "-i", str(vfile),
                "-vframes", str(clip_len),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}",
                "-"
            ]
            proc = subprocess.Popen(cmd_read, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
            raw_out, _ = proc.communicate()

            expected_bytes = clip_len * h * w * 3
            if len(raw_out) != expected_bytes:
                continue

            frames_np = np.frombuffer(raw_out, dtype=np.uint8).copy().reshape(clip_len, h, w, 3)
            tensor_clip = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

            collected_clips.append({
                "clip": tensor_clip,
                "video_name": vfile.stem,
                "clip_id": f"{vfile.stem}_clip{s_idx}",
                "start_frame": int(start_f),
                "resolution": f"{w}x{h}",
            })

    print(f"[Dataset] Extracted {len(collected_clips)} action clips across {len(video_files)} video sequences.")
    return collected_clips


def run_action_recognition_benchmark(
    video_dir: str = "data/action_videos",
    checkpoint: str = "checkpoints/adavcm_best.pth",
    num_clips: int = 25,
    codec_name: str = "h264",
    output_json: str = "results/real_action_recognition_results.json",
    log_file: str = "logs/action_recognition_benchmark.log",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AdaVCM AR] Running Action Recognition Benchmark on {device}...")

    # 1. Load Pretrained 3D Action Recognition Model (R3D-18 on Kinetics-400)
    print("[AdaVCM AR] Loading Torchvision R3D-18 Action Recognition Model (Kinetics-400 weights)...")
    weights = R3D_18_Weights.KINETICS400_V1
    ar_transforms = weights.transforms()
    ar_model = vmodels.r3d_18(weights=weights).to(device).eval()
    categories = weights.meta["categories"]

    # 2. Load Trained AdaVCM Preprocessor
    model = AdaVCM(learnable_policy=True).to(device)
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[AdaVCM AR] Loaded AdaVCM checkpoint: {ckpt_path}")
    model.eval()

    codec = StandardVideoCodec(codec_name=codec_name)
    clips = extract_video_clips(Path(video_dir), clip_len=8, max_total_clips=num_clips)
    if not clips:
        sys.exit("[Error] No valid video clips found for evaluation.")

    qps = [27, 32, 38, 43]

    # Pre-compute uncompressed Ground Truth action predictions
    print("[AdaVCM AR] Pre-computing ground-truth action predictions on uncompressed clips...")
    gt_predictions = []
    for c_info in clips:
        clip = c_info["clip"].to(device)
        # Transform takes [T, C, H, W]
        t_in = clip.permute(1, 0, 2, 3)
        m_in = ar_transforms(t_in).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = ar_model(m_in)
            probs = torch.softmax(logits, dim=-1)[0]
        top1_class = int(probs.argmax().item())
        gt_predictions.append({
            "top1_class": top1_class,
            "class_name": categories[top1_class],
            "confidence": float(probs[top1_class].item()),
        })

    anchor_rates, anchor_top1_accs, anchor_confidences = [], [], []
    adavcm_rates, adavcm_top1_accs, adavcm_confidences = [], [], []
    savings_list = []

    log_lines = [
        "=== Real Downstream Neural Action Recognition Benchmark Log ===",
        "Platform: Windows CPU / PyTorch",
        "Action Model: Torchvision R3D-18 (Kinetics-400 Pretrained)",
        f"Evaluated Clips: {len(clips)} action clips across UCF-101 / HMDB-51 / Kinetics",
        f"Codec: {codec_name.upper()} (libx264)",
        f"MPEG-VCM QPs: {qps}",
        "=" * 70,
        "",
    ]

    print(f"\n{'='*25} Beginning Rate-Accuracy Sweeps {'='*25}")
    for qp in qps:
        bpp_a_list, bpp_t_list = [], []
        top1_a_list, top1_t_list = [], []
        conf_a_list, conf_t_list = [], []

        pbar = tqdm(range(len(clips)), desc=f"AR Benchmark QP {qp:2d}")
        for idx in pbar:
            clip = clips[idx]["clip"].to(device)
            gt_class = gt_predictions[idx]["top1_class"]

            # --- Anchor Pipeline ---
            rec_a, bpp_a = codec.encode_decode_clip(clip, qp=qp)
            in_a = ar_transforms(rec_a.permute(1, 0, 2, 3)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs_a = torch.softmax(ar_model(in_a), dim=-1)[0]
            pred_a = int(probs_a.argmax().item())
            top1_a_match = float(pred_a == gt_class)
            conf_a = float(probs_a[gt_class].item())

            bpp_a_list.append(bpp_a)
            top1_a_list.append(top1_a_match)
            conf_a_list.append(conf_a)

            # --- AdaVCM Preprocessed Pipeline ---
            with torch.no_grad():
                out_prep = model(clip.unsqueeze(0), qp=float(qp))
                prep_clip = out_prep["preprocessed"].squeeze(0)

            rec_t, bpp_t = codec.encode_decode_clip(prep_clip, qp=qp)
            in_t = ar_transforms(rec_t.permute(1, 0, 2, 3)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs_t = torch.softmax(ar_model(in_t), dim=-1)[0]
            pred_t = int(probs_t.argmax().item())
            top1_t_match = float(pred_t == gt_class)
            conf_t = float(probs_t[gt_class].item())

            bpp_t_list.append(bpp_t)
            top1_t_list.append(top1_t_match)
            conf_t_list.append(conf_t)

        mean_bpp_a = float(np.mean(bpp_a_list))
        mean_bpp_t = float(np.mean(bpp_t_list))
        mean_top1_a = float(np.mean(top1_a_list))
        mean_top1_t = float(np.mean(top1_t_list))
        mean_conf_a = float(np.mean(conf_a_list))
        mean_conf_t = float(np.mean(conf_t_list))

        saving = (1.0 - mean_bpp_t / (mean_bpp_a + 1e-8)) * 100.0

        anchor_rates.append(mean_bpp_a)
        anchor_top1_accs.append(mean_top1_a)
        anchor_confidences.append(mean_conf_a)

        adavcm_rates.append(mean_bpp_t)
        adavcm_top1_accs.append(mean_top1_t)
        adavcm_confidences.append(mean_conf_t)
        savings_list.append(saving)

        summary_line = (
            f"QP {qp:2d} | Anchor: {mean_bpp_a:.4f} bpp, Top-1 Acc: {mean_top1_a*100:.1f}%, Conf: {mean_conf_a*100:.2f}% | "
            f"AdaVCM: {mean_bpp_t:.4f} bpp, Top-1 Acc: {mean_top1_t*100:.1f}%, Conf: {mean_conf_t*100:.2f}% | "
            f"Bit Saving: {saving:+.2f}%"
        )
        print(summary_line)
        log_lines.append(summary_line)

    # Compute Task BD-Rate (Rate vs Action Recognition Confidence)
    bd_conf = bd_rate(anchor_rates, anchor_confidences, adavcm_rates, adavcm_confidences)
    bd_top1 = bd_rate(anchor_rates, anchor_top1_accs, adavcm_rates, adavcm_top1_accs)
    avg_saving = float(np.mean(savings_list))

    print("=" * 70)
    print(f"[RESULT] Average Bitrate Saving (Delta R): {avg_saving:+.2f}%")
    print(f"[RESULT] Action Recognition Task BD-Rate (Top-1 Axis): {bd_top1:+.2f}%")
    print(f"[RESULT] Action Recognition Task BD-Rate (Confidence Axis): {bd_conf:+.2f}%")
    print("=" * 70)

    log_lines.append("")
    log_lines.append("=" * 70)
    log_lines.append(f"Average Bitrate Saving (Delta R): {avg_saving:+.2f}%")
    log_lines.append(f"Action Recognition Task BD-Rate (Top-1 Axis): {bd_top1:+.2f}%")
    log_lines.append(f"Action Recognition Task BD-Rate (Confidence Axis): {bd_conf:+.2f}%")
    log_lines.append("=" * 70)

    # Write log
    log_path = REPO_ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"[Done] Log written to {log_path}")

    # Write JSON results
    results_data = {
        "task": "Action Recognition",
        "model": "Torchvision R3D-18 (Kinetics-400)",
        "num_clips": len(clips),
        "qps": qps,
        "anchor_rates_bpp": anchor_rates,
        "adavcm_rates_bpp": adavcm_rates,
        "bitrate_savings_pct": savings_list,
        "avg_bitrate_saving_pct": avg_saving,
        "anchor_top1_acc": anchor_top1_accs,
        "adavcm_top1_acc": adavcm_top1_accs,
        "anchor_confidences": anchor_confidences,
        "adavcm_confidences": adavcm_confidences,
        "task_bd_rate_top1_pct": bd_top1,
        "task_bd_rate_confidence_pct": bd_conf,
    }
    out_p = REPO_ROOT / output_json
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"[Done] Results saved to {out_p}")


def main():
    p = argparse.ArgumentParser(description="Action Recognition Benchmark for AdaVCM")
    p.add_argument("--video-dir", default="data/action_videos")
    p.add_argument("--checkpoint", default="checkpoints/adavcm_best.pth")
    p.add_argument("--num-clips", type=int, default=25)
    p.add_argument("--codec", default="h264")
    p.add_argument("--output", default="results/real_action_recognition_results.json")
    p.add_argument("--log", default="logs/action_recognition_benchmark.log")
    args = p.parse_args()

    run_action_recognition_benchmark(
        video_dir=args.video_dir,
        checkpoint=args.checkpoint,
        num_clips=args.num_clips,
        codec_name=args.codec,
        output_json=args.output,
        log_file=args.log,
    )


if __name__ == "__main__":
    main()
