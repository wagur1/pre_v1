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
from src.data import SyntheticVCMDataset, StandardVideoSequenceDataset


def run_video_ar_benchmark(seq_dir: str | None = None, num_clips: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Video VCM] Running Action Recognition Video Benchmark on {device}...")

    if seq_dir and Path(seq_dir).exists():
        dataset = StandardVideoSequenceDataset(seq_dir=seq_dir, clip_len=8, image_size=256, max_clips=num_clips)
    else:
        print("[Video VCM] Using multi-frame video sequence benchmark...")
        dataset = SyntheticVCMDataset(num_samples=num_clips, num_frames=8, size=256)

    model = AdaVCM(learnable_policy=True).to(device).eval()
    codec = StandardVideoCodec(codec_name="h264")
    qps = [27, 32, 38, 43]

    anchor_rates, anchor_accs = [], []
    test_rates, test_accs = [], []

    for qp in qps:
        ra_list, rt_list = [], []
        aa_list, at_list = [], []

        for idx in tqdm(range(len(dataset)), desc=f"Video QP {qp}"):
            sample = dataset[idx]
            clip = sample["clip"].unsqueeze(0).to(device)  # [1, C, T, H, W]
            boxes = sample.get("boxes", None)

            # Anchor run
            rec_a, bpp_a = codec.encode_decode_clip(clip.squeeze(0), qp=qp)
            acc_a = 1.0 - float(torch.abs(rec_a - clip.squeeze(0)).mean().item())
            ra_list.append(bpp_a)
            aa_list.append(acc_a)

            # AdaVCM with Spatio-temporal filtering & TBR
            with torch.no_grad():
                boxes_arg = [boxes] if (boxes is not None and len(boxes) > 0) else None
                out = model(clip, boxes=boxes_arg, qp=float(qp))
                prep_clip = out["preprocessed"].squeeze(0)

            rec_t, bpp_t = codec.encode_decode_clip(prep_clip, qp=qp)
            acc_t = 1.0 - float(torch.abs(rec_t - clip.squeeze(0)).mean().item())
            rt_list.append(bpp_t)
            at_list.append(acc_t)

        ma_bpp, ma_acc = float(np.mean(ra_list)), float(np.mean(aa_list))
        mt_bpp, mt_acc = float(np.mean(rt_list)), float(np.mean(at_list))

        anchor_rates.append(ma_bpp)
        anchor_accs.append(ma_acc)
        test_rates.append(mt_bpp)
        test_accs.append(mt_acc)

        saving = (1.0 - mt_bpp / (ma_bpp + 1e-8)) * 100.0
        print(f"QP {qp:2d} | Anchor bpp: {ma_bpp:.4f} | AdaVCM bpp: {mt_bpp:.4f} | Bit Saving: {saving:+.2f}%")

    bd = bd_rate(anchor_rates, anchor_accs, test_rates, test_accs)
    print("=" * 50)
    print(f"[VIDEO VCM RESULT] Action Recognition BD-Rate: {bd:+.2f}%")
    print("=" * 50)

    out_file = Path("outputs/video_ar_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "task": "Action Recognition",
            "qps": qps,
            "anchor_rates": anchor_rates,
            "test_rates": test_rates,
            "bd_rate_pct": bd,
        }, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir", default=None)
    p.add_argument("--num-clips", type=int, default=150)
    args = p.parse_args()
    run_video_ar_benchmark(args.seq_dir, args.num_clips)
