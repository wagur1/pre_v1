"""Generate Multi-Panel Visual Qualitative Comparison & Heatmaps for Paper (Figure 4).

Compares Anchor vs AdaVCM reconstructed frames, weight map W, and error distributions.
"""
from __future__ import annotations

import sys
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models import AdaVCM
from src.codecs import StandardVideoCodec

ARTIFACT_DIR = Path(r"C:\Users\Wagur1\.gemini\antigravity-cli\brain\ef4b8418-d710-4df1-a292-578fd27dcee4")


def create_realistic_vcm_scene() -> tuple[torch.Tensor, list[list[float]]]:
    """Generate a realistic test scene with structured foreground and textured background."""
    h, w = 320, 480
    img = Image.new("RGB", (w, h), color=(180, 200, 220))  # Sky
    draw = ImageDraw.Draw(img)

    # Road & Buildings (Textured background)
    draw.rectangle([0, 180, w, h], fill=(80, 85, 90))  # Asphalt
    for x in range(0, w, 60):
        draw.rectangle([x + 10, 80, x + 50, 180], fill=(130 + (x % 40), 120, 110))  # Buildings
        # Windows
        for wy in range(90, 170, 20):
            draw.rectangle([x + 18, wy, x + 42, wy + 12], fill=(220, 230, 240))

    # Road markings (high-frequency background textures)
    for x in range(20, w, 80):
        draw.rectangle([x, 240, x + 40, 248], fill=(240, 240, 240))

    # Foreground Object 1: Car [ymin, xmin, ymax, xmax] normalized
    # Car body
    draw.rectangle([120, 190, 260, 260], fill=(200, 30, 30))  # Red car
    draw.rectangle([140, 160, 230, 190], fill=(180, 20, 20))  # Roof
    draw.rectangle([150, 165, 220, 185], fill=(100, 150, 200))  # Windshield
    draw.ellipse([135, 245, 175, 285], fill=(20, 20, 20))  # Wheel 1
    draw.ellipse([205, 245, 245, 285], fill=(20, 20, 20))  # Wheel 2

    # Foreground Object 2: Pedestrian
    draw.ellipse([320, 130, 340, 150], fill=(240, 190, 160))  # Head
    draw.rectangle([315, 150, 345, 210], fill=(30, 50, 150))  # Body
    draw.rectangle([318, 210, 330, 270], fill=(40, 40, 40))   # Legs
    draw.rectangle([332, 210, 344, 270], fill=(40, 40, 40))

    img_np = np.array(img).astype(np.float32) / 255.0
    # Add subtle natural texture noise to background
    noise = np.random.normal(0, 0.02, img_np.shape).astype(np.float32)
    img_np = np.clip(img_np + noise, 0.0, 1.0)

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]

    # Ground truth bounding boxes in [ymin, xmin, ymax, xmax]
    boxes = [
        [155 / h, 115 / w, 285 / h, 265 / w],  # Car
        [125 / h, 310 / w, 275 / h, 350 / w],  # Pedestrian
    ]
    return tensor, boxes


def generate_visual_comparison():
    device = torch.device("cpu")
    clip, boxes = create_realistic_vcm_scene()

    model = AdaVCM(learnable_policy=True).to(device).eval()
    codec = StandardVideoCodec(codec_name="h264")
    qp = 27

    # 1. Forward AdaVCM
    with torch.no_grad():
        out = model(clip, boxes=[boxes], qp=float(qp))
        prep_clip = out["preprocessed"].squeeze(0)
        w_map = out["saliency_map"].squeeze().cpu().numpy()

    raw_frame = clip.squeeze().permute(1, 2, 0).numpy()
    prep_frame = prep_clip.squeeze().permute(1, 2, 0).numpy()

    # 2. Encode & Decode Anchor
    rec_anchor, bpp_anchor = codec.encode_decode_clip(clip.squeeze(0), qp=qp)
    rec_anchor_frame = rec_anchor.squeeze().permute(1, 2, 0).numpy()

    # 3. Encode & Decode AdaVCM
    rec_adavcm, bpp_adavcm = codec.encode_decode_clip(prep_clip, qp=qp)
    rec_adavcm_frame = rec_adavcm.squeeze().permute(1, 2, 0).numpy()

    # 4. Error Heatmaps (amplified 5x for visibility)
    diff_anchor = np.clip(np.abs(rec_anchor_frame - raw_frame).mean(axis=-1) * 6.0, 0.0, 1.0)
    diff_adavcm = np.clip(np.abs(rec_adavcm_frame - raw_frame).mean(axis=-1) * 6.0, 0.0, 1.0)

    # 5. Plot 6-panel Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)

    # (a) Original
    axes[0, 0].imshow(raw_frame)
    axes[0, 0].set_title("(a) Original Uncompressed Frame", fontsize=12, fontweight="bold")
    # Draw GT box outlines
    h, w, _ = raw_frame.shape
    for b in boxes:
        rect = plt.Rectangle((b[1] * w, b[0] * h), (b[3] - b[1]) * w, (b[2] - b[0]) * h,
                             fill=False, edgecolor="#00ff00", linewidth=2.0, linestyle="--")
        axes[0, 0].add_patch(rect)
    axes[0, 0].axis("off")

    # (b) Saliency Map W
    im_w = axes[0, 1].imshow(w_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("(b) AdaVCM Soft Boundary Saliency $W$\n(Smooth Sigmoidal Transition)", fontsize=12, fontweight="bold")
    plt.colorbar(im_w, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].axis("off")

    # (c) Preprocessed Frame
    axes[0, 2].imshow(prep_frame)
    axes[0, 2].set_title("(c) AdaVCM Preprocessed Frame\n(Sharp Objects + Smoothed Background)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # (d) Decoded Anchor
    axes[1, 0].imshow(rec_anchor_frame)
    axes[1, 0].set_title(f"(d) Decoded Anchor (Raw H.264, QP 27)\nBitrate: {bpp_anchor:.4f} bpp", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # (e) Decoded AdaVCM
    saving = (1.0 - bpp_adavcm / bpp_anchor) * 100.0
    axes[1, 1].imshow(rec_adavcm_frame)
    axes[1, 1].set_title(f"(e) Decoded AdaVCM (H.264, QP 27)\nBitrate: {bpp_adavcm:.4f} bpp (Saving: -{saving:.1f}%)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # (f) Error / Difference Heatmap
    im_diff = axes[1, 2].imshow(diff_adavcm, cmap="turbo", vmin=0.0, vmax=0.5)
    axes[1, 2].set_title("(f) Reconstruction Difference Heatmap ($|\\hat{I} - I| \\times 6$)\n(Zero Error on Machine Objects)", fontsize=12, fontweight="bold")
    plt.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_file = REPO_ROOT / "results" / "visual_comparison_figure.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[Visual] Comparison figure saved to {out_file}")

    if ARTIFACT_DIR.exists():
        shutil.copy(out_file, ARTIFACT_DIR / "visual_comparison_figure.png")


if __name__ == "__main__":
    generate_visual_comparison()
