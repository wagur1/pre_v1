"""Generate Publication-Quality Architectural Pipeline Diagram for AdaVCM (Figure 1).

Renders a crisp, modern vector-style IEEE/ACM schematic of the end-to-end framework.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(r"C:\Users\Wagur1\.gemini\antigravity-cli\brain\ef4b8418-d710-4df1-a292-578fd27dcee4")


def draw_pipeline():
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, ax = plt.subplots(figsize=(18, 9), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Colors
    c_bg = "#f8f9fa"
    c_card_bg = "#ffffff"
    c_blue = "#1e40af"       # Primary
    c_blue_light = "#eff6ff"
    c_cyan = "#0284c7"
    c_emerald = "#059669"    # Policy / TBR
    c_emerald_light = "#ecfdf5"
    c_amber = "#d97706"      # Soft filter
    c_amber_light = "#fffbeb"
    c_purple = "#7c3aed"     # Codec
    c_purple_light = "#f5f3ff"
    c_rose = "#e11d48"       # Downstream tasks
    c_rose_light = "#fff1f2"
    c_border = "#cbd5e1"
    c_text_dark = "#0f172a"
    c_text_muted = "#475569"

    fig.patch.set_facecolor(c_bg)

    def draw_card(x, y, w, h, title, subtitle="", bg=c_card_bg, border=c_border, lw=1.5, radius=0.25):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={radius}",
                             facecolor=bg, edgecolor=border, linewidth=lw, zorder=2)
        ax.add_patch(box)
        if title:
            ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top",
                    fontsize=11.5, fontweight="bold", color=c_text_dark, zorder=3)
        if subtitle:
            ax.text(x + w / 2, y + h - 0.65, subtitle, ha="center", va="top",
                    fontsize=8.5, color=c_text_muted, zorder=3)

    def draw_arrow(x1, y1, x2, y2, label="", color="#475569", lw=2.0, style="->", rad=0.0):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                              color=color, linewidth=lw, connectionstyle=f"arc3,rad={rad}", zorder=4)
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.18
            ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=5)

    # ==================== 1. Big Container Boxes ====================
    # Proposed AdaVCM Preprocessor Container
    adavcm_box = FancyBboxPatch((3.4, 1.2), 8.8, 6.8, boxstyle="round,pad=0.0,rounding_size=0.4",
                                facecolor="#f1f5f9", edgecolor="#3b82f6", linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(adavcm_box)
    ax.text(3.6, 7.7, "PROPOSED ADAVCM FRAMEWORK (Edge / Lightweight Preprocessor)",
            fontsize=12, fontweight="bold", color="#1e3a8a", zorder=3)
    ax.text(3.6, 7.35, "Complexity: Only 4.9k Parameters (< 20 KB)  |  Throughput: 89 - 400+ FPS  |  Analyzer-Agnostic",
            fontsize=9.5, fontweight="semibold", color="#3b82f6", zorder=3)

    # ==================== 2. Nodes / Cards ====================
    # (A) Input Video Stream
    draw_card(0.4, 3.2, 2.4, 2.8, "Input Video", "Raw Stream I in R^{BxCxTxHxW}", bg=c_blue_light, border=c_blue)
    ax.text(1.6, 4.4, "[Video Sequence]\n+ Target QP\n(e.g., 27, 32, 38, 43)", ha="center", va="center",
            fontsize=9, color=c_text_dark, zorder=3)

    # (B) ST-SME Module
    draw_card(3.8, 4.6, 3.8, 2.4, "1. Spatio-Temporal Salience (ST-SME)",
              "Task Importance & Motion Estimation", bg=c_emerald_light, border=c_emerald)
    ax.text(5.7, 5.5, "• Task Prior Boxes / Gradient Maps\n• Motion Energy: Mt = ||It - It-1||\n• Continuous Sigmoid Expansion:\n   W ∈ [0, 1] (Boundary Ring)",
            ha="center", va="center", fontsize=8.5, color=c_text_dark, zorder=3)

    # (C) Adaptive Policy Network
    draw_card(3.8, 1.6, 3.8, 2.4, "2. Adaptive PolicyNet",
              "Lightweight Meta-Parameter Prediction", bg=c_card_bg, border="#6366f1")
    ax.text(5.7, 2.5, "• Param Count: 4,931 (~0.02 MB)\n• Inputs: Global Variance, Motion, QP\n• Optimal Lagrangian Meta-Outputs:\n   σ (spatial blur) & α (temporal reg)",
            ha="center", va="center", fontsize=8.5, color=c_text_dark, zorder=3)

    # (D) Temporal Background Regularizer (TBR)
    draw_card(8.0, 4.6, 3.8, 2.4, "3. Temporal Regularizer (TBR)",
              "Inter-Frame Background Stabilization", bg=c_emerald_light, border=c_emerald)
    ax.text(9.9, 5.5, "• State: Bt = (1 - α) Bt-1 + α It\n• Blended: It^{temp} = W ⊙ It + (1 - W) ⊙ Bt\n• Result: Null Motion Residuals in\n   Inter-Frame P/B-slices (-92.8% bits)",
            ha="center", va="center", fontsize=8.5, color=c_text_dark, zorder=3)

    # (E) Boundary-Aware Soft Transition Filter
    draw_card(8.0, 1.6, 3.8, 2.4, "4. Boundary-Aware Soft Filter",
              "Frequency Attenuation & Pass-Through", bg=c_amber_light, border=c_amber)
    ax.text(9.9, 2.5, "• Blurred Background: I^{blur} = Gσ * I^{temp}\n• Bit-Exact Pass: W = 1.0 ⟹ I^{out} ≡ I\n• Sigmoidal Ring: Eliminates DCT\n   Block Edge Penalties (No step artifacts)",
            ha="center", va="center", fontsize=8.5, color=c_text_dark, zorder=3)

    # (F) Standard Video Codec
    draw_card(12.7, 3.2, 2.2, 2.8, "Standard Codec", "Hardware / Off-the-Shelf", bg=c_purple_light, border=c_purple)
    ax.text(13.8, 4.4, "H.264 / AVC\nH.265 / HEVC\n(libx264 / libx265)\n\n⚡ -24.7% Bitrate\nSavings!", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#581c87", zorder=3)

    # (G) Downstream Vision Analyzers (Analyzer-Agnostic!)
    draw_card(15.4, 2.0, 2.3, 5.2, "Downstream Vision", "100% Analyzer-Agnostic", bg=c_rose_light, border=c_rose)
    ax.text(16.55, 5.8, "OBJECT DETECTION\n• YOLOv8 / YOLOv10\n• Faster R-CNN\n• Deformable DETR", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#9f1239", zorder=3)
    ax.text(16.55, 4.3, "ACTION RECOGNITION\n• SlowFast / SlowOnly\n• Video Swin / I3D", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#9f1239", zorder=3)
    ax.text(16.55, 2.9, "TRACKING / SURVEILLANCE\n• ByteTrack / KYS / DiMP\n\n[>98% Task Accuracy\nPreserved Bit-Exact!]", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#9f1239", zorder=3)

    # ==================== 3. Flow Arrows ====================
    # Input -> ST-SME
    draw_arrow(2.8, 4.9, 3.8, 5.6, "Raw Frames I", color=c_blue)
    # Input -> PolicyNet
    draw_arrow(2.8, 4.3, 3.8, 2.8, "Stats + QP", color=c_blue)
    # ST-SME -> PolicyNet (Weight map prior)
    draw_arrow(4.6, 4.6, 4.6, 4.0, "W", color=c_emerald)
    # PolicyNet -> TBR (alpha)
    draw_arrow(7.6, 3.2, 8.4, 4.6, "α (temporal)", color="#4f46e5")
    # PolicyNet -> Soft Filter (sigma)
    draw_arrow(7.6, 2.6, 8.0, 2.6, "σ (spatial)", color="#4f46e5")
    # ST-SME -> TBR (weight map W)
    draw_arrow(7.6, 5.8, 8.0, 5.8, "W map", color=c_emerald)
    # TBR -> Soft Filter (temporal frame I^{temp})
    draw_arrow(9.9, 4.6, 9.9, 4.0, "I^{temp}", color=c_emerald)
    # Soft Filter -> Standard Codec (Preprocessed stream)
    draw_arrow(11.8, 2.8, 12.7, 4.3, "I^{prep}", color=c_amber)
    # Standard Codec -> Downstream Tasks (Bitstream -> Decoded Frames)
    draw_arrow(14.9, 4.6, 15.4, 4.6, "Decoded I^", color=c_purple)

    plt.tight_layout()
    out_file = REPO_ROOT / "results" / "adavcm_pipeline_architecture.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[Pipeline] Diagram saved to {out_file}")

    if ARTIFACT_DIR.exists():
        shutil.copy(out_file, ARTIFACT_DIR / "adavcm_pipeline_architecture.png")


if __name__ == "__main__":
    draw_pipeline()
