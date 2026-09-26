"""Generate Publication-Quality Rate-Accuracy / RD Curves for AdaVCM vs Anchor.

Saves figures for IEEE/ACM paper submission (PNG at 300 DPI and PDF).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO_ROOT / "results" / "benchmark_1000_results.json"
ARTIFACT_DIR = Path(r"C:\Users\Wagur1\.gemini\antigravity-cli\brain\ef4b8418-d710-4df1-a292-578fd27dcee4")


def generate_plots():
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Styling settings
    color_anchor = "#1f77b4"  # Classic Blue
    color_adavcm = "#d62728"  # IEEE Crimson Red
    qps = [27, 32, 38, 43]

    codecs = [("h264", "H.264 / AVC", axes[0]), ("h265", "H.265 / HEVC", axes[1])]

    for c_key, c_label, ax in codecs:
        c_data = data[c_key]
        ra = np.array(c_data["anchor_bpp"])
        aa = np.array(c_data["anchor_acc"]) * 100.0
        rt = np.array(c_data["adavcm_bpp"])
        at = np.array(c_data["adavcm_acc"]) * 100.0

        # Plot curves
        ax.plot(ra, aa, "o-", color=color_anchor, label=f"Anchor ({c_label})", linewidth=2.5, markersize=8)
        ax.plot(rt, at, "s--", color=color_adavcm, label=f"Proposed AdaVCM", linewidth=2.5, markersize=8)

        # Annotate QPs and bitrate savings
        for i, qp in enumerate(qps):
            saving = (1.0 - rt[i] / ra[i]) * 100.0
            ax.annotate(f"QP {qp}", (ra[i], aa[i]), textcoords="offset points", xytext=(-15, 8),
                        fontsize=9, fontweight="bold", color=color_anchor)
            ax.annotate(f"QP {qp}\n(-{saving:.1f}%)", (rt[i], at[i]), textcoords="offset points", xytext=(8, -18),
                        fontsize=9, fontweight="bold", color=color_adavcm)

            # Draw dashed horizontal line connecting same QP to show bit savings
            ax.annotate("", xy=(rt[i], at[i]), xytext=(ra[i], at[i]),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, ls=":"))

        ax.set_title(f"Rate-Accuracy Trade-off: {c_label}\n(COCO-2017 CTC Benchmark, 1000 Images)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Bitrate [bpp] (Lower is better)", fontsize=12, fontweight="semibold")
        ax.set_ylabel("Task Accuracy Proxy [%] (Higher is better)", fontsize=12, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="lower right")

        # Set clean limits
        ax.set_ylim(93.0, 99.0)

    plt.tight_layout()

    out_png = REPO_ROOT / "results" / "rd_curve_comparison.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved to {out_png}")

    # Copy to artifact directory for rendering in Antigravity Chat UI
    if ARTIFACT_DIR.exists():
        art_png = ARTIFACT_DIR / "rd_curve_comparison.png"
        shutil.copy(out_png, art_png)
        print(f"[Plot] Copied to artifact dir: {art_png}")


if __name__ == "__main__":
    generate_plots()
