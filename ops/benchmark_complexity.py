"""Benchmark Computational Complexity, Parameter Count, Latency, and FPS of AdaVCM.

Generates the Complexity Table for Section IV (Edge Deployment & Real-Time Feasibility).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import AdaVCM


def benchmark_complexity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Complexity Benchmark] Running on {device}...")

    model = AdaVCM(learnable_policy=True).to(device).eval()

    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    policy_params = sum(p.numel() for p in model.policy_net.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"PolicyNet Parameters: {policy_params:,} ({policy_params / 1e3:.2f} kParams)")
    print(f"Model Checkpoint Size: {policy_params * 4 / (1024 * 1024):.3f} MB (FP32)")

    # 2. Latency & FPS across Resolutions
    resolutions = [
        ("WQVGA (256x256)", 256, 256),
        ("WVGA (480x320)", 320, 480),
        ("HD (720p - 1280x720)", 720, 1280),
        ("FHD (1080p - 1920x1080)", 1080, 1920),
    ]

    results = {
        "device": str(device),
        "total_params": total_params,
        "policy_params": policy_params,
        "memory_mb": round(policy_params * 4 / (1024 * 1024), 4),
        "benchmarks": [],
    }

    warmup_iters = 5
    num_iters = 20

    print("\n" + "=" * 65)
    print(f"{'Resolution':<25} | {'Latency (ms)':<15} | {'FPS':<10} | {'Status'}")
    print("=" * 65)

    for name, h, w in resolutions:
        dummy_clip = torch.randn(1, 3, 4, h, w, device=device)

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = model(dummy_clip, qp=32.0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_clip, qp=32.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)  # ms

        avg_latency = float(np.mean(times))
        # 4 frames per clip -> latency per frame = avg_latency / 4
        per_frame_latency = avg_latency / 4.0
        fps = 1000.0 / per_frame_latency
        realtime_status = "Real-Time (>30 FPS)" if fps >= 30.0 else "Near Real-Time"

        print(f"{name:<25} | {per_frame_latency:8.2f} ms/frame | {fps:7.1f}   | {realtime_status}")

        results["benchmarks"].append({
            "resolution": name,
            "height": h,
            "width": w,
            "latency_ms_per_frame": round(per_frame_latency, 2),
            "fps": round(fps, 1),
            "realtime": fps >= 30.0,
        })

    print("=" * 65)

    out_file = REPO_ROOT / "results" / "complexity_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Complexity results saved to {out_file}")
    return results


if __name__ == "__main__":
    benchmark_complexity()
