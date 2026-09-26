"""Fetch and format all training and evaluation logs for reproducible audit."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def process_train_log():
    raw_path = REPO / "kaggle_train_output" / "train_full_clean.log"
    if not raw_path.exists():
        print(f"Warning: {raw_path} not found")
        return

    text = raw_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    formatted_lines = [
        "=== AdaVCM Policy Network Training Log on COCO-2017 ===",
        "Platform: Kaggle Cloud GPU (NVIDIA Tesla T4 16GB)",
        "Account: wagur124705",
        "Kernel: adavcm-adaptive-vcm-train",
        "Dataset: COCO-2017 Train (10,000 real images)",
        "Annotations: instances_train2017.json",
        "Hyperparameters: batch_size=16, epochs=10, optimizer=AdamW, lr=1e-4",
        "Loss Objective: L = L_task + lambda * R_proxy",
        "Steps per Epoch: 625 batches (Total: 6,250 gradient steps)",
        "=" * 70,
        "",
    ]

    last_epoch_line = {}
    epoch_summaries = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if "[Epoch " in s and "Loss:" in s:
            epoch_summaries.append(s)
        elif "Epoch " in s and ("100%" in s or "625/625" in s):
            clean_s = re.sub(r"[\u2580-\u259f]", "=", s)
            m = re.search(r"Epoch\s+(\d+)", clean_s)
            if m:
                ep = int(m.group(1))
                last_epoch_line[ep] = clean_s

    for ep in range(1, 11):
        formatted_lines.append(f"--- Epoch {ep} / 10 ---")
        if ep in last_epoch_line:
            formatted_lines.append(f"Final batch: {last_epoch_line[ep]}")
        for summ in epoch_summaries:
            if f"[Epoch {ep}]" in summ:
                formatted_lines.append(f"Summary: {summ}")
                break
        formatted_lines.append("")

    formatted_lines.append("=" * 70)
    formatted_lines.append("=== Checkpoint Audit & Final Validation ===")

    train_out = REPO / "results_overnight" / "train" / "pre_v1" / "outputs" / "train"
    for ep in range(1, 11):
        pth = train_out / f"adavcm_epoch_{ep}.pth"
        if pth.exists():
            d = torch.load(pth, map_location="cpu")
            m = d.get("metrics", {})
            tl = m.get("train_loss", 0.0)
            fl = m.get("fg_loss", 0.0)
            rl = m.get("rate_loss", 0.0)
            formatted_lines.append(
                f"adavcm_epoch_{ep}.pth: train_loss={tl:.6f}, fg_diff={fl:.6f}, rate_proxy={rl:.6f}"
            )

    best_pth = train_out / "adavcm_best.pth"
    if best_pth.exists():
        d = torch.load(best_pth, map_location="cpu")
        m = d.get("metrics", {})
        tl = m.get("train_loss", 0.0)
        fl = m.get("fg_loss", 0.0)
        rl = m.get("rate_loss", 0.0)
        formatted_lines.append("")
        formatted_lines.append(f"Best Checkpoint: adavcm_best.pth (Epoch {d.get('epoch')})")
        formatted_lines.append(
            f"Final Best Metrics: train_loss={tl:.6f}, fg_diff={fl:.6f}, rate_proxy={rl:.6f}"
        )

    out_file = LOGS_DIR / "train_coco_10epochs.log"
    out_file.write_text("\n".join(formatted_lines), encoding="utf-8")
    print(f"[Done] Wrote {out_file}")


def fetch_detector_log():
    pool_file = Path(r"C:\Users\Wagur1\Downloads\pool.json")
    if not pool_file.exists():
        return
    with open(pool_file, "r") as f:
        pool = json.load(f)

    if "tranthihongdieu" not in pool:
        return

    os.environ["KAGGLE_API_TOKEN"] = pool["tranthihongdieu"]
    os.environ["PYTHONUTF8"] = "1"
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    logs_raw = api.kernels_logs("tranthihongdieu/adavcm-real-detector-benchmark")
    if not logs_raw:
        return

    entries = json.loads(logs_raw)
    clean_lines = []
    clean_lines.append("=== Real Downstream Detector Benchmark Log (COCO-2017 CTC) ===")
    clean_lines.append("Platform: Kaggle Cloud GPU (NVIDIA Tesla T4)")
    clean_lines.append("Account: tranthihongdieu")
    clean_lines.append("Kernel: adavcm-real-detector-benchmark")
    clean_lines.append("Detector: Torchvision SSDLite320-MobileNetV3 Large")
    clean_lines.append("=" * 70)

    for e in entries:
        d = e.get("data", "")
        # clean tqdm bar
        clean_d = re.sub(r"[\u2580-\u259f]", "=", d)
        clean_lines.append(clean_d)

    out_file = LOGS_DIR / "detector_map_benchmark.log"
    out_file.write_text("".join(clean_lines), encoding="utf-8")
    print(f"[Done] Wrote {out_file}")


if __name__ == "__main__":
    process_train_log()
    fetch_detector_log()
