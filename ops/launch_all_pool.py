"""Master Launcher: Spreads experimental workloads across the Kaggle pool."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POOL = Path(r"C:\Users\Wagur1\Downloads\pool.json")
DEFAULT_CLI = Path(r"C:\Users\Wagur1\AppData\Local\Python\pythoncore-3.14-64\Scripts\kaggle.exe")


def make_notebook(bash_src: str) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": bash_src.splitlines(keepends=True),
    }
    return {
        "cells": [cell],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


EXPERIMENTS = [
    {
        "account": "dngbolm",
        "slug": "adavcm-video-action-benchmark",
        "title": "AdaVCM Video Action Recognition Benchmark",
        "datasets": ["rohanmallick/kinetics-train-5per"],
        "script": r'''%%bash
set -euo pipefail
cd /kaggle/working
REPO=/kaggle/working/pre_v1
rm -rf "$REPO"
git clone https://github.com/wagur1/pre_v1.git "$REPO"
cd "$REPO"
pip install -q pyyaml tqdm

echo "=== Running Action Recognition & Spatiotemporal Video Benchmark ==="
python ops/run_ar_train.py --num-clips 200
echo "=== COMPLETED ==="
'''
    },
    {
        "account": "hieusunday0412",
        "slug": "adavcm-od-eval-1000-images",
        "title": "AdaVCM 1000 Image Detection Benchmark",
        "datasets": ["awsaf49/coco-2017-dataset"],
        "script": r'''%%bash
set -euo pipefail
cd /kaggle/working
REPO=/kaggle/working/pre_v1
rm -rf "$REPO"
git clone https://github.com/wagur1/pre_v1.git "$REPO"
cd "$REPO"
pip install -q pyyaml tqdm pycocotools

TR=$(find /kaggle/input -maxdepth 8 -type d -name "train2017" | head -1 || true)
ANN=$(find /kaggle/input -maxdepth 8 -type f -name "instances_train2017.json" | head -1 || true)

echo "=== Running 1000-Image Full Detection Benchmark across H.264 and H.265 ==="
if [ -n "$TR" ] && [ -n "$ANN" ] && [ -d "$TR" ] && [ -f "$ANN" ]; then
    python ops/run_large_eval.py --img-dir "$TR" --ann-file "$ANN" --num-samples 1000
else
    echo "Fallback to synthetic evaluation..."
    python evaluate.py --codec h264 --qps 27,32,38,43 --synthetic
fi
echo "=== COMPLETED ==="
'''
    },
    {
        "account": "vtk269",
        "slug": "adavcm-ablation-suite",
        "title": "AdaVCM Ablation Study Suite",
        "datasets": ["awsaf49/coco-2017-dataset"],
        "script": r'''%%bash
set -euo pipefail
cd /kaggle/working
REPO=/kaggle/working/pre_v1
rm -rf "$REPO"
git clone https://github.com/wagur1/pre_v1.git "$REPO"
cd "$REPO"
pip install -q pyyaml tqdm pycocotools

TR=$(find /kaggle/input -maxdepth 8 -type d -name "train2017" | head -1 || true)
ANN=$(find /kaggle/input -maxdepth 8 -type f -name "instances_train2017.json" | head -1 || true)

echo "=== Running Ablation Study (Full vs No-TBR vs Hard-Mask vs Fixed) ==="
if [ -n "$TR" ] && [ -n "$ANN" ] && [ -d "$TR" ] && [ -f "$ANN" ]; then
    python ops/run_ablation.py --img-dir "$TR" --ann-file "$ANN" --num-samples 300
else
    echo "Fallback..."
    python evaluate.py --codec h264 --qps 27,32,38,43 --synthetic
fi
echo "=== COMPLETED ==="
'''
    },
]


def launch_all(pool_path: Path):
    with open(pool_path, "r") as f:
        pool = json.load(f)

    kaggle_bin = str(DEFAULT_CLI) if DEFAULT_CLI.exists() else "kaggle"

    for exp in EXPERIMENTS:
        acct = exp["account"]
        slug = exp["slug"]
        if acct not in pool:
            print(f"[Skip] Account {acct} not found in pool")
            continue

        token = pool[acct]
        os.environ["KAGGLE_API_TOKEN"] = token

        push_dir = REPO / "ops" / "_push" / slug
        push_dir.mkdir(parents=True, exist_ok=True)

        nb = make_notebook(exp["script"])
        (push_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=2))

        meta = {
            "id": f"{acct}/{slug}",
            "title": slug,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": exp["datasets"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        cmd = [kaggle_bin, "kernels", "push", "-p", str(push_dir)]
        print(f"\n[Pushing] {acct}/{slug}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout.strip())
        if res.returncode != 0:
            print(f"Error pushing {slug}: {res.stderr}", file=sys.stderr)
        else:
            print(f"-> Launched: https://www.kaggle.com/code/{acct}/{slug}")


if __name__ == "__main__":
    launch_all(DEFAULT_POOL)
