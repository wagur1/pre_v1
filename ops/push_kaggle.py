"""Push AdaVCM training kernel to Kaggle using the token pool."""
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

BASH_SCRIPT = r'''%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== [1/4] Setting up working directory ==="
cd /kaggle/working
REPO=/kaggle/working/pre_v1

if [ -d "$REPO/.git" ]; then
    echo "Updating existing repository..."
    git -C "$REPO" pull origin main
else
    echo "Cloning AdaVCM repository..."
    git clone https://github.com/wagur1/pre_v1.git "$REPO"
fi

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "=== [2/4] Installing dependencies ==="
pip install -q pyyaml tqdm pycocotools

echo "=== [3/4] Running AdaVCM Real Data Training ==="
TR=$(find /kaggle/input -maxdepth 8 -type d -name "train2017" | head -1 || true)
ANN=$(find /kaggle/input -maxdepth 8 -type f -name "instances_train2017.json" | head -1 || true)

if [ -n "$TR" ] && [ -n "$ANN" ] && [ -d "$TR" ] && [ -f "$ANN" ]; then
    echo "Found COCO Dataset:"
    echo "  Images: $TR"
    echo "  Annotations: $ANN"
    echo "Training AdaVCM on 10,000 real images for 10 epochs..."
    python train.py --img-dir "$TR" --ann-file "$ANN" --max-samples 10000 --batch-size 16 --epochs 10 --device cuda
else
    echo "Running benchmark video training on synthetic spatiotemporal dataset..."
    python train.py --synthetic --epochs 16 --device cuda
fi

echo "=== [4/4] Evaluating Rate-Accuracy and BD-Rate across QPs [27, 32, 38, 43] ==="
python evaluate.py --codec h264 --qps 27,32,38,43 --synthetic
echo "=== Running Ablation Benchmark with Seed 42 ==="
python ops/run_ablation.py --synthetic --num-samples 50 --seed 42

echo "=== ALL TASKS COMPLETED SUCCESSFULLY ==="
'''


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


def main():
    p = argparse.ArgumentParser(description="Push AdaVCM to Kaggle")
    p.add_argument("--pool", default=str(DEFAULT_POOL), help="Path to pool.json")
    p.add_argument("--account", default="wagur124705", help="Kaggle username from pool")
    p.add_argument("--slug", default="adavcm-adaptive-vcm-train", help="Kernel slug")
    p.add_argument("--accelerator", default="NvidiaTeslaT4", help="GPU accelerator")
    args = p.parse_args()

    pool_path = Path(args.pool)
    if not pool_path.exists():
        sys.exit(f"Error: pool file not found at {pool_path}")

    with open(pool_path, "r") as f:
        pool = json.load(f)

    if args.account not in pool:
        sys.exit(f"Error: Account {args.account} not found in pool ({list(pool.keys())})")

    token = pool[args.account]
    os.environ["KAGGLE_API_TOKEN"] = token

    # Check CLI
    kaggle_bin = "kaggle"
    if DEFAULT_CLI.exists():
        kaggle_bin = str(DEFAULT_CLI)
    elif shutil.which("kaggle") is None:
        sys.exit("Error: kaggle executable not found")

    push_dir = REPO / "ops" / "_push" / args.slug
    push_dir.mkdir(parents=True, exist_ok=True)

    nb = make_notebook(BASH_SCRIPT)
    (push_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=2))

    meta = {
        "id": f"{args.account}/{args.slug}",
        "title": args.slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["awsaf49/coco-2017-dataset"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    cmd = [kaggle_bin, "kernels", "push", "-p", str(push_dir)]
    print(f"[Push] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"STDERR: {res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)

    print(f"[Success] Kernel pushed successfully to: https://www.kaggle.com/code/{args.account}/{args.slug}")


if __name__ == "__main__":
    main()
