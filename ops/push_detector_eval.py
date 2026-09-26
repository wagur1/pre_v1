import json
import os
import subprocess
from pathlib import Path

pool_path = Path(r"C:\Users\Wagur1\Downloads\pool.json")
with open(pool_path) as f:
    pool = json.load(f)

account = "tranthihongdieu"
slug = "adavcm-real-detector-benchmark"
token = pool[account]
os.environ["KAGGLE_API_TOKEN"] = token

cli = Path(r"C:\Users\Wagur1\AppData\Local\Python\pythoncore-3.14-64\Scripts\kaggle.exe")
push_dir = Path(r"C:\Users\Wagur1\pre_v1\ops\_push") / slug
push_dir.mkdir(parents=True, exist_ok=True)

script = """%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /kaggle/working
REPO=/kaggle/working/pre_v1
rm -rf "$REPO"
git clone https://github.com/wagur1/pre_v1.git "$REPO"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
pip install -q pyyaml tqdm pycocotools

echo "=== System & GPU Environment ==="
nvidia-smi

echo "=== Locating COCO Dataset in /kaggle/input ==="
VAL_DIR=$(find /kaggle/input -maxdepth 6 -type d -name "val2017" | head -1 || true)
VAL_ANN=$(find /kaggle/input -maxdepth 6 -type f -name "instances_val2017.json" | head -1 || true)
TRAIN_DIR=$(find /kaggle/input -maxdepth 6 -type d -name "train2017" | head -1 || true)
TRAIN_ANN=$(find /kaggle/input -maxdepth 6 -type f -name "instances_train2017.json" | head -1 || true)

IMG_DIR=""
ANN_FILE=""
if [ -n "$VAL_DIR" ] && [ -n "$VAL_ANN" ] && [ -d "$VAL_DIR" ] && [ -f "$VAL_ANN" ]; then
    echo "Found COCO val2017: $VAL_DIR"
    IMG_DIR="$VAL_DIR"
    ANN_FILE="$VAL_ANN"
elif [ -n "$TRAIN_DIR" ] && [ -n "$TRAIN_ANN" ] && [ -d "$TRAIN_DIR" ] && [ -f "$TRAIN_ANN" ]; then
    echo "Found COCO train2017: $TRAIN_DIR"
    IMG_DIR="$TRAIN_DIR"
    ANN_FILE="$TRAIN_ANN"
else
    echo "COCO not found in expected paths, searching /kaggle/input..."
    find /kaggle/input -maxdepth 4 || true
fi

if [ -n "$IMG_DIR" ] && [ -n "$ANN_FILE" ]; then
    echo "=== Running Real Neural Object Detector (SSDLite MobileNetV3) Evaluation on COCO ==="
    python ops/run_real_detector_map.py --img-dir "$IMG_DIR" --ann-file "$ANN_FILE" --num-samples 500
fi

echo "=== Running GPU Complexity & FPS Benchmark on Tesla T4 ==="
python ops/benchmark_complexity.py

echo "=== Saving Results to Working Dir for Direct Download ==="
cp -r results /kaggle/working/
ls -la /kaggle/working/results
echo "=== All Benchmarks Completed Successfully ==="
"""

nb = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": script.splitlines(keepends=True)
    }],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}
(push_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=2))

meta = {
    "id": f"{account}/{slug}",
    "title": slug,
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["awsaf49/coco-2017-dataset"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}
(push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

cmd = [str(cli), "kernels", "push", "-p", str(push_dir)]
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
if res.returncode == 0:
    print(f"[Success] Real Detector Benchmark launched: https://www.kaggle.com/code/{account}/{slug}")
