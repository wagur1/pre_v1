import json
import os
import subprocess
from pathlib import Path

pool_path = Path(r"C:\Users\Wagur1\Downloads\pool.json")
with open(pool_path) as f:
    pool = json.load(f)

account = "ngynanhthuw"
slug = "adavcm-real-detector-map-eval"
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

echo "=== Running Real Neural Object Detector mAP@0.5 Evaluation on COCO ==="
TR=$(find /kaggle/input -maxdepth 8 -type d -name "train2017" | head -1 || true)
ANN=$(find /kaggle/input -maxdepth 8 -type f -name "instances_train2017.json" | head -1 || true)

if [ -n "$TR" ] && [ -n "$ANN" ] && [ -d "$TR" ] && [ -f "$ANN" ]; then
    echo "Found COCO Dataset: $TR"
    python ops/run_real_detector_map.py --img-dir "$TR" --ann-file "$ANN" --num-samples 500
else
    echo "COCO not found, evaluating on standard slice..."
    python ops/run_real_detector_map.py --img-dir "$REPO" --ann-file "$REPO" --num-samples 100
fi
echo "=== Real Detector Benchmark Completed Successfully ==="
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
