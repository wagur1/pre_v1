"""Video and Image Data Pipeline for VCM.

Supports:
- Standard COCO Detection images (train2017 / val2017).
- Video datasets (Kinetics / MOT / Custom MP4 clips).
- Synthetic test generator for quick unit tests & verification.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class SyntheticVCMDataset(Dataset):
    """Generates synthetic video clips with foreground moving shapes for testing."""

    def __init__(self, num_samples: int = 50, num_frames: int = 8, size: int = 256):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Background: noisy texture
        bg = torch.rand(3, self.num_frames, self.size, self.size) * 0.3
        # Foreground: moving bright square
        x_start = (idx * 7) % (self.size - 60)
        y_start = (idx * 11) % (self.size - 60)
        boxes = []
        for t in range(self.num_frames):
            xt = x_start + t * 2
            yt = y_start + t * 2
            bg[:, t, yt : yt + 40, xt : xt + 40] = 0.95
            boxes.append([xt, yt, xt + 40, yt + 40])

        return {
            "clip": bg.clamp(0.0, 1.0),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.ones(self.num_frames, dtype=torch.float32),
            "target_qp": 35.0,
            "id": f"synth_{idx}",
        }


class VideoTaskDataset(Dataset):
    """Loads COCO detection dataset or image folders."""

    def __init__(
        self,
        img_dir: str | Path,
        ann_file: str | Path | None = None,
        image_size: int = 320,
        max_samples: int | None = None,
    ):
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.items = []

        if ann_file and Path(ann_file).exists():
            with open(ann_file, "r") as f:
                coco = json.load(f)
            img_map = {img["id"]: img for img in coco.get("images", [])}
            ann_map: dict[int, list] = {}
            for ann in coco.get("annotations", []):
                ann_map.setdefault(ann["image_id"], []).append(ann["bbox"])
            for img_id, img_info in img_map.items():
                file_path = self.img_dir / img_info["file_name"]
                if file_path.exists():
                    self.items.append((file_path, ann_map.get(img_id, [])))
        elif self.img_dir.exists():
            for p in sorted(self.img_dir.glob("*.jpg")) + sorted(self.img_dir.glob("*.png")):
                self.items.append((p, []))

        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, raw_boxes = self.items[idx]
        with Image.open(img_path).convert("RGB") as img:
            w_orig, h_orig = img.size
            img_resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            tensor = TF.to_tensor(img_resized)  # [C, H, W] in [0, 1]

        # Rescale boxes to resized image coords
        sx = self.image_size / float(w_orig)
        sy = self.image_size / float(h_orig)
        boxes_xyxy = []
        for b in raw_boxes:
            x, y, w, h = b
            boxes_xyxy.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])

        # Expand to video format [C, T=1, H, W]
        clip = tensor.unsqueeze(1)
        return {
            "clip": clip,
            "boxes": boxes_xyxy,
            "scores": [1.0] * len(boxes_xyxy),
            "target_qp": 35.0,
            "id": img_path.stem,
        }
