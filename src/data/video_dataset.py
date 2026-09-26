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

    def __init__(self, num_samples: int = 50, num_frames: int = 8, size: int = 256, seed: int | None = 42):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Background: noisy texture (seeded for bit-exact reproducibility)
        if self.seed is not None:
            gen = torch.Generator().manual_seed(self.seed + idx)
            bg = torch.rand(3, self.num_frames, self.size, self.size, generator=gen) * 0.3
        else:
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


def build_standard_coco_slice(
    images_dir: str | Path,
    ann_file: str | Path,
    out_json: str | Path,
    n_train: int = 5000,
    n_val: int = 500,
    seed: int = 42,
) -> dict:
    """Deterministically slice standard COCO subsets for reproducible VCM research."""
    images_dir, ann_file = Path(images_dir), Path(ann_file)
    with open(ann_file, "r") as f:
        ann = json.load(f)

    # Group annotations by image
    by_img: dict[int, list] = {}
    for a in ann.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a["bbox"])

    # Keep only images with at least 1 object and file exists
    valid_images = []
    for img in ann.get("images", []):
        img_id = img["id"]
        if img_id in by_img and len(by_img[img_id]) > 0:
            if (images_dir / img["file_name"]).exists():
                valid_images.append(img)

    import random
    rng = random.Random(seed)
    rng.shuffle(valid_images)

    def _format_entry(im):
        return {
            "id": im["id"],
            "file_name": im["file_name"],
            "path": str(images_dir / im["file_name"]),
            "boxes": by_img[im["id"]],
            "width": im.get("width", 0),
            "height": im.get("height", 0),
        }

    slice_data = {
        "metadata": {
            "dataset": "MS-COCO 2017",
            "seed": seed,
            "n_train": min(n_train, len(valid_images)),
            "n_val": min(n_val, max(0, len(valid_images) - n_train)),
        },
        "train": [_format_entry(im) for im in valid_images[:n_train]],
        "val": [_format_entry(im) for im in valid_images[n_train : n_train + n_val]],
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(slice_data, f, indent=2)

    print(f"[Dataset] Standard slice written: {len(slice_data['train'])} train, {len(slice_data['val'])} val to {out_path}")
    return slice_data


class StandardVideoSequenceDataset(Dataset):
    """Loads continuous video frame sequences (Kinetics / MOT / SFU-HW-Objects)."""

    def __init__(
        self,
        seq_dir: str | Path,
        clip_len: int = 8,
        image_size: int = 256,
        max_clips: int | None = None,
    ):
        self.seq_dir = Path(seq_dir)
        self.clip_len = clip_len
        self.image_size = image_size
        self.clips = []

        if self.seq_dir.exists():
            # Check for sequence subdirectories
            subdirs = [d for d in self.seq_dir.iterdir() if d.is_dir()]
            if not subdirs:
                subdirs = [self.seq_dir]

            for sdir in sorted(subdirs):
                frame_files = sorted(sdir.glob("*.jpg")) + sorted(sdir.glob("*.png"))
                if len(frame_files) >= clip_len:
                    # Stride by clip_len // 2 for continuous coverage
                    stride = max(1, clip_len // 2)
                    for start in range(0, len(frame_files) - clip_len + 1, stride):
                        self.clips.append(frame_files[start : start + clip_len])

        if max_clips is not None:
            self.clips = self.clips[:max_clips]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        frame_paths = self.clips[idx]
        tensors = []
        for p in frame_paths:
            with Image.open(p).convert("RGB") as img:
                resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                tensors.append(TF.to_tensor(resized))
        # Stack to [C, T, H, W]
        clip = torch.stack(tensors, dim=1)
        return {
            "clip": clip,
            "boxes": [],
            "scores": [],
            "target_qp": 35.0,
            "id": f"{frame_paths[0].parent.name}_{frame_paths[0].stem}",
        }
