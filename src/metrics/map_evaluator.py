"""Standard Object Detection Evaluation Engine (PASCAL VOC / COCO Style).

Computes true mAP@0.5 and mAP@0.5:0.95 using 101-point interpolated precision-recall curves.
"""
from __future__ import annotations

import numpy as np
import torch


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between N boxes and M boxes in [x1, y1, x2, y2] format."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-8, None)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve using COCO 101-point interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point sampling
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_thresholds, side="left")
    return float(np.mean(mpre[inds]))


class DetectionEvaluator:
    """Accumulates ground truth and predictions across a dataset and evaluates true mAP."""

    def __init__(self, iou_thresholds: list[float] | None = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.images = []

    def add_image_eval(self, gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray):
        """Add predictions and ground truth for a single image.

        Args:
            gt_boxes: [N, 4] in [x1, y1, x2, y2].
            pred_boxes: [M, 4] in [x1, y1, x2, y2].
            pred_scores: [M] confidence scores.
        """
        self.images.append({
            "gt_boxes": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
            "pred_boxes": np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4),
            "pred_scores": np.asarray(pred_scores, dtype=np.float32).reshape(-1),
        })

    def evaluate(self) -> dict[str, float]:
        """Compute mAP@0.5 and mAP@0.5:0.95 across all accumulated images."""
        if not self.images:
            return {"map50": 0.0, "map50_95": 0.0}

        # Gather all detections sorted by confidence
        all_matches = {iou_th: [] for iou_th in self.iou_thresholds}
        all_scores = []
        num_gt_total = sum(len(img["gt_boxes"]) for img in self.images)

        if num_gt_total == 0:
            return {"map50": 0.0, "map50_95": 0.0}

        for img in self.images:
            gt_b = img["gt_boxes"]
            pr_b = img["pred_boxes"]
            pr_s = img["pred_scores"]

            if len(pr_b) == 0:
                continue

            order = np.argsort(-pr_s)
            pr_b = pr_b[order]
            pr_s = pr_s[order]

            ious = box_iou(pr_b, gt_b)  # [M, N]

            for iou_th in self.iou_thresholds:
                matched_gt = set()
                matches = []
                for p_idx in range(len(pr_b)):
                    best_gt = -1
                    best_iou = iou_th
                    for g_idx in range(len(gt_b)):
                        if g_idx not in matched_gt and ious[p_idx, g_idx] >= best_iou:
                            best_iou = ious[p_idx, g_idx]
                            best_gt = g_idx
                    if best_gt >= 0:
                        matched_gt.add(best_gt)
                        matches.append(1)  # True Positive
                    else:
                        matches.append(0)  # False Positive
                all_matches[iou_th].extend(matches)

            all_scores.extend(pr_s.tolist())

        # Sort all global detections by confidence score
        all_scores = np.array(all_scores)
        global_order = np.argsort(-all_scores)

        ap_per_iou = []
        for iou_th in self.iou_thresholds:
            matches = np.array(all_matches[iou_th])[global_order]
            tp = np.cumsum(matches == 1)
            fp = np.cumsum(matches == 0)
            recalls = tp / float(num_gt_total)
            precisions = tp / np.clip(tp + fp, 1e-8, None)
            ap = compute_ap(recalls, precisions)
            ap_per_iou.append(ap)

        map50 = ap_per_iou[0]  # First threshold is 0.5
        map50_95 = float(np.mean(ap_per_iou))
        return {
            "map50": float(map50),
            "map50_95": float(map50_95),
            "num_gt": int(num_gt_total),
            "num_preds": len(all_scores),
        }
