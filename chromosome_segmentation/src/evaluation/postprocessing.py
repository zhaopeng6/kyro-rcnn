"""Inference-only instance-mask postprocessing."""
from __future__ import annotations

import torch

from .metrics import mask_iou


def _boxes_intersect(a, b):
    return min(float(a[2]), float(b[2])) > max(float(a[0]), float(b[0])) and min(float(a[3]), float(b[3])) > max(float(a[1]), float(b[1]))


def _box_iou(a, b):
    intersection = max(0., min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(0., min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    union = (float(a[2] - a[0]) * float(a[3] - a[1])) + (float(b[2] - b[0]) * float(b[3] - b[1])) - intersection
    return intersection / union if union else 0.


def mask_aware_nms(output: dict, iou_threshold=.85, mask_threshold=.5, box_iou_threshold=.8, class_aware=True):
    """Greedily suppress lower-score duplicate instances by predicted mask IoU.

    Box intersection gates costly pixel IoU. Distinct masks with overlapping
    boxes are retained, unlike ordinary box NMS.
    """
    count = len(output.get("scores", []))
    if count < 2:
        return output
    masks = output["masks"][:, 0] >= mask_threshold
    boxes, scores, labels = output["boxes"], output["scores"], output["labels"]
    kept = []
    for candidate in scores.argsort(descending=True).tolist():
        duplicate = False
        for accepted in kept:
            if class_aware and labels[candidate] != labels[accepted]:
                continue
            if _boxes_intersect(boxes[candidate], boxes[accepted]) and _box_iou(boxes[candidate], boxes[accepted]) >= box_iou_threshold and mask_iou(masks[candidate], masks[accepted]) > iou_threshold:
                duplicate = True; break
        if not duplicate:
            kept.append(candidate)
    indices = torch.tensor(kept, dtype=torch.long, device=scores.device)
    return {key: (value[indices] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count else value) for key, value in output.items()}


def build_postprocessor(config: dict):
    settings = config.get("postprocessing", {}).get("mask_nms", {})
    if not settings.get("enabled", False):
        return None
    return lambda output: mask_aware_nms(output, settings.get("iou_threshold", .85), settings.get("mask_threshold", .5), settings.get("box_iou_threshold", .8), settings.get("class_aware", True))
