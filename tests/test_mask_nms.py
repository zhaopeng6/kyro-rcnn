import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.evaluation.postprocessing import mask_aware_nms


def _output(masks, boxes):
    return {"masks": masks.unsqueeze(1).float(), "boxes": torch.tensor(boxes), "scores": torch.tensor([.9, .8]), "labels": torch.tensor([1, 1])}


def test_mask_nms_removes_duplicate_overlapping_mask():
    masks = torch.zeros((2, 8, 8)); masks[:, 1:5, 1:5] = 1
    filtered = mask_aware_nms(_output(masks, [[1., 1., 5., 5.], [1., 1., 5., 5.]]), iou_threshold=.5)
    assert len(filtered["scores"]) == 1 and abs(filtered["scores"].item() - .9) < 1e-6


def test_mask_nms_retains_distinct_masks_with_overlapping_boxes():
    masks = torch.zeros((2, 8, 8)); masks[0, 1:4, 1:4] = 1; masks[1, 4:7, 4:7] = 1
    filtered = mask_aware_nms(_output(masks, [[1., 1., 7., 7.], [1., 1., 7., 7.]]), iou_threshold=.5)
    assert len(filtered["scores"]) == 2


def test_mask_nms_preserves_overlapping_masks_when_boxes_are_not_near_identical():
    masks = torch.zeros((2, 10, 10)); masks[0, 1:7, 1:7] = 1; masks[1, 3:9, 3:9] = 1
    filtered = mask_aware_nms(_output(masks, [[1., 1., 7., 7.], [3., 3., 9., 9.]]), iou_threshold=.1, box_iou_threshold=.8)
    assert len(filtered["scores"]) == 2
