import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.transforms import RandomTranslation, RandomUniformScale
from src.evaluation.scale import disable_model_internal_resize
from src.models.mask_rcnn import build_mask_rcnn


def _sample():
    image = torch.zeros(3, 10, 20)
    masks = torch.zeros((1, 10, 20), dtype=torch.uint8); masks[:, 2:6, 4:10] = 1
    return image, {"boxes": torch.tensor([[4., 2., 10., 6.]]), "labels": torch.tensor([1]), "masks": masks,
                   "area": torch.tensor([24.]), "iscrowd": torch.tensor([0]), "image_id": torch.tensor([0]), "supervision_type": "polygon"}


def _assert_boxes_match_masks(target):
    for box, mask in zip(target["boxes"], target["masks"]):
        ys, xs = torch.where(mask.bool())
        assert box.tolist() == [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def test_uniform_scale_keeps_canvas_and_rebuilds_targets():
    image, target = _sample()
    with patch("src.datasets.transforms.random.uniform", return_value=.5):
        output, transformed = RandomUniformScale(.5, .5)(image, target)
    assert output.shape == image.shape and transformed["masks"].shape[-2:] == image.shape[-2:]
    assert len(transformed["boxes"]) == 1
    _assert_boxes_match_masks(transformed)


def test_translation_rebuilds_clipped_mask_boxes():
    image, target = _sample()
    with patch("src.datasets.transforms.random.uniform", side_effect=[.2, .2]):
        _, transformed = RandomTranslation(.2, 1.0)(image, target)
    assert len(transformed["boxes"]) == 1
    _assert_boxes_match_masks(transformed)


def test_box_only_scale_and_translation_keep_valid_detection_targets():
    image = torch.zeros(3, 10, 20)
    target = {"boxes": torch.tensor([[4., 2., 10., 6.]]), "labels": torch.tensor([1]), "area": torch.tensor([24.]),
              "iscrowd": torch.tensor([0]), "image_id": torch.tensor([99]), "supervision_type": "box"}
    with patch("src.datasets.transforms.random.uniform", return_value=.5):
        image, target = RandomUniformScale(.5, .5)(image, target)
    with patch("src.datasets.transforms.random.uniform", side_effect=[.2, .2]):
        _, target = RandomTranslation(.2, 1.0)(image, target)
    assert "masks" not in target and target["image_id"].item() == 99
    assert target["boxes"].shape == (1, 4) and (target["boxes"][:, 2:] > target["boxes"][:, :2]).all()


def test_no_resize_benchmark_transform_preserves_requested_shape():
    model = build_mask_rcnn({"model": {"num_classes": 2, "min_size": 64, "max_size": 64,
                                         "anchor_sizes": [[8], [16], [32], [64], [128]]}})
    disable_model_internal_resize(model)
    image = torch.rand(3, 43, 61)
    resized, _ = model.transform.resize(image)
    assert tuple(resized.shape[-2:]) == (43, 61)
