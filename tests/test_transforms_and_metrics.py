import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.transforms import RandomHorizontalFlip
from src.evaluation.metrics import match_masks
from src.evaluation.metrics import evaluate_polygon_dataset


def test_horizontal_flip_updates_box_and_mask():
    image = Image.new("RGB", (10, 8))
    target = {"boxes": torch.tensor([[1., 2., 4., 6.]]), "masks": torch.zeros((1, 8, 10), dtype=torch.uint8)}
    target["masks"][0, 2:6, 1:4] = 1
    _, result = RandomHorizontalFlip(1.0)(image, target)
    assert result["boxes"].tolist() == [[6., 2., 9., 6.]]
    assert result["masks"][0, 2:6, 6:9].all()


def test_matching_is_one_to_one():
    truth = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.bool)
    predictions = torch.tensor([[[1, 1], [0, 0]], [[1, 1], [0, 0]]], dtype=torch.bool)
    assert len(match_masks(predictions, truth, 0.5)) == 1


def test_polygon_evaluation_perfect_prediction():
    class Model:
        def eval(self): return self
        def __call__(self, _):
            return [{"scores": torch.tensor([.9]), "masks": torch.tensor([[[[1., 1.], [0., 0.]]]])}]
    target = {"masks": torch.tensor([[[1, 1], [0, 0]]], dtype=torch.uint8)}
    result = evaluate_polygon_dataset(Model(), [([torch.zeros(3, 2, 2)], [target])], "cpu")
    assert result["mask_ap50_proxy"] == 1.0 and result["instance_mean_iou"] == 1.0
