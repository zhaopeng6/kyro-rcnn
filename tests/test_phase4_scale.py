import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.evaluation.analysis import analyze_instance_predictions, summarize_analyses
from src.evaluation.scale import scale_polygon_sample


def test_scale_transform_preserves_box_mask_alignment():
    image = torch.zeros(3, 10, 20)
    mask = torch.zeros((1, 10, 20), dtype=torch.uint8); mask[:, 2:6, 4:10] = 1
    target = {"boxes": torch.tensor([[4., 2., 10., 6.]]), "masks": mask, "area": torch.tensor([24.])}
    scaled_image, scaled = scale_polygon_sample(image, target, .5)
    assert tuple(scaled_image.shape[-2:]) == (5, 10)
    assert scaled["boxes"].tolist() == [[2., 1., 5., 3.]]
    assert scaled["masks"].shape == (1, 5, 10) and scaled["area"].item() > 0


def test_overlap_and_error_analysis_reports_defined_cases():
    truth = torch.zeros((2, 8, 8), dtype=torch.bool)
    truth[0, 1:4, 1:4] = 1; truth[1, 1:4, 4:7] = 1
    boxes = torch.tensor([[1., 1., 4., 4.], [4., 1., 7., 4.]])
    predictions = truth[:1].clone(); pred_boxes = boxes[:1].clone()
    analysis = analyze_instance_predictions(predictions, pred_boxes, truth, boxes)
    summary = summarize_analyses([analysis])
    assert analysis["matches50"] == 1 and analysis["errors"]["missed"] == 1
    assert summary["instance_recall50"] == .5 and summary["subsets"]["isolated"]["instances"] == 2
