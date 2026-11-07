import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.models.mask_rcnn import build_mask_rcnn


def test_polygon_batch_produces_mask_loss_and_gradients():
    model = build_mask_rcnn({"model": {"num_classes": 2, "min_size": 64, "max_size": 64,
                                         "anchor_sizes": [[8], [16], [32], [64], [128]],
                                         "max_detections_per_image": 10}})
    image = torch.rand(3, 64, 64)
    mask = torch.zeros((1, 64, 64), dtype=torch.uint8); mask[:, 16:40, 20:38] = 1
    target = {"boxes": torch.tensor([[20., 16., 38., 40.]]), "labels": torch.tensor([1]), "masks": mask,
              "area": torch.tensor([432.]), "iscrowd": torch.tensor([0]), "image_id": torch.tensor([0]), "supervision_type": "polygon"}
    losses = model([image], [target])
    assert "loss_mask" in losses and torch.isfinite(sum(losses.values()))
    sum(losses.values()).backward()
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if "mask" in name)
