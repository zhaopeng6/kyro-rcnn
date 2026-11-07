import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.collate import detection_collate
from src.models.mask_rcnn import build_mask_rcnn
from src.training.mixed_trainer import HomogeneousBatchMixer, train_mixed_supervision


def _model():
    return build_mask_rcnn({"model": {"mixed_supervision": True, "num_classes": 2, "min_size": 64, "max_size": 64,
                                         "anchor_sizes": [[8], [16], [32], [64], [128]], "max_detections_per_image": 10}})


def _target(kind):
    target = {"boxes": torch.tensor([[20., 16., 38., 40.]]), "labels": torch.tensor([1]), "area": torch.tensor([432.]),
              "iscrowd": torch.tensor([0]), "image_id": torch.tensor([0]), "supervision_type": kind}
    if kind == "polygon":
        target["masks"] = torch.zeros((1, 64, 64), dtype=torch.uint8); target["masks"][:, 16:40, 20:38] = 1
    return target


def test_mask_loss_and_gradients_are_routed_by_supervision_type():
    model, image = _model(), torch.rand(3, 64, 64)
    polygon_losses = model([image], [_target("polygon")]); assert "loss_mask" in polygon_losses
    sum(polygon_losses.values()).backward()
    assert any(p.grad is not None for name, p in model.named_parameters() if "mask" in name)
    model.zero_grad(set_to_none=True)
    box_losses = model([image], [_target("box")])
    assert "loss_mask" not in box_losses and {"loss_classifier", "loss_box_reg"}.issubset(box_losses)
    sum(box_losses.values()).backward()
    assert all(p.grad is None for name, p in model.named_parameters() if "mask" in name)


class _Source(Dataset):
    def __init__(self, kind): self.kind = kind
    def __len__(self): return 1
    def __getitem__(self, _): return torch.rand(3, 64, 64), _target(self.kind)


def test_mixed_trainer_records_source_counts_and_checkpoint(tmp_path):
    polygon = DataLoader(_Source("polygon"), batch_size=1, collate_fn=detection_collate)
    box = DataLoader(_Source("box"), batch_size=1, collate_fn=detection_collate)
    model = _model(); optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    history, counts = train_mixed_supervision(model, polygon, box, optimizer, "cpu", 2, tmp_path, polygon_probability=.5, seed=42)
    assert len(history) == 2 and sum(counts.values()) == 2 and (tmp_path / "last.pt").exists()


def test_batch_mixer_returns_homogeneous_source():
    polygon = DataLoader(_Source("polygon"), batch_size=1, collate_fn=detection_collate)
    box = DataLoader(_Source("box"), batch_size=1, collate_fn=detection_collate)
    kind, (_, targets) = HomogeneousBatchMixer(polygon, box, .5, seed=1).next()
    assert all(target["supervision_type"] == kind for target in targets)
