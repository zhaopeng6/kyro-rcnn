import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.collate import detection_collate
from src.models.mask_rcnn import build_mask_rcnn
from src.training.trainer import train_polygon_baseline


class SyntheticPolygonDataset(Dataset):
    def __len__(self): return 1
    def __getitem__(self, _):
        mask = torch.zeros((1, 64, 64), dtype=torch.uint8); mask[:, 16:40, 20:38] = 1
        return torch.rand(3, 64, 64), {"boxes": torch.tensor([[20., 16., 38., 40.]]), "labels": torch.tensor([1]),
            "masks": mask, "area": torch.tensor([432.]), "iscrowd": torch.tensor([0]), "image_id": torch.tensor([0]), "supervision_type": "polygon"}


def test_one_polygon_step_saves_resumable_checkpoint(tmp_path):
    model = build_mask_rcnn({"model": {"num_classes": 2, "min_size": 64, "max_size": 64, "anchor_sizes": [[8], [16], [32], [64], [128]]}})
    loader = DataLoader(SyntheticPolygonDataset(), batch_size=1, collate_fn=detection_collate)
    history = train_polygon_baseline(model, loader, torch.optim.SGD(model.parameters(), lr=.001), "cpu", 1, tmp_path)
    assert len(history) == 1 and (tmp_path / "last.pt").exists() and (tmp_path / "train_history.json").exists()
