import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.collate import detection_collate
from src.models.mask_rcnn import build_mask_rcnn
from src.training.checkpoints import load_checkpoint
from src.training.trainer import train_polygon_baseline
from tests.test_phase2_trainer import SyntheticPolygonDataset


def _model():
    return build_mask_rcnn({"model": {"num_classes": 2, "min_size": 64, "max_size": 64,
                                         "anchor_sizes": [[8], [16], [32], [64], [128]]}})


def test_checkpoint_is_saved_and_training_resumes_global_step(tmp_path):
    loader = DataLoader(SyntheticPolygonDataset(), batch_size=1, collate_fn=detection_collate)
    model = _model(); optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    train_polygon_baseline(model, loader, optimizer, "cpu", 2, tmp_path, checkpoint_interval=1)
    checkpoint = tmp_path / "last.pt"; assert checkpoint.exists()
    resumed_model = _model(); resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=.001)
    state = load_checkpoint(checkpoint, resumed_model, resumed_optimizer, "cpu")
    history = train_polygon_baseline(resumed_model, loader, resumed_optimizer, "cpu", 3, tmp_path,
                                     start_step=state["step"], checkpoint_interval=1)
    assert state["step"] == 2 and history[0]["step"] == 3
    assert load_checkpoint(checkpoint, resumed_model, resumed_optimizer, "cpu")["step"] == 3
