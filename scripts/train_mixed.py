#!/usr/bin/env python3
"""Experiment-B trainer: homogeneous polygon and box batches with loss routing."""
import argparse, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.autokary import AutoKaryDataset
from src.datasets.box_dataset import LowqBoxDataset
from src.datasets.collate import detection_collate
from src.datasets.transforms import build_training_transform
from src.models.mask_rcnn import build_mask_rcnn
from src.training.mixed_trainer import train_mixed_supervision
from src.utils.config import dataset_path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--steps", type=int, default=None); args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); train = cfg["training"]
    torch.manual_seed(train.get("seed", 42)); transform = build_training_transform(cfg.get("augmentation", {}))
    polygon = AutoKaryDataset(dataset_path(cfg, cfg["datasets"]["autokary"]["root"]), transform=transform)
    lowq = cfg["datasets"]["lowq"]
    boxes = LowqBoxDataset(dataset_path(cfg, lowq["annotations"]), dataset_path(cfg, lowq["images"]), transform=transform)
    kwargs = {"batch_size": train.get("batch_size", 1), "num_workers": train.get("num_workers", 0), "collate_fn": detection_collate}
    polygon_loader, box_loader = DataLoader(polygon, shuffle=True, **kwargs), DataLoader(boxes, shuffle=True, **kwargs)
    model = build_mask_rcnn(cfg); optimizer = torch.optim.SGD(model.parameters(), lr=train.get("learning_rate", .005), momentum=.9, weight_decay=train.get("weight_decay", .0005))
    _, counts = train_mixed_supervision(model, polygon_loader, box_loader, optimizer, torch.device(train.get("device", "cpu")), args.steps or train.get("total_steps", 100), train.get("output_dir", "outputs/experiment_b"), train.get("polygon_batch_probability", .5), train.get("seed", 42), train.get("gradient_clip_norm", 1.0))
    print(f"Completed mixed training; polygon steps={counts['polygon']}, box steps={counts['box']}")


if __name__ == "__main__": main()
