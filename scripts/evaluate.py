#!/usr/bin/env python3
"""Evaluate a Phase-2 checkpoint on held-out polygon-labelled AutoKary data."""
import argparse, json, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.autokary import AutoKaryDataset
from src.datasets.collate import detection_collate
from src.datasets.transforms import ToTensor
from src.evaluation.metrics import evaluate_polygon_dataset
from src.evaluation.postprocessing import build_postprocessor
from src.models.mask_rcnn import build_mask_rcnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None); parser.add_argument("--score-threshold", type=float, default=.5)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); device = torch.device(cfg["training"].get("device", "cpu"))
    root = cfg["datasets"]["autokary"].get("validation_root")
    if not root: raise ValueError("datasets.autokary.validation_root is required for evaluation")
    dataset = AutoKaryDataset(root, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["training"].get("num_workers", 0), collate_fn=detection_collate)
    model = build_mask_rcnn(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    metrics = evaluate_polygon_dataset(model, loader, device, args.score_threshold, build_postprocessor(cfg))
    output = Path(args.output or Path(args.checkpoint).with_name("validation_metrics.json")); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2)); print(f"Wrote {output}")


if __name__ == "__main__": main()
