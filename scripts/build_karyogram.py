#!/usr/bin/env python3
"""Segment a metaphase image with Experiment A, classify instances, render a karyogram."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.transforms import ToTensor
from src.evaluation.postprocessing import build_postprocessor
from src.karyogram.classifier import class_name, classifier_logits, load_classifier
from src.karyogram.crops import classifier_tensor, masked_crop
from src.karyogram.device import resolve_device
from src.karyogram.layout import render_karyogram
from src.models.mask_rcnn import build_mask_rcnn


def _load_segmenter(cfg, checkpoint: Path, device):
    model = build_mask_rcnn(cfg).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp_karyogram.yaml")
    parser.add_argument("--image", required=True, help="Metaphase image (PNG/JPEG/etc.)")
    parser.add_argument("--segmentation-checkpoint", default=None)
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    options = cfg["mvp_inference"]
    device = resolve_device(args.device or options.get("device", cfg.get("training", {}).get("device", "cpu")))
    segmenter_path = Path(args.segmentation_checkpoint or options["segmentation_checkpoint"])
    classifier_path = Path(args.classifier_checkpoint or options["classifier_checkpoint"])
    if not segmenter_path.exists(): raise FileNotFoundError(f"Experiment-A checkpoint not found: {segmenter_path}")
    if not classifier_path.exists(): raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}. Train it with scripts/train_chromosome_classifier.py first.")
    output_dir = Path(args.output_dir or options.get("output_dir", "outputs/karyogram_mvp")) / Path(args.image).stem
    segments_dir = output_dir / "segments"; segments_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
    tensor, _ = ToTensor()(image, {})
    segmenter = _load_segmenter(cfg, segmenter_path, device)
    classifier = load_classifier(classifier_path, device)
    with torch.inference_mode():
        output = segmenter([tensor.to(device)])[0]
    threshold = float(options.get("segmentation_score_threshold", 0.5))
    keep = output["scores"] >= threshold
    output = {key: value[keep] if torch.is_tensor(value) and value.ndim and value.shape[0] == len(keep) else value for key, value in output.items()}
    postprocessor = build_postprocessor(cfg)
    if postprocessor: output = postprocessor(output)
    predictions = []
    mask_threshold = float(options.get("mask_threshold", 0.5))
    for index, (box, mask, segment_score) in enumerate(zip(output["boxes"], output["masks"], output["scores"])):
        binary_mask = mask[0].detach().cpu().numpy() >= mask_threshold
        if not binary_mask.any(): continue
        crop = masked_crop(image, binary_mask, int(options.get("crop_padding", 12)))
        with torch.inference_mode():
            probabilities = classifier_logits(classifier, classifier_tensor(crop, int(options.get("image_size", 224)), cfg["classifier_training"].get("normalization", "imagenet")).unsqueeze(0).to(device)).softmax(1)[0]
        label = int(probabilities.argmax())
        crop_path = segments_dir / f"segment_{index:03d}_{class_name(label)}.png"; crop.save(crop_path)
        predictions.append({"instance": index, "chromosome": class_name(label), "classification_score": round(float(probabilities[label]), 6), "segmentation_score": round(float(segment_score), 6), "box_xyxy": [round(float(value), 2) for value in box.detach().cpu()], "crop": crop, "crop_path": str(crop_path)})
    summary = render_karyogram(predictions, output_dir / "karyogram.png", columns=int(options.get("karyogram_columns", 6)))
    serializable = [{key: value for key, value in item.items() if key != "crop"} for item in predictions]
    report = {"image": str(Path(args.image)), "settings": {"segmentation_score_threshold": threshold, "mask_threshold": mask_threshold}, "summary": summary, "instances": serializable}
    (output_dir / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"karyogram": summary["output"], "instances": len(predictions), "counts": summary["counts"]}, indent=2))


if __name__ == "__main__": main()
