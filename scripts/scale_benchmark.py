#!/usr/bin/env python3
"""Phase-4 controlled scale robustness benchmark on held-out polygon data."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.autokary import AutoKaryDataset
from src.datasets.transforms import ToTensor
from src.evaluation.analysis import analyze_instance_predictions, summarize_analyses
from src.evaluation.postprocessing import build_postprocessor
from src.evaluation.scale import disable_model_internal_resize, scale_polygon_sample
from src.evaluation.visualization import save_prediction_comparison
from src.models.mask_rcnn import build_mask_rcnn
from src.utils.config import dataset_path


def write_plots(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is unavailable; JSON and CSV were written but plots were skipped")
        return
    metrics = [("mask_ap50_proxy", "Mask AP50 proxy"), ("instance_mean_dice", "Instance Dice"), ("instance_recall50", "Instance recall@0.5"), ("split_candidate_rate", "Split candidate rate"), ("merge_candidate_rate", "Merge candidate rate")]
    factors = [row["factor"] for row in rows]
    for key, title in metrics:
        figure, axis = plt.subplots(); axis.plot(factors, [row[key] for row in rows], marker="o")
        axis.set(xlabel="Synthetic scale factor", ylabel=title, title=f"{title} vs scale"); axis.grid(True, alpha=.3)
        figure.savefig(output_dir / f"{key}_vs_scale.png", dpi=150, bbox_inches="tight"); plt.close(figure)


def evaluate_factor(model, dataset, device, factor, score_threshold, selected_indices, visualization_dir, max_visualizations, postprocess=None):
    analyses, saved = [], 0; model.eval()
    with torch.inference_mode():
        for index in selected_indices:
            image, target = dataset[index]
            scaled_image, scaled_target = scale_polygon_sample(image, target, factor)
            output = model([scaled_image.to(device)])[0]
            output = {key: value.detach().cpu() for key, value in output.items()}
            if postprocess is not None: output = postprocess(output)
            keep = output["scores"] >= score_threshold
            analysis = analyze_instance_predictions(output["masks"][keep, 0] >= .5, output["boxes"][keep], scaled_target["masks"].bool(), scaled_target["boxes"])
            analyses.append(analysis)
            is_failure = analysis["errors"]["missed"] > 0 or analysis["errors"]["split_candidates"] > 0 or analysis["errors"]["merge_candidates"] > 0
            if visualization_dir and is_failure and saved < max_visualizations:
                save_prediction_comparison(visualization_dir / f"factor_{factor:g}" / f"sample_{index:04d}.png", scaled_image, scaled_target, output, keep)
                saved += 1
    result = summarize_analyses(analyses)
    result.update({"factor": factor, "images": len(analyses), "score_threshold": score_threshold,
                   "visualizations_saved": saved, "geometry": "uniform resize only; no padding or cropping"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None); parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--factors", help="Comma-separated scale factors; defaults to scale_test.factors")
    parser.add_argument("--score-threshold", type=float); args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); scale_cfg = cfg.get("scale_test", {})
    device = torch.device(args.device or cfg["training"].get("device", "cpu")); output_dir = Path(args.output_dir or Path(cfg["training"]["output_dir"]) / "scale_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = AutoKaryDataset(dataset_path(cfg, cfg["datasets"]["autokary"]["validation_root"]), transform=ToTensor())
    model = build_mask_rcnn(cfg).to(device); state = torch.load(args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(state["model"])
    resize_policy = scale_cfg.get("internal_resize_policy", "none")
    if resize_policy == "none":
        disable_model_internal_resize(model)
    elif resize_policy != "torchvision":
        raise ValueError("scale_test.internal_resize_policy must be 'none' or 'torchvision'")
    threshold = args.score_threshold if args.score_threshold is not None else scale_cfg.get("score_threshold", cfg["model"].get("score_threshold", .05))
    requested_images = args.max_images if args.max_images is not None else scale_cfg.get("max_images", 2)
    image_count = len(dataset) if requested_images is None else min(int(requested_images), len(dataset))
    selection_seed = int(scale_cfg.get("selection_seed", 42))
    postprocess = build_postprocessor(cfg)
    factors = [float(value) for value in args.factors.split(",")] if args.factors else scale_cfg.get("factors", [.6, .75, 1., 1.25, 1.5, 1.75])
    rows = []
    for factor_index, factor in enumerate(factors):
        indices = random.Random(selection_seed + factor_index).sample(range(len(dataset)), image_count)
        row = evaluate_factor(model, dataset, device, float(factor), threshold, indices, output_dir / "visualizations" if scale_cfg.get("save_visualizations", True) else None, int(scale_cfg.get("max_visualizations_per_factor", 12)), postprocess)
        row["selected_indices"] = indices; rows.append(row)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_step": state.get("step"), "score_threshold": threshold, "internal_resize_policy": resize_policy, "mask_nms": cfg.get("postprocessing", {}).get("mask_nms", {}), "selection_seed": selection_seed, "results": rows,
              "robustness_score_mean_mask_ap50_proxy": sum(row["mask_ap50_proxy"] for row in rows) / len(rows) if rows else None}
    (output_dir / "scale_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["factor", "images", "mask_ap50_proxy", "mask_ap75_proxy", "instance_precision50", "instance_recall50", "instance_mean_iou", "instance_mean_dice", "split_candidate_rate", "merge_candidate_rate", "predictions", "ground_truth_instances"]
    with (output_dir / "scale_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in rows])
    write_plots(rows, output_dir)
    print(json.dumps(report, indent=2)); print(f"Wrote Phase-4 artifacts to {output_dir}")


if __name__ == "__main__": main()
