"""Instance matching, overlap subsets, and explicitly-defined error counts."""
from __future__ import annotations

import torch

from .metrics import mask_iou


def _intersects(a, b):
    return min(float(a[2]), float(b[2])) > max(float(a[0]), float(b[0])) and min(float(a[3]), float(b[3])) > max(float(a[1]), float(b[1]))


def candidate_mask_ious(pred_masks, truth_masks, pred_boxes, truth_boxes):
    """Compute mask IoU only for box-intersecting pairs, avoiding dense HxW products."""
    return {(i, j): mask_iou(pred, truth) for i, pred in enumerate(pred_masks) for j, truth in enumerate(truth_masks)
            if _intersects(pred_boxes[i], truth_boxes[j])}


def _match(candidates, threshold):
    matches, used_pred, used_truth = [], set(), set()
    for (i, j), score in sorted(candidates.items(), key=lambda item: item[1], reverse=True):
        if score < threshold: break
        if i not in used_pred and j not in used_truth:
            matches.append((i, j, score)); used_pred.add(i); used_truth.add(j)
    return matches


def overlap_flags(masks, boxes):
    count = len(boxes); bbox = [False] * count; mask = [False] * count
    for i in range(count):
        for j in range(i + 1, count):
            if _intersects(boxes[i], boxes[j]):
                bbox[i] = bbox[j] = True
                if (masks[i].bool() & masks[j].bool()).any(): mask[i] = mask[j] = True
    return {"bbox_overlap": bbox, "mask_overlap": mask, "isolated": [not value for value in bbox]}


def analyze_instance_predictions(pred_masks, pred_boxes, truth_masks, truth_boxes):
    candidates = candidate_mask_ious(pred_masks, truth_masks, pred_boxes, truth_boxes)
    match50, match75 = _match(candidates, .5), _match(candidates, .75)
    matched_truth = {truth for _, truth, _ in match50}; matched_pred = {pred for pred, _, _ in match50}
    flags = overlap_flags(truth_masks, truth_boxes)
    subsets = {}
    for name, members in flags.items():
        selected = [index for index, member in enumerate(members) if member]
        hits = sum(index in matched_truth for index in selected)
        subsets[name] = {"instances": len(selected), "matched_at_50": hits, "recall_at_50": hits / len(selected) if selected else None}
    # A split/merge is defined at loose (0.1) mask IoU; attribution remains heuristic.
    split_truth = {truth for truth in range(len(truth_masks)) if sum(score >= .1 and j == truth for (_, j), score in candidates.items()) >= 2}
    merge_pred = {pred for pred in range(len(pred_masks)) if sum(score >= .1 and i == pred for (i, _), score in candidates.items()) >= 2}
    duplicates = sum(pred not in matched_pred and any(i == pred and score >= .5 for (i, _), score in candidates.items()) for pred in range(len(pred_masks)))
    return {"predictions": len(pred_masks), "ground_truth_instances": len(truth_masks), "matches50": len(match50), "matches75": len(match75),
            "matched_ious": [score for _, _, score in match50], "subsets": subsets,
            "errors": {"missed": len(truth_masks) - len(matched_truth), "duplicates": duplicates,
                       "split_candidates": len(split_truth), "merge_candidates": len(merge_pred)}, "matches": match50}


def summarize_analyses(analyses):
    total_pred = sum(row["predictions"] for row in analyses); total_truth = sum(row["ground_truth_instances"] for row in analyses)
    matches50, matches75 = sum(row["matches50"] for row in analyses), sum(row["matches75"] for row in analyses)
    ious = [score for row in analyses for score in row["matched_ious"]]
    precision = matches50 / total_pred if total_pred else 0.0; recall = matches50 / total_truth if total_truth else 0.0
    subsets = {}
    for name in ("isolated", "bbox_overlap", "mask_overlap"):
        instances = sum(row["subsets"][name]["instances"] for row in analyses)
        matched = sum(row["subsets"][name]["matched_at_50"] for row in analyses)
        subsets[name] = {"instances": instances, "matched_at_50": matched, "recall_at_50": matched / instances if instances else None}
    errors = {name: sum(row["errors"][name] for row in analyses) for name in ("missed", "duplicates", "split_candidates", "merge_candidates")}
    return {"mask_ap50_proxy": precision * recall, "mask_ap75_proxy": (matches75 / total_pred if total_pred else 0.0) * (matches75 / total_truth if total_truth else 0.0),
            "instance_precision50": precision, "instance_recall50": recall,
            "instance_mean_iou": sum(ious) / len(ious) if ious else 0.0,
            "instance_mean_dice": sum(2 * value / (1 + value) for value in ious) / len(ious) if ious else 0.0,
            "predictions": total_pred, "ground_truth_instances": total_truth, "subsets": subsets, "errors": errors,
            "split_candidate_rate": errors["split_candidates"] / total_truth if total_truth else 0.0,
            "merge_candidate_rate": errors["merge_candidates"] / total_pred if total_pred else 0.0,
            "definitions": {"bbox_overlap": "ground-truth boxes intersect", "mask_overlap": "ground-truth polygon masks share at least one pixel", "isolated": "no intersecting ground-truth box", "split_candidates": "one GT has >=2 predictions at mask IoU >=0.1", "merge_candidates": "one prediction has >=2 GT masks at mask IoU >=0.1"}}
