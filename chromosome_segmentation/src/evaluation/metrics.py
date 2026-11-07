"""Transparent, dependency-free instance matching metrics for polygon validation."""
from __future__ import annotations

import torch


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.bool(), b.bool()
    union = (a | b).sum().item()
    return 0.0 if union == 0 else (a & b).sum().item() / union


def match_masks(pred_masks, true_masks, threshold: float):
    candidates = sorted(((mask_iou(p, t), i, j) for i, p in enumerate(pred_masks) for j, t in enumerate(true_masks)), reverse=True)
    used_pred, used_truth, matches = set(), set(), []
    for score, i, j in candidates:
        if score < threshold: break
        if i not in used_pred and j not in used_truth:
            used_pred.add(i); used_truth.add(j); matches.append((i, j, score))
    return matches


@torch.inference_mode()
def evaluate_polygon_dataset(model, loader, device, score_threshold=0.5, postprocess=None):
    model.eval(); total_pred = total_truth = 0; matches50 = matches75 = 0; ious = []
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        if postprocess is not None: outputs = [postprocess(output) for output in outputs]
        for output, target in zip(outputs, targets):
            keep = output["scores"].detach().cpu() >= score_threshold
            preds = output["masks"].detach().cpu()[keep, 0] >= 0.5
            truth = target["masks"].detach().cpu().bool()
            total_pred += len(preds); total_truth += len(truth)
            m50 = match_masks(preds, truth, .5); m75 = match_masks(preds, truth, .75)
            matches50 += len(m50); matches75 += len(m75); ious.extend(score for _, _, score in m50)
    precision50 = matches50 / total_pred if total_pred else 0.0
    recall50 = matches50 / total_truth if total_truth else 0.0
    return {"mask_ap50_proxy": precision50 * recall50, "mask_ap75_proxy": (matches75 / total_pred if total_pred else 0.0) * (matches75 / total_truth if total_truth else 0.0),
            "instance_precision50": precision50, "instance_recall50": recall50,
            "instance_mean_iou": sum(ious) / len(ious) if ious else 0.0, "predictions": total_pred, "ground_truth_instances": total_truth}
