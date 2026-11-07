"""BoxInst-inspired weak mask losses for box-only Mask R-CNN batches.

This is an adaptation for ROI mask logits, not a byte-for-byte BoxInst port.
"""
from __future__ import annotations

import torch
from torchvision.ops import roi_align


def _dice(pred, target, eps=1e-6):
    return 1 - (2 * (pred * target).sum() + eps) / (pred.sum() + target.sum() + eps)


def boxinst_style_mask_losses(mask_logits, mask_proposals, labels, matched_idxs, targets, settings):
    """Projection loss + color-affinity pairwise loss without constructing GT masks."""
    projection_losses, pairwise_losses, offset = [], [], 0
    color_threshold = float(settings.get("pairwise_color_threshold", .12))
    pairwise_weight = float(settings.get("pairwise_weight", 1.0))
    for image_id, proposals in enumerate(mask_proposals):
        count = len(proposals)
        if not count:
            continue
        logits = mask_logits[offset:offset + count]; offset += count
        gt_indices = matched_idxs[image_id]
        gt_labels = targets[image_id]["labels"][gt_indices]
        probabilities = logits[torch.arange(count, device=logits.device), gt_labels].sigmoid()
        gt_boxes = targets[image_id]["boxes"][gt_indices]
        height, width = probabilities.shape[-2:]
        for index, probability in enumerate(probabilities):
            proposal, box = proposals[index], gt_boxes[index]
            xs = proposal[0] + (torch.arange(width, device=logits.device) + .5) * (proposal[2] - proposal[0]) / width
            ys = proposal[1] + (torch.arange(height, device=logits.device) + .5) * (proposal[3] - proposal[1]) / height
            target_x = ((xs >= box[0]) & (xs <= box[2])).float(); target_y = ((ys >= box[1]) & (ys <= box[3])).float()
            projection_losses.extend((_dice(probability.amax(0), target_x), _dice(probability.amax(1), target_y)))
            # Crop the original grayscale/RGB image by the unmodified GT box.
            raw_image = targets[image_id]["weak_image"].unsqueeze(0)
            raw_box = targets[image_id]["weak_boxes"][gt_indices[index]].unsqueeze(0)
            crop = roi_align(raw_image, [raw_box], output_size=(height, width), aligned=True)[0].mean(0)
            horizontal_similarity = (crop[:, 1:] - crop[:, :-1]).abs() <= color_threshold
            vertical_similarity = (crop[1:, :] - crop[:-1, :]).abs() <= color_threshold
            horizontal_same = probability[:, 1:] * probability[:, :-1] + (1 - probability[:, 1:]) * (1 - probability[:, :-1])
            vertical_same = probability[1:, :] * probability[:-1, :] + (1 - probability[1:, :]) * (1 - probability[:-1, :])
            terms = []
            if horizontal_similarity.any(): terms.append(-torch.log(horizontal_same[horizontal_similarity].clamp_min(1e-6)).mean())
            if vertical_similarity.any(): terms.append(-torch.log(vertical_same[vertical_similarity].clamp_min(1e-6)).mean())
            if terms: pairwise_losses.append(torch.stack(terms).mean())
    zero = mask_logits.sum() * 0
    projection = torch.stack(projection_losses).mean() if projection_losses else zero
    pairwise = torch.stack(pairwise_losses).mean() * pairwise_weight if pairwise_losses else zero
    return projection, pairwise
