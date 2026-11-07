"""A narrow torchvision RoIHeads extension for explicit mixed supervision."""
from __future__ import annotations

import torch
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_inference, maskrcnn_loss
from .weak_mask_loss import boxinst_style_mask_losses


class MixedSupervisionRoIHeads(RoIHeads):
    """Skip only the mask branch for homogeneous `supervision_type == 'box'` batches.

    Detection/RPN losses are still computed by torchvision. This is a component
    replacement, not a global monkey patch, and intentionally mirrors the
    torchvision 0.27 RoIHeads forward path.
    """
    def __init__(self, *args, weak_mask_supervision=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weak_mask_supervision = weak_mask_supervision or {"enabled": False}

    def check_targets(self, targets):
        """Relax torchvision's unconditional mask requirement for box batches."""
        if targets is None:
            raise ValueError("targets should not be None")
        if not all("boxes" in target for target in targets):
            raise ValueError("Every element of targets should have a boxes key")
        if not all("labels" in target for target in targets):
            raise ValueError("Every element of targets should have a labels key")

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            types = {target.get("supervision_type") for target in targets}
            if len(types) != 1 or None in types:
                raise ValueError("Mixed supervision requires homogeneous batches with supervision_type")
            supervision_type = next(iter(types))
            if supervision_type not in {"polygon", "box"}:
                raise ValueError(f"Unsupported supervision type: {supervision_type}")
            if supervision_type == "polygon" and any("masks" not in target for target in targets):
                raise ValueError("Polygon batches require genuine masks")
        else:
            supervision_type = None

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = regression_targets = matched_idxs = None

        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, output_labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            result = [{"boxes": boxes[i], "labels": output_labels[i], "scores": scores[i]} for i in range(len(boxes))]

        # Inference always needs masks; only box-only *training* intentionally skips them.
        weak_box_masks = self.training and supervision_type == "box" and self.weak_mask_supervision.get("enabled", False)
        run_mask = self.has_mask() and (not self.training or supervision_type == "polygon" or weak_box_masks)
        if run_mask:
            mask_proposals = [item["boxes"] for item in result]
            if self.training:
                mask_proposals, pos_matched_idxs = [], []
                for image_id in range(len(proposals)):
                    positive = torch.where(labels[image_id] > 0)[0]
                    mask_proposals.append(proposals[image_id][positive])
                    pos_matched_idxs.append(matched_idxs[image_id][positive])
            else:
                pos_matched_idxs = None
            mask_logits = self.mask_predictor(self.mask_head(self.mask_roi_pool(features, mask_proposals, image_shapes)))
            if self.training:
                if supervision_type == "polygon":
                    gt_masks = [target["masks"] for target in targets]
                    gt_labels = [target["labels"] for target in targets]
                    losses["loss_mask"] = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                else:
                    if any("weak_image" not in target or "weak_boxes" not in target for target in targets): raise ValueError("BoxInst-style loss requires weak_image and weak_boxes")
                    projection, pairwise = boxinst_style_mask_losses(mask_logits, mask_proposals, labels, pos_matched_idxs, targets, self.weak_mask_supervision)
                    losses["loss_mask_projection"] = projection * float(self.weak_mask_supervision.get("projection_weight", 1.0))
                    losses["loss_mask_pairwise"] = pairwise
            else:
                for masks, item in zip(maskrcnn_inference(mask_logits, [item["labels"] for item in result]), result):
                    item["masks"] = masks
        return result, losses


def replace_with_mixed_roi_heads(model, weak_mask_supervision=None):
    """Replace a Mask R-CNN instance's RoI component with the isolated router."""
    old = model.roi_heads
    model.roi_heads = MixedSupervisionRoIHeads(
        old.box_roi_pool, old.box_head, old.box_predictor,
        old.proposal_matcher.high_threshold, old.proposal_matcher.low_threshold,
        old.fg_bg_sampler.batch_size_per_image, old.fg_bg_sampler.positive_fraction,
        old.box_coder.weights, old.score_thresh, old.nms_thresh, old.detections_per_img,
        old.mask_roi_pool, old.mask_head, old.mask_predictor,
        weak_mask_supervision=weak_mask_supervision,
    )
    return model
