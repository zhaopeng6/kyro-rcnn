"""Phase-2 polygon-supervised Mask R-CNN construction."""
from __future__ import annotations

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from .mixed_roi_heads import replace_with_mixed_roi_heads


def build_mask_rcnn(config: dict):
    model_cfg = config.get("model", config)
    # Match torchvision's three-anchor-per-location pretrained RPN layout.
    # Altering the number of aspect ratios changes RPN-head tensor shapes and
    # makes COCO pretrained RPN weights incompatible.
    anchors = model_cfg.get("anchor_sizes", [[32], [64], [128], [256], [512]])
    aspect_ratios = model_cfg.get("anchor_aspect_ratios", [[0.5, 1.0, 2.0]] * len(anchors))
    if len(anchors) != len(aspect_ratios):
        raise ValueError("anchor_sizes and anchor_aspect_ratios must have one entry per FPN level")
    if model_cfg.get("pretrained", False) and any(len(ratios) != 3 for ratios in aspect_ratios):
        raise ValueError("COCO pretrained RPN requires exactly three anchor aspect ratios per FPN level. Use [0.5, 1.0, 2.0], or disable pretrained weights before changing the RPN layout.")
    anchor_generator = AnchorGenerator(tuple(tuple(x) for x in anchors), tuple(tuple(x) for x in aspect_ratios))
    pretrained = model_cfg.get("pretrained", False)
    model_kwargs = dict(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        min_size=int(model_cfg.get("min_size", 800)), max_size=int(model_cfg.get("max_size", 1333)),
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=float(model_cfg.get("score_threshold", 0.05)),
        box_nms_thresh=float(model_cfg.get("nms_threshold", 0.7)),
        box_detections_per_img=int(model_cfg.get("max_detections_per_image", 100)),
    )
    # Torchvision prevents passing a different num_classes value with loaded
    # COCO weights (91 classes). Load those transferable weights first, then
    # replace only the task-specific prediction heads.
    if not pretrained:
        model_kwargs["num_classes"] = int(model_cfg.get("num_classes", 2))
    model = maskrcnn_resnet50_fpn(**model_kwargs)
    if pretrained:
        num_classes = int(model_cfg.get("num_classes", 2))
        box_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
        mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
        mask_hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, mask_hidden, num_classes)
    return replace_with_mixed_roi_heads(model, model_cfg.get("weak_mask_supervision")) if model_cfg.get("mixed_supervision", False) else model
