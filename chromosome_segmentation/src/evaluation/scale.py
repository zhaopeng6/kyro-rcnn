"""Geometry-consistent synthetic scaling for polygon-labelled evaluation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def scale_polygon_sample(image: torch.Tensor, target: dict, factor: float):
    """Uniformly scale image, masks, boxes, and areas without padding or cropping."""
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    height, width = image.shape[-2:]
    new_height, new_width = max(1, round(height * factor)), max(1, round(width * factor))
    scaled_image = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False, antialias=True).squeeze(0)
    scaled = dict(target)
    scale_x, scale_y = new_width / width, new_height / height
    boxes = target["boxes"].clone()
    boxes[:, [0, 2]] *= scale_x; boxes[:, [1, 3]] *= scale_y
    scaled["boxes"] = boxes
    masks = F.interpolate(target["masks"].unsqueeze(1).float(), size=(new_height, new_width), mode="nearest").squeeze(1).to(target["masks"].dtype)
    scaled["masks"] = masks
    scaled["area"] = masks.flatten(1).sum(1).float()
    scaled["scale_transform"] = {"requested_factor": factor, "scale_x": scale_x, "scale_y": scale_y,
                                 "input_size": [height, width], "output_size": [new_height, new_width], "padding": [0, 0, 0, 0]}
    return scaled_image, scaled


class NoResizeGeneralizedRCNNTransform(GeneralizedRCNNTransform):
    """Keep requested benchmark scale intact while retaining normalize/pad/postprocess."""
    def resize(self, image, target=None):
        return image, target


def disable_model_internal_resize(model):
    """Replace only torchvision's resize step for a scale-controlled evaluation."""
    old = model.transform
    model.transform = NoResizeGeneralizedRCNNTransform(old.min_size, old.max_size, old.image_mean, old.image_std,
                                                        size_divisible=old.size_divisible, fixed_size=old.fixed_size)
    return model
