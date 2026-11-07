"""Geometry-safe image/target transforms for instance detection."""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ToTensor:
    def __call__(self, image: Image.Image, target: dict):
        data = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(data).permute(2, 0, 1).contiguous(), target


class RandomHorizontalFlip:
    def __init__(self, probability=0.5): self.probability = probability

    def __call__(self, image, target):
        if random.random() >= self.probability: return image, target
        width = image.width if isinstance(image, Image.Image) else image.shape[-1]
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if isinstance(image, Image.Image) else image.flip(-1)
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target = dict(target); target["boxes"] = boxes
        if "masks" in target:
            target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip:
    def __init__(self, probability=0.0): self.probability = probability

    def __call__(self, image, target):
        if random.random() >= self.probability: return image, target
        height = image.shape[-2:][0]
        target = dict(target); target["boxes"] = target["boxes"].clone()
        target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
        if "masks" in target: target["masks"] = target["masks"].flip(-2)
        return image.flip(-2), target


def _target_from_masks(target):
    """Recompute tight boxes/areas after a geometric mask transform or clipping."""
    target = dict(target); masks = target["masks"]
    valid, boxes = [], []
    for index, mask in enumerate(masks):
        ys, xs = torch.where(mask.bool())
        if len(xs):
            valid.append(index); boxes.append(torch.stack((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).float())
    target = _filter_instances(target, valid, len(masks))
    target["boxes"] = torch.stack(boxes).to(masks.device) if boxes else torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
    target["area"] = target["masks"].flatten(1).sum(1).float()
    return target


def _filter_instances(target, valid, instance_count):
    """Filter per-instance fields without accidentally changing image_id."""
    target = dict(target); indices = torch.tensor(valid, device=target["boxes"].device, dtype=torch.long)
    for key in ("boxes", "labels", "masks", "area", "iscrowd"):
        value = target.get(key)
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == instance_count:
            target[key] = value[indices]
    if isinstance(target.get("original_labels"), list) and len(target["original_labels"]) == instance_count:
        target["original_labels"] = [target["original_labels"][index] for index in valid]
    return target


def _target_from_boxes(target, boxes):
    valid = torch.where((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))[0].tolist()
    target = _filter_instances(target, valid, len(boxes)); target["boxes"] = boxes[valid]
    target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (target["boxes"][:, 3] - target["boxes"][:, 1])
    return target


def _center_fit(tensor, height, width):
    """Center crop or zero-pad NCHW/CHW tensor to a specified canvas."""
    source_height, source_width = tensor.shape[-2:]
    top = max(0, (source_height - height) // 2); left = max(0, (source_width - width) // 2)
    tensor = tensor[..., top:top + min(height, source_height), left:left + min(width, source_width)]
    pad_height, pad_width = height - tensor.shape[-2], width - tensor.shape[-1]
    return F.pad(tensor, (max(0, pad_width // 2), max(0, pad_width - pad_width // 2), max(0, pad_height // 2), max(0, pad_height - pad_height // 2)))


class RandomUniformScale:
    """Scale content, then center crop/pad back to its original canvas without warping."""
    def __init__(self, minimum=.65, maximum=1.5, probability=1.0): self.minimum, self.maximum, self.probability = minimum, maximum, probability

    def __call__(self, image, target):
        if random.random() >= self.probability: return image, target
        factor = random.uniform(self.minimum, self.maximum); height, width = image.shape[-2:]
        scaled_height, scaled_width = max(1, round(height * factor)), max(1, round(width * factor))
        image = F.interpolate(image.unsqueeze(0), size=(scaled_height, scaled_width), mode="bilinear", align_corners=False, antialias=True).squeeze(0)
        if "masks" in target:
            masks = F.interpolate(target["masks"].unsqueeze(1).float(), size=(scaled_height, scaled_width), mode="nearest").squeeze(1).to(target["masks"].dtype)
            target = dict(target); target["masks"] = _center_fit(masks, height, width)
            return _center_fit(image, height, width), _target_from_masks(target)
        crop_top, crop_left = max(0, (scaled_height - height) // 2), max(0, (scaled_width - width) // 2)
        kept_height, kept_width = min(height, scaled_height), min(width, scaled_width)
        pad_top, pad_left = max(0, (height - kept_height) // 2), max(0, (width - kept_width) // 2)
        boxes = target["boxes"].clone(); boxes[:, [0, 2]] = boxes[:, [0, 2]] * (scaled_width / width) - crop_left + pad_left; boxes[:, [1, 3]] = boxes[:, [1, 3]] * (scaled_height / height) - crop_top + pad_top
        boxes[:, [0, 2]].clamp_(0, width); boxes[:, [1, 3]].clamp_(0, height)
        return _center_fit(image, height, width), _target_from_boxes(target, boxes)


class RandomTranslation:
    """Translate content on a fixed canvas; off-canvas pixels/instances are clipped."""
    def __init__(self, max_fraction=.08, probability=.5): self.max_fraction, self.probability = max_fraction, probability

    def __call__(self, image, target):
        if random.random() >= self.probability: return image, target
        height, width = image.shape[-2:]
        dx, dy = round(random.uniform(-self.max_fraction, self.max_fraction) * width), round(random.uniform(-self.max_fraction, self.max_fraction) * height)
        def shift(tensor):
            shifted = torch.zeros_like(tensor)
            src_x1, src_x2 = max(0, -dx), min(width, width - dx); src_y1, src_y2 = max(0, -dy), min(height, height - dy)
            dst_x1, dst_x2 = max(0, dx), min(width, width + dx); dst_y1, dst_y2 = max(0, dy), min(height, height + dy)
            shifted[..., dst_y1:dst_y2, dst_x1:dst_x2] = tensor[..., src_y1:src_y2, src_x1:src_x2]
            return shifted
        if "masks" in target:
            target = dict(target); target["masks"] = shift(target["masks"])
            return shift(image), _target_from_masks(target)
        boxes = target["boxes"].clone(); boxes[:, [0, 2]] += dx; boxes[:, [1, 3]] += dy
        boxes[:, [0, 2]].clamp_(0, width); boxes[:, [1, 3]].clamp_(0, height)
        return shift(image), _target_from_boxes(target, boxes)


class RandomGaussianNoise:
    def __init__(self, std=.02, probability=.3): self.std, self.probability = std, probability

    def __call__(self, image, target):
        if random.random() < self.probability: image = (image + torch.randn_like(image) * self.std).clamp(0, 1)
        return image, target


def build_training_transform(augmentation: dict):
    """Configured realistic microscopy augmentation with geometry-safe targets."""
    transforms = [ToTensor()]
    scale = augmentation.get("scale", {})
    if scale.get("enabled", False): transforms.append(RandomUniformScale(scale.get("min", .65), scale.get("max", 1.5), scale.get("probability", 1.0)))
    translation = augmentation.get("translation", {})
    if translation.get("enabled", False): transforms.append(RandomTranslation(translation.get("max_fraction", .08), translation.get("probability", .5)))
    transforms += [RandomHorizontalFlip(augmentation.get("horizontal_flip_probability", .5)), RandomVerticalFlip(augmentation.get("vertical_flip_probability", 0.0))]
    noise = augmentation.get("gaussian_noise", {})
    if noise.get("enabled", False): transforms.append(RandomGaussianNoise(noise.get("std", .02), noise.get("probability", .3)))
    return Compose(transforms)


class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, image, target):
        for transform in self.transforms: image, target = transform(image, target)
        return image, target
