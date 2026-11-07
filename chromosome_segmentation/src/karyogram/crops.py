"""Mask-aware chromosome crop extraction shared by training and inference."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def masked_crop(image: Image.Image, mask: np.ndarray | torch.Tensor, padding: int = 12) -> Image.Image:
    """Return a square RGB crop whose non-chromosome pixels are white.

    The crop is based on the mask (not the detector box), so it removes nearby
    chromosomes even where detection boxes overlap.
    """
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2 or not foreground.any():
        raise ValueError("A non-empty 2-D instance mask is required for a chromosome crop")
    ys, xs = np.where(foreground)
    x1, x2 = max(0, int(xs.min()) - padding), min(image.width, int(xs.max()) + 1 + padding)
    y1, y2 = max(0, int(ys.min()) - padding), min(image.height, int(ys.max()) + 1 + padding)
    side = max(x2 - x1, y2 - y1)
    # Keep the crop square while remaining inside the source image.
    left = max(0, min(x1 - (side - (x2 - x1)) // 2, image.width - side))
    top = max(0, min(y1 - (side - (y2 - y1)) // 2, image.height - side))
    right, bottom = min(image.width, left + side), min(image.height, top + side)
    source = np.asarray(image.convert("RGB"))
    canvas = np.full((bottom - top, right - left, 3), 255, dtype=np.uint8)
    local_mask = foreground[top:bottom, left:right]
    canvas[local_mask] = source[top:bottom, left:right][local_mask]
    return Image.fromarray(canvas, mode="RGB")


def classifier_tensor(crop: Image.Image, size: int = 224, normalization: str = "imagenet") -> torch.Tensor:
    """Resize and normalize a crop for the configured classifier family."""
    tensor = TF.to_tensor(crop)
    tensor = TF.resize(tensor, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if normalization == "imagenet": return TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if normalization == "vit": return TF.normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    raise ValueError(f"Unsupported classifier normalization {normalization!r}; use 'imagenet' or 'vit'.")
