"""Lightweight failure rendering without changing source images."""
from __future__ import annotations

from PIL import Image, ImageDraw
import torch


_COLORS = [(230, 57, 70), (29, 161, 242), (42, 157, 143), (244, 162, 97), (131, 56, 236)]


def _base(image):
    data = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(data, "RGB")


def _panel(image, masks, boxes, scores=None):
    result = _base(image); overlay = result.copy(); pixels = overlay.load()
    for index, mask in enumerate(masks):
        color = _COLORS[index % len(_COLORS)]; array = mask.detach().cpu().bool().numpy()
        for y, x in zip(*array.nonzero()): pixels[x, y] = color
    result = Image.blend(result, overlay, .38); draw = ImageDraw.Draw(result)
    for index, box in enumerate(boxes):
        draw.rectangle(tuple(float(x) for x in box), outline=_COLORS[index % len(_COLORS)], width=2)
        if scores is not None: draw.text((float(box[0]), float(box[1])), f"{float(scores[index]):.2f}", fill=(255, 255, 0))
    return result


def save_prediction_comparison(path, image, target, output, keep):
    """Save input | GT | prediction panels for manual failure review."""
    original = _base(image)
    ground_truth = _panel(image, target["masks"], target["boxes"])
    prediction = _panel(image, output["masks"][keep, 0] >= .5, output["boxes"][keep], output["scores"][keep])
    canvas = Image.new("RGB", (original.width * 3, original.height))
    for index, panel in enumerate((original, ground_truth, prediction)): canvas.paste(panel, (index * original.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)
