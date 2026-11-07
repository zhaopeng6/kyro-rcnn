"""Render classified segmented crops in conventional 1-22, X, Y order."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageDraw, ImageFont

from .classifier import CLASS_NAMES


def _thumbnail(image: Image.Image, maximum: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB").copy()
    image.thumbnail(maximum, Image.Resampling.LANCZOS)
    return image


def render_karyogram(predictions: Iterable[dict], output: str | Path, columns: int = 6, crop_size: tuple[int, int] = (130, 155)) -> dict:
    """Place every accepted instance into a labelled karyogram cell.

    A cell expands vertically if a class has more than two detections. This is
    deliberate: an MVP should make potential aneuploidy/false positives visible
    instead of throwing those detections away to make a cosmetically normal map.
    """
    output = Path(output)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["chromosome"]].append(prediction)
    for name in CLASS_NAMES:
        grouped[name].sort(key=lambda item: float(item["classification_score"]), reverse=True)

    padding, header = 14, 28
    cell_width = crop_size[0] + 2 * padding
    rows = [CLASS_NAMES[start:start + columns] for start in range(0, len(CLASS_NAMES), columns)]
    row_heights = []
    for row in rows:
        max_count = max((len(grouped[name]) for name in row), default=0)
        # Two positions across, then add height for every additional pair.
        row_heights.append(header + max(1, (max_count + 1) // 2) * (crop_size[1] + padding) + padding)
    width, height = columns * cell_width + padding, sum(row_heights) + padding
    canvas = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    y = padding
    for row, row_height in zip(rows, row_heights):
        for column, name in enumerate(row):
            x = padding + column * cell_width
            draw.rectangle((x, y, x + cell_width - padding, y + row_height - padding), outline="#9aa0a6", width=1)
            draw.text((x + 4, y + 5), f"{name}  (n={len(grouped[name])})", fill="black", font=font)
            for position, item in enumerate(grouped[name]):
                crop = _thumbnail(item["crop"], crop_size)
                slot_x = x + padding + (position % 2) * (crop_size[0] // 2)
                slot_y = y + header + (position // 2) * (crop_size[1] + padding)
                canvas.paste(crop, (slot_x + (crop_size[0] // 2 - crop.width // 2), slot_y + (crop_size[1] - crop.height) // 2))
        y += row_height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {"output": str(output), "detected_instances": sum(len(items) for items in grouped.values()), "counts": {name: len(grouped[name]) for name in CLASS_NAMES}}
