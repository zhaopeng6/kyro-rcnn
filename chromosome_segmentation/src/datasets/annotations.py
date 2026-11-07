"""Format adapters shared by dataset loaders and inspection tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


CHROMOSOME_LABELS = {str(i): i for i in range(1, 23)} | {"X": 23, "Y": 24}


def chromosome_metadata_label(raw_label: str | None) -> int | None:
    """Return karyotype label when it is explicit; otherwise preserve as metadata."""
    if raw_label is None:
        return None
    value = str(raw_label).strip().upper()
    return CHROMOSOME_LABELS.get(value)


def rasterize_polygon(points: list[list[float]], width: int, height: int) -> Image.Image:
    """Rasterize a LabelMe polygon in its original image coordinate system."""
    mask = Image.new("L", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=1)
    return mask


def parse_labelme_annotation(path: str | Path, rasterize_masks: bool = True) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    width, height = int(payload["imageWidth"]), int(payload["imageHeight"])
    items, invalid = [], []
    for index, shape in enumerate(payload.get("shapes", [])):
        points = shape.get("points", [])
        if shape.get("shape_type", "polygon") != "polygon" or len(points) < 3:
            invalid.append({"index": index, "reason": "not_a_valid_polygon"})
            continue
        try:
            numeric_points = [(float(x), float(y)) for x, y in points]
            if not all(np.isfinite([x, y]).all() for x, y in numeric_points):
                raise ValueError("nonfinite_coordinates")
        except (TypeError, ValueError):
            invalid.append({"index": index, "reason": "invalid_coordinates"})
            continue
        if rasterize_masks:
            mask = rasterize_polygon(points, width, height)
            bbox = mask.getbbox()
            area = int(sum(mask.getdata()))
        else:
            mask = None
            xs, ys = zip(*numeric_points)
            bbox = (max(0, int(min(xs))), max(0, int(min(ys))), min(width, int(max(xs)) + 1), min(height, int(max(ys)) + 1))
            area = None
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            invalid.append({"index": index, "reason": "empty_rasterized_mask"})
            continue
        items.append({
            "label": shape.get("label", ""), "class_id": chromosome_metadata_label(shape.get("label")),
            "points": points, "mask": mask, "box": list(map(float, bbox)), "area": area,
            "instance_id": f"{path.stem}:{index}",
        })
    return {"path": str(path), "width": width, "height": height, "image_path": payload.get("imagePath"),
            "instances": items, "invalid": invalid}


def parse_voc_annotation(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {path}")
    width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
    items, invalid = [], []
    for index, obj in enumerate(root.findall("object")):
        box = obj.find("bndbox")
        try:
            coords = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")] if box is not None else None
            if coords is None or coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError("nonpositive_box")
        except (TypeError, ValueError):
            invalid.append({"index": index, "reason": "invalid_box"})
            continue
        label = obj.findtext("name", "")
        items.append({"label": label, "class_id": chromosome_metadata_label(label), "box": coords,
                      "area": (coords[2] - coords[0]) * (coords[3] - coords[1]),
                      "difficult": int(obj.findtext("difficult", "0")), "instance_id": f"{path.stem}:{index}"})
    return {"path": str(path), "filename": root.findtext("filename"), "width": width, "height": height,
            "instances": items, "invalid": invalid}
