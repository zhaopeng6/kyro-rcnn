"""Instance-crop dataset derived from the polygon-labelled AutoKary images."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.annotations import rasterize_polygon
from .classifier import label_index
from .crops import classifier_tensor, masked_crop


class AutoKaryCropDataset(Dataset):
    """One labelled, mask-isolated chromosome per item.

    Invalid/non-karyotype labels are excluded instead of being guessed.
    """
    def __init__(self, root: str | Path, crop_padding: int = 12, image_size: int = 224, normalization: str = "imagenet"):
        self.root = Path(root)
        self.crop_padding, self.image_size, self.normalization = int(crop_padding), int(image_size), normalization
        # Store LabelMe shape positions, not rasterized image masks. Rasterizing
        # every mask up front would retain tens of GB, and rasterizing every
        # image's full set of masks for every crop would make training needlessly
        # slow. A sample rasterizes just its own polygon when it is fetched.
        self.instances: list[tuple[Path, int, int]] = []
        for annotation_path in sorted(self.root.rglob("*.json")):
            payload = json.loads(annotation_path.read_text())
            for shape_index, shape in enumerate(payload.get("shapes", [])):
                if shape.get("shape_type", "polygon") != "polygon" or len(shape.get("points", [])) < 3:
                    continue
                target = label_index(shape.get("label"))
                if target is not None: self.instances.append((annotation_path, shape_index, target))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        annotation_path, shape_index, target = self.instances[index]
        annotation = json.loads(annotation_path.read_text())
        image_path = annotation_path.with_suffix(".png")
        if not image_path.exists():
            image_path = annotation_path.with_suffix(".jpg")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        shape = annotation["shapes"][shape_index]
        mask = rasterize_polygon(shape["points"], int(annotation["imageWidth"]), int(annotation["imageHeight"]))
        crop = masked_crop(image, np.asarray(mask), self.crop_padding)
        return classifier_tensor(crop, self.image_size, self.normalization), target
