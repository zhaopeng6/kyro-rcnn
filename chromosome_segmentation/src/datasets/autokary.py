from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import parse_labelme_annotation


class AutoKaryDataset(Dataset):
    """LabelMe polygons. Every target includes genuine instance masks."""
    supervision_type = "polygon"

    def __init__(self, root: str | Path, transform=None):
        self.root, self.transform = Path(root), transform
        self.annotations = sorted(self.root.rglob("*.json"))

    def __len__(self): return len(self.annotations)

    def __getitem__(self, index):
        ann = parse_labelme_annotation(self.annotations[index])
        image_path = self.annotations[index].with_suffix(".png")
        if not image_path.exists():
            image_path = self.annotations[index].with_suffix(".jpg")
        image = Image.open(image_path).convert("RGB")
        instances = ann["instances"]
        masks = np.stack([np.asarray(x["mask"], dtype=np.uint8) for x in instances]) if instances else np.zeros((0, ann["height"], ann["width"]), dtype=np.uint8)
        boxes = torch.tensor([x["box"] for x in instances], dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": boxes, "labels": torch.ones(len(instances), dtype=torch.int64),
                  "masks": torch.from_numpy(masks), "area": torch.tensor([x["area"] for x in instances], dtype=torch.float32),
                  "iscrowd": torch.zeros(len(instances), dtype=torch.int64), "image_id": torch.tensor([index]),
                  "supervision_type": self.supervision_type, "original_labels": [x["label"] for x in instances]}
        if self.transform: image, target = self.transform(image, target)
        return image, target
