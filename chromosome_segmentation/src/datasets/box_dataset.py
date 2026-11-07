from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import parse_voc_annotation


class LowqBoxDataset(Dataset):
    """Pascal VOC boxes. Deliberately does not create a `masks` target."""
    supervision_type = "box"

    def __init__(self, annotation_root: str | Path, image_root: str | Path, transform=None, skip_missing_images=True):
        self.annotation_root, self.image_root, self.transform = Path(annotation_root), Path(image_root), transform
        annotations = sorted(self.annotation_root.glob("*.xml"))
        self.records, self.skipped_missing_images = [], []
        for annotation in annotations:
            try:
                filename = ET.parse(annotation).getroot().findtext("filename")
            except ET.ParseError:
                # Leave malformed XML for the parser to report at access time.
                filename = None
            image_path = self.image_root / filename if filename else None
            if image_path is not None and image_path.is_file():
                self.records.append((annotation, image_path))
            elif skip_missing_images:
                self.skipped_missing_images.append(str(annotation))
            else:
                missing = image_path or self.image_root / "<missing-filename>"
                raise FileNotFoundError(f"Image referenced by {annotation} is unavailable: {missing}")
        self.annotations = [annotation for annotation, _ in self.records]

    def __len__(self): return len(self.annotations)

    def __getitem__(self, index):
        annotation_path, image_path = self.records[index]
        ann = parse_voc_annotation(annotation_path)
        image = Image.open(image_path).convert("RGB")
        instances = ann["instances"]
        boxes = torch.tensor([x["box"] for x in instances], dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": boxes, "labels": torch.ones(len(instances), dtype=torch.int64),
                  "area": torch.tensor([x["area"] for x in instances], dtype=torch.float32),
                  "iscrowd": torch.tensor([x["difficult"] for x in instances], dtype=torch.int64),
                  "image_id": torch.tensor([index]), "supervision_type": self.supervision_type,
                  "original_labels": [x["label"] for x in instances]}
        if self.transform: image, target = self.transform(image, target)
        return image, target
