from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class PseudoMaskLowqDataset(Dataset):
    """Accepted pseudo masks only; original XML boxes remain untouched."""
    supervision_type = "polygon"
    def __init__(self, manifest, transform=None):
        self.manifest, self.transform = Path(manifest), transform
        records = json.loads(self.manifest.read_text())["records"]
        self.records = [record for record in records if record["accepted"]]
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]; image = Image.open(record["image_path"]).convert("RGB")
        masks = np.load(self.manifest.parent / record["mask_file"])["masks"].astype(np.uint8)
        boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1,4)
        target = {"boxes": boxes, "labels": torch.ones(len(boxes),dtype=torch.int64), "masks": torch.from_numpy(masks),
                  "area": torch.tensor(masks.reshape(len(masks),-1).sum(1),dtype=torch.float32), "iscrowd":torch.zeros(len(boxes),dtype=torch.int64),
                  "image_id":torch.tensor([index]),"supervision_type":"polygon","mask_source":"pseudo"}
        if self.transform: image,target=self.transform(image,target)
        return image,target
