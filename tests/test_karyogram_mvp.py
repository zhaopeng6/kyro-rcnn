import numpy as np
from PIL import Image

from src.karyogram.classifier import class_name, label_index
from src.karyogram.crops import masked_crop
from src.karyogram.layout import render_karyogram


def test_autokary_class_mapping():
    assert label_index("1") == 0
    assert label_index("22") == 21
    assert label_index("23") == 22 and class_name(22) == "X"
    assert label_index("24") == 23 and class_name(23) == "Y"
    assert label_index("-1") is None


def test_masked_crop_removes_unmasked_pixels():
    image = Image.fromarray(np.full((10, 10, 3), 40, dtype=np.uint8))
    mask = np.zeros((10, 10), dtype=bool); mask[3:7, 3:7] = True
    crop = np.asarray(masked_crop(image, mask, padding=1))
    assert (crop == 40).all(axis=-1).any()
    assert (crop == 255).all(axis=-1).any()


def test_karyogram_keeps_all_instances(tmp_path):
    crop = Image.new("RGB", (20, 30), "black")
    items = [{"chromosome": "1", "classification_score": .9, "crop": crop}, {"chromosome": "1", "classification_score": .8, "crop": crop}, {"chromosome": "1", "classification_score": .7, "crop": crop}]
    result = render_karyogram(items, tmp_path / "karyogram.png")
    assert result["counts"]["1"] == 3
    assert (tmp_path / "karyogram.png").exists()
