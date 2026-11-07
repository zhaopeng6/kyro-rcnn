import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.box_dataset import LowqBoxDataset


def _xml(filename):
    return f"<annotation><filename>{filename}</filename><size><width>10</width><height>8</height></size><object><name>chromosomes</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>7</ymax></bndbox></object></annotation>"


def test_lowq_skips_annotation_when_jpeg_is_missing(tmp_path):
    annotations, images = tmp_path / "annotations", tmp_path / "JEPG"
    annotations.mkdir(); images.mkdir()
    (annotations / "present.xml").write_text(_xml("present.jpg"))
    (annotations / "missing.xml").write_text(_xml("missing.jpg"))
    Image.new("L", (10, 8)).save(images / "present.jpg")
    dataset = LowqBoxDataset(annotations, images)
    assert len(dataset) == 1 and len(dataset.skipped_missing_images) == 1
    _, target = dataset[0]
    assert target["supervision_type"] == "box" and "masks" not in target
