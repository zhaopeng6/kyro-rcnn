import json
from pathlib import Path

from src.datasets.annotations import parse_labelme_annotation, parse_voc_annotation


def test_labelme_polygon_is_tight_mask_box(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"imageWidth": 10, "imageHeight": 8, "shapes": [{"label": "X", "shape_type": "polygon", "points": [[2, 1], [7, 1], [7, 5]]}]}))
    result = parse_labelme_annotation(path)
    assert result["instances"][0]["box"] == [2.0, 1.0, 8.0, 6.0]
    assert result["instances"][0]["class_id"] == 23


def test_voc_box_is_not_mask(tmp_path: Path):
    path = tmp_path / "x.xml"
    path.write_text("<annotation><filename>x.jpg</filename><size><width>10</width><height>8</height></size><object><name>A1</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>7</ymax></bndbox></object></annotation>")
    result = parse_voc_annotation(path)
    assert result["instances"][0]["box"] == [1.0, 2.0, 5.0, 7.0]
    assert "mask" not in result["instances"][0]
