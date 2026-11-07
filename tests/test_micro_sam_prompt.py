import importlib.util
from pathlib import Path

def test_lowq_xyxy_is_converted_to_micro_sam_yxyx():
 spec=importlib.util.spec_from_file_location('generator',Path('scripts/generate_pseudo_masks.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 assert module.xyxy_to_micro_sam_yxyx([10,20,30,40]).tolist()==[20,10,40,30]
