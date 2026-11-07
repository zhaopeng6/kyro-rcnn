import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'chromosome_segmentation'))
from src.pseudo_masks.quality import score_mask

def test_rejects_mask_that_leaks_outside_box():
 mask=np.zeros((10,10),bool);mask[1:9,1:9]=1
 result=score_mask(mask,[3,3,7,7],.9,max_outside_fraction=.05)
 assert not result['accepted']

def test_accepts_compact_box_constrained_mask():
 mask=np.zeros((10,10),bool);mask[3:6,3:6]=1
 result=score_mask(mask,[3,3,7,7],.9)
 assert result['accepted']
