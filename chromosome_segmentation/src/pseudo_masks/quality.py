from __future__ import annotations
import numpy as np

def score_mask(mask, box, sam_score, min_area_fraction=.02, max_area_fraction=.95, max_outside_fraction=.05, min_sam_score=.0):
    x1, y1, x2, y2 = map(int, box); inside = np.zeros_like(mask, dtype=bool); inside[max(0,y1):min(mask.shape[0],y2), max(0,x1):min(mask.shape[1],x2)] = True
    area = int(mask.sum()); box_area = max(1, int(inside.sum())); outside = int((mask & ~inside).sum())
    fraction = area / box_area; outside_fraction = outside / max(1, area)
    accepted = min_area_fraction <= fraction <= max_area_fraction and outside_fraction <= max_outside_fraction and sam_score >= min_sam_score
    return {"accepted": bool(accepted), "sam_score": float(sam_score), "mask_area": area, "box_area": box_area, "box_fill_fraction": fraction, "outside_box_fraction": outside_fraction}
