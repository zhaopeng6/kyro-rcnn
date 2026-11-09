#!/usr/bin/env python3
"""Generate filtered, box-prompted micro-sam pseudo masks for Lowq."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"chromosome_segmentation"))
from src.datasets.annotations import parse_voc_annotation
from src.utils.config import dataset_path
from src.pseudo_masks.quality import score_mask

def xyxy_to_micro_sam_yxyx(box):
 x1,y1,x2,y2=box;return np.asarray([y1,x1,y2,x2])

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--max-images',type=int);a=p.parse_args()
 cfg=yaml.safe_load(Path(a.config).read_text()); d=cfg['datasets']['lowq']; settings=cfg['pseudo_masks']
 try:
  from micro_sam import util,prompt_based_segmentation
 except ModuleNotFoundError as exc: raise RuntimeError('Experiment D requires micro-sam. Install its documented package and model dependencies before generation.') from exc
 predictor=util.get_sam_model(model_type=settings.get('model_type','vit_b'), device=cfg['training'].get('device','cpu'))
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);records=[]; xmls=sorted(dataset_path(c, d['annotations']).glob('*.xml'))[:a.max_images]
 for xml in xmls:
  ann=parse_voc_annotation(xml); image_path=dataset_path(c, d['images'])/ann['filename']
  if not image_path.exists(): continue
  image=np.asarray(Image.open(image_path).convert('RGB')); predictor.set_image(image)
  accepted_masks=[];accepted_boxes=[];quality=[]
  for item in ann['instances']:
   # Lowq/VOC uses xyxy; micro-sam's prompt helper expects yxyx and
   # converts it internally to SAM's xyxy convention.
   micro_sam_box=xyxy_to_micro_sam_yxyx(item['box'])
   masks,scores,_=prompt_based_segmentation.segment_from_box(predictor,micro_sam_box,multimask_output=True,return_all=True)
   best=int(np.argmax(scores)); mask=np.asarray(masks[best],dtype=bool); q=score_mask(mask,item['box'],float(scores[best]),**settings.get('filter',{}));quality.append(q)
   if q['accepted']: accepted_masks.append(mask);accepted_boxes.append(item['box'])
  if accepted_masks:
   filename=f'{xml.stem}.npz';np.savez_compressed(out/filename,masks=np.stack(accepted_masks).astype(np.uint8));records.append({'image_path':str(image_path),'source_xml':str(xml),'mask_file':filename,'boxes':accepted_boxes,'accepted':True,'quality':quality})
 (out/'manifest.json').write_text(json.dumps({'records':records,'generator':'micro-sam box prompts','filter':settings.get('filter',{})},indent=2)+'\n');print(f'Wrote {len(records)} accepted image records to {out}')
if __name__=='__main__':main()
