#!/usr/bin/env python3
"""Render accepted Experiment-D pseudo masks over their original images."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

COLORS=[(230,57,70),(29,161,242),(42,157,143),(244,162,97),(131,56,236)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
 manifest=Path(a.manifest);records=json.loads(manifest.read_text())['records'];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 for index,record in enumerate(records[:a.limit]):
  image=Image.open(record['image_path']).convert('RGB'); base=np.asarray(image).copy();overlay=base.copy();masks=np.load(manifest.parent/record['mask_file'])['masks'].astype(bool)
  for i,mask in enumerate(masks): overlay[mask]=COLORS[i%len(COLORS)]
  right=Image.blend(Image.fromarray(base),Image.fromarray(overlay),.45);draw=ImageDraw.Draw(right)
  for i,box in enumerate(record['boxes']): draw.rectangle(tuple(box),outline=COLORS[i%len(COLORS)],width=2);draw.text((box[0],box[1]),str(i),fill=(255,255,0))
  canvas=Image.new('RGB',(image.width*2,image.height));canvas.paste(image,(0,0));canvas.paste(right,(image.width,0));canvas.save(out/f'{Path(record["image_path"]).stem}_pseudo.png')
 print(f'Wrote {min(len(records),a.limit or len(records))} pseudo-mask previews to {out}')
if __name__=='__main__':main()
