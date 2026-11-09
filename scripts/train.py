#!/usr/bin/env python3
"""Phase-2 experiment-A trainer (polygon-only)."""
import argparse, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.datasets.autokary import AutoKaryDataset
from src.datasets.box_dataset import LowqBoxDataset
from src.datasets.collate import detection_collate
from src.datasets.transforms import ToTensor, build_training_transform
from src.models.mask_rcnn import build_mask_rcnn
from src.training.trainer import train_polygon_baseline
from src.training.mixed_trainer import train_mixed_supervision
from src.training.checkpoints import load_checkpoint
from src.utils.config import dataset_path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--steps', type=int, default=None); p.add_argument('--resume'); p.add_argument('--checkpoint-interval', type=int, default=None); args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); training=cfg['training']
    torch.manual_seed(training.get('seed', 42)); device=torch.device(training.get('device', 'cpu'))
    train_tf=build_training_transform(cfg.get('augmentation', {}))
    dataset=AutoKaryDataset(dataset_path(cfg, cfg['datasets']['autokary']['root']), transform=train_tf)
    loader=DataLoader(dataset, batch_size=training['batch_size'], shuffle=True, num_workers=training.get('num_workers', 0), collate_fn=detection_collate)
    validation_root = cfg['datasets']['autokary'].get('validation_root')
    validation_loader = DataLoader(AutoKaryDataset(dataset_path(cfg, validation_root), transform=ToTensor()), batch_size=training.get('validation_batch_size', training['batch_size']), shuffle=False, num_workers=training.get('num_workers', 0), collate_fn=detection_collate) if validation_root else None
    model=build_mask_rcnn(cfg)
    if training.get('optimizer', 'sgd').lower() == 'adamw': optimizer=torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training.get('weight_decay', .0005))
    elif training.get('optimizer', 'sgd').lower() == 'sgd': optimizer=torch.optim.SGD(model.parameters(), lr=training['learning_rate'], momentum=training.get('momentum', .9), weight_decay=training.get('weight_decay', .0005))
    else: raise ValueError("training.optimizer must be 'sgd' or 'adamw'")
    resume_state = load_checkpoint(args.resume, model, optimizer, device) if args.resume else {}
    start_step = int(resume_state.get('step', 0)); checkpoint_interval = args.checkpoint_interval or training.get('checkpoint_interval', 10)
    if checkpoint_interval < 1: raise ValueError('checkpoint_interval must be at least 1')
    if args.resume: print(f"Resumed {args.resume} at completed step {start_step}")
    if cfg.get('model', {}).get('mixed_supervision', False):
        lowq = cfg['datasets']['lowq']
        box_dataset = LowqBoxDataset(dataset_path(cfg, lowq['annotations']), dataset_path(cfg, lowq['images']), transform=train_tf)
        if box_dataset.skipped_missing_images:
            print(f"Skipping {len(box_dataset.skipped_missing_images)} Lowq XML records with unavailable JPEGs")
        box_loader = DataLoader(box_dataset, batch_size=training['batch_size'], shuffle=True, num_workers=training.get('num_workers', 0), collate_fn=detection_collate)
        tensorboard = training.get('tensorboard', {})
        _, counts = train_mixed_supervision(model, loader, box_loader, optimizer, device, args.steps or training.get('total_steps', 100), Path(training.get('output_dir', 'outputs/experiment_b')), training.get('polygon_batch_probability', .5), training.get('seed', 42), training.get('gradient_clip_norm', 1.0), validation_loader, training.get('validation_interval', 1), training.get('validation_batches', 1), tensorboard.get('enabled', False), tensorboard.get('log_dir'), start_step, checkpoint_interval, resume_state.get('supervision_steps'))
        print(f"Completed mixed training; polygon steps={counts['polygon']}, box steps={counts['box']}")
        return
    tensorboard = training.get('tensorboard', {})
    history=train_polygon_baseline(model, loader, optimizer, device, args.steps or training.get('total_steps', 100), Path(training.get('output_dir', 'outputs/experiment_a')), training.get('seed', 42), training.get('gradient_clip_norm', 1.0), validation_loader, training.get('validation_interval', 1), training.get('validation_batches', 1), tensorboard.get('enabled', False), tensorboard.get('log_dir'), start_step, checkpoint_interval)
    print(f"Completed {len(history)} polygon-supervised optimizer steps; last loss={history[-1]['loss']:.4f}")
if __name__ == '__main__': main()
