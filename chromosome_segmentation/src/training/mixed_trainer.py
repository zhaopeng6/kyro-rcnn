"""Homogeneous-batch mixed-supervision training for Experiment B."""
from __future__ import annotations

import json
import random
from pathlib import Path
import torch
from .trainer import validation_loss
from src.utils.tensorboard import TrainingTensorBoard
from .checkpoints import save_checkpoint


class HomogeneousBatchMixer:
    def __init__(self, polygon_loader, box_loader, polygon_probability=.5, seed=42):
        if not 0 < polygon_probability < 1: raise ValueError("polygon_probability must be between 0 and 1")
        self.loaders = {"polygon": polygon_loader, "box": box_loader}; self.probability = polygon_probability; self.rng = random.Random(seed)
        self.iterators = {name: iter(loader) for name, loader in self.loaders.items()}

    def next(self):
        kind = "polygon" if self.rng.random() < self.probability else "box"
        try: return kind, next(self.iterators[kind])
        except StopIteration:
            self.iterators[kind] = iter(self.loaders[kind])
            return kind, next(self.iterators[kind])


def _to_device(target, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}


def train_mixed_supervision(model, polygon_loader, box_loader, optimizer, device, steps, checkpoint_dir, polygon_probability=.5, seed=42, max_grad_norm=1.0,
                            validation_loader=None, validation_interval=1, validation_batches=1, tensorboard_enabled=False, tensorboard_log_dir=None,
                            start_step=0, checkpoint_interval=10, initial_counts=None):
    checkpoint_dir = Path(checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).train(); mixer = HomogeneousBatchMixer(polygon_loader, box_loader, polygon_probability, seed)
    history, counts = [], dict(initial_counts or {"polygon": 0, "box": 0}); metrics_path = checkpoint_dir / "metrics.jsonl"
    if start_step == 0: metrics_path.write_text("")
    tensorboard = TrainingTensorBoard(tensorboard_log_dir or checkpoint_dir / "tensorboard", tensorboard_enabled)
    if steps <= start_step: raise ValueError(f"total steps ({steps}) must exceed resumed step ({start_step})")
    for step in range(start_step + 1, steps + 1):
        kind, (images, targets) = mixer.next()
        if any(target.get("supervision_type") != kind for target in targets): raise ValueError("Loader returned a non-homogeneous batch")
        if kind == "box" and any("masks" in target for target in targets): raise ValueError("Box batches must not contain masks")
        images = [image.to(device) for image in images]; targets = [_to_device(target, device) for target in targets]
        if kind == "box" and getattr(model.roi_heads, "weak_mask_supervision", {}).get("enabled", False):
            for image, target in zip(images, targets):
                target["weak_image"] = image
                target["weak_boxes"] = target["boxes"].clone()
        losses = model(images, targets)
        has_mask_loss = "loss_mask" in losses or "loss_mask_projection" in losses
        if kind == "polygon" and not has_mask_loss: raise RuntimeError("Polygon batch did not produce mask loss")
        if kind == "box" and getattr(model.roi_heads, "weak_mask_supervision", {}).get("enabled", False) != has_mask_loss: raise RuntimeError("Weak mask-loss router did not match supervision type")
        loss = sum(losses.values())
        if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True) if max_grad_norm is not None else torch.tensor(float("nan"), device=device)
        optimizer.step(); counts[kind] += 1
        row = {"step": step, "supervision_type": kind, "loss": float(loss.detach().cpu()), "gradient_norm": float(gradient_norm.detach().cpu()), **{k: float(v.detach().cpu()) for k, v in losses.items()}}
        if validation_loader is not None and step % validation_interval == 0:
            row["validation_loss"] = validation_loss(model, validation_loader, device, validation_batches)
        history.append(row)
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        tensorboard.log_step(row)
        print(json.dumps(row), flush=True)
        if step % checkpoint_interval == 0 or step == steps:
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, step, seed, counts)
    (checkpoint_dir / "train_history.json").write_text(json.dumps({"supervision_steps": counts, "steps": history}, indent=2) + "\n")
    tensorboard.close()
    return history, counts
