from __future__ import annotations

import json
from pathlib import Path
import torch
from src.utils.tensorboard import TrainingTensorBoard
from .checkpoints import save_checkpoint


def validation_loss(model, loader, device, batches=1):
    """Mean detection/mask loss on held-out polygon batches, without gradients."""
    was_training = model.training; model.train()  # torchvision detection returns losses only in train mode
    values, iterator = [], iter(loader)
    with torch.no_grad():
        for _ in range(batches):
            try: images, targets = next(iterator)
            except StopIteration: break
            images = [image.to(device) for image in images]
            targets = [{key: (value.to(device) if torch.is_tensor(value) else value) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            total = sum(losses.values())
            if not torch.isfinite(total): raise FloatingPointError(f"Non-finite validation loss: {total.item()}")
            values.append(float(total.detach().cpu()))
    model.train(was_training)
    return sum(values) / len(values) if values else None


def train_polygon_baseline(model, loader, optimizer, device, steps, checkpoint_dir: str | Path, seed=42, max_grad_norm=1.0,
                           validation_loader=None, validation_interval=1, validation_batches=1, tensorboard_enabled=False, tensorboard_log_dir=None,
                           start_step=0, checkpoint_interval=10):
    """Train only true polygon-mask batches; never accepts box-only samples."""
    checkpoint_dir = Path(checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.to(device); model.train(); history = []
    metrics_path = checkpoint_dir / "metrics.jsonl"
    if start_step == 0: metrics_path.write_text("")
    tensorboard = TrainingTensorBoard(tensorboard_log_dir or checkpoint_dir / "tensorboard", tensorboard_enabled)
    iterator = iter(loader)
    if steps <= start_step: raise ValueError(f"total steps ({steps}) must exceed resumed step ({start_step})")
    for step in range(start_step + 1, steps + 1):
        try: images, targets = next(iterator)
        except StopIteration: iterator = iter(loader); images, targets = next(iterator)
        if any(t.get("supervision_type") != "polygon" or "masks" not in t for t in targets):
            raise ValueError("Phase 2 accepts only polygon targets with real masks")
        images = [image.to(device) for image in images]
        targets = [{key: (value.to(device) if torch.is_tensor(value) else value) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets); loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            detail = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
            raise FloatingPointError(f"Non-finite loss at step {step}; components={detail}; image_ids={[t['image_id'].item() for t in targets]}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if max_grad_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        else:
            gradient_norm = torch.tensor(float("nan"), device=device)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach().cpu()), "gradient_norm": float(gradient_norm.detach().cpu()), **{name: float(value.detach().cpu()) for name, value in loss_dict.items()}}
        if validation_loader is not None and step % validation_interval == 0:
            row["validation_loss"] = validation_loss(model, validation_loader, device, validation_batches)
        history.append(row)
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        tensorboard.log_step(row)
        print(json.dumps(row), flush=True)
        if step % checkpoint_interval == 0 or step == steps:
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, step, seed)
    (checkpoint_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
    tensorboard.close()
    return history
