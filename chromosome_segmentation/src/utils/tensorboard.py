"""Optional TensorBoard writer with a clear dependency error."""
from __future__ import annotations

from pathlib import Path


class TrainingTensorBoard:
    def __init__(self, log_dir: str | Path, enabled: bool):
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise RuntimeError("TensorBoard logging is enabled but tensorboard is not installed. Run `python -m pip install tensorboard` or disable training.tensorboard.enabled.") from exc
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_step(self, row: dict):
        if self.writer is None:
            return
        step = row["step"]
        self.writer.add_scalar("train/total_loss", row["loss"], step)
        for name in ("loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg", "gradient_norm"):
            if name in row:
                self.writer.add_scalar(f"train/{name}", row[name], step)
        if row.get("validation_loss") is not None:
            self.writer.add_scalar("validation/loss", row["validation_loss"], step)
        if "supervision_type" in row:
            self.writer.add_scalar(f"training_steps/{row['supervision_type']}", 1, step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
