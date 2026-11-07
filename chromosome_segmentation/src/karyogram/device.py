"""Small, explicit device validation for command-line MVP tools."""
from __future__ import annotations
import torch


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch installation; use --device cpu or install an MPS-enabled PyTorch build.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    return device
