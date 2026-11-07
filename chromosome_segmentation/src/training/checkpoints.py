"""Atomic, resumable training checkpoint helpers."""
from __future__ import annotations

import os
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, step, seed, supervision_steps=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "seed": seed,
             "torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if supervision_steps is not None: state["supervision_steps"] = supervision_steps
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
    if "torch_rng_state" in state: torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state: torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return state
