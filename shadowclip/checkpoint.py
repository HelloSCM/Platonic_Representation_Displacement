"""Save resumable adapter training and export teacher-free model weights."""

import csv
import random
from pathlib import Path

import numpy as np
import torch

from .models import ShadowCLIP


def append_row(path, row):
    """Append one record to a CSV file and write its header when needed."""
    path = Path(path)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rng_state(loader, device):
    """Capture random generator states for training and data loading."""
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), *state[2:]),
        "torch": torch.get_rng_state(),
        "loader": loader.generator.get_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, loader, device):
    """Restore random generator states for training and data loading."""
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"])
    loader.generator.set_state(state["loader"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def write(path, payload):
    """Save a PyTorch payload atomically through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save(path, model, optimizer, scaler, epoch, args, runtime, loader):
    """Save adapter training state and per-process random states for resuming training."""
    states = runtime.gather(rng_state(loader, runtime.device))
    if runtime.main:
        write(
            path,
            {
                "format_version": 1,
                "config": model.core.config,
                "core": model.core.state_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "world_size": runtime.world_size,
                "rng": states,
            },
        )


def load_core(path, device="cpu"):
    """Load trained ShadowCLIP adapters and projections on the requested device."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    core = ShadowCLIP(payload["config"])
    core.load_state_dict(payload["core"])
    return core.to(device).eval()


def export(core, path):
    """Export the model configuration and core weights for teacher-free inference."""
    write(
        path,
        {
            "format_version": 1,
            "config": core.config,
            "core": {key: value.detach().cpu() for key, value in core.state_dict().items()},
        },
    )
