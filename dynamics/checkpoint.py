"""Save resumable training states and backbone feature weights."""

import csv
import random
from pathlib import Path

import numpy as np
import torch


def append_rows(path, rows):
    """Append tabular records to a CSV file and write its header when needed."""
    if not rows:
        return
    path = Path(path)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def rng_state(loader, device):
    """Capture random generator states for training and data loading."""
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), *state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "loader": loader.generator.get_state(),
    }


def restore_rng(state, loader, device):
    """Restore random generator states for training and data loading."""
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    loader.generator.set_state(state["loader"])


def save(path, student, optimizer, scaler, epoch, args, runtime, loader):
    """Save model, optimizer, scaler, and per-process random states for resuming training."""
    states = runtime.gather(rng_state(loader, runtime.device))
    if runtime.main:
        backbone = {
            key: value
            for key, value in student.backbone.state_dict().items()
            if not key.startswith("head.")
        }
        payload = {
            "model": student.state_dict(),
            "state_dict": backbone,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "world_size": runtime.world_size,
            "rng": states,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
