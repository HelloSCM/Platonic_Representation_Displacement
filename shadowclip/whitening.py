"""Initialize full-dimensional ZCA transforms on ImageNet-1K training features."""

import json
from pathlib import Path

import numpy as np

from representation.preprocessing import Preprocessor

from .data import Cache


def fit(args):
    """Fit and save full-dimensional ZCA transforms for student and teacher text features."""
    cache = Cache(args.input)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    values = {}
    for name, key in (("student", "clip_text"), ("teacher", "teacher")):
        proc = Preprocessor(method="zca", eps=args.eps).fit(
            cache.arrays[key], batch_size=args.batch_size
        )
        values[f"{name}_mean"] = proc.mu.astype(np.float32)
        values[f"{name}_matrix"] = ((proc.basis / proc.scale) @ proc.basis.T).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "reference": "imagenet1k-train",
        "source": str(cache.path.resolve()),
        "n": cache.n,
        "eps": args.eps,
        "encoders": cache.metadata["encoders"],
    }
    with output.open("wb") as f:
        np.savez(f, **values, metadata=json.dumps(info))


def load(path, cache=None):
    """Load whitening parameters and optionally validate their encoder settings against a cache."""
    with np.load(path) as saved:
        values = {key: saved[key].copy() for key in saved.files if key != "metadata"}
        info = json.loads(str(saved["metadata"]))
    if cache is not None:
        for key in ("clip_text", "teacher"):
            ref, train = info["encoders"][key], cache.metadata["encoders"][key]
            for field in (
                "backend",
                "model",
                "pretrained",
                "bidirectional",
                "instruction",
                "max_length",
            ):
                if ref.get(field) != train.get(field):
                    raise ValueError(
                        f"Whitening and training extraction settings differ: {key}/{field}"
                    )
        if len(values["student_mean"]) != cache.arrays["clip_text"].shape[1]:
            raise ValueError("Student whitening dimension differs from the CLIP text features.")
        if len(values["teacher_mean"]) != cache.arrays["teacher"].shape[1]:
            raise ValueError("Teacher whitening dimension differs from the teacher features.")
    return values
