"""Evaluate adapters against cached downstream benchmark features."""

import json
from pathlib import Path

import numpy as np
import torch

from . import benchmarks
from .data import Cache
from .models import ShadowCLIP


@torch.inference_mode()
def encode_cache(core, cache, modality, device, batch_size):
    """Encode cached image or text features into normalized ShadowCLIP embeddings in batches."""
    outputs = []
    core.eval()
    keys = ["clip_text"] if modality == "text" else ["clip_vision"]
    if modality == "image" and core.vision is not None:
        keys.append("dino")
    for start in range(0, cache.n, batch_size):
        batch = {
            key: torch.from_numpy(np.array(cache.arrays[key][start : start + batch_size]))
            .float()
            .to(device)
            for key in keys
        }
        features = (
            core.encode_text(batch["clip_text"])
            if modality == "text"
            else core.encode_image(batch["clip_vision"], batch.get("dino"))
        )
        outputs.append(features)
    return torch.cat(outputs)


def evaluate_spec(core, spec, device, batch_size=1024, chunk_size=512, baseline=False):
    """Evaluate cached classification, retrieval, or MMVP tasks from a JSON specification."""
    spec = Path(spec)
    tasks = json.loads(spec.read_text())
    scores = {}
    for task in tasks:
        name = task["name"]
        if name in scores:
            raise ValueError(f"Duplicate evaluation name: {name}")
        images = Cache(spec.parent / task["images"])
        texts = Cache(spec.parent / task["texts"])
        model = core
        if baseline:
            config = {**core.config, "variant": "clip"}
            model = ShadowCLIP(config, {**images.assets, **texts.assets}).to(device).eval()
        for key, cache in (("clip_vision", images), ("dino", images), ("clip_text", texts)):
            if key == "dino" and model.vision is None:
                continue
            source = core.config["encoders"][key]
            target = cache.metadata["encoders"][key]
            for field in ("backend", "model", "pretrained", "checkpoint", "pool"):
                if source.get(field) != target.get(field):
                    raise ValueError(
                        f"Evaluation encoder differs from training: {name}/{key}/{field}"
                    )
        image_features = encode_cache(model, images, "image", device, batch_size)
        text_features = encode_cache(model, texts, "text", device, batch_size)
        metric = getattr(benchmarks, task["task"])
        kwargs = {} if task["task"] == "mmvp" else {"chunk_size": chunk_size}
        scores[name] = metric(
            image_features, text_features, images.records(), texts.records(), **kwargs
        )
        if task["task"] == "classification":
            average = task.get("average", "sample")
            if average not in {"sample", "class"}:
                raise ValueError("Classification average must be sample or class.")
            key = "top1" if average == "sample" else "mean_class_top1"
            scores[name]["accuracy"] = scores[name][key]
        print(json.dumps({name: scores[name]}))
    return scores
