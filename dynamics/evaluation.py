"""Track held-out layer geometry against one frozen reference model."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader

from metric.core import METRICS
from metric.parallelism.pipeline import run as run_parallelism
from representation.data import load_records
from representation.extraction.common import to_numpy
from representation.extraction.pipeline import ImageDataset
from representation.storage import FeatureWriter

from .checkpoint import append_rows
from .models import block_hooks


@torch.inference_mode()
def extract(model, records, root, output, args, name, epoch=None):
    """Extract and save ordered block features for geometry evaluation."""
    model.eval()
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    loader = DataLoader(
        ImageDataset(records, root, transform),
        batch_size=args.metric_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )
    hooks = block_hooks(model, args.metric_layers)
    writer = FeatureWriter(
        output,
        records,
        {
            "model": name,
            "modality": "vision",
            "dataset": args.metric_dataset,
            "epoch": epoch,
            "layers": args.metric_layers,
            "normalized": False,
            "random": args.command == "scratch" and epoch == 0,
            "pool": "cls" if model.global_pool == "token" else "mean",
            "data_config": config,
        },
    )
    dtype = getattr(torch, args.dtype)
    device = next(model.parameters()).device
    for images in loader:
        hooks.clear()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            model.forward_features(images.to(device))
        writer.write(to_numpy(hooks.features))
    writer.finish()
    hooks.close()


def summarize(rows, depth_a, depth_b, epoch):
    """Pair layers by relative depth and average within thirds of the student."""
    names_a = sorted({row["layer_a"] for row in rows})
    names_b = sorted({row["layer_b"] for row in rows})
    groups = {"early": [], "middle": [], "deep": []}
    lookup = {(row["layer_a"], row["layer_b"]): row for row in rows}
    for a in names_a:
        index = int(a.removeprefix("layer_"))
        position = index / max(depth_a - 1, 1)
        b = min(
            names_b,
            key=lambda name: abs(int(name.removeprefix("layer_")) / max(depth_b - 1, 1) - position),
        )
        group = ("early", "middle", "deep")[min(3 * index // depth_a, 2)]
        groups[group].append(lookup[a, b])
    return [
        {
            "epoch": epoch,
            "stage": stage,
            "n_layer_pairs": len(values),
            **{metric: float(np.mean([row[metric] for row in values])) for metric in METRICS},
        }
        for stage, values in groups.items()
        if values
    ]


class GeometryTracker:
    """Track a student's held-out representation geometry against a frozen reference."""

    def __init__(self, args, student, reference, output):
        """Prepare evaluation records and cache the frozen reference model's features."""
        self.args, self.reference, self.output = args, reference, Path(output)
        self.records = load_records(
            args.metric_dataset, args.metric_input, caption_index=args.caption_index
        )
        self.root = args.metric_input if args.metric_dataset == "folder" else args.metric_image_root
        if self.root is None:
            raise ValueError("--metric-image-root is required for Flickr30k or manifest metrics.")
        if len(self.records) < args.fit_size + args.eval_size:
            raise ValueError("The metric dataset has fewer items than fit-size + eval-size.")
        if not 0 < args.dim < args.fit_size:
            raise ValueError("dim must be positive and smaller than fit-size.")
        self.depth_a, self.depth_b = len(student.blocks), len(reference.blocks)
        self.reference_store = Path(args.reference_store)
        if not self.reference_store.exists():
            extract(reference, self.records, self.root, self.reference_store, args, args.teacher)

    def evaluate(self, model, epoch):
        """Compare checkpoint features with the reference and record layer-stage metrics."""
        args = self.args
        with TemporaryDirectory(prefix="dynamics-") as temporary:
            current = (
                self.output / "features" / f"epoch_{epoch:03d}"
                if args.save_features
                else Path(temporary) / "current"
            )
            extract(model, self.records, self.root, current, args, args.student, epoch)
            output = self.output / "metrics" / f"epoch_{epoch:03d}"
            rows = run_parallelism(
                Namespace(
                    a=str(current),
                    b=str(self.reference_store),
                    output=str(output),
                    level="sample",
                    concept_map=None,
                    layers_a="all",
                    layers_b="all",
                    dim=args.dim,
                    eps=args.eps,
                    fit_size=args.fit_size,
                    eval_size=args.eval_size,
                    pairs=args.pairs,
                    seed=args.seed,
                    device=args.metric_device,
                    batch_size=4096,
                    plot=False,
                    save_transforms=args.save_transforms,
                )
            )
        summary = summarize(rows, self.depth_a, self.depth_b, epoch)
        append_rows(self.output / "geometry.csv", summary)
        return summary
