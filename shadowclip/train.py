"""Train lightweight adapters from frozen cached representations."""

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from . import checkpoint, whitening
from .data import CachedDataset, batch_identity, model_config
from .distributed import Runtime
from .losses import Objectives
from .models import ShadowCLIP, TrainingModel


def learning_rate(step, total, lr, warmup):
    """Compute a linear warmup followed by a cosine learning rate schedule."""
    warm = int(total * warmup)
    if step < warm:
        return lr * (step + 1) / max(warm, 1)
    progress = (step - warm) / max(total - warm - 1, 1)
    return lr * (1 + math.cos(math.pi * min(progress, 1))) / 2


def restore_args(args, argv):
    """Restore run settings before configuring visible training devices."""
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        explicit = {
            token.partition("=")[0].removeprefix("--").replace("-", "_")
            for token in argv
            if token.startswith("--")
        }
        allowed = {"resume", "output", "gpus", "device", "workers", "epochs", "eval_spec"}
        changed = explicit - allowed
        if changed:
            raise ValueError(
                f"Resume restores training settings; unsupported overrides: {sorted(changed)}"
            )
        for key, value in payload["args"].items():
            if key not in explicit and key != "resume":
                setattr(args, key, value)
    if not args.cache or not args.output:
        raise ValueError("Provide --cache and --output, or --resume.")
    if args.variant == "shadowclip" and not args.whitening and not args.resume:
        raise ValueError("Full ShadowCLIP requires ImageNet-1K training-set --whitening.")
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.warmup < 1:
        raise ValueError("Use positive epochs/batch size and a warmup fraction in [0, 1).")


def train_epoch(model, loss_fn, loader, optimizer, scaler, args, runtime, epoch):
    """Train adapters for one epoch and aggregate contrastive and distillation losses."""
    model.train()
    loader.sampler.set_epoch(epoch)
    stats = np.zeros(4, dtype=np.float64)
    dtype = getattr(torch, args.dtype)
    for index, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", disable=not runtime.main)):
        step = (epoch - 1) * len(loader) + index
        lr = learning_rate(step, args.epochs * len(loader), args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = {key: value.to(runtime.device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(runtime.device.type, dtype=dtype, enabled=dtype != torch.float32):
            out = model(batch)
        core = model.module.core if hasattr(model, "module") else model.core
        with torch.autocast(runtime.device.type, enabled=False):
            loss, contrastive, rkd = loss_fn(out, core.logit_scale, runtime.rank)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; check cached features and whitening.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        n = len(batch["clip_text"])
        stats += np.array([loss.item(), contrastive.item(), rkd.item(), 1]) * n
    values = runtime.sum(stats.tolist())
    return {
        "epoch": epoch,
        "loss": values[0] / values[3],
        "contrastive": values[1] / values[3],
        "rkd": values[2] / values[3],
        "lr": lr,
    }


def run(args):
    """Train or resume ShadowCLIP adapters with distributed data and checkpointing."""
    runtime = Runtime(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = CachedDataset(args.cache, args.variant)
    first = dataset.shards[0]
    config = model_config(first, args.variant, args.hidden_ratio)
    payload = (
        torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    )
    if payload and payload["world_size"] != runtime.world_size:
        raise ValueError("Resume requires the original number of training processes.")
    white = None
    if args.variant == "shadowclip":
        white = (
            whitening.load(args.whitening, first)
            if not payload
            else {
                f"{name}_{part}": payload["model"][f"{name}_white.{part}"]
                for name in ("student", "teacher")
                for part in ("mean", "matrix")
            }
        )
    core = ShadowCLIP(config, first.assets)
    student = TrainingModel(core, white).to(runtime.device)
    if payload:
        if config != payload["config"]:
            raise ValueError("Resume cache configuration differs from the saved model.")
        student.load_state_dict(payload["model"])
    sampler = DistributedSampler(
        dataset,
        num_replicas=runtime.world_size,
        rank=runtime.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        collate_fn=batch_identity,
        pin_memory=runtime.device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + runtime.rank),
    )
    if not len(loader):
        raise ValueError("The cache must contain at least one complete global batch.")
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(runtime.device.type, enabled=args.dtype == "float16")
    start = 1
    if payload:
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        restore = payload["rng"][runtime.rank]
        checkpoint.restore_rng(restore, loader, runtime.device)
        start = payload["epoch"] + 1
    if start > args.epochs:
        raise ValueError("The checkpoint already completed the requested number of epochs.")
    output = Path(args.output)
    if runtime.main:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()) and not payload:
            raise FileExistsError("Use an empty training output directory or --resume.")
        history = output / "history.csv"
        if payload and history.exists():
            with history.open(encoding="utf-8") as f:
                latest = max((int(row["epoch"]) for row in csv.DictReader(f)), default=0)
            if latest > payload["epoch"]:
                raise ValueError(
                    "Output contains later epochs; resume with a new --output directory."
                )
        settings = {
            **vars(args),
            "world_size": runtime.world_size,
            "global_batch_size": args.batch_size * runtime.world_size,
            "trainable_params": sum(p.numel() for p in student.parameters() if p.requires_grad),
            "trainable_names": [n for n, p in student.named_parameters() if p.requires_grad],
        }
        (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
        print(
            f"Global batch: {settings['global_batch_size']}; "
            f"trainable: {settings['trainable_params']:,}"
        )
    runtime.barrier()
    model = runtime.wrap(student)
    loss_fn = Objectives(args.chunk_size, args.temperature, args.rkd_weight)
    if args.eval_spec and not payload:
        evaluate_epoch(core, args, runtime, output, 0)
    for epoch in range(start, args.epochs + 1):
        row = train_epoch(model, loss_fn, loader, optimizer, scaler, args, runtime, epoch)
        if runtime.main:
            checkpoint.append_row(output / "history.csv", row)
            print(json.dumps(row))
        checkpoint.save(
            output / f"epoch_{epoch:03d}.pt",
            student,
            optimizer,
            scaler,
            epoch,
            args,
            runtime,
            loader,
        )
        if args.eval_spec:
            evaluate_epoch(core, args, runtime, output, epoch)
    if runtime.main:
        checkpoint.export(core, output / "shadowclip.pt")
    runtime.barrier()
    runtime.close()


def evaluate_epoch(core, args, runtime, output, epoch):
    """Evaluate the current adapters on the primary process and synchronize all processes."""
    if runtime.main:
        from .evaluation import evaluate_spec

        scores = evaluate_spec(core, args.eval_spec, runtime.device, args.eval_batch_size)
        (output / f"evaluation_{epoch:03d}.json").write_text(json.dumps(scores, indent=2) + "\n")
    runtime.barrier()
