"""Optimize classification and distillation losses and evaluate classifiers."""

import math

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .losses import feature_loss, relation_loss


def learning_rate(step, steps, warmup, lr, min_lr, schedule):
    """Compute a warmup learning rate followed by a constant or cosine schedule."""
    if warmup and step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    progress = (step - warmup) / max(steps - warmup - 1, 1)
    return min_lr + (lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


def train_epoch(model, teacher, teacher_hooks, loader, optimizer, scaler, epoch, args, runtime):
    """Train one epoch with classification or distillation and aggregate loss statistics."""
    model.train()
    if teacher is not None:
        teacher.eval()
    loader.sampler.set_epoch(epoch)
    total = [0.0, 0.0, 0.0, 0]
    steps = len(loader)
    dtype = getattr(torch, args.dtype)
    for i, (views, target) in enumerate(
        tqdm(loader, desc=f"Epoch {epoch}", disable=not runtime.main)
    ):
        lr = learning_rate(
            (epoch - 1) * steps + i,
            args.epochs * steps,
            round(args.warmup_epochs * steps),
            args.lr,
            args.min_lr,
            args.schedule,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        images, target = views[0].to(runtime.device), target.to(runtime.device)
        teacher_features = None
        if args.loss != "ce":
            teacher_hooks.clear()
            with (
                torch.no_grad(),
                torch.autocast(runtime.device.type, dtype=dtype, enabled=dtype != torch.float32),
            ):
                teacher.forward_features(views[1].to(runtime.device))
                teacher_features = next(iter(teacher_hooks.features.values()))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(runtime.device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits, features, projected = model(images)
            ce = (
                F.cross_entropy(logits.float(), target)
                if logits is not None
                else features.new_zeros(())
            )
            kd = features.new_zeros(())
            if args.loss == "fkd":
                kd = feature_loss(projected, teacher_features)
            elif args.loss == "rkd":
                kd = relation_loss(features, teacher_features, args.temperature)
            loss = args.ce_weight * ce + args.distill_weight * kd
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        n = len(target)
        for j, value in enumerate((loss, ce, kd)):
            total[j] += value.detach().item() * n
        total[3] += n
    loss, ce, kd, n = runtime.sum(total)
    return {"loss": loss / n, "ce_loss": ce / n, "distill_loss": kd / n, "lr": lr}


@torch.inference_mode()
def accuracy(model, loader, device, dtype):
    """Compute top-one classification accuracy over a data loader."""
    model.eval()
    correct, count = 0, 0
    for images, target in loader:
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(images.to(device))
        correct += (logits.argmax(-1).cpu() == target).sum().item()
        count += len(target)
    return correct / count
