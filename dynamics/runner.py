"""Run fine-tuning, distillation, and supervised training from scratch."""

import csv
from copy import deepcopy
from pathlib import Path
from time import monotonic

import torch
from torchvision.datasets import ImageFolder

from metric.io import make_output, write_csv, write_json
from representation.extraction.common import set_seed

from .checkpoint import append_rows, restore_rng, save
from .data import make_training_data, make_validation_data
from .distributed import Runtime
from .engine import accuracy, train_epoch
from .evaluation import GeometryTracker
from .models import Student, block_hooks, build_backbone


def prepare_output(args, state, runtime):
    """Prepare output paths and retained training history for a fresh or resumed run."""
    out = Path(args.output).resolve()
    if runtime.main:
        if state is None or out != Path(state["args"]["output"]):
            make_output(out)
            if state:
                source = Path(state["args"]["output"])
                for name in ("history.csv", "geometry.csv"):
                    if (source / name).exists():
                        with (source / name).open(encoding="utf-8") as f:
                            rows = [
                                row
                                for row in csv.DictReader(f)
                                if int(row["epoch"]) <= state["epoch"]
                            ]
                        if rows:
                            write_csv(out / name, rows)
        elif (out / "history.csv").exists():
            with (out / "history.csv").open(encoding="utf-8") as f:
                latest = max(int(row["epoch"]) for row in csv.DictReader(f))
            if latest > state["epoch"]:
                raise ValueError(
                    "Output contains later epochs. Use --output for a new resume directory."
                )
    runtime.barrier()
    args.output = str(out)
    for name in (
        "train_dir",
        "val_dir",
        "student_checkpoint",
        "teacher_checkpoint",
        "metric_input",
        "metric_image_root",
    ):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if state is None:
        args.initial_checkpoint = str(out / "checkpoints" / "epoch_000.pt")
        args.reference_store = str(out / "reference")
    return out


def evaluate(epoch, student, val_loader, tracker, args, runtime, stats=None):
    """Evaluate classification and representation geometry on the primary process."""
    if not runtime.main:
        return
    top1 = ""
    if val_loader is not None:
        top1 = accuracy(student.backbone, val_loader, runtime.device, getattr(torch, args.dtype))
    if tracker:
        tracker.evaluate(student.backbone, epoch)
    stats = stats or {"loss": "", "ce_loss": "", "distill_loss": "", "lr": args.lr, "seconds": 0.0}
    append_rows(Path(args.output) / "history.csv", [{"epoch": epoch, **stats, "top1": top1}])
    print(f"Epoch {epoch}: {stats}, top1={top1}", flush=True)


def run(args, state=None):
    """Run fine-tuning, distillation, or training from scratch with checkpointing."""
    runtime = Runtime(args.device)
    if state and state["world_size"] != runtime.world_size:
        raise ValueError("Resume with the same number of training processes.")
    if runtime.device.type == "cpu" and args.dtype == "float16":
        raise ValueError("Use float32 or bfloat16 for CPU training.")
    out = prepare_output(args, state, runtime)
    set_seed(args.seed)
    classify = args.command in {"finetune", "scratch"}
    classes = len(ImageFolder(args.train_dir).classes) if classify else 0
    backbone = build_backbone(
        args.student,
        classes,
        checkpoint=args.student_checkpoint if state is None else None,
        pretrained=args.command != "scratch" and state is None,
    )
    reference = None
    if args.loss != "ce" or not args.skip_metrics:
        if args.command == "finetune":
            reference = (
                build_backbone(args.student, checkpoint=args.initial_checkpoint, pretrained=False)
                if state
                else deepcopy(backbone)
            )
            args.teacher = args.student
        else:
            reference = build_backbone(args.teacher, checkpoint=args.teacher_checkpoint)
        reference.eval().requires_grad_(False).to(runtime.device)
    teacher_dim = reference.num_features if args.loss == "fkd" else None
    student = Student(backbone, classify, teacher_dim).to(runtime.device)
    if state:
        student.load_state_dict(state["model"])
    if args.dim is None:
        args.dim = min(500, backbone.num_features, reference.num_features if reference else 500)
    if reference and args.dim > min(backbone.num_features, reference.num_features):
        raise ValueError("dim exceeds a model's block feature width.")
    train, loader, sampler = make_training_data(
        args, backbone, reference, runtime.world_size, runtime.rank
    )
    val_loader = (
        make_validation_data(args, backbone, train.class_to_idx)
        if runtime.main and classify
        else None
    )
    tracker = (
        GeometryTracker(args, backbone, reference, out)
        if runtime.main and not args.skip_metrics
        else None
    )
    teacher_hooks = block_hooks(reference, "last") if args.loss != "ce" else None
    if args.lr is None:
        args.lr = args.base_lr * args.batch_size * runtime.world_size / args.base_batch_size
    if runtime.main:
        write_json(
            out / "config.json",
            {
                **vars(args),
                "world_size": runtime.world_size,
                "global_batch_size": args.batch_size * runtime.world_size,
                "classes": train.class_to_idx,
                "evaluation_transform_policy": "fit_per_checkpoint",
            },
        )
    model = runtime.wrap(student)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
    )
    scaler = torch.amp.GradScaler(runtime.device.type, enabled=args.dtype == "float16")
    start = state["epoch"] if state else 0
    set_seed(args.seed + runtime.rank)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"][runtime.rank], loader, runtime.device)
    else:
        evaluate(0, student, val_loader, tracker, args, runtime)
        save(args.initial_checkpoint, student, optimizer, scaler, 0, args, runtime, loader)
    runtime.barrier()
    for epoch in range(start + 1, args.epochs + 1):
        before = monotonic()
        stats = train_epoch(
            model, reference, teacher_hooks, loader, optimizer, scaler, epoch, args, runtime
        )
        stats["seconds"] = monotonic() - before
        evaluate(epoch, student, val_loader, tracker, args, runtime, stats)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save(
                out / "checkpoints" / f"epoch_{epoch:03d}.pt",
                student,
                optimizer,
                scaler,
                epoch,
                args,
                runtime,
                loader,
            )
        runtime.barrier()
        if runtime.main and args.plot and tracker:
            from .plotting import plot_run

            plot_run(out, out / "dynamics.pdf")
        runtime.barrier()
    if teacher_hooks:
        teacher_hooks.close()
    student.close()
    runtime.close()
