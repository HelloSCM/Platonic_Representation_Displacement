"""Command-line training, checkpoint resumption, and trajectory plots."""

import argparse
import sys

from .launch import add_device_args, launch

CLIP = "vit_base_patch16_clip_224.laion2b"
BASE = "vit_base_patch16_224.augreg_in1k"
SMALL = "vit_small_patch16_224"


def add_training(parser, mode):
    """Add training, checkpoint, and geometry evaluation options to an argument parser."""
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output", required=True, help="Empty output directory.")
    parser.add_argument(
        "--student", default=SMALL if mode == "scratch" else CLIP if mode == "finetune" else BASE
    )
    parser.add_argument("--teacher", default=BASE if mode == "scratch" else CLIP)
    parser.add_argument("--student-checkpoint", help="Local initial student weights.")
    parser.add_argument("--teacher-checkpoint", help="Local frozen teacher weights.")
    parser.add_argument("--epochs", type=int, default=100 if mode == "scratch" else 10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048 if mode == "scratch" else 128,
        help="Per-GPU training batch size.",
    )
    parser.add_argument("--workers", type=int, default=8)
    add_device_args(parser)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr",
        type=float,
        default=None if mode == "scratch" else 5e-5 if mode == "finetune" else 1e-4,
    )
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--base-batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument(
        "--schedule",
        choices=("constant", "cosine"),
        default="cosine" if mode == "scratch" else "constant",
    )
    parser.add_argument("--warmup-epochs", type=float, default=5.0 if mode == "scratch" else 0.0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0 if mode == "scratch" else 0.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument(
        "--metric-dataset", choices=("flickr30k", "folder", "manifest"), default="flickr30k"
    )
    parser.add_argument(
        "--metric-input", help="Flickr30k captions, an image folder, or a JSONL manifest."
    )
    parser.add_argument("--metric-image-root", help="Image root for caption or manifest records.")
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument(
        "--metric-layers", default="all", help="all, last, or comma-separated block indices."
    )
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--metric-device", default="cpu", help="Device for the OPA solve.")
    parser.add_argument(
        "--dim",
        type=int,
        help="PCA whitening dimension; defaults to min(500, student width, reference width).",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--fit-size", type=int, default=10000)
    parser.add_argument("--eval-size", type=int, default=10000)
    parser.add_argument("--pairs", type=int, default=10000)
    parser.add_argument(
        "--save-features", action="store_true", help="Retain current-model feature stores."
    )
    parser.add_argument("--save-transforms", action="store_true")
    parser.add_argument("--plot", action="store_true")
    if mode == "finetune":
        parser.set_defaults(loss="ce")
    else:
        parser.add_argument(
            "--loss",
            choices=("ce", "fkd", "rkd") if mode == "scratch" else ("fkd", "rkd"),
            default="ce" if mode == "scratch" else None,
            required=mode == "distill",
        )


def main(argv=None):
    """Parse command-line arguments and launch a training dynamics operation."""
    parser = argparse.ArgumentParser(
        description="Training dynamics for classification, FKD, and RKD."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for mode in ("finetune", "distill", "scratch"):
        add_training(commands.add_parser(mode), mode)
    resume = commands.add_parser(
        "resume", help="Resume with saved configuration and optimizer state."
    )
    resume.add_argument("--checkpoint", required=True)
    add_device_args(resume, default=None)
    resume.add_argument("--workers", type=int)
    resume.add_argument(
        "--output", help="Empty directory for a resumed run; defaults to the saved directory."
    )
    plot = commands.add_parser("plot", help="Plot saved geometry and accuracy trajectories.")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "plot":
        from .plotting import plot_run

        plot_run(args.input, args.output)
        return
    state = None
    if args.command == "resume":
        import torch

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        options = args
        args = argparse.Namespace(**state["args"])
        args.resume = options.checkpoint
        if options.device is not None:
            args.device = options.device
            if options.device == "cpu":
                args.gpus = None
        if options.gpus is not None:
            args.gpus = options.gpus
            args.device = "cpu" if options.device == "cpu" else "cuda"
        if options.workers is not None:
            args.workers = options.workers
        if options.output is not None:
            args.output = options.output
    else:
        args.resume = None
    if args.device == "cpu" and getattr(args, "gpus", None) is not None:
        parser.error(
            "--gpus selects CUDA devices; use --device cpu without --gpus for CPU training."
        )
    if args.command == "scratch" and args.student_checkpoint:
        parser.error("scratch starts from random weights; remove --student-checkpoint.")
    if not args.skip_metrics and not args.metric_input:
        parser.error("--metric-input is required unless --skip-metrics is specified.")
    if args.temperature <= 0 or args.epochs < 1 or args.save_every < 1:
        parser.error("temperature, epochs, and save-every must be positive.")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be nonnegative and smaller than epochs.")
    launch("dynamics", args, sys.argv[1:] if argv is None else argv)
    from .runner import run

    run(args, state)
