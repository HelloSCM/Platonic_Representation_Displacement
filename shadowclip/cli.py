"""Command-line entry points for ShadowCLIP experiments."""

import argparse
import json
import sys
from pathlib import Path

from .launch import add_device_args, launch


def parser():
    """Build the argument parser for ShadowCLIP caching, training, and evaluation."""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="Align and merge raw representation caches.")
    for key in ("clip-vision", "dino", "clip-text", "teacher"):
        cache.add_argument(
            f"--{key}", help="Feature store extracted with representation --layers final."
        )
    cache.add_argument("--output", required=True)
    cache.add_argument("--batch-size", type=int, default=4096)
    cache.add_argument("--save-dtype", choices=("float16", "float32"), default="float32")
    cache.add_argument("--logit-scale", type=float, help="Original CLIP log-scale parameter.")

    white = commands.add_parser("whiten", help="Fit ZCA on ImageNet-1K training-set text features.")
    white.add_argument(
        "--input", required=True, help="Paired CLIP text and teacher cache from ImageNet-1K train."
    )
    white.add_argument("--output", required=True, help="Output .npz file.")
    white.add_argument("--eps", type=float, default=1e-6)
    white.add_argument("--batch-size", type=int, default=4096)

    train = commands.add_parser("train", help="Train from cached CC12M shards.")
    train.add_argument("--cache", nargs="+", help="One or more paired training cache directories.")
    train.add_argument("--whitening", help="ImageNet-1K training-set whitening .npz.")
    train.add_argument("--output")
    train.add_argument(
        "--resume", help="Resume an epoch checkpoint, including optimizer and RNG states."
    )
    train.add_argument(
        "--variant", choices=("shadowclip", "clip-ft", "no-rkd"), default="shadowclip"
    )
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=8192, help="Batch size per GPU.")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--warmup", type=float, default=0.2)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--grad-clip", type=float, default=1.0, help="Set to zero to disable.")
    train.add_argument("--hidden-ratio", type=float, default=2)
    train.add_argument(
        "--temperature", type=float, default=0.1, help="RKD distribution temperature."
    )
    train.add_argument("--rkd-weight", type=float, default=1.0)
    train.add_argument(
        "--chunk-size", type=int, default=128, help="Rows per global similarity block."
    )
    train.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--eval-spec", help="JSON list of downstream evaluation tasks.")
    train.add_argument("--eval-batch-size", type=int, default=1024)
    add_device_args(train)

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate cached classification, retrieval, and MMVP tasks."
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--spec", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int, default=1024)
    evaluate.add_argument("--chunk-size", type=int, default=512)
    evaluate.add_argument(
        "--baseline", action="store_true", help="Use the original cached CLIP projections."
    )
    add_device_args(evaluate)

    export = commands.add_parser(
        "export", help="Export adapter weights without training-only whitening."
    )
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    return root


def main(argv=None):
    """Parse command-line arguments and run the selected ShadowCLIP operation."""
    argv = sys.argv[1:] if argv is None else argv
    root = parser()
    args = root.parse_args(argv)
    if args.command == "train":
        from .train import restore_args

        restore_args(args, argv)
        launch("shadowclip", args, argv)
        from .train import run

        run(args)
    elif args.command == "cache":
        from .data import pack

        pack(args)
    elif args.command == "whiten":
        from .whitening import fit

        fit(args)
    elif args.command == "evaluate":
        if args.gpus and len(args.gpus.split(",")) != 1:
            root.error("Evaluation uses one GPU; pass a single index to --gpus.")
        launch("shadowclip", args, argv)
        from .checkpoint import load_core
        from .evaluation import evaluate_spec

        core = load_core(args.checkpoint, args.device)
        scores = evaluate_spec(
            core, args.spec, args.device, args.batch_size, args.chunk_size, args.baseline
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(scores, indent=2) + "\n")
    else:
        from .checkpoint import export, load_core

        export(load_core(args.checkpoint), args.output)
