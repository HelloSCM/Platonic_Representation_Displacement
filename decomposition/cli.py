"""Fit GPA decompositions, transform features, and evaluate rank sweeps."""

import argparse
import sys

from .launch import add_device_args, launch

RANKS = "5%,10%,20%,30%,40%,50%,60%,70%,80%,90%,100%"


def main(argv=None):
    """Parse command-line arguments and run the selected decomposition operation."""
    parser = argparse.ArgumentParser(description="Cross-model consensus decomposition using GPA.")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit PCA whitening, GPA, and shared/private subspaces.")
    fit.add_argument("--inputs", nargs="+", required=True, help="Matched training feature stores.")
    fit.add_argument("--names", nargs="+", help="Unique model names in input order.")
    fit.add_argument("--layer", default="final", help="final, last, or a block name/index.")
    fit.add_argument("--output", required=True)
    fit.add_argument("--dim", type=int, default=500)
    fit.add_argument("--eps", type=float, default=1e-6)
    fit.add_argument(
        "--fit-size", type=int, help="Matched items to fit; defaults to all training items."
    )
    fit.add_argument("--iterations", type=int, default=30)
    fit.add_argument("--tol", type=float, default=1e-7)
    fit.add_argument("--batch-size", type=int, default=4096)
    fit.add_argument("--seed", type=int, default=0)
    add_device_args(fit)

    transform = commands.add_parser(
        "transform", help="Write component coordinates or reconstructions."
    )
    transform.add_argument("--params", required=True)
    transform.add_argument("--model", required=True, help="Fitted model name.")
    transform.add_argument("--input", required=True)
    transform.add_argument("--output", required=True)
    transform.add_argument(
        "--component", choices=("original", "shared", "private"), default="shared"
    )
    transform.add_argument("--rank", type=int)
    transform.add_argument("--space", choices=("coordinates", "original"), default="coordinates")
    transform.add_argument("--batch-size", type=int, default=4096)

    analyze = commands.add_parser(
        "analyze", help="Evaluate Cross-Model and normalized HSIC rank curves."
    )
    analyze.add_argument("--params", required=True)
    analyze.add_argument(
        "--inputs", nargs="+", required=True, help="Held-out stores in fitted model order."
    )
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--ranks", default=RANKS)
    analyze.add_argument("--level", choices=("sample", "concept"), default="concept")
    analyze.add_argument(
        "--eval-size", type=int, default=10000, help="Maximum held-out items at sample level."
    )
    analyze.add_argument("--pairs", type=int, default=10000)
    analyze.add_argument("--hsic-samples", type=int, default=3000)
    analyze.add_argument("--batch-size", type=int, default=4096)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--plot", action="store_true")
    add_device_args(analyze)

    probe = commands.add_parser("probe", help="Train and evaluate linear classification heads.")
    probe.add_argument("--params", required=True)
    probe.add_argument(
        "--train-inputs", nargs="+", help="Training stores; defaults to the fitted stores."
    )
    probe.add_argument(
        "--inputs", nargs="+", required=True, help="Held-out stores in fitted model order."
    )
    probe.add_argument("--output", required=True)
    probe.add_argument("--ranks", default=RANKS)
    probe.add_argument("--protocol", choices=("refit", "frozen"), default="refit")
    probe.add_argument("--epochs", type=int, default=10)
    probe.add_argument("--batch-size", type=int, default=8192, help="Per-GPU batch size.")
    probe.add_argument("--workers", type=int, default=0)
    probe.add_argument("--lr", type=float, default=1e-3)
    probe.add_argument("--weight-decay", type=float, default=0.0)
    probe.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--plot", action="store_true")
    add_device_args(probe)

    plot = commands.add_parser("plot", help="Regenerate plots from analysis or probe CSVs.")
    plot.add_argument("--input", required=True)
    plot.add_argument("--kind", choices=("analysis", "probe"), required=True)
    args = parser.parse_args(argv)
    if args.command == "plot":
        from .plotting import plot_analysis, plot_probe

        (plot_analysis if args.kind == "analysis" else plot_probe)(args.input)
        return
    if args.command == "transform":
        if args.component != "original" and args.rank is None:
            parser.error("--rank is required for shared and private components.")
        from .pipeline import transform as run

        run(args)
        return
    if args.device == "cpu" and args.gpus is not None:
        parser.error("Use --device cpu without --gpus for CPU execution.")
    if args.command != "probe" and args.gpus and "," in args.gpus:
        parser.error("fit and analyze use one device; select one GPU index.")
    if args.command == "probe" and (
        args.epochs < 1 or args.device == "cpu" and args.dtype == "float16"
    ):
        parser.error("Use positive epochs and float32 or bfloat16 for CPU probing.")
    if args.command == "fit" and (args.iterations < 1 or args.tol <= 0):
        parser.error("iterations and tol must be positive.")
    launch("decomposition", args, sys.argv[1:] if argv is None else argv)
    if args.command == "fit":
        from .pipeline import fit as run
    elif args.command == "analyze":
        from .analysis import run
    else:
        from .probe import run
    run(args)
