"""Run concept-geometry and whitening demonstrations."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    """Parse command-line arguments and run the selected synthetic experiment."""
    parser = argparse.ArgumentParser(
        description="Synthetic concept geometry and whitening experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    relationship = commands.add_parser(
        "relationship", help="Sweep concept geometry and sample parallelism."
    )
    relationship.add_argument("--output", required=True)
    relationship.add_argument(
        "--simulation", choices=("parallelism", "orthogonality", "both"), default="both"
    )
    relationship.add_argument("--dim", type=int, default=512)
    relationship.add_argument("--roots", type=int, default=4)
    relationship.add_argument("--branching", type=int, default=4)
    relationship.add_argument("--root-norm", type=float, default=1.0)
    relationship.add_argument("--increment-norm", type=float, default=2.0)
    relationship.add_argument("--samples", type=int, default=2048)
    relationship.add_argument("--pairs", type=int, default=10000)
    relationship.add_argument(
        "--mix", type=int, default=3, help="Subset size in the parallelism simulation."
    )
    relationship.add_argument("--grid", type=int, default=25)
    relationship.add_argument("--noise-min", type=float, default=0.01)
    relationship.add_argument("--noise-max", type=float, default=10**1.5)
    relationship.add_argument("--shift-min", type=float, default=0.01)
    relationship.add_argument("--shift-max", type=float, default=10**1.5)
    relationship.add_argument(
        "--angles",
        type=int,
        default=41,
        help="Odd number of parent-increment cosines over [-1, 1].",
    )
    relationship.add_argument(
        "--offset",
        type=float,
        default=10.0,
        help="Global offset norm in the orthogonality simulation.",
    )
    relationship.add_argument(
        "--coupling", type=float, default=1.5, help="Per-coordinate increment perturbation scale."
    )
    relationship.add_argument("--seed", type=int, default=42)
    relationship.add_argument("--repeats", type=int, default=1)
    relationship.add_argument("--no-plot", action="store_true")

    whitening = commands.add_parser(
        "whitening", help="Compare additive factorization before and after whitening."
    )
    whitening.add_argument("--output", required=True)
    whitening.add_argument(
        "--levels", type=int, default=8, help="Number of values for each concept."
    )
    whitening.add_argument("--noise-std", type=float, default=0.15)
    whitening.add_argument("--missing-std", type=float, default=0.05)
    whitening.add_argument("--secondary-scale", type=float, default=0.01)
    whitening.add_argument("--eps", type=float, default=1e-10)
    whitening.add_argument("--seed", type=int, default=42)
    whitening.add_argument("--no-plot", action="store_true")

    plot = commands.add_parser("plot", help="Regenerate figures from saved experiment data.")
    plot.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.command == "plot":
        with (Path(args.input) / "config.json").open(encoding="utf-8") as f:
            experiment = json.load(f)["experiment"]
        if experiment == "relationship":
            from .relationship.plotting import plot_run
        else:
            from .whitening.plotting import plot_run
        plot_run(args.input)
        return
    if args.command == "relationship":
        count = args.roots * (1 + args.branching + args.branching**2)
        if args.dim < count + 1 or not 1 <= args.mix <= count:
            parser.error("dim must exceed the concept count, and mix must fit within that count.")
        if args.angles < 3 or args.angles % 2 != 1:
            parser.error("angles must be odd and at least 3 to include zero.")
        if args.repeats < 1 or args.grid < 1 or args.roots < 1 or args.branching < 1:
            parser.error("repeats, grid, roots, and branching must be positive.")
        if min(args.noise_min, args.shift_min, args.root_norm, args.increment_norm) <= 0:
            parser.error("Noise/shift minima and vector norms must be positive.")
        if args.noise_max < args.noise_min or args.shift_max < args.shift_min:
            parser.error("Noise/shift maxima must not be smaller than the minima.")
        from .relationship.experiment import run
    else:
        if args.levels < 2 or min(args.noise_std, args.missing_std, args.secondary_scale) < 0:
            parser.error("Use at least two concept values and nonnegative scales.")
        from .whitening.experiment import run
    run(args)
