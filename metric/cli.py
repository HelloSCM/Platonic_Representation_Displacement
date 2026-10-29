"""Command-line entry points for directional representation metrics."""

import argparse
from pathlib import Path


def add_preprocessing(parser):
    """Add PCA whitening, transform-saving, and plotting options to an argument parser."""
    parser.add_argument("--dim", type=int, default=500, help="PCA whitening dimension.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Covariance eigenvalue floor.")
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Covariance accumulation batch size."
    )
    parser.add_argument(
        "--save-transforms",
        action="store_true",
        help="Save preprocessing and alignment parameters.",
    )
    parser.add_argument("--plot", action="store_true", help="Save PDF figures.")


def main(argv=None):
    """Parse command-line arguments and run the selected metric operation."""
    parser = argparse.ArgumentParser(
        description="Cross-model parallelism and within-model concept geometry."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    concepts = commands.add_parser(
        "concepts", help="Build a shared ImageNet-CommonWords concept map."
    )
    concepts.add_argument("--vision", required=True, help="ImageNet concept feature store.")
    concepts.add_argument("--language", required=True, help="CommonWords feature store.")
    concepts.add_argument("--output", required=True, help="Output JSONL concept map.")
    concepts.add_argument(
        "--wordnet-dir", help="NLTK data directory containing the WordNet corpus."
    )

    parallel = commands.add_parser(
        "parallelism", help="Evaluate held-out sample or concept parallelism."
    )
    parallel.add_argument("--a", required=True, help="Model A feature store.")
    parallel.add_argument("--b", required=True, help="Model B feature store.")
    parallel.add_argument("--output", required=True, help="Empty output directory.")
    parallel.add_argument("--level", choices=("sample", "concept"), default="sample")
    parallel.add_argument("--concept-map", help="Shared JSON/JSONL map with id and word fields.")
    parallel.add_argument(
        "--layers-a", default="all", help="all, last, final, or comma-separated layers."
    )
    parallel.add_argument(
        "--layers-b", default="all", help="all, last, final, or comma-separated layers."
    )
    parallel.add_argument("--fit-size", type=int, default=10000)
    parallel.add_argument("--eval-size", type=int, default=10000)
    parallel.add_argument(
        "--pairs", type=int, default=10000, help="Distinct unordered evaluation pairs."
    )
    parallel.add_argument("--seed", type=int, default=42)
    parallel.add_argument("--device", default="cpu", help="OPA device, such as cpu or cuda:0.")
    add_preprocessing(parallel)

    orthogonal = commands.add_parser(
        "orthogonality", help="Evaluate parent-child-sibling concept geometry."
    )
    orthogonal.add_argument("--input", required=True, help="Concept feature store.")
    orthogonal.add_argument("--output", required=True, help="Empty output directory.")
    orthogonal.add_argument(
        "--layers", default="all", help="all, last, final, or comma-separated layers."
    )
    orthogonal.add_argument(
        "--hierarchy", help="JSON/JSONL id-parent mapping or shared concept map."
    )
    orthogonal.add_argument(
        "--wordnet-dir", help="NLTK data directory containing the WordNet corpus."
    )
    orthogonal.add_argument("--fit-store", help="Raw feature store used to fit PCA whitening.")
    orthogonal.add_argument(
        "--fit-indices", help="NumPy array of fitting rows in the fitting store."
    )
    orthogonal.add_argument("--parent-store", help="Optional feature store indexed by parent IDs.")
    orthogonal.add_argument("--min-children", type=int, default=3)
    orthogonal.add_argument("--reference", choices=("parent", "random", "both"), default="both")
    orthogonal.add_argument("--seed", type=int, default=42)
    add_preprocessing(orthogonal)

    aggregate = commands.add_parser(
        "aggregate", help="Summarize model pairs and paired statistical tests."
    )
    aggregate.add_argument(
        "--inputs", nargs="+", required=True, help="Parallelism output directories."
    )
    aggregate.add_argument("--output", required=True, help="Empty output directory.")
    aggregate.add_argument(
        "--layers", choices=("mean", "last"), default="mean", help="Layer-pair reduction."
    )
    aggregate.add_argument("--bootstrap", type=int, default=2000)
    aggregate.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "concepts":
        from .concepts import make_concept_map
        from .io import write_jsonl

        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        rows = make_concept_map(args.vision, args.language, args.wordnet_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output, rows)
        print(f"Matched {len(rows)} concepts.")
    elif args.command == "parallelism":
        from .parallelism.pipeline import run

        run(args)
    elif args.command == "orthogonality":
        from .orthogonality.pipeline import run

        run(args)
    elif args.command == "aggregate":
        from .parallelism.aggregate import run

        run(args)
