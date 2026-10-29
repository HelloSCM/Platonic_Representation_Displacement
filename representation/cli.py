"""Command-line entry points for feature extraction and preprocessing."""

import argparse
from pathlib import Path

import numpy as np

from .data import load_records
from .preprocessing import Preprocessor, transform_file


def run_extraction(args):
    """Extract and save image or text representations with the selected encoder."""
    from .extraction.clip import CLIPEncoder
    from .extraction.language import LanguageEncoder, LLM2VecEncoder
    from .extraction.pipeline import extract_language, extract_vision
    from .extraction.vision import VisionEncoder

    if args.random and args.backend not in {"timm", "transformers"}:
        raise ValueError("--random is supported by the timm and transformers backends.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    records = load_records(args.dataset, args.input, args.caption_index, args.split)
    if args.limit:
        records = records[: args.limit]
    metadata = {
        "dataset": args.dataset,
        "input": args.input,
        "caption_index": args.caption_index,
        "split": args.split,
        "device": args.device,
        "limit": args.limit,
    }
    if args.backend == "openclip":
        if args.layers != "final" or not args.pretrained:
            raise ValueError("OpenCLIP requires --layers final and --pretrained.")
        encoder = CLIPEncoder(
            args.model, args.pretrained, args.command, args.device, args.dtype, args.projected
        )
    elif args.command == "vision":
        encoder = VisionEncoder(
            args.model,
            args.device,
            args.layers,
            args.pool,
            args.dtype,
            args.checkpoint,
            args.random,
            args.seed,
        )
    elif args.backend == "llm2vec":
        if args.layers != "final":
            raise ValueError("LLM2Vec-Gen requires --layers final.")
        encoder = LLM2VecEncoder(
            args.model, args.device, args.dtype, args.max_length, args.instruction
        )
    else:
        encoder = LanguageEncoder(
            args.model,
            args.device,
            args.layers,
            args.dtype,
            args.max_length,
            args.trust_remote_code,
            args.random,
            args.seed,
        )

    if args.command == "vision":
        root = args.input if args.dataset == "folder" else args.image_root
        if root is None:
            raise ValueError("--image-root is required for Flickr30k and manifest images.")
        extract_vision(
            encoder,
            records,
            root,
            args.output,
            args.batch_size,
            args.workers,
            args.save_dtype,
            metadata,
        )
    else:
        extract_language(encoder, records, args.output, args.batch_size, args.save_dtype, metadata)
    encoder.close()


def run_centroids(args):
    """Compute and save concept centroids from a feature store."""
    from .extraction.pipeline import compute_centroids

    compute_centroids(args.input, args.output, args.batch_size)


def run_fit(args):
    """Fit and save a PCA or whitening transform from command-line settings."""
    x = np.load(args.input, mmap_mode="r")
    idx = np.load(args.indices) if args.indices else None
    method = args.method.replace("-", "_")
    dim = args.dim if args.dim is not None or method == "zca" else 500
    proc = Preprocessor(dim, method, args.eps).fit(x, idx, args.batch_size)
    proc.save(args.output)
    print(f"Fitted {proc.method} on {len(proc.fit_idx)} items: {len(proc.mu)} -> {proc.dim}")


def run_transform(args):
    """Apply a saved preprocessing transform or its inverse to a feature matrix."""
    proc = Preprocessor.load(args.params)
    transform_file(proc, args.input, args.output, args.batch_size, args.inverse)


def build_parser():
    """Build the argument parser for representation extraction and preprocessing."""
    parser = argparse.ArgumentParser(description="Representation extraction and preprocessing")
    sub = parser.add_subparsers(dest="command", required=True)
    for modality, backends, datasets in [
        ("vision", ["timm", "openclip"], ["folder", "flickr30k", "manifest"]),
        (
            "language",
            ["transformers", "openclip", "llm2vec"],
            ["flickr30k", "commonwords", "manifest"],
        ),
    ]:
        cmd = sub.add_parser(modality, help=f"Extract {modality} representations")
        cmd.add_argument("--input", required=True, help="Dataset path or CommonWords79k dataset ID")
        cmd.add_argument("--output", required=True, type=Path, help="Empty output directory")
        cmd.add_argument("--dataset", required=True, choices=datasets)
        cmd.add_argument("--model", required=True, help="Model identifier or local model directory")
        cmd.add_argument("--backend", choices=backends, default=backends[0])
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
        cmd.add_argument("--save-dtype", choices=["float32", "float16"], default="float32")
        cmd.add_argument(
            "--layers", default="all", help="all, last, final, or zero-based block indices: 0,5,11"
        )
        cmd.add_argument("--batch-size", type=int, default=32)
        cmd.add_argument(
            "--caption-index",
            type=int,
            default=0,
            help="Flickr30k caption index; -1 averages all captions",
        )
        cmd.add_argument("--split", default="whole", help="CommonWords79k dataset split")
        cmd.add_argument("--limit", type=int, default=None, help="Extract the first N records")
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument(
            "--random", action="store_true", help="Initialize timm/Transformers weights randomly"
        )
        cmd.add_argument("--pretrained", default=None, help="OpenCLIP pretrained tag or checkpoint")
        cmd.add_argument(
            "--projected", action="store_true", help="Include the OpenCLIP final projection"
        )
        if modality == "vision":
            cmd.add_argument("--image-root", type=Path, default=None)
            cmd.add_argument("--checkpoint", type=Path, default=None, help="Local timm checkpoint")
            cmd.add_argument("--workers", type=int, default=4)
            cmd.add_argument("--pool", choices=["auto", "cls", "mean"], default="auto")
        else:
            cmd.add_argument("--max-length", type=int, default=128)
            cmd.add_argument("--trust-remote-code", action="store_true")
            cmd.add_argument("--instruction", default="", help="LLM2Vec-Gen instruction prefix")
        cmd.set_defaults(handler=run_extraction)

    cmd = sub.add_parser("centroids", help="Average features by concept")
    cmd.add_argument("--input", required=True, type=Path, help="Feature store with concept fields")
    cmd.add_argument("--output", required=True, type=Path)
    cmd.add_argument("--batch-size", type=int, default=128)
    cmd.set_defaults(handler=run_centroids)

    cmd = sub.add_parser("fit", help="Fit PCA or whitening on a reference set")
    cmd.add_argument("--input", required=True, type=Path, help="Feature matrix (.npy)")
    cmd.add_argument("--output", required=True, type=Path, help="Fitted parameters (.npz)")
    cmd.add_argument(
        "--indices",
        type=Path,
        default=None,
        help="Fitting row indices (.npy); default: all input rows",
    )
    cmd.add_argument("--method", choices=["pca", "pca-whiten", "zca"], default="pca-whiten")
    cmd.add_argument(
        "--dim",
        type=int,
        default=None,
        help="PCA dimension (default: 500); ZCA retains all dimensions",
    )
    cmd.add_argument("--eps", type=float, default=1e-6, help="Whitening eigenvalue floor")
    cmd.add_argument("--batch-size", type=int, default=4096)
    cmd.set_defaults(handler=run_fit)

    cmd = sub.add_parser("transform", help="Apply fitted preprocessing or its inverse")
    cmd.add_argument("--input", required=True, type=Path, help="Feature matrix (.npy)")
    cmd.add_argument("--output", required=True, type=Path, help="Output matrix (.npy)")
    cmd.add_argument("--params", required=True, type=Path, help="Fitted parameters (.npz)")
    cmd.add_argument("--batch-size", type=int, default=4096)
    cmd.add_argument("--inverse", action="store_true")
    cmd.set_defaults(handler=run_transform)
    return parser


def main(argv=None):
    """Parse command-line arguments and run the selected representation operation."""
    args = build_parser().parse_args(argv)
    args.handler(args)
