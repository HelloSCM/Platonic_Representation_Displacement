"""Evaluate all selected layer pairs with held-out rigid alignment."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from representation import FeatureStore, Preprocessor

from ..concepts import match_items
from ..core import compare, sample_pairs, split_items
from ..io import make_output, select_layers, write_csv, write_json, write_jsonl
from .alignment import fit_opa


def prepare_layer(store, name, indices, fit_idx, eval_idx, args, save_dir=None):
    """Fit preprocessing on matched fitting rows and transform both splits."""
    x = store.load(name)
    proc = Preprocessor(args.dim, eps=args.eps).fit(x, indices[fit_idx], batch_size=args.batch_size)
    if save_dir is not None:
        proc.save(save_dir / f"{name}.npz")
    return proc.transform(x[indices[fit_idx]]), proc.transform(x[indices[eval_idx]])


def run(args):
    """Fit pairwise alignment and evaluate sample-level or concept-level parallelism."""
    a, b = FeatureStore(args.a), FeatureStore(args.b)
    layers_a, layers_b = select_layers(a, args.layers_a), select_layers(b, args.layers_b)
    matches = match_items(a, b, args.level, args.concept_map)
    fit_idx, eval_idx = split_items(len(matches), args.fit_size, args.eval_size, args.seed)
    if args.dim >= args.fit_size:
        raise ValueError("dim must be smaller than the number of fitting items.")
    pairs = sample_pairs(len(eval_idx), args.pairs, args.seed + 1)
    indices_a = np.array([row["index_a"] for row in matches])
    indices_b = np.array([row["index_b"] for row in matches])
    out = make_output(args.output)
    write_json(
        out / "config.json",
        {
            "analysis": "parallelism",
            "level": args.level,
            "a": {"path": str(a.path.resolve()), "metadata": a.metadata},
            "b": {"path": str(b.path.resolve()), "metadata": b.metadata},
            "layers_a": layers_a,
            "layers_b": layers_b,
            "dim": args.dim,
            "eps": args.eps,
            "method": "pca_whiten",
            "fit_size": args.fit_size,
            "eval_size": args.eval_size,
            "pairs": args.pairs,
            "seed": args.seed,
            "device": args.device,
            "concept_map": str(Path(args.concept_map).resolve()) if args.concept_map else None,
            "save_transforms": args.save_transforms,
        },
    )
    write_jsonl(out / "matches.jsonl", matches)
    np.savez(out / "split.npz", fit_idx=fit_idx, eval_idx=eval_idx, pairs=pairs)
    params_a = out / "preprocess" / "a" if args.save_transforms else None
    params_b = out / "preprocess" / "b" if args.save_transforms else None
    rotations = out / "rotations"
    if args.save_transforms:
        rotations.mkdir()
    results = []
    # Cache one side on disk to keep memory independent of the number of layers.
    with TemporaryDirectory(prefix="metric-") as cache:
        cache = Path(cache)
        for name in layers_b:
            fit, evaluation = prepare_layer(b, name, indices_b, fit_idx, eval_idx, args, params_b)
            np.save(cache / f"{name}_fit.npy", fit)
            np.save(cache / f"{name}_eval.npy", evaluation)
        for name_a in layers_a:
            fit_a, eval_a = prepare_layer(a, name_a, indices_a, fit_idx, eval_idx, args, params_a)
            for name_b in layers_b:
                fit_b = np.load(cache / f"{name_b}_fit.npy", mmap_mode="r")
                eval_b = np.load(cache / f"{name_b}_eval.npy", mmap_mode="r")
                rotation = fit_opa(fit_a, fit_b, args.device)
                values = compare(eval_a @ rotation, eval_b, pairs)
                results.append({"layer_a": name_a, "layer_b": name_b, **values})
                if args.save_transforms:
                    np.save(rotations / f"{name_a}__{name_b}.npy", rotation)
            print(f"Evaluated {name_a}: {len(layers_b)} layer pairs.", flush=True)
    write_csv(out / "results.csv", results)
    if args.plot:
        from ..plotting import plot_parallelism

        plot_parallelism(results, layers_a, layers_b, out)
    return results
