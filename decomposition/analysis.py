"""Evaluate cross-model displacement geometry and shared-private dependence."""

from itertools import combinations
from pathlib import Path

import numpy as np

from metric.core import compare, sample_pairs
from metric.io import make_output, write_csv, write_json, write_jsonl

from .data import concepts, held_out_ids, open_inputs, parse_ranks
from .dependence import normalized_hsic
from .model import Decomposition


def evaluation_features(bundle, stores, ids, indices, arrays, args):
    """Prepare aligned sample features or concept centroids for decomposition evaluation."""
    if args.level == "sample":
        chosen = np.sort(
            np.random.default_rng(args.seed).choice(
                len(ids), min(args.eval_size, len(ids)), replace=False
            )
        )
        features = []
        for i, (array, rows) in enumerate(zip(arrays, indices)):
            batches = [
                bundle.encode(i, array[rows[chosen[start : start + args.batch_size]]])
                for start in range(0, len(chosen), args.batch_size)
            ]
            features.append(np.concatenate(batches))
        return features, [{"id": ids[i], "match_index": int(i)} for i in chosen]
    keys = [str(key) for key in concepts(stores, indices)]
    names = sorted(set(keys))
    lookup = {key: i for i, key in enumerate(names)}
    group = np.array([lookup[key] for key in keys])
    counts = np.bincount(group, minlength=len(names))
    features = []
    for i, (array, rows) in enumerate(zip(arrays, indices)):
        sums = np.zeros((len(names), bundle.dim), dtype=np.float64)
        for start in range(0, len(rows), args.batch_size):
            z = bundle.encode(i, array[rows[start : start + args.batch_size]])
            np.add.at(sums, group[start : start + len(z)], z)
        features.append((sums / counts[:, None]).astype(np.float32))
    return features, [{"id": key, "count": int(count)} for key, count in zip(names, counts)]


def run(args):
    """Evaluate cross-model geometry and shared-private dependence across component ranks."""
    bundle = Decomposition.load(args.params)
    stores, ids, indices, arrays = open_inputs(bundle, args.inputs)
    if any(
        store.path.resolve() == Path(model["path"]) for store, model in zip(stores, bundle.models)
    ):
        raise ValueError("Use held-out feature stores for evaluation.")
    held_out_ids(args.params, ids)
    ranks = parse_ranks(args.ranks, bundle.dim)
    features, units = evaluation_features(bundle, stores, ids, indices, arrays, args)
    pairs = sample_pairs(len(units), args.pairs, args.seed + 1)
    subset = np.sort(
        np.random.default_rng(args.seed + 2).choice(
            len(units), min(args.hsic_samples, len(units)), replace=False
        )
    )
    out = make_output(args.output)
    write_json(
        out / "config.json",
        {
            **vars(args),
            "params": str(Path(args.params).resolve()),
            "inputs": [str(store.path.resolve()) for store in stores],
            "ranks": ranks,
            "evaluation_items": len(units),
        },
    )
    write_jsonl(
        out / "matches.jsonl",
        [{"id": key, "indices": [int(rows[i]) for rows in indices]} for i, key in enumerate(ids)],
    )
    write_jsonl(out / "items.jsonl", units)
    np.savez(out / "sampling.npz", pairs=pairs, hsic_indices=subset)
    baseline = {
        (a, b): compare(features[a], features[b], pairs)
        for a, b in combinations(range(len(features)), 2)
    }
    similarities, dependence = [], []
    for rank in ranks:
        projected = {}
        for component in ("shared", "private"):
            basis = bundle.subspace(component, rank)
            projected[component] = [x @ basis for x in features]
        for (a, b), original in baseline.items():
            base = {
                "model_a": bundle.models[a]["name"],
                "model_b": bundle.models[b]["name"],
                "rank": rank,
                "ratio": rank / bundle.dim,
                "level": args.level,
            }
            similarities.append({**base, "component": "original", **original})
            for component in ("shared", "private"):
                values = compare(projected[component][a], projected[component][b], pairs)
                similarities.append({**base, "component": component, **values})
        if rank <= bundle.dim // 2:
            for i, model in enumerate(bundle.models):
                for kernel in ("linear", "rbf"):
                    value = normalized_hsic(
                        projected["shared"][i][subset],
                        projected["private"][i][subset],
                        kernel,
                        device=args.device,
                    )
                    dependence.append(
                        {
                            "model": model["name"],
                            "rank": rank,
                            "ratio": rank / bundle.dim,
                            "level": args.level,
                            "kernel": kernel,
                            "hsic": value,
                            "n": len(subset),
                        }
                    )
        print(f"Evaluated rank {rank}/{bundle.dim}.", flush=True)
    write_csv(out / "cross_model.csv", similarities)
    write_csv(
        out / "hsic.csv",
        dependence,
        fields=["model", "rank", "ratio", "level", "kernel", "hsic", "n"],
    )
    if args.plot:
        from .plotting import plot_analysis

        plot_analysis(out)
    return similarities, dependence
