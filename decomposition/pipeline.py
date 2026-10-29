"""Fit and apply a reusable multi-model consensus decomposition."""

from pathlib import Path

import numpy as np
import torch

from metric.io import make_output, select_layers, write_csv, write_json, write_jsonl
from representation import FeatureStore, Preprocessor
from representation.storage import FeatureWriter

from .core import consensus, cross_covariance, fit_gpa
from .data import match_stores
from .model import Decomposition


def fit(args):
    """Fit whitening, GPA alignment, and consensus eigenspaces on matched model features."""
    stores, ids, indices = match_stores(args.inputs)
    if len(stores) < 2:
        raise ValueError("GPA decomposition requires at least two models.")
    names = args.names or [store.metadata.get("model", store.path.name) for store in stores]
    if len(names) != len(stores) or len(set(names)) != len(names):
        raise ValueError("Provide one unique model name per input store.")
    layers = [select_layers(store, args.layer) for store in stores]
    if any(len(selected) != 1 for selected in layers):
        raise ValueError("Select one feature layer per model.")
    layers = [selected[0] for selected in layers]
    size = len(ids) if args.fit_size is None else args.fit_size
    if size > len(ids) or size <= args.dim:
        raise ValueError("fit-size must exceed dim and not exceed the number of matched items.")
    fit_idx = np.sort(np.random.default_rng(args.seed).choice(len(ids), size, replace=False))
    arrays = [store.load(layer) for store, layer in zip(stores, layers)]
    out = make_output(args.output)
    processors = [
        Preprocessor(args.dim, eps=args.eps).fit(array, rows[fit_idx], args.batch_size)
        for array, rows in zip(arrays, indices)
    ]
    models = [
        {
            "name": name,
            "path": str(store.path.resolve()),
            "layer": layer,
            "metadata": store.metadata,
        }
        for name, store, layer in zip(names, stores, layers)
    ]
    device = args.device
    means = [torch.as_tensor(proc.mu, device=device) for proc in processors]
    weights = [torch.as_tensor(proc.basis / proc.scale, device=device) for proc in processors]

    def batches():
        """Yield matched batches of whitened features for every model."""
        for start in range(0, size, args.batch_size):
            batch = fit_idx[start : start + args.batch_size]
            yield [
                (
                    torch.as_tensor(np.array(array[rows[batch]], dtype=np.float64), device=device)
                    - mean
                )
                @ weight
                for array, rows, mean, weight in zip(arrays, indices, means, weights)
            ]

    cov = cross_covariance(batches(), len(stores), args.dim, device)
    rotations, history = fit_gpa(cov, args.iterations, args.tol, device)
    matrix, values, basis = consensus(cov, rotations)
    bundle = Decomposition(models, processors, rotations, basis, values)
    bundle.save(out)
    write_json(
        out / "config.json",
        {
            "analysis": "decomposition",
            "models": models,
            "dim": args.dim,
            "method": "pca_whiten",
            "eps": args.eps,
            "fit_size": size,
            "seed": args.seed,
            "iterations": args.iterations,
            "tol": args.tol,
            "device": device,
            "gpus": args.gpus,
        },
    )
    write_jsonl(
        out / "matches.jsonl",
        [{"id": key, "indices": [int(rows[i]) for rows in indices]} for i, key in enumerate(ids)],
    )
    np.save(out / "fit_indices.npy", fit_idx)
    np.save(out / "consensus.npy", matrix)
    write_csv(out / "gpa.csv", history)
    write_csv(
        out / "spectrum.csv",
        [{"index": i, "eigenvalue": float(value)} for i, value in enumerate(values)],
    )
    print(
        f"Fitted {len(stores)} models on {size} matched items; GPA iterations: {len(history) - 1}."
    )
    return bundle


def transform(args):
    """Apply a fitted component transform and save the transformed feature store."""
    bundle = Decomposition.load(args.params)
    index = bundle.index(args.model)
    model = bundle.models[index]
    store = FeatureStore(args.input)
    x = store.load(model["layer"])
    metadata = {
        **store.metadata,
        "decomposition": str(Path(args.params).resolve()),
        "component": args.component,
        "rank": args.rank,
        "space": args.space,
    }
    writer = FeatureWriter(args.output, store.records, metadata)
    for start in range(0, len(x), args.batch_size):
        z = bundle.transform(
            index, x[start : start + args.batch_size], args.component, args.rank, args.space
        )
        writer.write({model["layer"]: z})
    writer.finish()
