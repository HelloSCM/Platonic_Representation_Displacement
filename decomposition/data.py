"""Match feature stores and select model-specific matrices."""

import json
from pathlib import Path

import numpy as np

from representation import FeatureStore


def match_stores(paths):
    """Match feature stores by common sample IDs and return their row indices."""
    stores = [FeatureStore(path) for path in paths]
    indices = [{key: i for i, key in enumerate(store.ids)} for store in stores]
    ids = sorted(set.intersection(*(set(index) for index in indices)))
    if len(ids) < 2:
        raise ValueError("At least two common feature IDs are required.")
    rows = [np.array([index[key] for key in ids], dtype=np.int64) for index in indices]
    return stores, ids, rows


def open_inputs(bundle, paths):
    """Load matched feature arrays and validate them against the fitted encoders."""
    if len(paths) != len(bundle.models):
        raise ValueError("Supply one feature store per fitted model, in the fitted model order.")
    stores, ids, indices = match_stores(paths)
    for store, model, proc in zip(stores, bundle.models, bundle.preprocessors):
        for key in ("model", "backend", "pretrained", "checkpoint", "random"):
            if store.metadata.get(key) != model["metadata"].get(key):
                raise ValueError(f"Input encoder differs from fitted model {model['name']}: {key}")
        if store.metadata.get("random") and store.metadata.get("seed") != model["metadata"].get(
            "seed"
        ):
            raise ValueError(f"Random initialization differs for fitted model {model['name']}.")
        if store.load(model["layer"]).shape[1] != len(proc.mu):
            raise ValueError(f"Input feature width differs for fitted model {model['name']}.")
    arrays = [store.load(model["layer"]) for store, model in zip(stores, bundle.models)]
    return stores, ids, indices, arrays


def held_out_ids(params, ids):
    """Reject evaluation samples used to fit the decomposition."""
    params = Path(params)
    selected = set(np.load(params / "fit_indices.npy").tolist())
    with (params / "matches.jsonl").open(encoding="utf-8") as f:
        fitted = {json.loads(line)["id"] for i, line in enumerate(f) if i in selected}
    if fitted.intersection(ids):
        raise ValueError("Evaluation IDs overlap the decomposition fitting set.")


def concepts(stores, indices):
    """Read matched concept or label identities consistently across models."""

    def keys(store, rows):
        """Read concept identities or fallback labels for selected feature rows."""
        return [store.records[i].get("concept", store.records[i].get("label")) for i in rows]

    values = keys(stores[0], indices[0])
    if any(value is None for value in values):
        raise ValueError("Concept-level analysis requires concept or label fields.")
    if any(keys(store, rows) != values for store, rows in zip(stores[1:], indices[1:])):
        raise ValueError("Matched concept annotations differ across models.")
    return values


def parse_ranks(value, dim):
    """Convert integer or percentage rank selections to unique valid dimensions."""
    ranks = []
    for item in value.split(","):
        item = item.strip()
        rank = round(float(item[:-1]) * dim / 100) if item.endswith("%") else int(item)
        if not 1 <= rank <= dim:
            raise ValueError(f"Requested rank must lie between 1 and {dim}: {item}")
        ranks.append(rank)
    return sorted(set(ranks))
