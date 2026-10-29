"""Cosine metrics, held-out splits, and unordered pair sampling."""

import numpy as np

METRICS = ("pointwise", "cross_object", "cross_model")


def cosine(a, b, eps=1e-12):
    """Compute cosine along the last axis, assigning zero to zero-length vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.clip(np.sum(a * b, axis=-1) / np.maximum(norm, eps), -1, 1)


def compare(a, b, pairs):
    """Evaluate aligned [items, dimension] matrices on one common evaluation pool."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Aligned feature matrices must have the same two-dimensional shape.")
    i, j = np.asarray(pairs, dtype=np.int64).T
    return {
        "pointwise": float(cosine(a, b).mean()),
        "cross_object": float(cosine(a[i] - b[i], a[j] - b[j]).mean()),
        "cross_model": float(cosine(a[i] - a[j], b[i] - b[j]).mean()),
    }


def split_items(n, fit_size=10000, eval_size=10000, seed=42):
    """Draw disjoint fitting and evaluation items without replacement."""
    if fit_size < 2 or eval_size < 2 or fit_size + eval_size > n:
        raise ValueError(f"Need {fit_size + eval_size} matched items; received {n}.")
    indices = np.random.default_rng(seed).permutation(n)
    return indices[:fit_size], indices[fit_size : fit_size + eval_size]


def sample_pairs(n, count=10000, seed=42):
    """Sample distinct unordered pairs uniformly without constructing an n-by-n matrix."""
    total = n * (n - 1) // 2
    if not 1 <= count <= total:
        raise ValueError(f"Pair count must lie between 1 and {total}.")
    ranks = np.random.default_rng(seed).choice(total, size=count, replace=False)
    rows = np.arange(n, dtype=np.int64)
    starts = rows * (2 * n - rows - 1) // 2
    i = np.searchsorted(starts, ranks, side="right") - 1
    j = i + 1 + ranks - starts[i]
    return np.column_stack([i, j])
