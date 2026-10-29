"""Generate balanced concept grids and measure additive factorization."""

import numpy as np


def make_cases(levels=8, noise_std=0.15, missing_std=0.05, secondary_scale=0.01, seed=42):
    """Construct sound, noisy, and missing-concept two-dimensional representations."""
    c1, c2 = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    labels = np.column_stack([c1.ravel(), c2.ravel()])
    centered = labels - (levels - 1) / 2
    rng = np.random.default_rng(seed)
    sound = centered @ np.array([[1.0, 0.3], [0.4, 1.0]])
    noisy = centered * np.array([1.0, secondary_scale]) + rng.normal(0, noise_std, centered.shape)
    missing = np.column_stack([centered[:, 0], np.zeros(len(centered))])
    missing += rng.normal(0, missing_std, centered.shape)
    return labels, {"sound": sound, "noisy": noisy, "missing": missing}


def additive_factors(x, labels):
    """Estimate additive concept effects by conditional means on a balanced grid."""
    x, labels = np.asarray(x, dtype=np.float64), np.asarray(labels)
    mean = x.mean(axis=0)
    reconstruction = np.broadcast_to(mean, x.shape).copy()
    factors = []
    for column in labels.T:
        _, group = np.unique(column, return_inverse=True)
        counts = np.bincount(group)
        sums = np.zeros((len(counts), x.shape[1]))
        np.add.at(sums, group, x)
        effect = sums / counts[:, None] - mean
        reconstruction += effect[group]
        factors.append(effect)
    return reconstruction, mean, factors


def factorization_score(x, labels):
    """Return descriptive multivariate R-squared and the additive reconstruction."""
    x = np.asarray(x, dtype=np.float64)
    reconstruction, mean, factors = additive_factors(x, labels)
    residual = float(np.sum((x - reconstruction) ** 2))
    total = float(np.sum((x - mean) ** 2))
    return (
        {"r2": 1 - residual / total, "residual_ss": residual, "total_ss": total},
        reconstruction,
        factors,
        mean,
    )
