"""Fit PCA and whitening transforms on a designated reference set."""

from pathlib import Path

import numpy as np


def mean_cov(x, indices=None, batch_size=4096):
    """Accumulate a centered covariance matrix in float64."""
    indices = np.arange(len(x)) if indices is None else np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("At least two fitting samples are required.")
    n = 0
    mu = np.zeros(x.shape[1], dtype=np.float64)
    scatter = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        batch = np.asarray(x[indices[start : start + batch_size]], dtype=np.float64)
        b = len(batch)
        center = batch.mean(axis=0)
        xc = batch - center
        delta = center - mu
        scatter += xc.T @ xc + np.outer(delta, delta) * n * b / (n + b)
        mu += delta * b / (n + b)
        n += b
    return mu, scatter / (n - 1)


class Preprocessor:
    """Apply PCA, PCA whitening, or full-dimensional ZCA whitening."""

    def __init__(self, dim=None, method="pca_whiten", eps=1e-6):
        """Configure the preprocessing method, retained dimension, and eigenvalue floor."""
        if method not in {"pca", "pca_whiten", "zca"}:
            raise ValueError(f"Unknown preprocessing method: {method}")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.dim = dim
        self.method = method
        self.eps = eps

    def fit(self, x, indices=None, batch_size=4096):
        """Estimate the mean and transform using only the selected rows."""
        if x.ndim != 2:
            raise ValueError("Expected an [items, dimension] matrix.")
        d = x.shape[1]
        dim = d if self.dim is None else self.dim
        if not 1 <= dim <= d:
            raise ValueError(f"dim must lie between 1 and {d}.")
        if self.method == "zca" and dim != d:
            raise ValueError("ZCA whitening preserves the full feature dimension.")
        self.mu, cov = mean_cov(x, indices, batch_size)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1][:dim]
        self.values = np.maximum(values[order], 0)
        self.basis = vectors[:, order]
        # Fix eigenvector signs for reproducible coordinates.
        pivot = np.abs(self.basis).argmax(axis=0)
        self.basis *= np.where(self.basis[pivot, np.arange(dim)] < 0, -1, 1)
        self.scale = np.sqrt(np.maximum(self.values, self.eps))
        self.fit_idx = np.arange(len(x)) if indices is None else np.asarray(indices, dtype=np.int64)
        self.dim = dim
        return self

    def transform(self, x):
        """Center and project inputs using the fitted parameters."""
        z = (np.asarray(x, dtype=np.float64) - self.mu) @ self.basis
        if self.method != "pca":
            z /= self.scale
        if self.method == "zca":
            z = z @ self.basis.T
        return z.astype(np.float32)

    def inverse_transform(self, z):
        """Reconstruct retained components in the original feature space."""
        z = np.asarray(z, dtype=np.float64)
        if self.method == "zca":
            z = z @ self.basis
        if self.method != "pca":
            z = z * self.scale
        return (z @ self.basis.T + self.mu).astype(np.float32)

    def save(self, path):
        """Save the fitted transform parameters and fitting row indices."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez(
                f,
                method=self.method,
                eps=self.eps,
                dim=self.dim,
                mu=self.mu,
                basis=self.basis,
                values=self.values,
                scale=self.scale,
                fit_idx=self.fit_idx,
            )

    @classmethod
    def load(cls, path):
        """Load a fitted preprocessing transform from a NumPy archive."""
        with np.load(path) as data:
            obj = cls(int(data["dim"]), str(data["method"]), float(data["eps"]))
            for key in ("mu", "basis", "values", "scale", "fit_idx"):
                setattr(obj, key, data[key].copy())
        return obj


def transform_file(proc, source, output, batch_size=4096, inverse=False):
    """Apply a fitted transform to a NumPy matrix in batches."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Input and output must be different files.")
    x = np.load(source, mmap_mode="r")
    dim = len(proc.mu) if inverse or proc.method == "zca" else proc.dim
    output.parent.mkdir(parents=True, exist_ok=True)
    z = np.lib.format.open_memmap(output, mode="w+", dtype="float32", shape=(len(x), dim))
    transform = proc.inverse_transform if inverse else proc.transform
    for start in range(0, len(x), batch_size):
        z[start : start + batch_size] = transform(x[start : start + batch_size])
    z.flush()
