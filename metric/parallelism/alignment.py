"""Fit a rigid rotation or reflection between corresponding representations."""

import numpy as np


def fit_opa(a, b, device="cpu"):
    """Return R minimizing ||a R - b|| subject to R.T R = I."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("OPA inputs must have the same two-dimensional shape.")
    if device == "cpu":
        cross = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
        u, _, vt = np.linalg.svd(cross, full_matrices=False)
        return u @ vt
    import torch

    with torch.no_grad():
        a = torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)
        b = torch.as_tensor(np.asarray(b), dtype=torch.float64, device=device)
        u, _, vt = torch.linalg.svd(a.T @ b, full_matrices=False)
        return (u @ vt).cpu().numpy()
