"""Joint rigid alignment and cross-model agreement eigenspaces."""

import numpy as np
import torch


def cross_covariance(batches, models, dim, device="cpu"):
    """Accumulate normalized cross-products of centered, whitened model batches."""
    gram = torch.zeros(models * dim, models * dim, dtype=torch.float64, device=device)
    n = 0
    for batch in batches:
        x = torch.cat(
            [torch.as_tensor(x, dtype=torch.float64, device=device) for x in batch], dim=1
        )
        gram.addmm_(x.T, x)
        n += len(x)
    if n < 2:
        raise ValueError("At least two matched fitting items are required.")
    return (gram / (n - 1)).reshape(models, dim, models, dim).permute(0, 2, 1, 3).cpu().numpy()


def fit_gpa(cov, iterations=30, tol=1e-7, device="cpu"):
    """Fit all rigid transforms jointly, initializing each rotation to identity."""
    models, _, dim, _ = cov.shape
    c = torch.as_tensor(
        cov.transpose(0, 2, 1, 3).reshape(models * dim, models * dim),
        dtype=torch.float64,
        device=device,
    )
    rotations = torch.eye(dim, dtype=torch.float64, device=device).repeat(models, 1, 1)
    target = (c @ rotations.reshape(models * dim, dim)).reshape(models, dim, dim)
    diagonal = torch.trace(c)

    def objective(r, t):
        """Compute the mean pairwise alignment objective from rotations and covariance products."""
        return float(((r * t).sum() - diagonal).item() / (models * (models - 1)))

    previous = objective(rotations, target)
    history = [{"iteration": 0, "objective": previous, "rotation_change": 0.0}]
    for iteration in range(1, iterations + 1):
        before = rotations.clone()
        for model in range(models):
            block = slice(model * dim, (model + 1) * dim)
            other = target[model] - c[block, block] @ rotations[model]
            u, _, vt = torch.linalg.svd(other, full_matrices=False)
            updated = u @ vt
            delta = updated - rotations[model]
            rotations[model] = updated
            target += (c[:, block] @ delta).reshape(models, dim, dim)
        change = float(torch.mean((rotations - before) ** 2).sqrt().item())
        value = objective(rotations, target)
        history.append({"iteration": iteration, "objective": value, "rotation_change": change})
        if abs(value - previous) <= tol * max(1.0, abs(previous)) and change <= np.sqrt(tol):
            break
        previous = value
    return rotations.cpu().numpy(), history


def consensus(cov, rotations):
    """Average symmetrized aligned cross-covariances over distinct ordered models."""
    models, _, dim, _ = cov.shape
    matrix = np.zeros((dim, dim), dtype=np.float64)
    for a in range(models):
        for b in range(a + 1, models):
            cross = rotations[a].T @ cov[a, b] @ rotations[b]
            matrix += cross + cross.T
    matrix /= models * (models - 1)
    values, basis = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    basis = basis[:, order]
    pivot = np.abs(basis).argmax(axis=0)
    basis *= np.where(basis[pivot, np.arange(dim)] < 0, -1, 1)
    return matrix, values[order], basis
