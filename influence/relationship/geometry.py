"""Construct three-level concept forests and weighted sample representations."""

import numpy as np


def make_forest(rng, dim=512, roots=4, branching=4):
    """Return orthogonal increments, direct parents, and root-to-leaf paths."""
    count = roots * (1 + branching + branching**2)
    if dim < count + 1:
        raise ValueError(
            f"The hierarchy and offset direction require at least {count + 1} dimensions."
        )
    basis, _ = np.linalg.qr(rng.normal(size=(dim, count + 1)), mode="reduced")
    parents = np.full(count, -1, dtype=np.int64)
    middle, paths = [], []
    index = roots
    for root in range(roots):
        for _ in range(branching):
            parents[index] = root
            middle.append((root, index))
            index += 1
    for root, parent in middle:
        for _ in range(branching):
            parents[index] = parent
            paths.append((root, parent, index))
            index += 1
    return basis[:, :count].T, parents, np.array(paths), basis[:, count]


def concept_vectors(basis, parents, parent_cos=0.0, root_norm=1.0, increment_norm=2.0):
    """Control each non-root increment's cosine with its direct parent."""
    vectors = np.empty_like(basis)
    for i, parent in enumerate(parents):
        if parent < 0:
            vectors[i] = root_norm * basis[i]
        else:
            direction = vectors[parent] / np.linalg.norm(vectors[parent])
            increment = increment_norm * (
                parent_cos * direction + np.sqrt(1 - parent_cos**2) * basis[i]
            )
            vectors[i] = vectors[parent] + increment
    return vectors


def perturbed_forest(reference, parents, shift, noise, strength):
    """Accumulate perturbed child increments from shifted roots."""
    vectors = np.empty_like(reference)
    for i, parent in enumerate(parents):
        if parent < 0:
            vectors[i] = reference[i] + shift
        else:
            increment = reference[i] - reference[parent]
            vectors[i] = vectors[parent] + increment + strength * noise[i]
    return vectors


def sample_mixtures(rng, count, concepts, paths=None, mix=3):
    """Sample normalized uniform weights for arbitrary subsets or hierarchy paths."""
    if paths is None:
        indices = np.array([rng.choice(concepts, mix, replace=False) for _ in range(count)])
    else:
        indices = paths[rng.integers(len(paths), size=count)]
    weights = rng.random(indices.shape)
    weights /= weights.sum(axis=1, keepdims=True)
    return indices, weights


def mix_concepts(vectors, indices, weights):
    """Form sample representations as weighted sums of selected concept vectors."""
    return np.einsum("nk,nkd->nd", weights, vectors[indices])
