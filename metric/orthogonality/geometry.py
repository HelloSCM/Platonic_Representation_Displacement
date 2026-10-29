"""Measure concept geometry using parent or unrelated reference vectors."""

import numpy as np

from ..core import cosine

METRICS = ("child_ref", "child_sibling", "delta_ref", "delta_sibling")


def sample_references(ids, parents, groups, seed=42):
    """Choose one unrelated concept per child, shared across all layers."""
    rng = np.random.default_rng(seed)
    references = {}
    for parent, children in groups.items():
        candidates = [i for i, key in enumerate(ids) if key in parents and parents[key] != parent]
        if not candidates:
            raise ValueError("Random references require at least two distinct parent families.")
        for child in children:
            references[child] = int(rng.choice(candidates))
    return references


def concept_geometry(x, groups, references=None, parent_vectors=None):
    """Yield per-child cosines, retaining each ordered sibling comparison."""
    if references is None and parent_vectors is None:
        raise ValueError(
            "Parent-reference geometry requires independently encoded parent features."
        )
    for parent, children in groups.items():
        for child in children:
            siblings = [i for i in children if i != child]
            c, s = x[child], x[siblings]
            ref = references[child] if references is not None else parent
            r = x[ref] if references is not None else parent_vectors[parent]
            values = {
                "child_ref": np.atleast_1d(cosine(c, r)),
                "child_sibling": cosine(c, s),
                "delta_ref": np.atleast_1d(cosine(c - r, r)),
                "delta_sibling": cosine(c - r, s - r),
            }
            yield {
                "child": child,
                "parent": parent,
                "ref": ref,
                "siblings": siblings,
                "values": values,
            }
