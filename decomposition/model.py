"""Project aligned features and reconstruct components in original coordinates."""

import json
from pathlib import Path

import numpy as np

from representation import Preprocessor


class Decomposition:
    """Encode and reconstruct shared or private components using fitted multi-model transforms."""

    def __init__(self, models, preprocessors, rotations, basis, values):
        """Store model metadata, preprocessors, rotations, and the consensus eigensystem."""
        self.models = models
        self.preprocessors = preprocessors
        self.rotations = rotations
        self.basis = basis
        self.values = values
        self.dim = len(values)

    def index(self, model):
        """Resolve a model name or integer to its fitted model index."""
        if isinstance(model, int):
            return model
        return [row["name"] for row in self.models].index(model)

    def subspace(self, component="original", rank=None):
        """Select the full basis or the leading shared or trailing private directions."""
        if component == "original":
            return np.eye(self.dim)
        if rank is None or not 1 <= rank <= self.dim:
            raise ValueError(f"rank must lie between 1 and {self.dim}.")
        if component == "shared":
            return self.basis[:, :rank]
        if component == "private":
            return self.basis[:, -rank:]
        raise ValueError(f"Unknown component: {component}")

    def encode(self, model, x, component="original", rank=None):
        """Return coordinates in the aligned, whitened component space."""
        i = self.index(model)
        z = self.preprocessors[i].transform(x) @ self.rotations[i]
        return (z @ self.subspace(component, rank)).astype(np.float32)

    def decode(self, model, coordinates, component="original", rank=None):
        """Undo component projection, rotation, and whitening, then restore the mean."""
        i = self.index(model)
        z = np.asarray(coordinates, dtype=np.float64) @ self.subspace(component, rank).T
        return self.preprocessors[i].inverse_transform(z @ self.rotations[i].T)

    def transform(self, model, x, component="original", rank=None, space="coordinates"):
        """Project features to component coordinates or reconstruct them in the original space."""
        coordinates = self.encode(model, x, component, rank)
        if space == "coordinates":
            return coordinates
        if space != "original":
            raise ValueError("space must be coordinates or original.")
        return self.decode(model, coordinates, component, rank)

    def affine(self, model, component="original", rank=None):
        """Return low-rank matrices for original-space reconstruction in Torch."""
        i = self.index(model)
        proc, rotation = self.preprocessors[i], self.rotations[i]
        basis = self.subspace(component, rank)
        forward = (proc.basis / proc.scale) @ rotation @ basis
        inverse = basis.T @ rotation.T @ (proc.scale[:, None] * proc.basis.T)
        return proc.mu, forward, inverse

    def save(self, output):
        """Save fitted preprocessing parameters, rotations, and the consensus eigensystem."""
        output = Path(output)
        for i, proc in enumerate(self.preprocessors):
            proc.save(output / "preprocess" / f"{i:03d}.npz")
        np.savez(
            output / "subspaces.npz", rotations=self.rotations, basis=self.basis, values=self.values
        )

    @classmethod
    def load(cls, source):
        """Load model metadata and fitted decomposition parameters from a saved directory."""
        source = Path(source)
        with (source / "config.json").open(encoding="utf-8") as f:
            config = json.load(f)
        processors = [
            Preprocessor.load(source / "preprocess" / f"{i:03d}.npz")
            for i in range(len(config["models"]))
        ]
        with np.load(source / "subspaces.npz") as data:
            return cls(
                config["models"], processors, data["rotations"], data["basis"], data["values"]
            )
