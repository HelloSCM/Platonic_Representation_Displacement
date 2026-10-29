"""Store feature matrices with their row identities and extraction metadata."""

import json
from pathlib import Path

import numpy as np


class FeatureWriter:
    """Write feature batches to NumPy memory-mapped arrays."""

    def __init__(self, path, records, metadata, dtype="float32"):
        """Create an empty feature store and write its ordered item records."""
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise FileExistsError(f"Use an empty output directory: {self.path}")
        self.n = len(records)
        self.pos = 0
        self.dtype = np.dtype(dtype)
        self.arrays = {}
        self.metadata = dict(metadata)
        with (self.path / "items.jsonl").open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write(self, features):
        """Append matching [batch, dimension] arrays for each feature name."""
        size = next(iter(features.values())).shape[0]
        if self.pos + size > self.n:
            raise ValueError("Feature rows exceed the number of records.")
        if self.arrays and features.keys() != self.arrays.keys():
            raise ValueError("Feature names changed between batches.")
        for name, x in features.items():
            x = np.asarray(x, dtype=self.dtype)
            if x.ndim != 2 or len(x) != size or not np.isfinite(x).all():
                raise ValueError(f"Invalid feature batch: {name}")
            if name not in self.arrays:
                self.arrays[name] = np.lib.format.open_memmap(
                    self.path / f"{name}.npy",
                    mode="w+",
                    dtype=self.dtype,
                    shape=(self.n, x.shape[1]),
                )
            self.arrays[name][self.pos : self.pos + size] = x
        self.pos += size

    def finish(self, assets=None):
        """Flush feature arrays and save projection assets and metadata."""
        if self.pos != self.n:
            raise ValueError(f"Expected {self.n} feature rows, received {self.pos}.")
        features = {}
        for name, x in self.arrays.items():
            x.flush()
            features[name] = {"file": f"{name}.npy", "shape": list(x.shape), "dtype": str(x.dtype)}
        if assets:
            np.savez(self.path / "projections.npz", **assets)
        metadata = {**self.metadata, "format_version": 1, "n": self.n, "features": features}
        with (self.path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


class FeatureStore:
    """Read stored features in the order specified by items.jsonl."""

    def __init__(self, path):
        """Load feature metadata and validate the stored item identities."""
        self.path = Path(path)
        with (self.path / "metadata.json").open(encoding="utf-8") as f:
            self.metadata = json.load(f)
        with (self.path / "items.jsonl").open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        if len(self.records) != self.metadata["n"] or len(set(self.ids)) != len(self.records):
            raise ValueError("Feature records must have unique IDs and match the stored row count.")

    @property
    def ids(self):
        """Return item IDs in stored row order."""
        return [row["id"] for row in self.records]

    @property
    def names(self):
        """Return the names of the stored feature matrices."""
        return list(self.metadata["features"])

    def load(self, name):
        """Open a named feature matrix as a read-only memory map."""
        return np.load(self.path / self.metadata["features"][name]["file"], mmap_mode="r")
