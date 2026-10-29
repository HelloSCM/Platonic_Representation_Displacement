"""Extract ordered image and text features and aggregate concept centroids."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..storage import FeatureStore, FeatureWriter
from .common import to_numpy


class ImageDataset(Dataset):
    """Load and transform images in feature-record order."""

    def __init__(self, records, root, transform):
        """Store image records, their root directory, and the image transform."""
        self.records = records
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        """Return the number of image records."""
        return len(self.records)

    def __getitem__(self, index):
        """Load one image as RGB and apply the configured transform."""
        with Image.open(self.root / self.records[index]["image"]) as image:
            return self.transform(image.convert("RGB"))


def extract_vision(
    encoder, records, root, output, batch_size=32, workers=4, save_dtype="float32", metadata=None
):
    """Write image features without changing the input record order."""
    dataset = ImageDataset(records, root, encoder.transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    info = {**encoder.metadata, **(metadata or {}), "modality": "vision", "image_root": str(root)}
    writer = FeatureWriter(output, records, info, save_dtype)
    for images in tqdm(loader, desc="Vision features"):
        writer.write(to_numpy(encoder(images)))
    writer.finish(encoder.assets)


def extract_language(encoder, records, output, batch_size=32, save_dtype="float32", metadata=None):
    """Encode sentences in batches and average embeddings within each record."""
    if any(not row.get("texts") for row in records):
        raise ValueError("Every language record must contain at least one text.")
    info = {
        **encoder.metadata,
        **(metadata or {}),
        "modality": "language",
        "aggregation": "sentence_mean",
    }
    writer = FeatureWriter(output, records, info, save_dtype)
    for start in tqdm(range(0, len(records), batch_size), desc="Language features"):
        rows = records[start : start + batch_size]
        texts = [text for row in rows for text in row["texts"]]
        groups = np.repeat(np.arange(len(rows)), [len(row["texts"]) for row in rows])
        counts = np.bincount(groups, minlength=len(rows))[:, None]
        sums = {}
        for offset in range(0, len(texts), batch_size):
            features = to_numpy(encoder(texts[offset : offset + batch_size]))
            for name, x in features.items():
                if name not in sums:
                    sums[name] = np.zeros((len(rows), x.shape[1]), dtype=np.float64)
                np.add.at(sums[name], groups[offset : offset + len(x)], x)
        writer.write({name: total / counts for name, total in sums.items()})
    writer.finish(encoder.assets)


def compute_centroids(source, output, batch_size=128):
    """Average member representations for each concept at every stored layer."""
    store = FeatureStore(source)
    groups = {}
    for index, row in enumerate(store.records):
        groups.setdefault(row["concept"], []).append(index)
    concepts = sorted(groups)
    rows = []
    for concept in concepts:
        first = store.records[groups[concept][0]]
        row = {"id": concept, "concept": concept, "count": len(groups[concept])}
        if "label" in first:
            row["label"] = first["label"]
        rows.append(row)
    info = {**store.metadata, "aggregation": "concept_mean", "source": str(source)}
    writer = FeatureWriter(output, rows, info)
    arrays = {name: store.load(name) for name in store.names}
    for start in tqdm(range(0, len(concepts), batch_size), desc="Concept centroids"):
        batch = concepts[start : start + batch_size]
        features = {
            name: np.stack([x[groups[concept]].mean(axis=0, dtype=np.float64) for concept in batch])
            for name, x in arrays.items()
        }
        writer.write(features)
    proj = store.path / "projections.npz"
    assets = dict(np.load(proj)) if proj.exists() else None
    writer.finish(assets)
