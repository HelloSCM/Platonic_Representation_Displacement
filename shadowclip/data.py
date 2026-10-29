"""Align backbone caches and read memory-mapped training shards."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from representation.storage import FeatureStore, FeatureWriter


def encoder_settings(metadata):
    """Select encoder configuration fields used to compare feature extraction settings."""
    fields = (
        "backend",
        "model",
        "pretrained",
        "checkpoint",
        "layers",
        "pool",
        "dtype",
        "max_length",
        "instruction",
        "bidirectional",
        "normalized",
        "projected",
    )
    return {name: {key: info.get(key) for key in fields} for name, info in metadata.items()}


def pack(args):
    """Merge final backbone features by sample ID without averaging captions."""
    paths = {
        key: getattr(args, key)
        for key in ("clip_vision", "dino", "clip_text", "teacher")
        if getattr(args, key)
    }
    if not paths or ("clip_vision" in paths) != ("dino" in paths):
        raise ValueError("Provide text caches and/or both CLIP and DINO image caches.")
    stores = {key: FeatureStore(path) for key, path in paths.items()}
    reference = next(iter(stores.values()))
    ids = reference.ids
    indices, arrays, assets = {}, {}, {}
    for key, store in stores.items():
        if len(set(store.ids)) != len(store.ids) or set(store.ids) != set(ids):
            raise ValueError(f"Cache IDs must be unique and match exactly: {key}")
        lookup = {name: i for i, name in enumerate(store.ids)}
        indices[key] = np.array([lookup[name] for name in ids])
        info = store.metadata
        if info.get("normalized") or info.get("projected") or info.get("preprocessing"):
            raise ValueError(f"Use raw, unnormalized, pre-projection final features: {key}")
        if info.get("layers") != "final":
            raise ValueError(f"Extract --layers final for {key}.")
        if key in {"clip_text", "teacher"} and any(
            len(row.get("texts", [])) != 1 for row in store.records
        ):
            raise ValueError("Use one caption or prompt per record, with a unique ID per text.")
        arrays[key] = store.load("final")
        if key.startswith("clip_"):
            with np.load(store.path / "projections.npz") as saved:
                modality = "vision" if key == "clip_vision" else "language"
                assets[modality] = saved[modality].copy()
                if "logit_scale" in saved:
                    scale = saved["logit_scale"].copy()
                    if "logit_scale" in assets and not np.array_equal(assets["logit_scale"], scale):
                        raise ValueError("CLIP image and text caches must use the same weights.")
                    assets["logit_scale"] = scale
    if args.logit_scale is not None:
        assets["logit_scale"] = np.array(args.logit_scale, dtype=np.float32)
    if "logit_scale" not in assets:
        raise ValueError(
            "Re-extract CLIP features or provide the original --logit-scale (log space)."
        )
    clip_info = [s.metadata for k, s in stores.items() if k.startswith("clip_")]
    if len(clip_info) == 2 and any(
        clip_info[0].get(key) != clip_info[1].get(key) for key in ("model", "pretrained")
    ):
        raise ValueError(
            "CLIP image and text caches must use the same model and pretrained weights."
        )
    metadata = {
        "shadowclip_cache": True,
        "sources": {key: str(Path(path).resolve()) for key, path in paths.items()},
        "encoders": {key: store.metadata for key, store in stores.items()},
    }
    writer = FeatureWriter(args.output, reference.records, metadata, args.save_dtype)
    for start in range(0, len(ids), args.batch_size):
        sl = slice(start, start + args.batch_size)
        writer.write({key: x[indices[key][sl]] for key, x in arrays.items()})
    writer.finish(assets)


class Cache:
    """Read feature arrays without loading per-sample metadata into memory."""

    def __init__(self, path):
        """Open cached feature arrays and load their metadata and CLIP projection assets."""
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        if not self.metadata.get("shadowclip_cache"):
            raise ValueError(f"Create a ShadowCLIP cache with the cache command: {path}")
        self.arrays = {
            key: np.load(self.path / spec["file"], mmap_mode="r")
            for key, spec in self.metadata["features"].items()
        }
        with np.load(self.path / "projections.npz") as saved:
            self.assets = {key: saved[key].copy() for key in saved.files}
        self.n = self.metadata["n"]
        if any(len(x) != self.n for x in self.arrays.values()):
            raise ValueError("Cache arrays have different sample counts.")

    def records(self):
        """Load ordered item records from the cache."""
        with (self.path / "items.jsonl").open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]


def model_config(cache, variant="shadowclip", ratio=2):
    """Validate cached feature dimensions and build the ShadowCLIP model configuration."""
    vision, text = cache.assets["vision"], cache.assets["language"]
    if vision.shape[1] != text.shape[1]:
        raise ValueError("CLIP projection output dimensions must match.")
    if cache.arrays["clip_vision"].shape[1] != vision.shape[0]:
        raise ValueError("CLIP image features must be before projection.")
    if cache.arrays["clip_text"].shape[1] != text.shape[0]:
        raise ValueError("CLIP text features must be before projection.")
    dino_dim = cache.arrays["dino"].shape[1] if "dino" in cache.arrays else vision.shape[0]
    if variant in {"shadowclip", "no-rkd"} and dino_dim != vision.shape[0]:
        raise ValueError("Use CLIP and DINO backbones with matched representation dimensions.")
    return {
        "variant": variant,
        "vision_dim": vision.shape[0],
        "text_dim": text.shape[0],
        "embed_dim": text.shape[1],
        "dino_dim": dino_dim,
        "hidden_ratio": ratio,
        "encoders": cache.metadata["encoders"],
    }


def batch_identity(batch):
    """Return a vectorized feature batch without additional collation."""
    return batch


class CachedDataset(Dataset):
    """Load vectorized batches from one or more aligned cached shards."""

    def __init__(self, paths, variant):
        """Load cache shards and validate their features, projections, and encoder settings."""
        self.shards = [Cache(path) for path in paths]
        self.ends = np.cumsum([shard.n for shard in self.shards])
        self.names = ["clip_vision", "clip_text"]
        if variant in {"shadowclip", "no-rkd"}:
            self.names.append("dino")
        if variant == "shadowclip":
            self.names.append("teacher")
        first = self.shards[0]
        for shard in self.shards:
            for key in self.names:
                if (
                    key not in shard.arrays
                    or shard.arrays[key].shape[1] != first.arrays[key].shape[1]
                ):
                    raise ValueError(f"All training shards must contain matching {key} features.")
            if shard.assets.keys() != first.assets.keys() or any(
                not np.array_equal(shard.assets[key], first.assets[key]) for key in first.assets
            ):
                raise ValueError(
                    "All shards must use the same original CLIP projections and scale."
                )
            if encoder_settings(shard.metadata["encoders"]) != encoder_settings(
                first.metadata["encoders"]
            ):
                raise ValueError("All shards must use identical extraction settings.")

    def __len__(self):
        """Return the total number of samples across all cache shards."""
        return int(self.ends[-1])

    def __getitems__(self, indices):
        """Load selected feature rows across shards as an ordered batch of tensors."""
        indices = np.asarray(indices)
        groups = np.searchsorted(self.ends, indices, side="right")
        out = {
            key: np.empty((len(indices), self.shards[0].arrays[key].shape[1]), dtype=np.float32)
            for key in self.names
        }
        for group in np.unique(groups):
            mask = groups == group
            offset = self.ends[group - 1] if group else 0
            rows = indices[mask] - offset
            for key in self.names:
                out[key][mask] = self.shards[group].arrays[key][rows]
        return {key: torch.from_numpy(value) for key, value in out.items()}

    def __getitem__(self, index):
        """Load one sample's named feature tensors from the cached shards."""
        return {key: value[0] for key, value in self.__getitems__([index]).items()}
