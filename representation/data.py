"""Load image, caption, and concept records."""

import csv
import json
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_json(path):
    """Read a JSON document or a sequence of JSONL records."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def load_manifest(path):
    """Read records with unique IDs and optional image, text, and concept fields."""
    records = read_json(path)
    for row in records:
        row["id"] = str(row["id"])
        text = row.pop("text", row.get("texts", []))
        row["texts"] = [text] if isinstance(text, str) else text
    return records


def load_folder(path):
    """Index a class-organized image directory in sorted order."""
    root = Path(path)
    records = []
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    for label, folder in enumerate(classes):
        for image in sorted(folder.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                name = image.relative_to(root).as_posix()
                records.append({"id": name, "image": name, "concept": folder.name, "label": label})
    return records


def load_flickr30k(path, caption_index=0):
    """Read token files, image_name/comment CSVs, or Karpathy JSON annotations."""
    path = Path(path)
    captions = defaultdict(list)
    if path.suffix == ".json":
        for row in read_json(path)["images"]:
            name = row["filename"]
            captions[name].extend(s["raw"] for s in row["sentences"])
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig") as f:
            sep = "|" if "|" in f.readline() else ","
            f.seek(0)
            for row in csv.DictReader(f, delimiter=sep, skipinitialspace=True):
                row = {key.strip(): value for key, value in row.items()}
                captions[row["image_name"].strip()].append(
                    (int(row["comment_number"]), row["comment"].strip())
                )
        captions = {name: [text for _, text in sorted(rows)] for name, rows in captions.items()}
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                name, text = line.rstrip("\n").split("\t", 1)
                name, index = name.rsplit("#", 1)
                captions[name].append((int(index), text.strip()))
        captions = {name: [text for _, text in sorted(rows)] for name, rows in captions.items()}

    records = []
    for name in sorted(captions):
        texts = captions[name] if caption_index == -1 else [captions[name][caption_index]]
        records.append({"id": name, "image": name, "texts": texts})
    return records


def load_commonwords(source, split="whole"):
    """Merge sentence lists by normalized CommonWords79k alias."""
    path = Path(source)
    if path.is_file():
        rows = read_json(path)
    elif path.is_dir():
        from datasets import DatasetDict, load_from_disk

        rows = load_from_disk(str(path))
        if isinstance(rows, DatasetDict):
            rows = rows[split]
    else:
        from datasets import load_dataset

        rows = load_dataset(source, split=split)

    words = defaultdict(list)
    for row in rows:
        alias = row["alias"].strip().lower().replace("_", " ")
        texts = [text.strip() for text in row["sentences"] if text.strip()]
        if alias and texts:
            words[alias].extend(texts)
    return [{"id": word, "concept": word, "texts": words[word]} for word in sorted(words)]


def load_records(dataset, source, caption_index=0, split="whole"):
    """Load image, caption, or concept records for the selected dataset."""
    if dataset == "folder":
        records = load_folder(source)
    elif dataset == "flickr30k":
        records = load_flickr30k(source, caption_index)
    elif dataset == "commonwords":
        records = load_commonwords(source, split)
    else:
        records = load_manifest(source)
    if not records:
        raise ValueError("The dataset has no records.")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Record IDs must be unique.")
    return records
