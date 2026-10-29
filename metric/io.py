"""Read feature selections and write experiment records."""

import csv
import json
from pathlib import Path


def make_output(path):
    """Create an empty experiment output directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {path}")
    return path


def write_json(path, data):
    """Write data to an indented JSON file."""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    """Write one JSON object per record line."""
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows, fields=None):
    """Write tabular records to a CSV file with a header."""
    fields = fields or list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_layers(store, selection="all"):
    """Resolve a layer selection to feature names available in the store."""
    names = store.names
    if selection == "all":
        return names
    if selection == "last":
        blocks = [name for name in names if name.startswith("layer_")]
        return [sorted(blocks)[-1] if blocks else names[-1]]
    selected = [
        f"layer_{int(name):03d}" if name.isdigit() else name for name in selection.split(",")
    ]
    missing = set(selected) - set(names)
    if missing:
        raise ValueError(f"Unknown features: {sorted(missing)}")
    return selected
