"""Evaluate layer-wise hierarchical concept geometry and reference controls."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from representation import FeatureStore, Preprocessor
from representation.data import read_json

from ..concepts import group_children, make_hierarchy
from ..io import make_output, select_layers, write_csv, write_json, write_jsonl
from .geometry import METRICS, concept_geometry, sample_references


def evaluate_layer(x, ids, groups, references, parent_vectors, layer, reference, writer):
    """Stream raw cosines and summarize with equal weight per child."""
    values, counts, child_rows = defaultdict(list), defaultdict(int), []
    for row in concept_geometry(x, groups, references, parent_vectors):
        child, parent, ref = row["child"], row["parent"], row["ref"]
        ref = ids[ref] if references is not None else ref
        base = {
            "layer": layer,
            "reference": reference,
            "child": ids[child],
            "parent": parent,
            "ref": ref,
        }
        child_row = dict(base)
        for metric, cosines in row["values"].items():
            values[metric].append(float(cosines.mean()))
            counts[metric] += len(cosines)
            child_row[metric] = float(cosines.mean())
            siblings = row["siblings"] if metric in {"child_sibling", "delta_sibling"} else [None]
            for sibling, value in zip(siblings, cosines):
                writer.writerow(
                    {
                        **base,
                        "metric": metric,
                        "sibling": ids[sibling] if sibling is not None else "",
                        "cosine": float(value),
                    }
                )
        child_rows.append(child_row)
    summary = [
        {
            "layer": layer,
            "reference": reference,
            "metric": metric,
            "mean": float(np.mean(values[metric])),
            "std": float(np.std(values[metric])),
            "n_children": len(values[metric]),
            "n_comparisons": counts[metric],
        }
        for metric in METRICS
    ]
    return summary, child_rows


def run(args):
    """Evaluate within-model concept orthogonality across the selected layers."""
    if args.min_children < 3:
        raise ValueError("Parent families must contain at least three child concepts.")
    store = FeatureStore(args.input)
    fit_store = FeatureStore(args.fit_store) if args.fit_store else store
    parent_store = FeatureStore(args.parent_store) if args.parent_store else store
    layers = select_layers(store, args.layers)
    ids = store.ids
    parents = make_hierarchy(ids, args.hierarchy, args.wordnet_dir)
    groups = group_children(ids, parents, args.min_children)
    if not groups:
        raise ValueError("No parent families contain the required number of child concepts.")
    fit_idx = np.load(args.fit_indices) if args.fit_indices else None
    parent_idx = {key: i for i, key in enumerate(parent_store.ids)}
    if args.hierarchy:
        for row in read_json(args.hierarchy):
            if row.get("word") in parent_idx:
                parent_idx.setdefault(row["id"], parent_idx[row["word"]])
    missing = sorted(set(groups) - parent_idx.keys())
    if args.parent_store and missing:
        raise ValueError("The parent store must contain every selected parent ID.")
    groups = {parent: children for parent, children in groups.items() if parent in parent_idx}
    if not groups:
        raise ValueError("No encoded parent concepts are available; provide --parent-store.")
    references = (
        sample_references(ids, parents, groups, args.seed) if args.reference != "parent" else {}
    )
    out = make_output(args.output)
    write_json(
        out / "config.json",
        {
            "analysis": "orthogonality",
            "input": str(store.path.resolve()),
            "metadata": store.metadata,
            "fit_store": str(fit_store.path.resolve()),
            "fit_indices": str(Path(args.fit_indices).resolve()) if args.fit_indices else None,
            "parent_store": str(parent_store.path.resolve()),
            "parent_mode": "features",
            "excluded_parents": missing,
            "hierarchy": str(Path(args.hierarchy).resolve()) if args.hierarchy else "wordnet",
            "reference": args.reference,
            "min_children": args.min_children,
            "layers": layers,
            "dim": args.dim,
            "eps": args.eps,
            "method": "pca_whiten",
            "seed": args.seed,
            "save_transforms": args.save_transforms,
        },
    )
    write_jsonl(
        out / "hierarchy.jsonl", [{"id": key, "parent": parent} for key, parent in parents.items()]
    )
    write_jsonl(
        out / "references.jsonl",
        [{"child": ids[child], "ref": ids[ref]} for child, ref in references.items()],
    )
    summary, children = [], []
    fields = ["layer", "reference", "child", "parent", "ref", "metric", "sibling", "cosine"]
    with (out / "raw.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for layer in layers:
            proc = Preprocessor(args.dim, eps=args.eps).fit(
                fit_store.load(layer), fit_idx, args.batch_size
            )
            x = proc.transform(store.load(layer))
            px = parent_store.load(layer)
            parent_vectors = {key: proc.transform(px[parent_idx[key]]) for key in groups}
            if args.save_transforms:
                proc.save(out / "preprocess" / f"{layer}.npz")
            for reference in ("parent", "random"):
                if args.reference not in {reference, "both"}:
                    continue
                stats, rows = evaluate_layer(
                    x,
                    ids,
                    groups,
                    references if reference == "random" else None,
                    parent_vectors,
                    layer,
                    reference,
                    writer,
                )
                summary.extend(stats)
                children.extend(rows)
            print(
                f"Evaluated {layer}: {sum(map(len, groups.values()))} child concepts.", flush=True
            )
    write_csv(out / "summary.csv", summary)
    write_csv(out / "children.csv", children)
    # Preserve per-child layer averages for concept-pair distributions.
    pooled = defaultdict(list)
    for row in children:
        key = tuple(row[name] for name in ("reference", "child", "parent", "ref"))
        pooled[key].append([row[metric] for metric in METRICS])
    distribution = [
        {
            **dict(zip(("reference", "child", "parent", "ref"), key)),
            **dict(zip(METRICS, np.mean(values, axis=0).tolist())),
        }
        for key, values in pooled.items()
    ]
    write_csv(out / "distribution.csv", distribution)
    if args.plot:
        from ..plotting import plot_orthogonality

        plot_orthogonality(summary, distribution, layers, out)
    return summary
