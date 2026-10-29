"""Plot sample geometry against controlled concept perturbations."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_table(path):
    """Read a CSV table and convert each field to a floating-point value."""
    with path.open(encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def draw_parallelism(ax, rows):
    """Plot concept parallelism colored by sample-level Cross-Model similarity."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["noise"], row["shift"])].append(row)
    values = np.array(
        [
            [
                np.mean([row[key] for row in group])
                for key in ("concept_cross_model", "concept_cross_object", "sample_cross_model")
            ]
            for group in groups.values()
        ]
    )
    points = ax.scatter(
        *values[:, :2].T, c=values[:, 2], cmap="coolwarm", vmin=0, vmax=1, s=28, edgecolors="none"
    )
    ax.set(
        xlabel="Concept-level Cross-Model similarity",
        ylabel="Concept-level Cross-Object similarity",
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
    )
    ax.figure.colorbar(points, ax=ax, label="Sample-level Cross-Model similarity", shrink=0.85)


def draw_orthogonality(ax, rows):
    """Plot sample parallelism against parent-increment cosine with repeat variability."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_cos"]].append(row)
    angles = sorted(groups)
    for metric, label, color, style in (
        ("sample_cross_model", "Cross-Model", "#4E79A7", "-"),
        ("sample_cross_object", "Cross-Object", "#C16E71", "--"),
    ):
        mean = np.array([np.mean([row[metric] for row in groups[angle]]) for angle in angles])
        std = np.array([np.std([row[metric] for row in groups[angle]]) for angle in angles])
        ax.plot(angles, mean, color=color, linestyle=style, label=label)
        ax.fill_between(angles, mean - std, mean + std, color=color, alpha=0.15)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Parent-increment cosine", ylabel="Sample-level similarity", ylim=(-0.1, 1.05))
    ax.legend(frameon=False)


def plot_run(source):
    """Plot the available synthetic parallelism and orthogonality sweeps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(source)
    plots = [
        (source / f"{name}.csv", draw)
        for name, draw in (("parallelism", draw_parallelism), ("orthogonality", draw_orthogonality))
        if (source / f"{name}.csv").exists()
    ]
    fig, axes = plt.subplots(
        1, len(plots), figsize=(6 * len(plots), 4), squeeze=False, constrained_layout=True
    )
    for ax, (path, draw) in zip(axes[0], plots):
        draw(ax, read_table(path))
    fig.savefig(source / "relationship.pdf")
    plt.close(fig)
