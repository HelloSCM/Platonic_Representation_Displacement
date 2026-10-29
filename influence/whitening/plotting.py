"""Plot computed factorization scores over paired raw and whitened point clouds."""

import csv
from pathlib import Path

import numpy as np

MARKERS = ("o", "s", "^", "v", "<", ">", "P", "X", "D", "*", "h")


def plot_run(source):
    """Plot raw and whitened synthetic point clouds with their additive factorization scores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(source)
    with (source / "scores.csv").open(encoding="utf-8") as f:
        scores = {(row["case"], row["space"]): float(row["r2"]) for row in csv.DictReader(f)}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for col, case in enumerate(("sound", "noisy", "missing")):
        with np.load(source / f"{case}.npz") as data:
            labels = data["labels"]
            for row, space in enumerate(("raw", "whitened")):
                ax, x = axes[row, col], data[space]
                for label in np.unique(labels[:, 0]):
                    selected = labels[:, 0] == label
                    ax.scatter(
                        x[selected, 0],
                        x[selected, 1],
                        c=labels[selected, 1],
                        cmap="coolwarm",
                        vmin=labels[:, 1].min(),
                        vmax=labels[:, 1].max(),
                        marker=MARKERS[int(label) % len(MARKERS)],
                        s=32,
                        edgecolors="none",
                    )
                ax.text(
                    0.03,
                    0.96,
                    f"$R^2$={scores[case, space]:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                )
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 1:
                    ax.set_xlabel("Dim 1")
                if col == 0:
                    ax.set_ylabel(f"{'Original' if space == 'raw' else 'Whitened'}\nDim 2")
    fig.savefig(source / "whitening.pdf")
    plt.close(fig)
