"""Plot layer-pair metrics and hierarchical concept geometry."""

import numpy as np

from .core import METRICS

LABELS = {
    "pointwise": "Pointwise",
    "cross_object": "Cross-Object",
    "cross_model": "Cross-Model",
    "child_ref": "C-R",
    "child_sibling": "C-S",
    "delta_ref": "(C-R)-R",
    "delta_sibling": "(C-R)-(S-R)",
}


def plot_parallelism(rows, layers_a, layers_b, output):
    """Plot layer-pair heatmaps for the three parallelism metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["layer_a"], row["layer_b"]): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        matrix = [[lookup[a, b][metric] for b in layers_b] for a in layers_a]
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto", origin="lower")
        ax.set(title=LABELS[metric], xlabel="Model B layer", ylabel="Model A layer")
        for axis, labels in ((ax.xaxis, layers_b), (ax.yaxis, layers_a)):
            ticks = np.unique(np.linspace(0, len(labels) - 1, min(6, len(labels))).astype(int))
            axis.set_ticks(ticks, labels=[labels[i] for i in ticks])
        ax.tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=axes, label="Cosine similarity", shrink=0.8)
    fig.savefig(output / "layer_pairs.pdf")
    plt.close(fig)


def plot_orthogonality(summary, distribution, layers, output):
    """Plot concept geometry across layers and its layer-averaged distribution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    references = list(dict.fromkeys(row["reference"] for row in summary))
    metrics = ("child_ref", "child_sibling", "delta_ref", "delta_sibling")
    fig, axes = plt.subplots(
        1, len(references), figsize=(6 * len(references), 4), squeeze=False, constrained_layout=True
    )
    for ax, reference in zip(axes[0], references):
        for metric in metrics:
            rows = [
                row for row in summary if row["reference"] == reference and row["metric"] == metric
            ]
            mean = np.array([row["mean"] for row in rows])
            std = np.array([row["std"] for row in rows])
            (line,) = ax.plot(range(len(layers)), mean, label=LABELS[metric])
            ax.fill_between(
                range(len(layers)), mean - std, mean + std, color=line.get_color(), alpha=0.12
            )
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(
            title=f"{reference.capitalize()} reference",
            xlabel="Layer",
            ylabel="Cosine similarity",
            ylim=(-1, 1),
        )
        ticks = np.unique(np.linspace(0, len(layers) - 1, min(6, len(layers))).astype(int))
        ax.set_xticks(ticks, labels=[layers[i] for i in ticks], rotation=45)
        ax.legend(frameon=False)
    fig.savefig(output / "layers.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    for reference in references:
        values = [row["delta_ref"] for row in distribution if row["reference"] == reference]
        ax.hist(values, bins=np.linspace(-1, 1, 41), density=True, alpha=0.5, label=reference)
    ax.set(xlabel="Layer-averaged cos(child - reference, reference)", ylabel="Density")
    ax.legend(frameon=False)
    fig.savefig(output / "distribution.pdf")
    plt.close(fig)
