"""Plot rank sweeps with model-pair or model-level uncertainty."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from metric.io import write_csv

LABELS = {
    "original": "Original",
    "shared": "Shared",
    "private": "Private",
    "linear": "Linear Kernel",
    "rbf": "RBF Kernel",
}


def plot_table(source, output, value, series, ylabel):
    """Summarize a rank-sweep table and plot grouped means with standard deviations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(source).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    groups = defaultdict(list)
    for row in rows:
        groups[(row[series], float(row["ratio"]))].append(float(row[value]))
    summary = [
        {
            "series": name,
            "ratio": ratio,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }
        for (name, ratio), values in sorted(groups.items())
    ]
    write_csv(Path(output).with_suffix(".csv"), summary)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    for name in dict.fromkeys(row["series"] for row in summary):
        selected = [row for row in summary if row["series"] == name]
        x = np.array([row["ratio"] for row in selected])
        mean = np.array([row["mean"] for row in selected])
        std = np.array([row["std"] for row in selected])
        (line,) = ax.plot(x, mean, label=LABELS[name], marker="o", markersize=3)
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set(xlabel="Ratio", ylabel=ylabel)
    ax.legend(frameon=False)
    fig.savefig(output)
    plt.close(fig)


def plot_analysis(source):
    """Plot cross-model similarity and shared-private dependence across component ranks."""
    source = Path(source)
    plot_table(
        source / "cross_model.csv",
        source / "cross_model_plot.pdf",
        "cross_model",
        "component",
        "Cross-Model similarity",
    )
    plot_table(source / "hsic.csv", source / "hsic_plot.pdf", "hsic", "kernel", "Normalized HSIC")


def plot_probe(source):
    """Plot linear-probe accuracy across shared and private component ranks."""
    source = Path(source)
    plot_table(source / "probe.csv", source / "probe_plot.pdf", "accuracy", "component", "Accuracy")
