"""Plot Pointwise, Cross-Model, and accuracy trajectories."""

import csv
from pathlib import Path


def plot_run(source, output):
    """Plot Pointwise, Cross-Model, and classification accuracy trajectories over epochs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, output = Path(source), Path(output)
    with (source / "geometry.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    colors = {"early": "#4E79A7", "middle": "#59A14F", "deep": "#E39C37"}
    for stage, color in colors.items():
        selected = [row for row in rows if row["stage"] == stage]
        for metric, label, style in (
            ("pointwise", "Pointwise", "--"),
            ("cross_model", "Cross-Model", "-"),
        ):
            ax.plot(
                [int(row["epoch"]) for row in selected],
                [float(row[metric]) for row in selected],
                color=color,
                linestyle=style,
                label=f"{stage.capitalize()} {label}",
            )
    history = source / "history.csv"
    if history.exists():
        with history.open(encoding="utf-8") as f:
            scores = [row for row in csv.DictReader(f) if row["top1"] != ""]
        if scores:
            ax.plot(
                [int(row["epoch"]) for row in scores],
                [float(row["top1"]) for row in scores],
                color="gray",
                label="Accuracy",
            )
    ax.set(xlabel="Epoch", ylabel="Similarity / Accuracy", ylim=(-1, 1))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
