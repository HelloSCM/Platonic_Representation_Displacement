"""Evaluate additive factorization before and after full-dimensional ZCA whitening."""

from pathlib import Path

import numpy as np

from metric.io import make_output, write_csv, write_json
from representation import Preprocessor

from .factorization import factorization_score, make_cases


def run(args):
    """Evaluate additive factorization of synthetic features before and after whitening."""
    out = make_output(args.output)
    write_json(
        out / "config.json",
        {
            **vars(args),
            "experiment": "whitening",
            "output": str(Path(args.output).resolve()),
            "method": "zca",
            "score": "conditional_mean_additive_r2",
        },
    )
    labels, cases = make_cases(
        args.levels, args.noise_std, args.missing_std, args.secondary_scale, args.seed
    )
    np.save(out / "labels.npy", labels)
    rows = []
    for name, raw in cases.items():
        proc = Preprocessor(method="zca", eps=args.eps).fit(raw)
        whitened = proc.transform(raw)
        proc.save(out / "transforms" / f"{name}.npz")
        assets = {"raw": raw, "whitened": whitened, "labels": labels}
        for space, x in (("raw", raw), ("whitened", whitened)):
            score, reconstruction, factors, mean = factorization_score(x, labels)
            rows.append({"case": name, "space": space, "n": len(x), **score})
            assets[f"{space}_reconstruction"] = reconstruction
            assets[f"{space}_mean"] = mean
            for i, factor in enumerate(factors):
                assets[f"{space}_factor_{i}"] = factor
        np.savez(out / f"{name}.npz", **assets)
        print(f"{name}: raw R2={rows[-2]['r2']:.6f}, whitened R2={rows[-1]['r2']:.6f}.", flush=True)
    write_csv(out / "scores.csv", rows)
    if not args.no_plot:
        from .plotting import plot_run

        plot_run(out)
    return rows
