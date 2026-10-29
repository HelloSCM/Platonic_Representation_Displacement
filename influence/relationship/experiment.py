"""Sweep concept noise, cross-model offsets, and parent-increment cosines."""

from pathlib import Path

import numpy as np

from metric.core import compare, cosine, sample_pairs
from metric.io import make_output, write_csv, write_json

from .geometry import concept_vectors, make_forest, mix_concepts, perturbed_forest, sample_mixtures


def prefix(values, name):
    """Prefix dictionary keys with the specified metric group name."""
    return {f"{name}_{key}": value for key, value in values.items()}


def parent_cosine(vectors, parents):
    """Compute the mean cosine between child increments and their parent vectors."""
    children = np.flatnonzero(parents >= 0)
    reference = vectors[parents[children]]
    return float(cosine(vectors[children] - reference, reference).mean())


def parallelism(args, seed, out):
    """Sweep concept perturbations and model offsets to compare concept and sample parallelism."""
    rng = np.random.default_rng(seed)
    basis, parents, paths, offset = make_forest(rng, args.dim, args.roots, args.branching)
    reference = concept_vectors(
        basis, parents, root_norm=args.root_norm, increment_norm=args.increment_norm
    )
    noise = rng.normal(size=reference.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    indices, weights = sample_mixtures(rng, args.samples, len(reference), mix=args.mix)
    samples = mix_concepts(reference, indices, weights)
    sample_noise = mix_concepts(noise, indices, weights)
    concept_pairs = sample_pairs(
        len(reference), min(args.pairs, len(reference) * (len(reference) - 1) // 2), seed
    )
    pairs = sample_pairs(args.samples, args.pairs, seed + 1)
    noises = np.geomspace(args.noise_min, args.noise_max, args.grid)
    shifts = np.geomspace(args.shift_min, args.shift_max, args.grid)
    np.savez(
        out / f"parallelism_seed_{seed}.npz",
        reference=reference,
        noise=noise,
        offset=offset,
        parents=parents,
        paths=paths,
        indices=indices,
        weights=weights,
        concept_pairs=concept_pairs,
        sample_pairs=pairs,
    )
    rows = []
    for strength in noises:
        sample_score = compare(samples + strength * sample_noise, samples, pairs)["cross_model"]
        for shift in shifts:
            current = reference + strength * noise + shift * offset
            values = compare(current, reference, concept_pairs)
            rows.append(
                {
                    "seed": seed,
                    "noise": float(strength),
                    "shift": float(shift),
                    **prefix(values, "concept"),
                    "parent_cos_a": parent_cosine(current, parents),
                    "parent_cos_b": parent_cosine(reference, parents),
                    "sample_cross_model": sample_score,
                }
            )
        print(f"Parallelism seed {seed}: noise={strength:.4g}.", flush=True)
    return rows


def orthogonality(args, seed, out):
    """Sweep parent-increment cosines with coupled perturbations and measure geometry."""
    rng = np.random.default_rng(seed)
    basis, parents, paths, offset = make_forest(rng, args.dim, args.roots, args.branching)
    noise = rng.normal(size=basis.shape)
    indices, weights = sample_mixtures(rng, args.samples, len(parents), paths=paths)
    pairs = sample_pairs(args.samples, args.pairs, seed + 1)
    concept_pairs = sample_pairs(
        len(parents), min(args.pairs, len(parents) * (len(parents) - 1) // 2), seed
    )
    angles = np.linspace(-1.0, 1.0, args.angles)
    np.savez(
        out / f"orthogonality_seed_{seed}.npz",
        basis=basis,
        noise=noise,
        offset=offset,
        parents=parents,
        paths=paths,
        indices=indices,
        weights=weights,
        concept_pairs=concept_pairs,
        sample_pairs=pairs,
    )
    rows = []
    for angle in angles:
        reference = concept_vectors(basis, parents, angle, args.root_norm, args.increment_norm)
        current = perturbed_forest(
            reference, parents, args.offset * offset, noise, args.coupling * angle
        )
        a = mix_concepts(current, indices, weights)
        b = mix_concepts(reference, indices, weights)
        measured = {
            f"parent_cos_{name}": parent_cosine(vectors, parents)
            for name, vectors in (("a", current), ("b", reference))
        }
        rows.append(
            {
                "seed": seed,
                "parent_cos": float(angle),
                **measured,
                **prefix(compare(current, reference, concept_pairs), "concept"),
                **prefix(compare(a, b, pairs), "sample"),
            }
        )
    print(f"Orthogonality seed {seed}: {len(angles)} parent-increment cosines.", flush=True)
    return rows


def run(args):
    """Run the selected synthetic concept-geometry sweeps and save their data and plots."""
    out = make_output(args.output)
    write_json(
        out / "config.json",
        {
            **vars(args),
            "experiment": "relationship",
            "output": str(Path(args.output).resolve()),
            "coordinates": "constructed_common_frame",
        },
    )
    for name, simulation in (("parallelism", parallelism), ("orthogonality", orthogonality)):
        if args.simulation not in {name, "both"}:
            continue
        rows = []
        for repeat in range(args.repeats):
            rows.extend(simulation(args, args.seed + repeat, out))
        write_csv(out / f"{name}.csv", rows)
    if not args.no_plot:
        from .plotting import plot_run

        plot_run(out)
