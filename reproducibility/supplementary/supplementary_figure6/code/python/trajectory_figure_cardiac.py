#!/usr/bin/env python3
"""Evaluate TF target-program coherence along computational PHATE A* paths.

This script is the reproducible generator for the cardiac path-association
figure. It expects the exact 378,802-cell analysis-ready AnnData object as an
explicit input because that large file is deposited separately from the compact
USB bundle. The bundled human DoRothEA A-C prior and bundled ``graph_utils.py``
are resolved relative to this script, so no machine-specific project path is
required.

For a TF X, the control-referenced target-program score of cell c is

    s_c(X) = mean_t sign(w_X,t) * (x_c,t - mean_control,t) / sd_control,t

For each TF and seed, A* is run on at most 300 TF-knockdown cells and a seeded
control sample of at most three times that number. The 30,000-cell gray PHATE
background is sampled separately for visualization and is not used by A*.
Coherence is the mean absolute Spearman association between path position and
``s_c`` over the sampled paths. The statistic is direction-agnostic and is not
interpreted as chronological ordering.

Example
-------
python trajectory_figure_cardiac.py \
    --h5ad /path/to/cardio_perturb_phate.h5ad \
    --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_DIR = SCRIPT_DIR.parents[1]
BUNDLE_DIR = REPRO_DIR.parent
DEFAULT_DOROTHEA = BUNDLE_DIR / "shared_inputs" / "dorothea_ABC_human.parquet"
DEFAULT_RESULTS_DIR = REPRO_DIR / "results" / "trajectory_figure_cardiac_reproduced"

# Use the graph implementation archived with this reproducibility bundle.
sys.path.insert(0, str(SCRIPT_DIR))
from graph_utils import run_astar_for_cluster  # noqa: E402


METHODS = [
    "astar_path",
    "random_paths",
    "phate1_order",
    "n_genes_order",
    "cluster_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        required=True,
        help="Exact 378,802-cell analysis-ready cardio_perturb_phate.h5ad.",
    )
    parser.add_argument(
        "--dorothea",
        type=Path,
        default=DEFAULT_DOROTHEA,
        help="Bundled human DoRothEA A-C prior.",
    )
    parser.add_argument("--tfs", default="MEF2C,MEF2A,HIF1A")
    parser.add_argument(
        "--control-ratio",
        type=float,
        default=3.0,
        help="Maximum number of sampled controls per retained KO cell.",
    )
    parser.add_argument(
        "--max-ko",
        type=int,
        default=300,
        help="Maximum number of KO cells retained per TF.",
    )
    parser.add_argument(
        "--min-targets",
        type=int,
        default=3,
        help="Minimum number of expressed regulon targets required per TF.",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=20,
        help="Number of paths used for method comparison and program shuffling.",
    )
    parser.add_argument(
        "--bg-cells",
        type=int,
        default=30_000,
        help="Number of cells sampled only for the gray PHATE background.",
    )
    parser.add_argument("--n-shuffle", type=int, default=20)
    parser.add_argument("--n-rand-pool", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory inside the reproducibility bundle by default.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in (("analysis H5AD", args.h5ad), ("DoRothEA prior", args.dorothea)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")


def make_paths(method, mini, n_ref: int, rng: np.random.Generator) -> list[list[int]]:
    """Return paths indexed relative to the TF-specific mini population."""
    n_cells = mini.n_obs
    indices = np.arange(n_cells)

    def split(order: np.ndarray) -> list[list[int]]:
        chunks = np.array_split(order, max(1, n_ref))
        return [list(map(int, chunk)) for chunk in chunks if len(chunk)]

    if method == "astar_path":
        # A* walks from low- to high-n_genes endpoints in PHATE coordinates.
        return run_astar_for_cluster(
            mini,
            "0",
            leiden_key="leiden",
            gene_col="n_genes",
            min_paths=n_ref,
            fallback_splits=10,
            max_iter_ratio=0.8,
            verbose=False,
        )
    if method == "cluster_mean":
        # A single unordered group used as the cluster-level baseline.
        return [list(map(int, indices))]
    if method == "random_paths":
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return split(shuffled)
    if method == "n_genes_order":
        order = np.argsort(mini.obs["n_genes"].to_numpy())
        return split(indices[order])
    if method == "phate1_order":
        order = np.argsort(np.asarray(mini.obsm["X_phate"])[:, 0])
        return split(indices[order])
    raise ValueError(f"Unknown path method: {method}")


def path_metrics(paths: list[list[int]], scores: np.ndarray) -> dict[str, float | int]:
    """Calculate direction-agnostic path-position coherence."""
    from scipy.stats import spearmanr

    coherences = []
    for path in paths:
        if len(path) < 5:
            continue
        path_scores = scores[path]
        positions = np.arange(len(path))
        if np.std(path_scores) == 0:
            continue
        rho = spearmanr(positions, path_scores).correlation
        if not np.isnan(rho):
            coherences.append(abs(rho))

    return {
        "coherence": float(np.mean(coherences)) if coherences else np.nan,
        "n_paths": len(paths),
    }


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tfs = [value.strip().upper() for value in args.tfs.split(",") if value.strip()]

    # Independent deterministic streams prevent unrelated random operations
    # from changing another TF's result when the implementation is edited.
    seed_streams = np.random.SeedSequence(args.seed).spawn(len(tfs) + 2)
    setup_rng = np.random.default_rng(seed_streams[0])
    background_rng = np.random.default_rng(seed_streams[1])
    tf_rngs = [np.random.default_rng(stream) for stream in seed_streams[2:]]

    import anndata as ad

    print(f"[load] {args.h5ad} (backed mode; loading required genes only)", flush=True)
    adata = ad.read_h5ad(args.h5ad, backed="r")
    var_positions = {gene.upper(): index for index, gene in enumerate(adata.var_names)}
    obs = adata.obs
    is_control = obs["is_control_use"].to_numpy().astype(bool)
    is_perturbation = obs["is_perturb_use"].to_numpy().astype(bool)
    target_gene = obs["target_gene"].astype(str).str.upper().to_numpy()
    phate = np.asarray(adata.obsm["X_phate"])
    n_genes = pd.to_numeric(obs["n_genes"], errors="coerce").to_numpy()
    all_control_indices = np.where(is_control)[0]

    dorothea = pd.read_parquet(args.dorothea)
    dorothea["source"] = dorothea["source"].astype(str).str.upper()
    dorothea["target"] = dorothea["target"].astype(str).str.upper()

    # Load only the union of expressed regulon targets required by the TF list.
    target_union = sorted(
        {
            gene
            for tf in tfs
            for gene in dorothea.loc[dorothea["source"] == tf, "target"]
            if gene in var_positions
        }
    )
    print(f"[expression] loading {len(target_union)} regulon target genes", flush=True)
    target_expression = adata[:, target_union].to_memory().X
    if hasattr(target_expression, "todense"):
        target_expression = np.asarray(target_expression.todense())
    else:
        target_expression = np.asarray(target_expression)
    target_column = {gene: index for index, gene in enumerate(target_union)}

    # Standardize each target against all retained control cells.
    control_mean = target_expression[all_control_indices].mean(axis=0)
    control_sd = target_expression[all_control_indices].std(axis=0) + 1e-9
    target_z = (target_expression - control_mean) / control_sd

    # Construct a seeded random-gene pool for matched program-shuffle controls.
    random_indices = setup_rng.choice(
        adata.n_vars,
        size=min(args.n_rand_pool, adata.n_vars),
        replace=False,
    )
    print(f"[expression] loading {len(random_indices)} random-null genes", flush=True)
    random_expression = adata[:, random_indices].to_memory().X
    if hasattr(random_expression, "todense"):
        random_expression = np.asarray(random_expression.todense())
    else:
        random_expression = np.asarray(random_expression)
    random_z = (
        random_expression - random_expression[all_control_indices].mean(axis=0)
    ) / (random_expression[all_control_indices].std(axis=0) + 1e-9)

    # This subsample is used only to render the full-embedding context.
    background = background_rng.choice(
        adata.n_obs,
        size=min(args.bg_cells, adata.n_obs),
        replace=False,
    )

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figure, axes = plt.subplots(
        len(tfs),
        1,
        figsize=(6.4, 3.8 * len(tfs)),
        squeeze=False,
    )

    rows = []
    for tf_index, tf in enumerate(tfs):
        rng = tf_rngs[tf_index]
        started = time.time()

        regulon = dorothea[dorothea["source"] == tf]
        target_set = [
            (gene, np.sign(weight) or 1)
            for gene, weight in zip(regulon["target"], regulon["weight"])
            if gene in target_column
        ]
        if len(target_set) < args.min_targets:
            print(
                f"[skip] {tf}: {len(target_set)} expressed targets "
                f"< required {args.min_targets}",
                flush=True,
            )
            continue

        target_columns = [target_column[gene] for gene, _ in target_set]
        target_signs = np.array([sign for _, sign in target_set], dtype=float)
        scores_all = (target_z[:, target_columns] * target_signs[None, :]).mean(axis=1)

        # TF-specific A* population: retained KO cells plus seeded controls.
        ko = np.where(is_perturbation & (target_gene == tf))[0]
        if len(ko) > args.max_ko:
            ko = rng.choice(ko, args.max_ko, replace=False)
        n_controls = min(
            len(all_control_indices),
            int(args.control_ratio * len(ko)),
        )
        controls = rng.choice(all_control_indices, n_controls, replace=False)
        cells = np.concatenate([controls, ko])
        labels = np.concatenate(
            [np.zeros(len(controls), dtype=int), np.ones(len(ko), dtype=int)]
        )

        # Remove any dependence on source row order or condition block order.
        permutation = rng.permutation(len(cells))
        cells = cells[permutation]
        labels = labels[permutation]
        scores = scores_all[cells]

        # A* uses X_phate, n_genes, and the single analysis-group label. The
        # expression column is only a placeholder required by AnnData.
        mini = ad.AnnData(
            X=target_expression[cells, :1].copy(),
            obs=pd.DataFrame({"n_genes": n_genes[cells], "leiden": "0"}),
            obsm={"X_phate": phate[cells]},
        )

        metrics = {}
        all_astar_paths = None
        sampled_astar_paths = None
        for method in METHODS:
            paths = make_paths(method, mini, args.n_paths, rng)
            if method == "astar_path":
                all_astar_paths = paths
                if len(paths) > args.n_paths:
                    selected = rng.choice(len(paths), args.n_paths, replace=False)
                    paths = [paths[index] for index in selected]
                sampled_astar_paths = paths
            metrics[method] = path_metrics(paths, scores)
            rows.append(
                {
                    "seed": args.seed,
                    "tf": tf,
                    "method": method,
                    "n_ko": int(labels.sum()),
                    "n_control": len(controls),
                    "n_cells": len(cells),
                    "n_targets": len(target_set),
                    **metrics[method],
                }
            )

        # Preserve the sampled A* paths and replace only the TF program with
        # same-size random signed gene sets.
        shuffled_coherences = []
        for _ in range(args.n_shuffle):
            random_genes = rng.choice(
                random_z.shape[1],
                size=len(target_set),
                replace=False,
            )
            random_signs = rng.choice([1, -1], size=len(target_set)).astype(float)
            random_scores = (
                random_z[cells][:, random_genes] * random_signs[None, :]
            ).mean(axis=1)
            shuffled_coherences.append(
                path_metrics(sampled_astar_paths, random_scores)["coherence"]
            )

        shuffled_coherences = np.asarray(shuffled_coherences)
        astar_metrics = metrics["astar_path"]
        shuffle_mean = float(shuffled_coherences.mean())
        shuffle_sd = float(shuffled_coherences.std())
        coherence_z = (astar_metrics["coherence"] - shuffle_mean) / (shuffle_sd + 1e-9)
        rows.append(
            {
                "seed": args.seed,
                "tf": tf,
                "method": "ASTAR_vs_SHUFFLE",
                "n_ko": int(labels.sum()),
                "n_control": len(controls),
                "n_cells": len(cells),
                "n_targets": len(target_set),
                "coherence": astar_metrics["coherence"],
                "coh_shuffle_mean": shuffle_mean,
                "coh_shuffle_std": shuffle_sd,
                "coh_z": float(coherence_z),
            }
        )
        print(
            f"[{tf} program shuffle] real={astar_metrics['coherence']:.3f}; "
            f"null={shuffle_mean:.3f} +/- {shuffle_sd:.3f}; z={coherence_z:.1f}",
            flush=True,
        )

        random_path_metrics = metrics["random_paths"]
        axis = axes[tf_index][0]
        axis.scatter(
            phate[background, 0],
            phate[background, 1],
            s=2,
            c="#dcdcdc",
            linewidths=0,
            rasterized=True,
        )
        vmax = np.nanpercentile(np.abs(scores), 98) or 1.0
        point_order = np.argsort(np.abs(scores))
        colored_points = axis.scatter(
            phate[cells][point_order, 0],
            phate[cells][point_order, 1],
            c=scores[point_order],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
            s=7,
            linewidths=0,
            rasterized=True,
        )

        # Display only the first 15 returned paths to limit visual crowding.
        mini_phate = phate[cells]
        for path in (all_astar_paths[:15] if all_astar_paths else []):
            if len(path) > 1:
                axis.plot(
                    mini_phate[path, 0],
                    mini_phate[path, 1],
                    "-",
                    color="#333333",
                    linewidth=0.4,
                    alpha=0.4,
                    zorder=3,
                )

        axis.set_title(
            f"{tf} | A* coherence={astar_metrics['coherence']:.2f} "
            f"(program shuffle={shuffle_mean:.2f}, z={coherence_z:.1f})\n"
            f"colored: {len(ko)} KO + {len(controls)} control cells; "
            f"gray: {len(background):,}-cell subsample of {adata.n_obs:,}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(colored_points, ax=axis, fraction=0.046, pad=0.02).set_label(
            "control-referenced signed target-program score, s_c",
            fontsize=7,
        )
        print(
            f"[{tf}] KO={int(labels.sum())}; controls={len(controls)}; "
            f"cells={len(cells)}; A* paths={astar_metrics['n_paths']}; "
            f"coherence A*={astar_metrics['coherence']:.3f}, "
            f"random={random_path_metrics['coherence']:.3f}; "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    figure.suptitle(
        "Cardiac TF target programs along computational PHATE A* paths",
        y=1.005,
        fontsize=11,
    )
    figure.tight_layout()
    figure_stem = args.out_dir / f"trajectory_figure_cardiac_seed{args.seed}"
    for extension in ("pdf", "png"):
        figure.savefig(
            figure_stem.with_suffix(f".{extension}"),
            dpi=180,
            bbox_inches="tight",
        )

    metrics_path = args.out_dir / f"trajectory_cardiac_metrics_seed{args.seed}.csv"
    results = pd.DataFrame(rows)
    results.to_csv(metrics_path, index=False)
    print(f"[done] figure: {figure_stem}.pdf/.png", flush=True)
    print(f"[done] metrics: {metrics_path}", flush=True)

    print("\nA* versus random paths by TF")
    for tf in tfs:
        subset = results[results["tf"] == tf].set_index("method")
        if "astar_path" in subset.index and "random_paths" in subset.index:
            print(
                f"  {tf}: A*={subset.loc['astar_path', 'coherence']:.3f}; "
                f"random={subset.loc['random_paths', 'coherence']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
