#!/usr/bin/env python3
"""Plot cardiac TF expression across PHATE with early/right and mature/left groups."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

EARLY_RIGHT = ["GATA4", "GATA6", "HAND2", "ISL1"]
MATURE_LEFT = ["MEF2A", "ESRRA", "PPARA", "IRX4"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument("--max-cells", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    genes = EARLY_RIGHT + MATURE_LEFT
    adata = ad.read_h5ad(args.h5ad, backed="r")
    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise ValueError(f"Genes missing from AnnData: {missing}")

    expression = adata[:, genes].to_memory().X
    expression = expression.toarray() if sp.issparse(expression) else np.asarray(expression)
    phate = np.asarray(adata.obsm["X_phate"])
    maturation = pd.to_numeric(adata.obs["score_maturation"], errors="coerce").to_numpy()

    rows = []
    for index, gene in enumerate(genes):
        values = expression[:, index]
        rows.append({
            "group": "early_right" if gene in EARLY_RIGHT else "mature_left",
            "gene": gene,
            "mean_expression": float(values.mean()),
            "detection_fraction": float((values > 0).mean()),
            "corr_phate1": float(np.corrcoef(values, phate[:, 0])[0, 1]),
            "corr_maturation": float(np.corrcoef(values, maturation)[0, 1]),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "cardiac_tf_phate_correlations.csv", index=False)

    if args.max_cells and len(phate) > args.max_cells:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(phate), size=args.max_cells, replace=False))
    else:
        selected = np.arange(len(phate))

    expression_cmap = LinearSegmentedColormap.from_list(
        "soft_expression", ["#eeeeee", "#fee0d2", "#fc9272", "#de2d26"]
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.7), sharex=True, sharey=True)
    group_specs = [
        (EARLY_RIGHT, "Early/progenitor-associated TFs — PHATE right"),
        (MATURE_LEFT, "Maturation-associated TFs — PHATE left"),
    ]

    for row_index, (group, row_title) in enumerate(group_specs):
        for column_index, gene in enumerate(group):
            axis = axes[row_index, column_index]
            gene_index = genes.index(gene)
            values = expression[selected, gene_index]
            order = np.argsort(values)
            vmax = float(np.nanpercentile(values, 99.5))
            vmax = vmax if vmax > 0 else 1.0
            scatter = axis.scatter(
                phate[selected[order], 0],
                phate[selected[order], 1],
                c=values[order],
                s=1.7,
                cmap=expression_cmap,
                vmin=0,
                vmax=vmax,
                linewidths=0,
                rasterized=True,
            )
            metric = metrics.set_index("gene").loc[gene]
            axis.set_title(
                f"{gene}\nr(PHATE1)={metric.corr_phate1:+.2f}; "
                f"r(maturation)={metric.corr_maturation:+.2f}",
                fontsize=9,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.02)
        axes[row_index, 0].set_ylabel(row_title, fontsize=10, labelpad=14)

    fig.suptitle(
        "Cardiac transcription-factor expression across PHATE\n"
        "PHATE1 is negatively associated with the maturation score",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out_dir / "supplementary_figure3_cardiac_tf_phate.pdf", dpi=220, bbox_inches="tight")
    fig.savefig(args.out_dir / "supplementary_figure3_cardiac_tf_phate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
