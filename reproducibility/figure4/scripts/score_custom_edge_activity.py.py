#!/usr/bin/env python3
"""Score a user-supplied custom edge set per cell and plot its UMAP activity.

This mirrors the cell-scale custom_pathway_calc objective:
  edge_L2 = sqrt(sum((alpha_i * alpha_j / alpha_G^2) * abs(w + alpha_i + alpha_j)))

The edge set must be supplied as CSV or JSON. No biological module is embedded
as a default, which keeps the recorded score tied to an explicit input file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_edges(edges_json: str | None, edges_csv: str | None) -> list[tuple[str, str, float]]:
    if edges_json:
        raw = json.loads(Path(edges_json).read_text())
        if isinstance(raw, dict):
            raw = raw.get("edges", [])
        edges = [(str(e[0]), str(e[1]), float(e[2]) if len(e) > 2 else 1.0) for e in raw]
        if not edges:
            raise SystemExit("The JSON edge set is empty.")
        return edges
    if edges_csv:
        out = []
        with Path(edges_csv).open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                keep = str(row.get("keep", "True")).lower() not in {"0", "false", "no"}
                if not keep:
                    continue
                src = row.get("source") or row.get("src")
                tgt = row.get("target") or row.get("tgt")
                if not src or not tgt:
                    raise SystemExit(f"CSV row missing source/target: {row}")
                out.append((str(src), str(tgt), float(row.get("weight", 1.0))))
        if not out:
            raise SystemExit("The CSV edge set has no retained edges.")
        return out
    raise SystemExit("Supply exactly one of --edges-json or --edges-csv.")


def ensure_csr_float(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(X, dtype=np.float64))


def custom_edge_l2_scores(adata: ad.AnnData, edges: list[tuple[str, str, float]]) -> np.ndarray:
    try:
        from cwg_rust import KEGGPathway, compute_all_kegg_norms_sparse

        n_edges = len(edges)
        pathway = KEGGPathway(
            name="custom_edge_set",
            sources=[e[0] for e in edges],
            targets=[e[1] for e in edges],
            weights=[float(e[2]) for e in edges],
            modifications=[""] * n_edges,
            effects=[0] * n_edges,
            types=[""] * n_edges,
            indirects=[False] * n_edges,
        )
        X = ensure_csr_float(adata.X)
        out = compute_all_kegg_norms_sparse(X, list(adata.var_names), [pathway])
        return np.asarray(out["custom_edge_set"], dtype=np.float64)
    except Exception as exc:
        print(f"[warn] cwg_rust scoring unavailable; falling back to numpy: {type(exc).__name__}: {exc}")

    X = ensure_csr_float(adata.X)
    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}

    alpha_g = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    alpha_g_sq = alpha_g * alpha_g
    alpha_g_sq[alpha_g_sq <= 1e-20] = 1.0

    total = np.zeros(adata.n_obs, dtype=np.float64)
    for src, tgt, weight in edges:
        if src not in var_to_idx or tgt not in var_to_idx:
            print(f"[skip] missing edge gene: {src}->{tgt}")
            continue
        ai = X[:, var_to_idx[src]].toarray().ravel()
        aj = X[:, var_to_idx[tgt]].toarray().ravel()
        total += (ai * aj / alpha_g_sq) * np.abs(float(weight) + ai + aj)
    return np.sqrt(np.maximum(total, 0.0))


def plot_umap(
    adata: ad.AnnData,
    score_col: str,
    cluster_key: str,
    cluster_id: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    if "X_umap" not in adata.obsm:
        raise SystemExit("adata.obsm['X_umap'] is missing; cannot plot UMAP.")

    xy = np.asarray(adata.obsm["X_umap"])
    scores = adata.obs[score_col].astype(float).to_numpy()
    scope = adata.obs[cluster_key].astype(str).to_numpy() == str(cluster_id)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.scatter(xy[:, 0], xy[:, 1], s=5, c="#d0d0d0", linewidths=0, rasterized=True)
    ax.scatter(xy[scope, 0], xy[scope, 1], s=7, c="#2f6fbb", linewidths=0, rasterized=True)
    ax.set_title(f"{cluster_key} {cluster_id} scope")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.scatter(xy[:, 0], xy[:, 1], s=5, c="#d8d8d8", linewidths=0, rasterized=True)
    finite = np.isfinite(scores)
    positive = finite & (scores > 0)
    if positive.any():
        vmax = float(np.nanpercentile(scores[positive], 99))
        if vmax <= 0:
            vmax = float(np.nanmax(scores[positive]))
    else:
        vmax = 1.0
    sc = ax.scatter(
        xy[finite, 0],
        xy[finite, 1],
        s=10,
        c=scores[finite],
        cmap="magma",
        vmin=0,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(score_col)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="score")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--cluster-id", default="18")
    parser.add_argument("--score-col", required=True)
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument(
        "--color-all-cells",
        action="store_true",
        help="Store and plot pathway scores for all cells. Default keeps scores outside the requested cluster as NaN.",
    )
    edge_group = parser.add_mutually_exclusive_group(required=True)
    edge_group.add_argument(
        "--edges-json",
        help="JSON file containing [[src, tgt, weight], ...] or {'edges': ...}.",
    )
    edge_group.add_argument(
        "--edges-csv",
        help="CSV with columns pathway,source,target,weight,keep. Only source/target/weight/keep are used.",
    )
    args = parser.parse_args()

    h5ad = Path(args.h5ad)
    out_h5ad = Path(args.out_h5ad)
    out_prefix = Path(args.out_prefix)

    adata = ad.read_h5ad(h5ad)
    edges = load_edges(args.edges_json, args.edges_csv)
    if args.cluster_key not in adata.obs.columns:
        raise SystemExit(f"cluster key not found in obs: {args.cluster_key}")

    scores_all = custom_edge_l2_scores(adata, edges)
    scope = adata.obs[args.cluster_key].astype(str).to_numpy() == str(args.cluster_id)

    if args.color_all_cells:
        adata.obs[args.score_col] = scores_all
        score_scope = "all_cells"
    else:
        scoped_scores = np.full(adata.n_obs, np.nan, dtype=np.float64)
        scoped_scores[scope] = scores_all[scope]
        adata.obs[args.score_col] = scoped_scores
        score_scope = "cluster_scope_only"
    adata.uns[f"{args.score_col}_edges"] = [list(edge) for edge in edges]
    adata.uns[f"{args.score_col}_scope"] = {
        "cluster_key": args.cluster_key,
        "cluster_id": str(args.cluster_id),
        "n_scope": int(scope.sum()),
        "score_scope": score_scope,
    }

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_h5ad)

    if args.color_all_cells:
        top_source = pd.DataFrame({
            "cell_id": adata.obs_names,
            "score": scores_all,
            args.cluster_key: adata.obs[args.cluster_key].astype(str).to_numpy(),
        })
    else:
        top_source = pd.DataFrame({
            "cell_id": adata.obs_names[scope],
            "score": scores_all[scope],
            args.cluster_key: adata.obs.loc[scope, args.cluster_key].astype(str).to_numpy(),
        })
    top = top_source.sort_values("score", ascending=False).head(10)
    top_csv = out_prefix.with_name(out_prefix.name + "_top10.csv")
    top.to_csv(top_csv, index=False)

    plot_umap(
        adata,
        score_col=args.score_col,
        cluster_key=args.cluster_key,
        cluster_id=str(args.cluster_id),
        out_pdf=out_prefix.with_suffix(".pdf"),
        out_png=out_prefix.with_suffix(".png"),
    )

    print(f"[ok] wrote h5ad: {out_h5ad}")
    print(f"[ok] wrote top10: {top_csv}")
    print(f"[ok] wrote plot: {out_prefix.with_suffix('.pdf')}")
    print(f"[ok] wrote plot: {out_prefix.with_suffix('.png')}")
    print("[top10]")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
