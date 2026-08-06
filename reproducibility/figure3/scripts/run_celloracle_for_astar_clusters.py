#!/usr/bin/env python3
"""Run CellOracle GRN inference for selected A* benchmark clusters.

The output is a long edge table compatible with benchmark_collectri_methods.py:
  sampleid, cluster_id, method, source, target, score

CellOracle's ranking score is coef_abs; coef_mean and p values are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# GimmeMotifs/GenomePy read XDG paths at import time through the xdg package.
# Set writable defaults before importing anndata/celloracle dependencies.
_DEFAULT_RUNTIME_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "celloracle_runtime"
os.environ.setdefault("HOME", str(_DEFAULT_RUNTIME_CACHE / "home"))
os.environ.setdefault("XDG_CONFIG_HOME", str(_DEFAULT_RUNTIME_CACHE / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(_DEFAULT_RUNTIME_CACHE / "cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_RUNTIME_CACHE / "matplotlib"))
for _p in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    Path(os.environ[_p]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import numpy as np
import pandas as pd


def configure_celloracle_cache(cache_dir: Path) -> None:
    os.environ.setdefault("XDG_CONFIG_HOME", str(cache_dir / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    cache_dir.mkdir(parents=True, exist_ok=True)


def clean_gene(x) -> str:
    value = str(x).strip()
    if value.endswith("(+)") or value.endswith("(-)"):
        value = value[:-3]
    return value.upper()


def load_collectri_gene_sets(path: Path) -> tuple[set[str], set[str]]:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    src = "source" if "source" in df.columns else "tf"
    tgt = "target" if "target" in df.columns else "gene"
    return set(df[src].map(clean_gene)), set(df[tgt].map(clean_gene))


def hvg_order(adata: ad.AnnData) -> pd.DataFrame:
    score = adata.var["hvg_score"].values if "hvg_score" in adata.var else np.zeros(adata.n_vars)
    out = pd.DataFrame({"gene": adata.var_names.astype(str), "hvg_score": score})
    out["upper"] = out["gene"].map(clean_gene)
    return out.sort_values("hvg_score", ascending=False)


def pick_genes(adata: ad.AnnData, collectri_tfs: set[str], collectri_targets: set[str], max_tfs: int, max_targets: int) -> list[str]:
    var = hvg_order(adata)
    tf_genes = var[var["upper"].isin(collectri_tfs)].head(max_tfs)["gene"].tolist()
    target_genes = var[var["upper"].isin(collectri_targets)].head(max_targets)["gene"].tolist()
    return sorted(set(tf_genes) | set(target_genes))


def restrict_base_grn(base: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    gene_upper = {clean_gene(g) for g in genes}
    base = base[base["gene_short_name"].map(clean_gene).isin(gene_upper)].copy()
    tf_cols = [c for c in base.columns[2:] if clean_gene(c) in gene_upper]
    return base[["peak_id", "gene_short_name"] + tf_cols]


def subset_cluster(adata: ad.AnnData, cluster: str, genes: list[str], max_cells: int, seed: int) -> ad.AnnData:
    idx = np.where(adata.obs["leiden"].astype(str).values == str(cluster))[0]
    if max_cells and len(idx) > max_cells:
        rng = np.random.default_rng(seed + int(cluster))
        idx = np.sort(rng.choice(idx, size=max_cells, replace=False))
    sub = adata[idx, genes].copy()
    if "counts" in sub.layers:
        sub.X = sub.layers["counts"].copy()
    sub.obs["celloracle_cluster"] = str(cluster)
    return sub


def links_to_edges(links, sampleid: str) -> pd.DataFrame:
    frames = []
    for cluster, df in links.links_dict.items():
        if df is None or len(df) == 0:
            continue
        out = df.copy()
        out["sampleid"] = sampleid
        out["cluster_id"] = str(cluster)
        out["method"] = "celloracle"
        out["source"] = out["source"].map(clean_gene)
        out["target"] = out["target"].map(clean_gene)
        out["score"] = pd.to_numeric(out["coef_abs"], errors="coerce")
        frames.append(out[["sampleid", "cluster_id", "method", "source", "target", "score", "coef_mean", "coef_abs", "p", "-logp"]])
    if not frames:
        return pd.DataFrame(columns=["sampleid", "cluster_id", "method", "source", "target", "score", "coef_mean", "coef_abs", "p", "-logp"])
    edges = pd.concat(frames, ignore_index=True)
    edges = edges.dropna(subset=["score"])
    return (
        edges.sort_values("score", ascending=False)
        .groupby(["sampleid", "cluster_id", "method", "source", "target"], as_index=False)
        .first()
        .sort_values(["cluster_id", "score"], ascending=[True, False])
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--sampleid", required=True)
    p.add_argument("--collectri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--clusters", required=True, help="Comma-separated leiden clusters.")
    p.add_argument("--celloracle-cache", default="SC_Graph_Agent/data/celloracle")
    p.add_argument("--runtime-cache", default="SC_Graph_Agent/.cache/celloracle_runtime")
    p.add_argument("--base-grn-version", default="hg38_gimmemotifsv5_fpr2")
    p.add_argument("--max-tfs", type=int, default=500)
    p.add_argument("--max-targets", type=int, default=1500)
    p.add_argument("--max-cells", type=int, default=0, help="0 means use all cells in each cluster.")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--bagging-number", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-level", type=int, default=0)
    args = p.parse_args()

    configure_celloracle_cache(Path(args.runtime_cache))

    import celloracle as co
    import celloracle.data.config as cfg
    import celloracle.data.load_promoter_base_GRN as loader

    cfg.CELLORACLE_DATA_DIR = str(Path(args.celloracle_cache).resolve())
    loader.CELLORACLE_DATA_DIR = cfg.CELLORACLE_DATA_DIR
    Path(cfg.CELLORACLE_DATA_DIR).mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    if "leiden" not in adata.obs:
        raise KeyError("adata.obs['leiden'] is required.")
    if "X_phate" not in adata.obsm:
        raise KeyError("adata.obsm['X_phate'] is required.")

    collectri_tfs, collectri_targets = load_collectri_gene_sets(Path(args.collectri))
    genes = pick_genes(adata, collectri_tfs, collectri_targets, args.max_tfs, args.max_targets)

    base = co.data.load_human_promoter_base_GRN(version=args.base_grn_version)
    base = restrict_base_grn(base, genes)
    if base.shape[1] <= 2:
        raise RuntimeError("No TF columns remain after restricting CellOracle base GRN.")
    if base.empty:
        raise RuntimeError("No target rows remain after restricting CellOracle base GRN.")

    summary_rows = []
    edge_paths = []
    for cluster in [c.strip() for c in args.clusters.split(",") if c.strip()]:
        cluster_dir = out_dir / f"{args.sampleid}__{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        edge_path = cluster_dir / "celloracle_edges.csv"
        t0 = time.time()
        rec = {
            "sampleid": args.sampleid,
            "cluster_id": cluster,
            "status": "started",
            "n_genes": len(genes),
            "n_base_targets": int(base["gene_short_name"].nunique()),
            "n_base_tfs": int(base.shape[1] - 2),
            "max_cells": args.max_cells,
            "alpha": args.alpha,
            "bagging_number": args.bagging_number,
            "n_jobs": args.n_jobs,
        }
        try:
            sub = subset_cluster(adata, cluster, genes, args.max_cells, args.seed)
            rec["n_cells"] = int(sub.n_obs)
            oracle = co.Oracle()
            oracle.import_anndata_as_raw_count(
                sub,
                cluster_column_name="celloracle_cluster",
                embedding_name="X_phate",
            )
            n_components = min(30, sub.n_obs - 1, sub.n_vars - 1)
            n_pca_dims = min(20, sub.n_obs - 1, sub.n_vars - 1)
            oracle.import_TF_data(TF_info_matrix=base)
            oracle.perform_PCA(n_components=n_components)
            oracle.knn_imputation(
                k=min(30, max(5, sub.n_obs // 20)),
                n_pca_dims=n_pca_dims,
                n_jobs=args.n_jobs,
            )
            links = oracle.get_links(
                cluster_name_for_GRN_unit="celloracle_cluster",
                alpha=args.alpha,
                bagging_number=args.bagging_number,
                verbose_level=args.verbose_level,
                n_jobs=args.n_jobs,
            )
            edges = links_to_edges(links, args.sampleid)
            edges.to_csv(edge_path, index=False)
            rec["status"] = "ok"
            rec["n_edges"] = int(len(edges))
            rec["seconds"] = round(time.time() - t0, 3)
            edge_paths.append(edge_path)
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = repr(exc)
            rec["seconds"] = round(time.time() - t0, 3)
            pd.DataFrame([rec]).to_csv(cluster_dir / "error.csv", index=False)
        summary_rows.append(rec)
        pd.DataFrame(summary_rows).to_csv(out_dir / "celloracle_cluster_run_summary.csv", index=False)

    all_edges = []
    for path in edge_paths:
        all_edges.append(pd.read_csv(path))
    if all_edges:
        pd.concat(all_edges, ignore_index=True).to_csv(out_dir / "celloracle_edges_combined.csv", index=False)

    config = vars(args).copy()
    config["genes_used"] = len(genes)
    config["base_grn_shape"] = list(base.shape)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
