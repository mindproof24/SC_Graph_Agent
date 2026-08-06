#!/usr/bin/env python3
"""
Run a small pySCENIC workflow only for clusters where the existing A* benchmark
used real path search instead of fallback splitting.

The default cluster criterion is:
  benchmark_summary.csv row where method == astar_path and n_cells_cluster > min_paths

For runtime control, expression is restricted to CollecTRI-overlapping genes and
then to top variable genes within each cluster, while retaining expressed TFs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _clean_gene(value) -> str:
    return str(value).strip().upper()


def _run(cmd: list[str], timeout: int | None, log_path: Path) -> tuple[int, float]:
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout if timeout and timeout > 0 else None,
        )
    return proc.returncode, time.time() - t0


def _ctx_failed(ctx_csv: Path, log_path: Path) -> bool:
    if not ctx_csv.exists() or ctx_csv.stat().st_size == 0:
        return True
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if " - ERROR - " in text or "\nERROR" in text:
            return True
    return False


def select_clusters(summary_path: Path, sampleid: str, min_paths: int, requested: str) -> list[str]:
    if requested:
        return [x.strip() for x in requested.split(",") if x.strip()]
    summary = pd.read_csv(summary_path)
    sub = summary[
        summary["sampleid"].astype(str).eq(str(sampleid))
        & summary["method"].astype(str).eq("astar_path")
        & (summary["n_cells_cluster"].astype(float) > float(min_paths))
    ]
    return sub.sort_values("cluster_id")["cluster_id"].astype(str).tolist()


def collectri_gene_sets(path: Path, adata) -> tuple[set[str], set[str], set[str]]:
    ct = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    source_col = "source" if "source" in ct.columns else "tf"
    target_col = "target" if "target" in ct.columns else "gene"
    var_upper = {_clean_gene(g): str(g) for g in adata.var_names}
    tfs = {_clean_gene(x) for x in ct[source_col]} & set(var_upper)
    targets = {_clean_gene(x) for x in ct[target_col]} & set(var_upper)
    return tfs, targets, set(var_upper)


def choose_genes(adata, indices: np.ndarray, tfs: set[str], targets: set[str], max_genes: int, min_expr_frac: float) -> list[str]:
    var_upper_to_original = {_clean_gene(g): str(g) for g in adata.var_names}
    candidate_upper = sorted((tfs | targets) & set(var_upper_to_original))
    candidate_genes = [var_upper_to_original[g] for g in candidate_upper]
    sub = adata[indices, candidate_genes]
    x = sub.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        expr_frac = np.asarray((x > 0).mean(axis=0)).ravel()
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        arr = np.asarray(x)
        mean = arr.mean(axis=0)
        expr_frac = (arr > 0).mean(axis=0)
        mean_sq = (arr * arr).mean(axis=0)
    var = mean_sq - mean * mean
    stats = pd.DataFrame({
        "gene": candidate_genes,
        "upper": candidate_upper,
        "mean": mean,
        "expr_frac": expr_frac,
        "var": var,
    })
    stats = stats[(stats["expr_frac"] >= min_expr_frac) & (stats["mean"] > 0)]
    expressed_tfs = stats[stats["upper"].isin(tfs)].sort_values("var", ascending=False)
    non_tf = stats[~stats["upper"].isin(tfs)].sort_values("var", ascending=False)
    if max_genes and max_genes > 0:
        tf_budget = min(len(expressed_tfs), max(50, max_genes // 4))
        selected = pd.concat([expressed_tfs.head(tf_budget), non_tf.head(max_genes - tf_budget)])
    else:
        selected = stats
    genes = selected.drop_duplicates("gene")["gene"].tolist()
    return genes


def export_cluster_expression(adata, cluster_id: str, genes: list[str], out_csv: Path, leiden_key: str) -> int:
    mask = adata.obs[leiden_key].astype(str).values == str(cluster_id)
    indices = np.where(mask)[0]
    x = adata[indices, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    expr = pd.DataFrame(np.asarray(x), index=adata.obs_names[indices], columns=genes)
    expr.to_csv(out_csv)
    return len(indices)


def write_tf_list(genes: list[str], tfs: set[str], out_txt: Path) -> int:
    tf_genes = [g for g in genes if _clean_gene(g) in tfs]
    out_txt.write_text("\n".join(tf_genes) + "\n", encoding="utf-8")
    return len(tf_genes)


def main() -> int:
    p = argparse.ArgumentParser(description="Run pySCENIC for real-A* clusters only.")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--sampleid", required=True)
    p.add_argument("--benchmark-summary", required=True)
    p.add_argument("--collectri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--clusters", default="")
    p.add_argument("--leiden-key", default="leiden")
    p.add_argument("--min-paths", type=int, default=300)
    p.add_argument("--max-genes", type=int, default=800)
    p.add_argument("--min-expr-frac", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--grn-timeout", type=int, default=0)
    p.add_argument("--ctx-timeout", type=int, default=0)
    p.add_argument("--skip-grn", action="store_true")
    p.add_argument("--skip-ctx", action="store_true")
    p.add_argument("--pyscenic-bin", default="pyscenic")
    p.add_argument("--rankings", nargs="*", default=[])
    p.add_argument("--annotations", default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = select_clusters(Path(args.benchmark_summary), args.sampleid, args.min_paths, args.clusters)
    if not clusters:
        raise SystemExit("No clusters selected.")

    adata = ad.read_h5ad(args.h5ad)
    tfs, targets, _ = collectri_gene_sets(Path(args.collectri), adata)
    rows = []
    for cluster_id in clusters:
        cdir = out_dir / f"{args.sampleid}__{cluster_id}"
        cdir.mkdir(parents=True, exist_ok=True)
        mask = adata.obs[args.leiden_key].astype(str).values == str(cluster_id)
        indices = np.where(mask)[0]
        genes = choose_genes(adata, indices, tfs, targets, args.max_genes, args.min_expr_frac)
        expr_csv = cdir / "expression.csv"
        tf_txt = cdir / "tfs.txt"
        n_cells = export_cluster_expression(adata, cluster_id, genes, expr_csv, args.leiden_key)
        n_tfs = write_tf_list(genes, tfs, tf_txt)
        adj_csv = cdir / "adjacencies.csv"
        ctx_csv = cdir / "ctx.csv"
        row = {
            "sampleid": args.sampleid,
            "cluster_id": cluster_id,
            "n_cells": n_cells,
            "n_genes": len(genes),
            "n_tfs": n_tfs,
            "expression_csv": str(expr_csv),
            "tf_txt": str(tf_txt),
            "adjacencies_csv": str(adj_csv),
            "ctx_csv": str(ctx_csv),
            "grn_status": "skipped" if args.skip_grn else "pending",
            "ctx_status": "skipped" if args.skip_ctx else "pending",
        }
        if n_tfs == 0 or len(genes) == 0:
            row["grn_status"] = "no_tfs_or_genes"
            row["ctx_status"] = "no_tfs_or_genes"
            rows.append(row)
            continue

        if not args.skip_grn:
            cmd = [
                args.pyscenic_bin,
                "grn",
                str(expr_csv),
                str(tf_txt),
                "-o",
                str(adj_csv),
                "-m",
                "grnboost2",
                "--seed",
                str(args.seed),
                "--num_workers",
                str(args.num_workers),
            ]
            try:
                code, seconds = _run(cmd, args.grn_timeout, cdir / "grn.log")
                row["grn_status"] = "ok" if code == 0 else f"exit_{code}"
                row["grn_seconds"] = seconds
            except subprocess.TimeoutExpired:
                row["grn_status"] = "timeout"
                row["grn_seconds"] = args.grn_timeout

        if not args.skip_ctx and row.get("grn_status") == "ok":
            if not args.rankings or not args.annotations:
                row["ctx_status"] = "missing_resources"
            else:
                cmd = [
                    args.pyscenic_bin,
                    "ctx",
                    str(adj_csv),
                    *[str(x) for x in args.rankings],
                    "--annotations_fname",
                    str(args.annotations),
                    "--expression_mtx_fname",
                    str(expr_csv),
                    "-o",
                    str(ctx_csv),
                    "--num_workers",
                    str(args.num_workers),
                ]
                try:
                    ctx_log = cdir / "ctx.log"
                    code, seconds = _run(cmd, args.ctx_timeout, ctx_log)
                    if code == 0 and not _ctx_failed(ctx_csv, ctx_log):
                        row["ctx_status"] = "ok"
                    elif code == 0:
                        row["ctx_status"] = "empty_or_error"
                    else:
                        row["ctx_status"] = f"exit_{code}"
                    row["ctx_seconds"] = seconds
                except subprocess.TimeoutExpired:
                    row["ctx_status"] = "timeout"
                    row["ctx_seconds"] = args.ctx_timeout
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "scenic_cluster_run_summary.csv", index=False)

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
