#!/usr/bin/env python3
"""
Benchmark the A*-based TF-target tool without an LLM in the loop.

The script evaluates the tool along three axes:

1. Measure ablation:
   compare A* path measure against cluster mean, random paths, n_genes-order
   paths, and PHATE-axis paths while reusing the same conservative graph scorer.
2. Optional biological validation:
   if driver genes or independent TF-target truth edges are supplied, report
   driver ranks and precision/recall/AP@K.
3. Path geometry sanity:
   report path counts, length, cell coverage, and n_genes monotonicity.

Outputs:
  <out_dir>/<sampleid>__<cluster_id>__edges.csv
  <out_dir>/<sampleid>__<cluster_id>__summary.json
  <out_dir>/benchmark_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "server" / "graph_utils.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the SC_Graph_Agent repository root.")


ROOT = _find_repo_root(Path(__file__).resolve().parent)
SERVER_DIR = ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from graph_utils import (  # noqa: E402
    _ensure_csr,
    detect_organism,
    path_filter_process_,
    run_astar_for_cluster,
)
from cwg_rust import ClusterWeightedGraphRust, build_conservative_graph  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "tf_target"
DEFAULT_TOPK = (10, 20, 50, 100)


def _read_dorothea(data_dir: Path, organism: str) -> pd.DataFrame:
    path = data_dir / f"dorothea_ABC_{organism}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"DoRothEA parquet not found: {path}. "
            "Pass --data-dir or set up SC_Graph_Agent/data first."
        )
    df = pd.read_parquet(path)
    required = {"source", "target", "weight", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def _split_evenly(indices: Iterable[int], n_paths: int) -> list[list[int]]:
    values = list(map(int, indices))
    if not values:
        return []
    n_paths = max(1, min(int(n_paths), len(values)))
    return [list(map(int, x)) for x in np.array_split(values, n_paths) if len(x)]


def _random_paths(indices: Iterable[int], n_paths: int, seed: int) -> list[list[int]]:
    values = list(map(int, indices))
    rng = random.Random(seed)
    rng.shuffle(values)
    return _split_evenly(values, n_paths)


def _ordered_paths(values: np.ndarray, indices: np.ndarray, n_paths: int) -> list[list[int]]:
    order = np.argsort(values)
    return _split_evenly(indices[order], n_paths)


def _cluster_mean_path(indices: Iterable[int]) -> list[list[int]]:
    values = list(map(int, indices))
    return [values] if values else []


def _paths_for_method(
    method: str,
    adata,
    cluster_id: str,
    leiden_key: str,
    gene_col: str,
    embedding_key: str,
    astar_paths: list[list[int]] | None,
    seed: int,
) -> list[list[int]]:
    mask = adata.obs[leiden_key].astype(str).values == str(cluster_id)
    indices = np.where(mask)[0]
    n_ref = len(astar_paths or [])
    if n_ref == 0:
        n_ref = min(10, max(1, len(indices)))

    if method == "astar_path":
        return astar_paths or []
    if method == "cluster_mean":
        return _cluster_mean_path(indices)
    if method == "random_paths":
        return _random_paths(indices, n_ref, seed)
    if method == "n_genes_order":
        vals = pd.to_numeric(adata.obs[gene_col], errors="coerce").to_numpy()
        return _ordered_paths(vals[indices], indices, n_ref)
    if method == "phate1_order":
        coords = np.asarray(adata.obsm[embedding_key])
        return _ordered_paths(coords[indices, 0], indices, n_ref)
    raise ValueError(f"Unknown method: {method}")


def _build_graph_df(
    adata,
    dorothea_df: pd.DataFrame,
    cluster_id: str,
    paths: list[list[int]],
    beta_threshold: float,
    edge_frequency_threshold: float,
) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "count",
                "freq",
                "mean_beta",
                "mean_beta_raw",
                "score",
                "score_raw",
                "mean_contribution",
                "alpha_i",
                "alpha_j",
            ]
        )

    unique_cells = np.array(list({cell for path in paths for cell in path}), dtype=int)
    cluster_mask = np.zeros(adata.n_obs, dtype=bool)
    cluster_mask[unique_cells] = True
    x_csr = _ensure_csr(adata.X)

    cwg = ClusterWeightedGraphRust.new_sparse(
        sparse_matrix=x_csr,
        gene_names=list(adata.var_names),
        cluster_mask=cluster_mask,
        dorothea_sources=list(dorothea_df["source"]),
        dorothea_targets=list(dorothea_df["target"]),
        dorothea_weights=dorothea_df["weight"].values.astype(np.float64),
        dorothea_confidences=list(dorothea_df["confidence"]),
        cluster_id=str(cluster_id),
        beta_mode="dynamic",
        tf_expr_threshold=0.001,
        target_expr_threshold=0.001,
        require_both_expressed=True,
    )

    edge_data = build_conservative_graph(
        cwg,
        paths,
        x_csr,
        use_greedy=False,
        beta_threshold=beta_threshold,
        threshold=edge_frequency_threshold,
    )

    df = pd.DataFrame({
        "source": edge_data["source"],
        "target": edge_data["target"],
        "count": edge_data["count"],
        "freq": edge_data["freq"],
        "mean_beta": edge_data["mean_beta"],
        "mean_contribution": edge_data["mean_contribution"],
        "alpha_i": edge_data["mean_alpha_i"],
        "alpha_j": edge_data["mean_alpha_j"],
    })
    if df.empty:
        df["mean_beta_raw"] = []
        df["score_raw"] = []
        df["score"] = []
        return df

    df["mean_beta_raw"] = df["mean_beta"].astype(float)
    beta_max = float(df["mean_beta_raw"].max()) if len(df) else 0.0
    beta_scale = beta_max if beta_max > 0 else 1.0
    df["mean_beta"] = df["mean_beta_raw"] / beta_scale
    df["score_raw"] = df["freq"] * df["mean_beta_raw"]
    df["score"] = df["freq"] * df["mean_beta"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _prepare_reduced_paths(paths: list[list[int]]) -> list[list[int]]:
    if len(paths) > 30:
        return path_filter_process_(paths)
    return paths


def _path_stats(adata, paths: list[list[int]], cluster_indices: np.ndarray, gene_col: str) -> dict:
    if not paths:
        return {
            "n_paths": 0,
            "mean_path_len": 0.0,
            "median_path_len": 0.0,
            "unique_cell_fraction": 0.0,
            "mean_n_genes_step_positive_fraction": None,
        }

    lengths = np.array([len(p) for p in paths], dtype=float)
    unique_cells = {int(c) for p in paths for c in p}
    n_genes = pd.to_numeric(adata.obs[gene_col], errors="coerce").to_numpy()
    step_fracs = []
    for path in paths:
        if len(path) < 2:
            continue
        vals = n_genes[np.asarray(path, dtype=int)]
        finite = np.isfinite(vals)
        vals = vals[finite]
        if len(vals) >= 2:
            step_fracs.append(float(np.mean(np.diff(vals) >= 0)))

    return {
        "n_paths": int(len(paths)),
        "mean_path_len": float(np.mean(lengths)),
        "median_path_len": float(np.median(lengths)),
        "unique_cell_fraction": float(len(unique_cells) / max(1, len(cluster_indices))),
        "mean_n_genes_step_positive_fraction": (
            float(np.mean(step_fracs)) if step_fracs else None
        ),
    }


def _load_driver_genes(args) -> dict[tuple[str | None, str | None], set[str]]:
    drivers: dict[tuple[str | None, str | None], set[str]] = defaultdict(set)
    if args.driver_genes:
        drivers[(None, None)].update(_split_gene_arg(args.driver_genes))
    if args.driver_genes_file:
        df = pd.read_csv(args.driver_genes_file)
        if "gene" not in df.columns:
            raise ValueError("--driver-genes-file must contain a 'gene' column")
        for row in df.itertuples(index=False):
            sample = getattr(row, "sampleid", None)
            cluster = getattr(row, "cluster_id", None)
            drivers[(str(sample) if sample is not None and not pd.isna(sample) else None,
                     str(cluster) if cluster is not None and not pd.isna(cluster) else None)].add(
                str(getattr(row, "gene")).strip()
            )
    return drivers


def _split_gene_arg(value: str) -> set[str]:
    return {x.strip() for x in value.replace(";", ",").split(",") if x.strip()}


def _driver_set(
    drivers: dict[tuple[str | None, str | None], set[str]],
    sampleid: str,
    cluster_id: str,
) -> set[str]:
    out = set()
    for key in ((None, None), (sampleid, None), (None, str(cluster_id)), (sampleid, str(cluster_id))):
        out.update(drivers.get(key, set()))
    return out


def _load_truth_edges(path: str | None) -> set[tuple[str, str]]:
    if not path:
        return set()
    df = pd.read_csv(path)
    required = {"source", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"--truth-edges must contain columns: {sorted(required)}")
    return {(str(r.source), str(r.target)) for r in df.itertuples(index=False)}


def _average_precision(labels: list[bool]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    hits = 0
    precisions = []
    for i, ok in enumerate(labels, start=1):
        if ok:
            hits += 1
            precisions.append(hits / i)
    return float(sum(precisions) / positives) if precisions else 0.0


def _edge_validation_metrics(
    df: pd.DataFrame,
    truth_edges: set[tuple[str, str]],
    topk: tuple[int, ...],
) -> dict:
    if not truth_edges:
        return {}
    pairs = list(zip(df["source"].astype(str), df["target"].astype(str)))
    labels = [p in truth_edges for p in pairs]
    metrics = {
        "truth_edges_total": int(len(truth_edges)),
        "truth_edges_in_ranking": int(sum(labels)),
        "edge_average_precision": _average_precision(labels),
    }
    for k in topk:
        top = labels[:k]
        hits = int(sum(top))
        metrics[f"edge_precision_at_{k}"] = hits / k if k else None
        metrics[f"edge_recall_at_{k}"] = hits / len(truth_edges) if truth_edges else None
    return metrics


def _driver_metrics(df: pd.DataFrame, drivers: set[str], topk: tuple[int, ...]) -> dict:
    if not drivers:
        return {}
    sources = df["source"].astype(str).tolist()
    targets = df["target"].astype(str).tolist()
    source_hits = [g in drivers for g in sources]
    endpoint_hits = [(s in drivers) or (t in drivers) for s, t in zip(sources, targets)]
    ranks = [i + 1 for i, ok in enumerate(source_hits) if ok]
    endpoint_ranks = [i + 1 for i, ok in enumerate(endpoint_hits) if ok]
    metrics = {
        "driver_genes": ",".join(sorted(drivers)),
        "driver_source_best_rank": min(ranks) if ranks else None,
        "driver_endpoint_best_rank": min(endpoint_ranks) if endpoint_ranks else None,
        "driver_source_average_precision": _average_precision(source_hits),
        "driver_endpoint_average_precision": _average_precision(endpoint_hits),
    }
    for k in topk:
        metrics[f"driver_source_hits_at_{k}"] = int(sum(source_hits[:k]))
        metrics[f"driver_endpoint_hits_at_{k}"] = int(sum(endpoint_hits[:k]))
    return metrics


def _rank_overlap_metrics(
    by_method: dict[str, pd.DataFrame],
    reference_method: str,
    topk: tuple[int, ...],
) -> dict[str, dict]:
    if reference_method not in by_method:
        return {}
    ref_pairs = list(zip(by_method[reference_method]["source"], by_method[reference_method]["target"]))
    out = {}
    for method, df in by_method.items():
        if method == reference_method:
            continue
        pairs = list(zip(df["source"], df["target"]))
        md = {}
        for k in topk:
            a = set(ref_pairs[:k])
            b = set(pairs[:k])
            denom = len(a | b)
            md[f"top{k}_jaccard_vs_{reference_method}"] = float(len(a & b) / denom) if denom else None
        out[method] = md
    return out


def _parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_topk(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in _parse_csv_list(value))


def _discover_samples(data_dir: Path, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    return sorted(p.stem for p in data_dir.glob("*.h5ad"))


def _select_clusters(adata, leiden_key: str, requested: list[str], min_cells: int) -> list[str]:
    counts = adata.obs[leiden_key].astype(str).value_counts()
    if requested:
        return [c for c in requested if c in counts.index and counts[c] >= min_cells]
    return [str(c) for c, n in counts.items() if n >= min_cells]


def run_benchmark(args) -> pd.DataFrame:
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = _parse_csv_list(args.methods)
    topk = _parse_topk(args.topk)
    sampleids = _discover_samples(data_dir, _parse_csv_list(args.samples))
    requested_clusters = _parse_csv_list(args.clusters)
    drivers_by_key = _load_driver_genes(args)
    truth_edges = _load_truth_edges(args.truth_edges)

    all_rows = []

    for sampleid in sampleids:
        h5ad_path = data_dir / f"{sampleid}.h5ad"
        if not h5ad_path.exists():
            print(f"[skip] missing h5ad: {h5ad_path}", file=sys.stderr)
            continue

        print(f"[sample] {sampleid}", flush=True)
        adata = ad.read_h5ad(h5ad_path)
        if args.leiden_key not in adata.obs:
            raise KeyError(f"{sampleid}: obs['{args.leiden_key}'] not found")
        if args.gene_col not in adata.obs:
            raise KeyError(f"{sampleid}: obs['{args.gene_col}'] not found")
        if args.embedding_key not in adata.obsm:
            raise KeyError(f"{sampleid}: obsm['{args.embedding_key}'] not found")

        # Existing A* implementation expects these canonical names.
        # Keep the source h5ad untouched; only alias in memory for this run.
        if args.gene_col != "n_genes":
            adata.obs["n_genes"] = adata.obs[args.gene_col]
        if args.embedding_key != "X_phate":
            adata.obsm["X_phate"] = adata.obsm[args.embedding_key]

        organism = args.organism if args.organism != "auto" else detect_organism(adata)
        dorothea_df = _read_dorothea(data_dir, organism)
        cluster_ids = _select_clusters(adata, args.leiden_key, requested_clusters, args.min_cells)
        if args.max_clusters:
            cluster_ids = cluster_ids[: args.max_clusters]

        for cluster_id in cluster_ids:
            print(f"  [cluster] {cluster_id}", flush=True)
            cluster_mask = adata.obs[args.leiden_key].astype(str).values == str(cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            astar_paths = []
            astar_seconds = None
            if "astar_path" in methods:
                t0 = time.time()
                astar_paths_raw = run_astar_for_cluster(
                    adata,
                    cluster_id,
                    leiden_key=args.leiden_key,
                    gene_col="n_genes",
                    q_low=args.q_low,
                    q_high=args.q_high,
                    min_paths=args.min_paths,
                    fallback_splits=args.fallback_splits,
                    max_iter_ratio=args.max_iter_ratio,
                    verbose=args.verbose,
                )
                astar_paths = _prepare_reduced_paths(astar_paths_raw)
                astar_seconds = time.time() - t0

            by_method = {}
            cluster_rows = []
            for method in methods:
                raw_paths = _paths_for_method(
                    method=method,
                    adata=adata,
                    cluster_id=cluster_id,
                    leiden_key=args.leiden_key,
                    gene_col=args.gene_col,
                    embedding_key=args.embedding_key,
                    astar_paths=astar_paths,
                    seed=args.seed,
                )
                paths = raw_paths if method == "astar_path" else _prepare_reduced_paths(raw_paths)
                t0 = time.time()
                df = _build_graph_df(
                    adata=adata,
                    dorothea_df=dorothea_df,
                    cluster_id=cluster_id,
                    paths=paths,
                    beta_threshold=args.beta_threshold,
                    edge_frequency_threshold=args.edge_frequency_threshold,
                )
                graph_seconds = time.time() - t0
                df.insert(0, "method", method)
                df.insert(0, "cluster_id", str(cluster_id))
                df.insert(0, "sampleid", sampleid)
                by_method[method] = df

                stats = _path_stats(adata, paths, cluster_indices, args.gene_col)
                drivers = _driver_set(drivers_by_key, sampleid, str(cluster_id))
                row = {
                    "sampleid": sampleid,
                    "cluster_id": str(cluster_id),
                    "organism": organism,
                    "method": method,
                    "n_cells_cluster": int(len(cluster_indices)),
                    "n_edges": int(len(df)),
                    "graph_seconds": float(graph_seconds),
                    "astar_seconds": float(astar_seconds) if method == "astar_path" and astar_seconds is not None else None,
                    **stats,
                    **_driver_metrics(df, drivers, topk),
                    **_edge_validation_metrics(df, truth_edges, topk),
                }
                cluster_rows.append(row)

            overlaps = _rank_overlap_metrics(by_method, "astar_path", topk)
            for row in cluster_rows:
                row.update(overlaps.get(row["method"], {}))
                all_rows.append(row)

            edge_out = pd.concat(by_method.values(), ignore_index=True) if by_method else pd.DataFrame()
            edge_path = out_dir / f"{sampleid}__{cluster_id}__edges.csv"
            edge_out.to_csv(edge_path, index=False)
            summary_path = out_dir / f"{sampleid}__{cluster_id}__summary.json"
            summary_path.write_text(json.dumps(cluster_rows, indent=2, ensure_ascii=False))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    return summary


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Benchmark A*-based TF-target ranking without model/agent calls."
    )
    p.add_argument("--data-dir", default=os.getenv("MCP_DATA_DIR", str(DEFAULT_DATA_DIR)))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--samples", default="", help="Comma-separated sampleids. Default: all h5ad files.")
    p.add_argument("--clusters", default="", help="Comma-separated cluster ids. Default: all clusters.")
    p.add_argument("--max-clusters", type=int, default=0, help="Limit clusters per sample after filtering.")
    p.add_argument("--min-cells", type=int, default=2, help="Skip clusters smaller than this.")
    p.add_argument("--leiden-key", default="leiden")
    p.add_argument("--gene-col", default="n_genes")
    p.add_argument("--embedding-key", default="X_phate")
    p.add_argument("--organism", choices=["auto", "human", "mouse"], default="auto")
    p.add_argument(
        "--methods",
        default="astar_path,cluster_mean,random_paths,n_genes_order,phate1_order",
        help="Comma-separated methods.",
    )
    p.add_argument("--beta-threshold", type=float, default=1.45)
    p.add_argument("--edge-frequency-threshold", type=float, default=0.8)
    p.add_argument("--q-low", type=float, default=0.1)
    p.add_argument("--q-high", type=float, default=0.9)
    p.add_argument("--min-paths", type=int, default=300)
    p.add_argument("--fallback-splits", type=int, default=10)
    p.add_argument("--max-iter-ratio", type=float, default=0.8)
    p.add_argument(
        "--process-paths",
        action="store_true",
        help="Deprecated compatibility flag. Path reduction is applied once for every method when needed.",
    )
    p.add_argument("--topk", default=",".join(map(str, DEFAULT_TOPK)))
    p.add_argument("--driver-genes", default="", help="Comma-separated driver genes applied to all samples/clusters.")
    p.add_argument("--driver-genes-file", default="", help="CSV with gene and optional sampleid,cluster_id columns.")
    p.add_argument("--truth-edges", default="", help="Independent truth CSV with source,target columns.")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args)
    print(f"[done] wrote {len(summary)} summary rows to {Path(args.out_dir) / 'benchmark_summary.csv'}")
    if len(summary):
        cols = ["sampleid", "cluster_id", "method", "n_cells_cluster", "n_edges", "n_paths", "mean_path_len"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
