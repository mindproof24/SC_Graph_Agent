#!/usr/bin/env python3
"""Generate the A*-ranked TF-target edges used in Main Figure 3.

For each requested Leiden cluster, this script:

1. finds A* paths between low- and high-``n_genes`` endpoint sets in PHATE
   coordinates;
2. reduces large path ensembles with the server's path-filtering procedure;
3. scores prior TF-target edges with the conservative graph calculation; and
4. writes ranked edge and per-cluster runtime tables.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
DEFAULT_OUT_DIR = ROOT / "reproducibility" / "figure3" / "work" / "astar"


def _read_tf_target_prior(data_dir: Path, organism: str) -> pd.DataFrame:
    path = data_dir / f"dorothea_ABC_{organism}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"TF-target prior not found: {path}")
    prior = pd.read_parquet(path)
    required = {"source", "target", "weight", "confidence"}
    missing = required - set(prior.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return prior


def _reduce_paths(paths: list[list[int]]) -> list[list[int]]:
    return path_filter_process_(paths) if len(paths) > 30 else paths


def _score_edges(
    adata: ad.AnnData,
    prior: pd.DataFrame,
    cluster_id: str,
    paths: list[list[int]],
    beta_threshold: float,
    frequency_threshold: float,
) -> pd.DataFrame:
    columns = [
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
    if not paths:
        return pd.DataFrame(columns=columns)

    visited = np.fromiter(
        {cell for path in paths for cell in path}, dtype=int
    )
    cluster_mask = np.zeros(adata.n_obs, dtype=bool)
    cluster_mask[visited] = True
    x_csr = _ensure_csr(adata.X)

    graph = ClusterWeightedGraphRust.new_sparse(
        sparse_matrix=x_csr,
        gene_names=list(adata.var_names),
        cluster_mask=cluster_mask,
        dorothea_sources=list(prior["source"]),
        dorothea_targets=list(prior["target"]),
        dorothea_weights=prior["weight"].to_numpy(dtype=np.float64),
        dorothea_confidences=list(prior["confidence"]),
        cluster_id=str(cluster_id),
        beta_mode="dynamic",
        tf_expr_threshold=0.001,
        target_expr_threshold=0.001,
        require_both_expressed=True,
    )
    edge_data = build_conservative_graph(
        graph,
        paths,
        x_csr,
        use_greedy=False,
        beta_threshold=beta_threshold,
        threshold=frequency_threshold,
    )

    edges = pd.DataFrame(
        {
            "source": edge_data["source"],
            "target": edge_data["target"],
            "count": edge_data["count"],
            "freq": edge_data["freq"],
            "mean_beta": edge_data["mean_beta"],
            "mean_contribution": edge_data["mean_contribution"],
            "alpha_i": edge_data["mean_alpha_i"],
            "alpha_j": edge_data["mean_alpha_j"],
        }
    )
    if edges.empty:
        return pd.DataFrame(columns=columns)

    edges["mean_beta_raw"] = edges["mean_beta"].astype(float)
    beta_max = float(edges["mean_beta_raw"].max())
    edges["mean_beta"] = edges["mean_beta_raw"] / (beta_max if beta_max > 0 else 1.0)
    edges["score_raw"] = edges["freq"] * edges["mean_beta_raw"]
    edges["score"] = edges["freq"] * edges["mean_beta"]
    return edges[columns].sort_values("score", ascending=False).reset_index(drop=True)


def _path_summary(paths: list[list[int]]) -> dict[str, float | int]:
    if not paths:
        return {"n_paths": 0, "mean_path_len": 0.0, "median_path_len": 0.0}
    lengths = np.asarray([len(path) for path in paths], dtype=float)
    return {
        "n_paths": int(len(paths)),
        "mean_path_len": float(lengths.mean()),
        "median_path_len": float(np.median(lengths)),
    }


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_clusters(
    adata: ad.AnnData,
    leiden_key: str,
    requested: list[str],
    min_cells: int,
) -> list[str]:
    counts = adata.obs[leiden_key].astype(str).value_counts()
    if requested:
        return [cluster for cluster in requested if counts.get(cluster, 0) >= min_cells]
    return [str(cluster) for cluster, count in counts.items() if count >= min_cells]


def run(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sampleids = _parse_list(args.samples)
    if not sampleids:
        sampleids = sorted(path.stem for path in data_dir.glob("*.h5ad"))
    requested_clusters = _parse_list(args.clusters)
    summaries = []

    for sampleid in sampleids:
        h5ad_path = data_dir / f"{sampleid}.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(h5ad_path)
        adata = ad.read_h5ad(h5ad_path)
        if args.leiden_key not in adata.obs:
            raise KeyError(f"{sampleid}: obs[{args.leiden_key!r}] not found")
        if args.gene_col not in adata.obs:
            raise KeyError(f"{sampleid}: obs[{args.gene_col!r}] not found")
        if args.embedding_key not in adata.obsm:
            raise KeyError(f"{sampleid}: obsm[{args.embedding_key!r}] not found")

        if args.gene_col != "n_genes":
            adata.obs["n_genes"] = adata.obs[args.gene_col]
        if args.embedding_key != "X_phate":
            adata.obsm["X_phate"] = adata.obsm[args.embedding_key]

        organism = args.organism if args.organism != "auto" else detect_organism(adata)
        prior = _read_tf_target_prior(data_dir, organism)
        clusters = _select_clusters(
            adata, args.leiden_key, requested_clusters, args.min_cells
        )

        for cluster_id in clusters:
            cluster_mask = adata.obs[args.leiden_key].astype(str).eq(str(cluster_id))

            started = time.perf_counter()
            raw_paths = run_astar_for_cluster(
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
            paths = _reduce_paths(raw_paths)
            astar_seconds = time.perf_counter() - started

            started = time.perf_counter()
            edges = _score_edges(
                adata,
                prior,
                cluster_id,
                paths,
                args.beta_threshold,
                args.edge_frequency_threshold,
            )
            graph_seconds = time.perf_counter() - started

            edges.insert(0, "method", "astar_path")
            edges.insert(0, "cluster_id", str(cluster_id))
            edges.insert(0, "sampleid", sampleid)
            edges.to_csv(
                out_dir / f"{sampleid}__{cluster_id}__edges.csv", index=False
            )

            summary = {
                "sampleid": sampleid,
                "cluster_id": str(cluster_id),
                "organism": organism,
                "method": "astar_path",
                "n_cells_cluster": int(cluster_mask.sum()),
                "n_edges": int(len(edges)),
                "graph_seconds": float(graph_seconds),
                "astar_seconds": float(astar_seconds),
                **_path_summary(paths),
            }
            summaries.append(summary)
            (out_dir / f"{sampleid}__{cluster_id}__summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(out_dir / "benchmark_summary.csv", index=False)
    return summary_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the A*-ranked TF-target edges used in Main Figure 3."
    )
    parser.add_argument(
        "--data-dir", default=os.getenv("MCP_DATA_DIR", str(DEFAULT_DATA_DIR))
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--samples", default="imm_atlas_TT_p01")
    parser.add_argument("--clusters", default="0,1,2,3,6,9,10")
    parser.add_argument("--min-cells", type=int, default=2)
    parser.add_argument("--leiden-key", default="leiden")
    parser.add_argument("--gene-col", default="n_genes")
    parser.add_argument("--embedding-key", default="X_phate")
    parser.add_argument("--organism", choices=["auto", "human", "mouse"], default="auto")
    parser.add_argument("--beta-threshold", type=float, default=1.45)
    parser.add_argument("--edge-frequency-threshold", type=float, default=0.8)
    parser.add_argument("--q-low", type=float, default=0.1)
    parser.add_argument("--q-high", type=float, default=0.9)
    parser.add_argument("--min-paths", type=int, default=300)
    parser.add_argument("--fallback-splits", type=int, default=10)
    parser.add_argument("--max-iter-ratio", type=float, default=0.8)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
