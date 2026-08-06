#!/usr/bin/env python3
"""
Compare A*, SCENIC, and CellOracle TF-target rankings on a shared CollecTRI
edge universe.

Expected normalized edge schema:
  sampleid, cluster_id, method, source, target, score

The A* edge directory produced by benchmark_tf_target_tool.py is read directly.
SCENIC and CellOracle inputs can be any CSV/TSV/parquet long edge table with
inferable source/target/score columns, or explicit column names can be supplied.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SOURCE_ALIASES = ("source", "tf", "TF", "regulator", "transcription_factor", "motif", "regulon")
TARGET_ALIASES = ("target", "gene", "target_gene", "Target", "target_genes", "TargetGenes")
SCORE_ALIASES = ("score", "importance", "weight", "coef", "coef_mean", "activity", "auc", "AUC", "NES")
SAMPLE_ALIASES = ("sampleid", "sample", "dataset")
CLUSTER_ALIASES = ("cluster_id", "cluster", "leiden", "cell_type", "celltype")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _pick_col(df: pd.DataFrame, explicit: str | None, aliases: Iterable[str], required: bool) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Column {explicit!r} not found. Available: {list(df.columns)}")
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    if required:
        raise KeyError(f"Could not infer required column from aliases {aliases}. Available: {list(df.columns)}")
    return None


def _clean_gene(x) -> str:
    value = str(x).strip()
    if value.endswith("(+)") or value.endswith("(-)"):
        value = value[:-3]
    return value.strip().upper()


def _explode_target_column(df: pd.DataFrame, source_col: str, target_col: str, score_col: str | None) -> pd.DataFrame:
    """Handle SCENIC-like TargetGenes columns containing lists or list-of-tuples."""
    if target_col not in {"TargetGenes", "target_genes"}:
        return df

    rows = []
    keep_cols = [c for c in df.columns if c != target_col]
    for row in df.itertuples(index=False):
        data = row._asdict()
        raw = data.pop(target_col)
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            parsed = raw
        if not isinstance(parsed, (list, tuple, set)):
            parsed = [parsed]
        for item in parsed:
            out = {c: data[c] for c in keep_cols}
            if isinstance(item, (list, tuple)) and item:
                out[target_col] = item[0]
                if score_col is None and len(item) > 1:
                    out["_target_score"] = item[1]
            else:
                out[target_col] = item
            rows.append(out)
    return pd.DataFrame(rows)


def normalize_edges(
    df: pd.DataFrame,
    method: str,
    source_col: str | None = None,
    target_col: str | None = None,
    score_col: str | None = None,
    sample_col: str | None = None,
    cluster_col: str | None = None,
    default_sample: str = "all",
    default_cluster: str = "all",
) -> pd.DataFrame:
    src = _pick_col(df, source_col, SOURCE_ALIASES, required=True)
    tgt = _pick_col(df, target_col, TARGET_ALIASES, required=True)
    score = _pick_col(df, score_col, SCORE_ALIASES, required=False)
    sample = _pick_col(df, sample_col, SAMPLE_ALIASES, required=False)
    cluster = _pick_col(df, cluster_col, CLUSTER_ALIASES, required=False)

    df = _explode_target_column(df, src, tgt, score)
    if score is None and "_target_score" in df.columns:
        score = "_target_score"

    out = pd.DataFrame({
        "sampleid": df[sample].astype(str) if sample else default_sample,
        "cluster_id": df[cluster].astype(str) if cluster else default_cluster,
        "method": method,
        "source": df[src].map(_clean_gene),
        "target": df[tgt].map(_clean_gene),
        "score": pd.to_numeric(df[score], errors="coerce") if score else 1.0,
    })
    out = out.dropna(subset=["source", "target", "score"])
    out = out[(out["source"] != "") & (out["target"] != "")]
    return (
        out.groupby(["sampleid", "cluster_id", "method", "source", "target"], as_index=False)
        .agg(score=("score", "max"))
    )


def load_astar_edges(edge_dir: Path, method_name: str, astar_method: str, samples: set[str] | None) -> pd.DataFrame:
    frames = []
    for path in sorted(edge_dir.glob("*__edges.csv")):
        df = pd.read_csv(path)
        if "method" in df.columns:
            df = df[df["method"].astype(str).eq(astar_method)]
        if df.empty:
            continue
        norm = normalize_edges(df, method=method_name)
        if samples:
            norm = norm[norm["sampleid"].isin(samples)]
        frames.append(norm)
    if not frames:
        return pd.DataFrame(columns=["sampleid", "cluster_id", "method", "source", "target", "score"])
    return pd.concat(frames, ignore_index=True)


def load_collectri(args) -> pd.DataFrame:
    if args.collectri:
        df = _read_table(Path(args.collectri))
    else:
        try:
            import decoupler.op as op
        except Exception as exc:
            raise RuntimeError("--collectri was not supplied and decoupler.op.collectri is unavailable") from exc
        df = op.collectri(
            organism=args.organism,
            remove_complexes=args.remove_complexes,
            license=args.license,
            verbose=args.verbose,
        )
        if args.cache_collectri:
            Path(args.cache_collectri).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(args.cache_collectri, index=False)

    src = _pick_col(df, args.collectri_source_col, SOURCE_ALIASES, required=True)
    tgt = _pick_col(df, args.collectri_target_col, TARGET_ALIASES, required=True)
    out = pd.DataFrame({
        "source": df[src].map(_clean_gene),
        "target": df[tgt].map(_clean_gene),
    }).drop_duplicates()
    return out[(out["source"] != "") & (out["target"] != "")]


def rank_edges(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(["sampleid", "cluster_id", "method", "score"], ascending=[True, True, True, False]).copy()
    ranked["rank"] = ranked.groupby(["sampleid", "cluster_id", "method"])["score"].rank(
        method="first", ascending=False
    ).astype(int)
    return ranked


def topk_sets(ranked: pd.DataFrame, k: int) -> dict[tuple[str, str, str], set[tuple[str, str]]]:
    sets = {}
    for key, sub in ranked[ranked["rank"] <= k].groupby(["sampleid", "cluster_id", "method"]):
        sets[key] = set(zip(sub["source"], sub["target"]))
    return sets


def pairwise_jaccard(ranked: pd.DataFrame, topks: list[int]) -> pd.DataFrame:
    rows = []
    methods = sorted(ranked["method"].unique())
    contexts = ranked[["sampleid", "cluster_id"]].drop_duplicates()
    for k in topks:
        sets = topk_sets(ranked, k)
        for ctx in contexts.itertuples(index=False):
            sampleid, cluster_id = str(ctx.sampleid), str(ctx.cluster_id)
            for m1, m2 in combinations(methods, 2):
                a = sets.get((sampleid, cluster_id, m1), set())
                b = sets.get((sampleid, cluster_id, m2), set())
                union = a | b
                rows.append({
                    "sampleid": sampleid,
                    "cluster_id": cluster_id,
                    "topk": k,
                    "method_a": m1,
                    "method_b": m2,
                    "n_a": len(a),
                    "n_b": len(b),
                    "intersection": len(a & b),
                    "union": len(union),
                    "jaccard": len(a & b) / len(union) if union else np.nan,
                })
    return pd.DataFrame(rows)


def summarize_methods(ranked: pd.DataFrame, pre_filter: pd.DataFrame, topks: list[int]) -> pd.DataFrame:
    rows = []
    before = pre_filter.groupby(["sampleid", "cluster_id", "method"]).size().rename("n_edges_before_collectri")
    after = ranked.groupby(["sampleid", "cluster_id", "method"]).size().rename("n_edges_collectri")
    contexts = ranked[["sampleid", "cluster_id", "method"]].drop_duplicates()
    for row in contexts.itertuples(index=False):
        key = (row.sampleid, row.cluster_id, row.method)
        sub = ranked[
            ranked["sampleid"].eq(row.sampleid)
            & ranked["cluster_id"].eq(row.cluster_id)
            & ranked["method"].eq(row.method)
        ]
        rec = {
            "sampleid": row.sampleid,
            "cluster_id": row.cluster_id,
            "method": row.method,
            "n_edges_before_collectri": int(before.get(key, 0)),
            "n_edges_collectri": int(after.get(key, 0)),
            "collectri_retention": int(after.get(key, 0)) / max(1, int(before.get(key, 0))),
        }
        for k in topks:
            rec[f"top{k}_available"] = int((sub["rank"] <= k).sum())
            rec[f"top{k}_mean_score"] = float(sub.loc[sub["rank"] <= k, "score"].mean()) if len(sub) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["sampleid", "cluster_id", "method"])


def edge_specificity(ranked: pd.DataFrame, topks: list[int]) -> pd.DataFrame:
    rows = []
    max_k = max(topks)
    for (sampleid, method), sample_df in ranked.groupby(["sampleid", "method"]):
        score_map = sample_df.set_index(["cluster_id", "source", "target"])["score"].to_dict()
        clusters = sorted(sample_df["cluster_id"].astype(str).unique(), key=str)
        top = sample_df[sample_df["rank"] <= max_k]
        for r in top.itertuples(index=False):
            vals = {
                c: float(score_map.get((c, r.source, r.target), 0.0))
                for c in clusters
            }
            this = vals[str(r.cluster_id)]
            others = [v for c, v in vals.items() if c != str(r.cluster_id)]
            max_other = max(others) if others else 0.0
            rank = 1 + sum(v > this for v in vals.values())
            rows.append({
                "sampleid": sampleid,
                "cluster_id": r.cluster_id,
                "method": method,
                "source": r.source,
                "target": r.target,
                "rank": int(r.rank),
                "score": float(r.score),
                "cluster_rank_for_edge": int(rank),
                "is_top_cluster_for_edge": rank == 1,
                "fold_vs_max_other_cluster": this / (max_other + 1e-12),
                "present_clusters": int(sum(v > 0 for v in vals.values())),
                "total_clusters": int(len(clusters)),
            })
    return pd.DataFrame(rows)


def marker_enrichment(ranked: pd.DataFrame, marker_path: str | None, topks: list[int]) -> pd.DataFrame:
    if not marker_path:
        return pd.DataFrame()
    markers = _read_table(Path(marker_path))
    gene_col = _pick_col(markers, None, ("gene", "marker", "target"), required=True)
    sample_col = _pick_col(markers, None, SAMPLE_ALIASES, required=False)
    cluster_col = _pick_col(markers, None, CLUSTER_ALIASES, required=True)
    markers = pd.DataFrame({
        "sampleid": markers[sample_col].astype(str) if sample_col else "all",
        "cluster_id": markers[cluster_col].astype(str),
        "gene": markers[gene_col].map(_clean_gene),
    }).drop_duplicates()
    marker_sets = markers.groupby(["sampleid", "cluster_id"])["gene"].apply(set).to_dict()
    rows = []
    for (sampleid, cluster_id, method), sub in ranked.groupby(["sampleid", "cluster_id", "method"]):
        marker_set = marker_sets.get((sampleid, cluster_id), marker_sets.get(("all", cluster_id), set()))
        for k in topks:
            top = sub[sub["rank"] <= k]
            hits = top["target"].isin(marker_set).sum() if marker_set else 0
            rows.append({
                "sampleid": sampleid,
                "cluster_id": cluster_id,
                "method": method,
                "topk": k,
                "n_top_edges": int(len(top)),
                "n_marker_targets": int(hits),
                "marker_target_fraction": hits / len(top) if len(top) else np.nan,
                "n_markers_available": int(len(marker_set)),
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark A*, SCENIC, and CellOracle on CollecTRI universe.")
    p.add_argument("--collectri", default="", help="CSV/TSV/parquet CollecTRI edge table. If omitted, decoupler fetches it.")
    p.add_argument("--cache-collectri", default="", help="Optional parquet path to cache fetched CollecTRI.")
    p.add_argument("--organism", default="human")
    p.add_argument("--license", default="academic", choices=["academic", "commercial", "nonprofit"])
    p.add_argument("--remove-complexes", action="store_true")
    p.add_argument("--collectri-source-col", default="")
    p.add_argument("--collectri-target-col", default="")

    p.add_argument("--astar-dir", required=True, help="Directory containing *__edges.csv from benchmark_tf_target_tool.py.")
    p.add_argument("--astar-method", default="astar_path")
    p.add_argument("--scenic", default="", help="SCENIC edge ranking CSV/TSV/parquet.")
    p.add_argument("--celloracle", default="", help="CellOracle edge ranking CSV/TSV/parquet.")
    p.add_argument("--samples", default="", help="Optional comma-separated sample filter.")

    p.add_argument("--scenic-source-col", default="")
    p.add_argument("--scenic-target-col", default="")
    p.add_argument("--scenic-score-col", default="")
    p.add_argument("--scenic-sample-col", default="")
    p.add_argument("--scenic-cluster-col", default="")
    p.add_argument("--celloracle-source-col", default="")
    p.add_argument("--celloracle-target-col", default="")
    p.add_argument("--celloracle-score-col", default="")
    p.add_argument("--celloracle-sample-col", default="")
    p.add_argument("--celloracle-cluster-col", default="")

    p.add_argument("--default-sample", default="all")
    p.add_argument("--default-cluster", default="all")
    p.add_argument("--topk", default="10,20,50,100")
    p.add_argument("--markers", default="", help="Optional marker CSV with cluster_id,gene and optional sampleid.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = [int(x) for x in args.topk.split(",") if x.strip()]
    samples = {x.strip() for x in args.samples.split(",") if x.strip()} or None

    collectri = load_collectri(args)
    collectri.to_csv(out_dir / "collectri_universe_edges.csv", index=False)
    universe = set(zip(collectri["source"], collectri["target"]))

    frames = [load_astar_edges(Path(args.astar_dir), "astar_path", args.astar_method, samples)]
    if args.scenic:
        scenic = normalize_edges(
            _read_table(Path(args.scenic)),
            "scenic",
            args.scenic_source_col or None,
            args.scenic_target_col or None,
            args.scenic_score_col or None,
            args.scenic_sample_col or None,
            args.scenic_cluster_col or None,
            args.default_sample,
            args.default_cluster,
        )
        frames.append(scenic)
    if args.celloracle:
        oracle = normalize_edges(
            _read_table(Path(args.celloracle)),
            "celloracle",
            args.celloracle_source_col or None,
            args.celloracle_target_col or None,
            args.celloracle_score_col or None,
            args.celloracle_sample_col or None,
            args.celloracle_cluster_col or None,
            args.default_sample,
            args.default_cluster,
        )
        frames.append(oracle)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(["sampleid", "cluster_id", "method", "source", "target"])
    raw.to_csv(out_dir / "normalized_edges_before_collectri.csv", index=False)
    filtered = raw[raw.apply(lambda r: (r["source"], r["target"]) in universe, axis=1)].copy()
    ranked = rank_edges(filtered)
    ranked.to_csv(out_dir / "normalized_edges_collectri_ranked.csv", index=False)

    summarize_methods(ranked, raw, topks).to_csv(out_dir / "method_cluster_summary.csv", index=False)
    pairwise_jaccard(ranked, topks).to_csv(out_dir / "pairwise_topk_jaccard.csv", index=False)
    edge_specificity(ranked, topks).to_csv(out_dir / "topk_edge_cluster_specificity.csv", index=False)
    markers = marker_enrichment(ranked, args.markers or None, topks)
    if not markers.empty:
        markers.to_csv(out_dir / "marker_target_enrichment.csv", index=False)

    meta = {
        "collectri_edges": int(len(collectri)),
        "raw_edges": int(len(raw)),
        "collectri_filtered_edges": int(len(ranked)),
        "methods": sorted(raw["method"].unique().tolist()),
        "topk": topks,
        "outputs": [
            "collectri_universe_edges.csv",
            "normalized_edges_before_collectri.csv",
            "normalized_edges_collectri_ranked.csv",
            "method_cluster_summary.csv",
            "pairwise_topk_jaccard.csv",
            "topk_edge_cluster_specificity.csv",
            "marker_target_enrichment.csv" if not markers.empty else None,
        ],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
