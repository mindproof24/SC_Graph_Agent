#!/usr/bin/env python3
"""Plot 3-method TF-target edge Venn diagrams from ranked edge tables.

The Venn regions are computed from top-N directed TF-target edges, while the
optional edge labels are limited to edges that are also in each method's top-M.
This keeps the figure readable but preserves exact region counts in CSV files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_DEFAULT_MPL_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPL_CACHE))
_DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.patches import Patch


DEFAULT_COLORS = {
    "astar_path": "#4C78A8",
    "scenic": "#F58518",
    "celloracle": "#54A24B",
}


REGION_POS = {
    "100": (-0.78, 0.18),
    "010": (0.78, 0.18),
    "001": (0.0, -0.78),
    "110": (0.0, 0.48),
    "101": (-0.42, -0.42),
    "011": (0.42, -0.42),
    "111": (0.0, -0.02),
}

PLOT_Y_OFFSET = 0.0


def edge_key(source: str, target: str) -> str:
    return f"{source}->{target}"


def method_top_edges(df: pd.DataFrame, methods: list[str], topk: int) -> dict[str, set[str]]:
    out = {}
    for method in methods:
        sub = df[df["method"].eq(method)].sort_values("rank").head(topk)
        out[method] = {edge_key(r.source, r.target) for r in sub.itertuples(index=False)}
    return out


def method_rank_maps(df: pd.DataFrame, methods: list[str]) -> dict[str, dict[str, int]]:
    maps = {}
    for method in methods:
        sub = df[df["method"].eq(method)]
        maps[method] = {
            edge_key(r.source, r.target): int(r.rank)
            for r in sub.itertuples(index=False)
        }
    return maps


def region_code(edge: str, sets: dict[str, set[str]], methods: list[str]) -> str:
    return "".join("1" if edge in sets[m] else "0" for m in methods)


def build_region_table(
    df: pd.DataFrame,
    sampleid: str,
    cluster_id: str,
    methods: list[str],
    topk: int,
    label_topk: int,
) -> pd.DataFrame:
    sets = method_top_edges(df, methods, topk)
    label_sets = method_top_edges(df, methods, label_topk)
    rank_maps = method_rank_maps(df, methods)
    universe = sorted(set().union(*sets.values()))
    rows = []
    for edge in universe:
        code = region_code(edge, sets, methods)
        in_label_methods = [m for m in methods if edge in label_sets[m]]
        row = {
            "sampleid": sampleid,
            "cluster_id": cluster_id,
            "edge": edge,
            "region": code,
            "n_methods_topk": code.count("1"),
            "in_any_label_topk": bool(in_label_methods),
            "label_topk_methods": ";".join(in_label_methods),
        }
        for method in methods:
            row[f"in_{method}_top{topk}"] = edge in sets[method]
            row[f"in_{method}_top{label_topk}"] = edge in label_sets[method]
            row[f"{method}_rank"] = rank_maps[method].get(edge)
        rows.append(row)
    return pd.DataFrame(rows)


def format_region_label(
    region_edges: pd.DataFrame,
    label_topk: int,
    max_label_edges: int,
) -> str:
    n_topk = len(region_edges)
    label_edges = region_edges[region_edges["in_any_label_topk"]].copy()
    n_label = len(label_edges)
    lines = [f"n={n_topk}", f"top{label_topk}={n_label}"]
    if max_label_edges > 0 and n_label:
        label_edges = label_edges.sort_values(["n_methods_topk", "edge"], ascending=[False, True])
        edges = label_edges["edge"].head(max_label_edges).tolist()
        lines.extend(edges)
        if n_label > max_label_edges:
            lines.append(f"+{n_label - max_label_edges} more")
    return "\n".join(lines)


def plot_cluster_venn(
    region_df: pd.DataFrame,
    methods: list[str],
    topk: int,
    label_topk: int,
    max_label_edges: int,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8.6))
    ax.set_aspect("equal")
    ax.axis("off")

    circles = {
        methods[0]: (-0.55, 0.2 + PLOT_Y_OFFSET),
        methods[1]: (0.55, 0.2 + PLOT_Y_OFFSET),
        methods[2]: (0.0, -0.45 + PLOT_Y_OFFSET),
    }
    radius = 1.05
    for method in methods:
        x, y = circles[method]
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=DEFAULT_COLORS.get(method, "#999999"),
                edgecolor=DEFAULT_COLORS.get(method, "#555555"),
                alpha=0.28,
                linewidth=2.0,
            )
        )
    legend_handles = [
        Patch(
            facecolor=DEFAULT_COLORS.get(method, "#999999"),
            edgecolor=DEFAULT_COLORS.get(method, "#555555"),
            alpha=0.38,
            label=method,
        )
        for method in methods
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    for code, (x, y) in REGION_POS.items():
        sub = region_df[region_df["region"].eq(code)]
        label = format_region_label(sub, label_topk, max_label_edges)
        ax.text(
            x,
            y + PLOT_Y_OFFSET,
            label,
            ha="center",
            va="center",
            fontsize=8.1,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
        )

    ax.text(
        0,
        -1.70,
        f"Venn regions use top{topk} directed TF-target edges. Edge labels are restricted to top{label_topk}.",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.95, 1.52)
    ax.set_title(title, fontsize=10.5, pad=12, linespacing=1.15)
    fig.subplots_adjust(top=0.78, bottom=0.04, left=0.04, right=0.96)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def load_identity_labels(path: str) -> dict[tuple[str, str], str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    labels = {}
    for r in df.itertuples(index=False):
        sampleid = str(getattr(r, "sampleid", ""))
        cluster_id = str(getattr(r, "cluster_id"))
        parts = []
        for i in (1, 2, 3):
            ct = getattr(r, f"celltype{i}", None)
            frac = getattr(r, f"frac{i}", None)
            if pd.notna(ct) and pd.notna(frac):
                short = str(ct)
                short = short.replace("thymus-derived ", "")
                short = short.replace(", alpha-beta", "")
                short = short.replace("-positive", "+")
                short = short.replace("positive", "+")
                short = short.replace("-", " ")
                short = " ".join(short.split())
                parts.append(f"{short} {float(frac) * 100:.1f}%")
        if parts:
            labels[(sampleid, cluster_id)] = "\n".join(parts)
            labels[("", cluster_id)] = labels[(sampleid, cluster_id)]
    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ranked-edges", required=True, help="normalized_edges_collectri_ranked.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--methods", default="astar_path,scenic,celloracle")
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--label-topk", type=int, default=10)
    p.add_argument("--max-label-edges", type=int, default=3)
    p.add_argument("--samples", default="", help="Optional comma-separated sample filter.")
    p.add_argument("--clusters", default="", help="Optional comma-separated cluster filter.")
    p.add_argument("--cluster-identity", default="", help="Optional CSV with sampleid, cluster_id, celltype1/frac1 ... celltype3/frac3.")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if len(methods) != 3:
        raise ValueError("--methods must contain exactly three method names.")

    df = pd.read_csv(args.ranked_edges)
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["sampleid"] = df["sampleid"].astype(str)
    if args.samples:
        samples = {s.strip() for s in args.samples.split(",") if s.strip()}
        df = df[df["sampleid"].isin(samples)]
    if args.clusters:
        clusters = {c.strip() for c in args.clusters.split(",") if c.strip()}
        df = df[df["cluster_id"].isin(clusters)]
    identity_labels = load_identity_labels(args.cluster_identity)

    all_regions = []
    summary_rows = []
    for (sampleid, cluster_id), sub in df.groupby(["sampleid", "cluster_id"], sort=True):
        if not set(methods).issubset(set(sub["method"])):
            continue
        region_df = build_region_table(sub, sampleid, cluster_id, methods, args.topk, args.label_topk)
        all_regions.append(region_df)

        for code in sorted(REGION_POS):
            rsub = region_df[region_df["region"].eq(code)]
            summary_rows.append({
                "sampleid": sampleid,
                "cluster_id": cluster_id,
                "region": code,
                "methods_in_region": ";".join(m for bit, m in zip(code, methods) if bit == "1"),
                f"n_top{args.topk}_edges": len(rsub),
                f"n_edges_also_in_any_top{args.label_topk}": int(rsub["in_any_label_topk"].sum()),
                "example_label_edges": ";".join(rsub[rsub["in_any_label_topk"]]["edge"].head(args.max_label_edges).tolist()),
            })

        safe_sample = sampleid.replace("/", "_")
        safe_cluster = cluster_id.replace("/", "_")
        out_path = out_dir / f"{safe_sample}__cluster_{safe_cluster}__top{args.topk}_venn.{args.format}"
        identity = identity_labels.get((sampleid, cluster_id), identity_labels.get(("", cluster_id), ""))
        title = f"{sampleid} cluster {cluster_id}"
        if identity:
            title = f"{title}\n{identity}"
        plot_cluster_venn(
            region_df,
            methods,
            args.topk,
            args.label_topk,
            args.max_label_edges,
            title,
            out_path,
        )

    if not all_regions:
        raise RuntimeError("No sample/cluster contexts had all three requested methods.")
    pd.concat(all_regions, ignore_index=True).to_csv(out_dir / f"top{args.topk}_venn_region_edges.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / f"top{args.topk}_venn_region_summary.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
