#!/usr/bin/env python3
"""Assemble the seven clean Venn panels into Supplementary Figure 1."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Patch


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

_spec = importlib.util.spec_from_file_location(
    "venn_base", ROOT / "scripts" / "plot_topk_edge_venn.py"
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load Venn plotting helpers")
venn_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(venn_base)


METHODS = ["astar_path", "scenic", "celloracle"]
METHOD_LABELS = {"astar_path": "A*", "scenic": "SCENIC", "celloracle": "CellOracle"}
CLUSTERS = ["0", "1", "2", "3", "6", "9", "10"]
LETTERS = "abcdefg"


def draw_panel(
    ax: plt.Axes,
    regions: pd.DataFrame,
    title: str,
    letter: str,
) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    centers = {
        "astar_path": (-0.55, 0.20),
        "scenic": (0.55, 0.20),
        "celloracle": (0.0, -0.45),
    }
    for method in METHODS:
        color = venn_base.DEFAULT_COLORS[method]
        ax.add_patch(
            Circle(
                centers[method],
                1.05,
                facecolor=color,
                edgecolor=color,
                alpha=0.28,
                linewidth=1.5,
            )
        )
    for code, (x, y) in venn_base.REGION_POS.items():
        subset = regions[regions["region"].eq(code)]
        label = venn_base.format_region_label(subset, label_topk=10, max_label_edges=3)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=5.25,
            linespacing=1.05,
        )
    ax.text(-1.72, 1.43, letter, fontsize=13, fontweight="bold", ha="left", va="top")
    ax.set_title(title, fontsize=7.2, pad=4, linespacing=1.1)
    ax.set_xlim(-1.78, 1.78)
    ax.set_ylim(-1.62, 1.42)


def main() -> None:
    data_dir = ROOT / "data"
    plots_dir = ROOT / "results"
    ranked = pd.read_csv(data_dir / "normalized_edges_collectri_ranked.csv")
    ranked["sampleid"] = ranked["sampleid"].astype(str)
    ranked["cluster_id"] = ranked["cluster_id"].astype(str)
    identities = venn_base.load_identity_labels(str(data_dir / "cluster_identity_for_venn.csv"))

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    axes = axes.ravel()
    sampleid = "imm_atlas_TT_p01"
    for ax, cluster_id, letter in zip(axes, CLUSTERS, LETTERS):
        subset = ranked[
            ranked["sampleid"].eq(sampleid) & ranked["cluster_id"].eq(cluster_id)
        ]
        regions = venn_base.build_region_table(
            subset,
            sampleid,
            cluster_id,
            METHODS,
            topk=200,
            label_topk=10,
        )
        identity = identities.get((sampleid, cluster_id), "")
        title = f"Cluster {cluster_id}"
        if identity:
            title += f"\n{identity}"
        draw_panel(ax, regions, title, letter)

    axes[-1].axis("off")
    handles = [
        Patch(
            facecolor=venn_base.DEFAULT_COLORS[method],
            edgecolor=venn_base.DEFAULT_COLORS[method],
            alpha=0.38,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    axes[-1].legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=13,
        ncol=1,
        title="Top-200 directed TF-target edges",
        title_fontsize=12,
    )
    axes[-1].text(
        0.5,
        0.27,
        "Region labels retain complete top-200 counts and\nrepresentative edges from each method's top 10.",
        transform=axes[-1].transAxes,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.3,
    )
    fig.subplots_adjust(left=0.035, right=0.98, top=0.965, bottom=0.025, hspace=0.22, wspace=0.08)

    pdf_path = plots_dir / "supplementary_figure1_top200_venn.pdf"
    png_path = plots_dir / "supplementary_figure1_top200_venn.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=240)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
