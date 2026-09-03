"""
cwg_rust - Rust-accelerated ClusterWeightedGraph

Python exports for Rust graph activity and A* path calculations.
"""

# Maturin builds the compiled extension as ``_cwg_rust``.
from ._cwg_rust import (
    ClusterWeightedGraphRust,
    compute_all_cwg_norms,
    compute_cluster_cwg_norms,
    compute_all_cwg_norms_sparse,
    compute_cluster_cwg_norms_sparse,
    KEGGPathway,
    make_kegg_edges_bidirectional,
    compute_all_kegg_norms_sparse,
    compute_all_kegg_norms_cluster_mean,
    astar_all_pairs,
    build_conservative_graph,
)

# Python convenience wrapper.
from .wrapper import ClusterWeightedGraph

__all__ = [
    # DoRothEA graph activity.
    "ClusterWeightedGraphRust",
    "compute_all_cwg_norms",
    "compute_cluster_cwg_norms",
    "compute_all_cwg_norms_sparse",
    "compute_cluster_cwg_norms_sparse",
    # KEGG graph activity.
    "compute_all_kegg_norms_sparse",
    "compute_all_kegg_norms_cluster_mean",
    "KEGGPathway",
    "make_kegg_edges_bidirectional",
    # PHATE-coordinate A* path search.
    "astar_all_pairs",
    # Conservative TF-target graph construction.
    "build_conservative_graph",
    # Python convenience wrapper.
    "ClusterWeightedGraph",
]

__version__ = "0.2.0"
