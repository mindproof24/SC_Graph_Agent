"""
cwg_rust - Rust-accelerated ClusterWeightedGraph

Rust로 구현된 빠른 G2 norm 계산 및 TF-TF cascade 분석
"""

# Rust 컴파일된 모듈 임포트 (maturin이 _cwg_rust로 빌드)
from ._cwg_rust import (
    ClusterWeightedGraphRust,
    compute_all_cwg_norms,
    compute_cluster_cwg_norms,
    compute_all_cwg_norms_sparse,
    compute_cluster_cwg_norms_sparse,
    KEGGPathway,
    CascadePath,
    SelectiveIntegratedGraph,
    make_kegg_edges_bidirectional,
    build_all_integrated_graphs,
    compute_all_kegg_norms_sparse,
    compute_all_kegg_norms_cluster_mean,
    compute_all_integrated_norms_sparse,
    astar_all_pairs,
    build_conservative_graph,

    #test done
    #astar_all_pairs_legacy
)

# Python wrapper
from cwg_rust.wrapper import ClusterWeightedGraph

__all__ = [
    # Rust — lib.rs (DoRothEA CWG)
    "ClusterWeightedGraphRust",
    "compute_all_cwg_norms",
    "compute_cluster_cwg_norms",
    "compute_all_cwg_norms_sparse",
    "compute_cluster_cwg_norms_sparse",
    # Rust — selective_integrated.rs (batch sparse)
    "compute_all_kegg_norms_sparse",
    "compute_all_integrated_norms_sparse",
    "compute_all_kegg_norms_cluster_mean",
    "KEGGPathway",
    "CascadePath",
    "SelectiveIntegratedGraph",
    "make_kegg_edges_bidirectional",
    "build_all_integrated_graphs",
    # Rust — astar_phate.rs (trajectory sensitive graph finder)
    "astar_all_pairs",
    
    # testing.. - done/
    #"astar_all_pairs_legacy",
    
    
    
    # Rust — lib.rs (conservative graph builder)
    "build_conservative_graph",
    # Python wrapper
    "ClusterWeightedGraph",
]

__version__ = "0.2.0"
