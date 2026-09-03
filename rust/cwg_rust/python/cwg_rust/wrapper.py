"""
Python interface for the Rust-accelerated ClusterWeighted implementation.
Usage:
    pip install maturin
    cd cwg_rust && maturin develop --release
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from typing import Dict, List, Optional, Union, Tuple
import time

# Import the compiled Rust extension when available.
try:
    from cwg_rust._cwg_rust import (
        ClusterWeightedGraphRust,
        compute_all_cwg_norms,
        compute_cluster_cwg_norms,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: cwg_rust._cwg_rust not available. Install with 'cd cwg_rust && maturin develop --release'")


class ClusterWeightedGraph:
    """
    Python wrapper for Rust-accelerated graph-norm calculations.

    The wrapper constructs a graph from prior TF-target edges and exposes
    per-cell and cluster-restricted edge-L2 activity calculations.
    """
    
    def __init__(
        self,
        adata,
        dorothea_df: pd.DataFrame,
        cluster_id,
        cluster_key: str = 'leiden',
        confidence_levels: List[str] = ['A', 'B', 'C'],
        tf_expr_threshold: float = 0.0,
        target_expr_threshold: float = 0.0,
        require_both_expressed: bool = True,
        beta_mode: str = 'dynamic',
    ):
        """
        Construct a ClusterWeightedGraph.
        
        Args:
            adata: AnnData object.
            dorothea_df: DoRothEA DataFrame
            cluster_id: Cluster identifier.
            cluster_key: Column in ``adata.obs`` containing cluster labels.
            confidence_levels: DoRothEA confidence levels
            tf_expr_threshold: Minimum TF expression.
            target_expr_threshold: Minimum target expression.
            require_both_expressed: Retain only edges with both genes expressed.
            beta_mode: 'dynamic' or 'static'
        """
        if not RUST_AVAILABLE:
            raise ImportError(
                "cwg_rust module not available. "
                "Install with: cd cwg_rust && maturin develop --release"
            )
        
        self.adata = adata
        self.cluster_id = str(cluster_id)
        self.cluster_key = cluster_key
        self.beta_mode = beta_mode
        self.pathway_name = f"adata_{cluster_key}{cluster_id}"
        
        # Prepare the expression matrix
        if issparse(adata.X):
            self._expr_matrix = np.ascontiguousarray(
                adata.X.toarray(), dtype=np.float64
            )
        else:
            self._expr_matrix = np.ascontiguousarray(
                adata.X, dtype=np.float64
            )
        
        # Preserve the AnnData gene order used by the expression matrix.
        self._gene_names = list(adata.var_names)
        
        # Build the cluster mask.
        cluster_mask = (
            adata.obs[cluster_key].astype(str) == self.cluster_id
        ).values
        
        # Normalize DoRothEA columns and apply confidence filtering
        df = self._prepare_dorothea(dorothea_df, confidence_levels)
        
        # Construct the Rust graph object.
        self._rust_cwg = ClusterWeightedGraphRust(
            expr_matrix=self._expr_matrix,
            gene_names=self._gene_names,
            cluster_mask=np.ascontiguousarray(cluster_mask),
            dorothea_sources=df['source'].tolist(),
            dorothea_targets=df['target'].tolist(),
            dorothea_weights=np.ascontiguousarray(
                df['weight'].values, dtype=np.float64
            ),
            dorothea_confidences=df['confidence'].tolist(),
            cluster_id=self.cluster_id,
            cluster_key=self.cluster_key,
            confidence_levels=confidence_levels,
            tf_expr_threshold=tf_expr_threshold,
            target_expr_threshold=target_expr_threshold,
            require_both_expressed=require_both_expressed,
            beta_mode=beta_mode,
        )
        
        # Expose selected Rust properties through the Python wrapper.
        self.pathway_genes = self._rust_cwg.get_pathway_genes()
        self.n_genes = self._rust_cwg.n_genes
        self.n_edges = self._rust_cwg.n_edges
        self.n_tf_tf_edges = self._rust_cwg.n_tf_tf_edges
        
        # Retain the indices of cells included in the cluster.
        self._cluster_cells = self._rust_cwg.get_cluster_cells()
        
        # Cache edge-level values used by convience methods.
        self._cache_edge_data()
    
    def _prepare_dorothea(
        self, 
        df: pd.DataFrame, 
        confidence_levels: List[str]
    ) -> pd.DataFrame:
        """Normalize a DoRothEA table for the Rust constructor."""
        df = df.copy()
        
        # Normalize supported column aliases.
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['source', 'tf', 'regulator']:
                col_map[col] = 'source'
            elif col_lower in ['target', 'gene']:
                col_map[col] = 'target'
            elif col_lower in ['weight', 'mor']:
                col_map[col] = 'weight'
            elif col_lower in ['confidence', 'level']:
                col_map[col] = 'confidence'
        
        df = df.rename(columns=col_map)
        
        if 'weight' not in df.columns:
            df['weight'] = 1.0
        if 'confidence' not in df.columns:
            df['confidence'] = 'A'
        
        # Filtering confidence levels.
        df = df[df['confidence'].isin(confidence_levels)]
        
        return df
    
    def _cache_edge_data(self):
        """Cache edge values and source/target expression summaries"""
        edges_data = self._rust_cwg.get_edges_data()
        
        self.edge_weights = {}
        self.dorothea_weights = {}
        
        for i in range(len(edges_data['source'])):
            key = (edges_data['source'][i], edges_data['target'][i])
            self.edge_weights[key] = edges_data['beta'][i]
            self.dorothea_weights[key] = edges_data['dorothea_weight'][i]
        
        self.source_expr = dict(self._rust_cwg.get_source_expr())
        self.target_expr = dict(self._rust_cwg.get_target_expr())
    
    # =========================================================
    # G2 Norm 계산
    # =========================================================
    
    def compute_graph_norm(self, cell_id: str) -> float:
        """Calculate the graph norm for one cell."""
        cell_idx = self.adata.obs.index.get_loc(cell_id)
        return self._rust_cwg.compute_graph_norm(self._expr_matrix, cell_idx)
    
    def compute_all_norms(self, add_to_obs: bool = True) -> np.ndarray:
        """Calculate graph norms for all cells in parallel."""
        norms = np.array(self._rust_cwg.compute_all_norms(self._expr_matrix))
        
        if add_to_obs:
            col_name = f"{self.pathway_name}_G2"
            self.adata.obs[col_name] = norms
        
        return norms
    
    def compute_cluster_norms(
        self, 
        add_to_obs: bool = True,
        return_with_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Calculate graph norms for cells in the selected cluster.
        
        Args:
            add_to_obs: Add the result to ``adata.obs``.
            return_with_indices: Return the corresponding cell indices.
            
        Returns:
            norms: Graph norms for cluster cells.
            indices: Cluster-cell indices when requested.
        """
        norms = np.array(
            self._rust_cwg.compute_cluster_norms(self._expr_matrix)
        )
        
        if add_to_obs:
            col_name = f"{self.pathway_name}_cluster_G2"
            # Assign norms to cluster cells and leave all other cells as NaN.
            full_norms = np.full(len(self.adata), np.nan)
            full_norms[self._cluster_cells] = norms
            self.adata.obs[col_name] = full_norms
        
        if return_with_indices:
            return norms, self._cluster_cells
        return norms
    
    def get_cluster_activity_summary(self) -> Dict:
        """Summarize graph activity within the selected cluster"""
        norms = self.compute_cluster_norms(add_to_obs=False)
        
        return {
            'pathway_name': self.pathway_name,
            'n_cluster_cells': len(self._cluster_cells),
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'median_norm': float(np.median(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'n_genes': self.n_genes,
            'n_edges': self.n_edges,

        }

    # =========================================================
    # Edge and expression sumaries
    # =========================================================
    def get_edges_df(self) -> pd.DataFrame:
        """Return all retained edges as a DataFrame."""
        return pd.DataFrame(self._rust_cwg.get_edges_data())
    
    def get_expression_summary(self) -> pd.DataFrame:
        """Return source and target expression for each retained edge."""
        edges_data = self._rust_cwg.get_edges_data()
        
        records = []
        for i in range(len(edges_data['source'])):
            src = edges_data['source'][i]
            tgt = edges_data['target'][i]
            
            records.append({
                'source': src,
                'target': tgt,
                'source_expr': self.source_expr.get(src, 0),
                'target_expr': self.target_expr.get(tgt, 0),
                'dorothea_weight': edges_data['dorothea_weight'][i],
                'beta': edges_data['beta'][i],
            })
        
        return pd.DataFrame(records)
    
    def __repr__(self):
        return (
            f"ClusterWeightedGraph('{self.pathway_name}', "
            f"genes={self.n_genes}, edges={self.n_edges})"
        )



# =============================================================
# Batch helpers
# =============================================================

def build_all_cluster_graphs(
    adata,
    dorothea_df: pd.DataFrame,
    cluster_key: str = 'leiden',
    cluster_ids: Optional[List] = None,
    confidence_levels: List[str] = ['A', 'B', 'C'],
    tf_expr_threshold: float = 0.0,
    target_expr_threshold: float = 0.0,
    require_both_expressed: bool = True,
    beta_mode: str = 'dynamic',
    verbose: bool = True,
) -> Dict[str, ClusterWeightedGraph]:
    """
    Build a ClusterWeightedGraph for every selected cluster.
    
    Args:
        adata: AnnData object.
        dorothea_df: DoRothEA DataFrame
        cluster_key: Column in ``adata.obs`` containing cluster labels.
        cluster_ids: Optional subset of cluster identifiers.
        confidence_levels: DoRothEA confidence levels
        tf_expr_threshold: Minimum TF expression.
        target_expr_threshold: Minimum target expression.
        require_both_expressed: Require both genes to be expressed.
        beta_mode: 'dynamic' or 'static'
        verbose: Print progress information.
        
    Returns:
        {pathway_name: ClusterWeightedGraph}
    """
    all_clusters = adata.obs[cluster_key].unique()
    
    if cluster_ids is None:
        selected_clusters = sorted(all_clusters, key=str)
    else:
        cluster_ids_str = [str(c) for c in cluster_ids]
        selected_clusters = [
            c for c in all_clusters if str(c) in cluster_ids_str
        ]
        selected_clusters = sorted(selected_clusters, key=str)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Building ClusterWeightedGraphs")
        print(f"{'='*60}")
        print(f"Selected clusters: {len(selected_clusters)}")
    
    start_time = time.time()
    cwg_dict = {}
    
    for cluster_id in selected_clusters:
        cwg = ClusterWeightedGraph(
            adata=adata,
            dorothea_df=dorothea_df,
            cluster_id=cluster_id,
            cluster_key=cluster_key,
            confidence_levels=confidence_levels,
            tf_expr_threshold=tf_expr_threshold,
            target_expr_threshold=target_expr_threshold,
            require_both_expressed=require_both_expressed,
            beta_mode=beta_mode,
        )
        cwg_dict[cwg.pathway_name] = cwg
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Built {len(cwg_dict)} graphs in {elapsed:.2f}s")
    
    return cwg_dict


def compute_all_cluster_norms(
    adata,
    cwg_dict: Dict[str, ClusterWeightedGraph],
    add_to_obs: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate graph norms for every cell and graph.
    
    Args:
        adata: AnnData object.
        cwg_dict: Mapping of graph names to wrappers.
        add_to_obs: Add each result to ``adata.obs``.
        verbose: Print progress information.
        
    Returns:
        DataFrame (rows=cells, cols=graph norms)
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Computing G2 Norms for All Cells")
        print(f"{'='*60}")
        print(f"Total cells: {len(adata)}")
        print(f"Total graphs: {len(cwg_dict)}")
    
    # Prepare the expression matrix.
    if issparse(adata.X):
        expr_matrix = np.ascontiguousarray(
            adata.X.toarray(), dtype=np.float64
        )
    else:
        expr_matrix = np.ascontiguousarray(adata.X, dtype=np.float64)
    
    # Collect the Rust graph objects.
    rust_cwg_list = [cwg._rust_cwg for cwg in cwg_dict.values()]
    
    # Calculate all norms in one batch.
    results = compute_all_cwg_norms(expr_matrix, rust_cwg_list)
    
    # Store results by graph name.
    norm_data = {}
    for col_name, norms in results.items():
        norms = np.array(norms)
        norm_data[col_name] = norms
        
        if add_to_obs:
            adata.obs[col_name] = norms
        
        if verbose:
            print(f"  {col_name}: mean={np.mean(norms):.4f}")
    
    norm_df = pd.DataFrame(norm_data, index=adata.obs.index)
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✓ Completed in {elapsed:.2f}s")
    
    return norm_df


def compute_cluster_activity_norms(
    adata,
    cwg_dict: Dict[str, ClusterWeightedGraph],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate graph activity over the cells assigned to each graph's cluster.
    
    Args:
        adata: AnnData object.
        cwg_dict: Mapping of graph names to wrappers.
        verbose: Print progress information.
        
    Returns:
        DataFrame (rows=graphs, cols=activity metrics)
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Computing Cluster Activity")
        print(f"{'='*60}")
    
    results = []
    
    for pathway_name, cwg in cwg_dict.items():
        summary = cwg.get_cluster_activity_summary()
        results.append(summary)
        
        if verbose:
            print(
                f"  {pathway_name}: "
                f"cells={summary['n_cluster_cells']}, "
                f"mean_norm={summary['mean_norm']:.4f}"
            )
    
    df = pd.DataFrame(results)
    df = df.set_index('pathway_name')
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✓ Completed in {elapsed:.2f}s")
    
    return df


# Usage example

if __name__ == "__main__":
    print("""
    Usage Example:
    ==============
    
    import scanpy as sc
    from cwg_wrapper import (
        ClusterWeightedGraph, 
        build_all_cluster_graphs,
        compute_all_cluster_norms,
        compute_cluster_activity_norms,
    )
    
    # 1. Build one graph.
    cwg = ClusterWeightedGraph(
        adata=adata,
        dorothea_df=dorothea,
        cluster_id='13',
        cluster_key='leiden',
        require_both_expressed=True,
    )
    print(cwg)
    
    # 2. Calculate norms for all cells.
    norms = cwg.compute_all_norms(add_to_obs=True)
    
    # 3. Calculate activity only over cells in the selected cluster.
    cluster_norms = cwg.compute_cluster_norms()
    summary = cwg.get_cluster_activity_summary()
    
    # 4. Build graphs for all clusters.
    cwg_dict = build_all_cluster_graphs(
        adata=adata,
        dorothea_df=dorothea,
        cluster_key='leiden',
    )
    
    # 5. Calculate all graph norms in one batch.
    norm_df = compute_all_cluster_norms(adata, cwg_dict)
    
    # 6. Compare cluster activity.
    activity_df = compute_cluster_activity_norms(adata, cwg_dict)
    
    """)
