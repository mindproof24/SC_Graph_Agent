"""
ClusterWeightedGraph - Rust 가속 버전

Rust로 구현된 빠른 G2 norm 계산 및 TF-TF cascade 분석

사용법:
    pip install maturin
    cd cwg_rust && maturin develop --release
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from typing import Dict, List, Optional, Union, Tuple
import time

# Rust 모듈 임포트 시도
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
    Rust 가속 ClusterWeightedGraph
    
    Features:
    - 빠른 G2 norm 계산 (병렬)
    - TF-TF cascade 분석
    - Greedy max-beta path 탐색
    - 클러스터 activity 계산
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
        ClusterWeightedGraph 생성
        
        Args:
            adata: AnnData 객체
            dorothea_df: DoRothEA DataFrame
            cluster_id: 클러스터 ID
            cluster_key: 클러스터 컬럼명
            confidence_levels: DoRothEA confidence levels
            tf_expr_threshold: TF 최소 발현량
            target_expr_threshold: Target 최소 발현량
            require_both_expressed: 둘 다 발현된 edge만
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
        
        # Expression matrix 준비
        if issparse(adata.X):
            self._expr_matrix = np.ascontiguousarray(
                adata.X.toarray(), dtype=np.float64
            )
        else:
            self._expr_matrix = np.ascontiguousarray(
                adata.X, dtype=np.float64
            )
        
        # 유전자 이름 리스트
        self._gene_names = list(adata.var_names)
        
        # 클러스터 마스크
        cluster_mask = (
            adata.obs[cluster_key].astype(str) == self.cluster_id
        ).values
        
        # DoRothEA 컬럼 감지 및 데이터 준비
        df = self._prepare_dorothea(dorothea_df, confidence_levels)
        
        # Rust CWG 생성
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
        
        # Python 속성 동기화
        self.pathway_genes = self._rust_cwg.get_pathway_genes()
        self.n_genes = self._rust_cwg.n_genes
        self.n_edges = self._rust_cwg.n_edges
        self.n_tf_tf_edges = self._rust_cwg.n_tf_tf_edges
        
        # 클러스터 세포 인덱스
        self._cluster_cells = self._rust_cwg.get_cluster_cells()
        
        # Edge 데이터 캐싱
        self._cache_edge_data()
    
    def _prepare_dorothea(
        self, 
        df: pd.DataFrame, 
        confidence_levels: List[str]
    ) -> pd.DataFrame:
        """DoRothEA DataFrame 준비 및 정규화"""
        df = df.copy()
        
        # 컬럼명 정규화
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
        
        # 필수 컬럼 확인
        if 'weight' not in df.columns:
            df['weight'] = 1.0
        if 'confidence' not in df.columns:
            df['confidence'] = 'A'
        
        # Confidence 필터링 (Python에서 먼저)
        df = df[df['confidence'].isin(confidence_levels)]
        
        return df
    
    def _cache_edge_data(self):
        """Edge 데이터 캐싱"""
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
        """단일 세포의 G2 norm 계산"""
        cell_idx = self.adata.obs.index.get_loc(cell_id)
        return self._rust_cwg.compute_graph_norm(self._expr_matrix, cell_idx)
    
    def compute_all_norms(self, add_to_obs: bool = True) -> np.ndarray:
        """모든 세포의 G2 norm 계산 (병렬)"""
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
        클러스터 세포들만의 G2 norm 계산 (클러스터 activity)
        
        Args:
            add_to_obs: adata.obs에 추가할지
            return_with_indices: 세포 인덱스와 함께 반환할지
            
        Returns:
            norms: 클러스터 세포들의 G2 norm 배열
            (optional) indices: 클러스터 세포 인덱스
        """
        norms = np.array(
            self._rust_cwg.compute_cluster_norms(self._expr_matrix)
        )
        
        if add_to_obs:
            col_name = f"{self.pathway_name}_cluster_G2"
            # 클러스터 세포만 값 할당
            full_norms = np.full(len(self.adata), np.nan)
            full_norms[self._cluster_cells] = norms
            self.adata.obs[col_name] = full_norms
        
        if return_with_indices:
            return norms, self._cluster_cells
        return norms
    
    def get_cluster_activity_summary(self) -> Dict:
        """클러스터 activity 요약 통계"""
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
            'n_tf_tf_edges': self.n_tf_tf_edges,
        }
    
    # =========================================================
    # Greedy Max-Beta Path
    # =========================================================
    
    def find_greedy_max_beta_path(
        self,
        beta_threshold: float = 0.5,
        max_length: int = 50,
        top_n_starts: int = 1,
    ) -> Dict:
        """
        가장 activity가 큰 TF-TF edge에서 시작하여
        최대 beta 방향으로 이어지는 경로 탐색
        
        Args:
            beta_threshold: 최소 beta 임계값 (이하면 멈춤)
            max_length: 최대 경로 길이
            top_n_starts: 시작점 후보 수
            
        Returns:
            {
                'path': ['TF1', 'TF2', 'TF3', ...],
                'betas': [1.5, 1.2, 0.9, ...],
                'total_beta': 3.6,
                'length': 4,
                'all_paths': [...],
            }
        """
        result = self._rust_cwg.find_greedy_max_beta_path(
            beta_threshold, max_length, top_n_starts
        )
        return dict(result)
    
    def find_all_greedy_paths(
        self,
        beta_threshold: float = 0.5,
        min_path_length: int = 3,
        max_length: int = 50,
    ) -> pd.DataFrame:
        """
        모든 TF에서 시작하는 Greedy Path 탐색
        
        Args:
            beta_threshold: 최소 beta 임계값
            min_path_length: 최소 경로 길이
            max_length: 최대 경로 길이
            
        Returns:
            DataFrame with columns:
            - start_tf: 시작 TF
            - path: 경로 문자열 (TF1 -> TF2 -> ...)
            - length: 경로 길이
            - total_beta: 총 beta 합
            - avg_beta: 평균 beta
        """
        result = self._rust_cwg.find_all_greedy_paths(
            beta_threshold, min_path_length, max_length
        )
        return pd.DataFrame(result)
    
    # =========================================================
    # Edge 및 그래프 정보
    # =========================================================
    
    def get_edges_df(self) -> pd.DataFrame:
        """모든 edge 정보를 DataFrame으로"""
        return pd.DataFrame(self._rust_cwg.get_edges_data())
    
    def get_tf_tf_edges_df(self) -> pd.DataFrame:
        """TF-TF edge만 DataFrame으로"""
        return pd.DataFrame(self._rust_cwg.get_tf_tf_edges())
    
    def get_expression_summary(self) -> pd.DataFrame:
        """Edge별 발현량 요약"""
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
            f"genes={self.n_genes}, edges={self.n_edges}, "
            f"tf_tf_edges={self.n_tf_tf_edges})"
        )


# =============================================================
# 배치 처리 함수들
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
    모든/특정 클러스터에 대해 ClusterWeightedGraph 생성
    
    Args:
        adata: AnnData 객체
        dorothea_df: DoRothEA DataFrame
        cluster_key: 클러스터 컬럼명
        cluster_ids: 특정 클러스터만 (None이면 전체)
        confidence_levels: DoRothEA confidence levels
        tf_expr_threshold: TF 최소 발현량
        target_expr_threshold: Target 최소 발현량
        require_both_expressed: 둘 다 발현 필수
        beta_mode: 'dynamic' or 'static'
        verbose: 진행상황 출력
        
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
    모든 CWG에 대해 전체 세포의 G2 norm 계산
    
    Args:
        adata: AnnData 객체
        cwg_dict: CWG 딕셔너리
        add_to_obs: adata.obs에 추가
        verbose: 진행상황 출력
        
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
    
    # Expression matrix 준비
    if issparse(adata.X):
        expr_matrix = np.ascontiguousarray(
            adata.X.toarray(), dtype=np.float64
        )
    else:
        expr_matrix = np.ascontiguousarray(adata.X, dtype=np.float64)
    
    # Rust CWG 리스트
    rust_cwg_list = [cwg._rust_cwg for cwg in cwg_dict.values()]
    
    # 배치 계산
    results = compute_all_cwg_norms(expr_matrix, rust_cwg_list)
    
    # 결과 저장
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
    각 CWG에 대해 해당 클러스터 세포들만의 G2 norm 계산
    (클러스터 activity 측정)
    
    Args:
        adata: AnnData 객체
        cwg_dict: CWG 딕셔너리
        verbose: 진행상황 출력
        
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


def find_top_greedy_paths(
    cwg_dict: Dict[str, ClusterWeightedGraph],
    beta_threshold: float = 0.5,
    min_path_length: int = 3,
    top_n_per_cluster: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    모든 클러스터에서 상위 Greedy Path 탐색
    
    Args:
        cwg_dict: CWG 딕셔너리
        beta_threshold: 최소 beta 임계값
        min_path_length: 최소 경로 길이
        top_n_per_cluster: 클러스터당 상위 N개
        verbose: 진행상황 출력
        
    Returns:
        DataFrame with top paths across all clusters
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Finding Top Greedy Paths")
        print(f"{'='*60}")
    
    all_paths = []
    
    for pathway_name, cwg in cwg_dict.items():
        paths_df = cwg.find_all_greedy_paths(
            beta_threshold=beta_threshold,
            min_path_length=min_path_length,
        )
        
        if len(paths_df) > 0:
            paths_df['cluster'] = pathway_name
            top_paths = paths_df.head(top_n_per_cluster)
            all_paths.append(top_paths)
            
            if verbose:
                print(f"  {pathway_name}: {len(paths_df)} paths found")
    
    if all_paths:
        result_df = pd.concat(all_paths, ignore_index=True)
        result_df = result_df.sort_values('total_beta', ascending=False)
        return result_df
    
    return pd.DataFrame()


# =============================================================
# 시각화 함수
# =============================================================

def visualize_greedy_path(
    cwg: ClusterWeightedGraph,
    path_info: Optional[Dict] = None,
    beta_threshold: float = 0.5,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Greedy Path 시각화
    
    Args:
        cwg: ClusterWeightedGraph
        path_info: find_greedy_max_beta_path 결과 (없으면 자동 계산)
        beta_threshold: beta 임계값
        figsize: 그림 크기
    """
    import matplotlib.pyplot as plt
    
    if path_info is None:
        path_info = cwg.find_greedy_max_beta_path(
            beta_threshold=beta_threshold
        )
    
    path = path_info['path']
    betas = path_info['betas']
    
    if len(path) < 2:
        print("Path too short to visualize")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 노드 위치 (선형 배치)
    x_pos = list(range(len(path)))
    y_pos = [0] * len(path)
    
    # Edge 그리기
    for i in range(len(betas)):
        x = [x_pos[i], x_pos[i+1]]
        y = [0, 0]
        width = betas[i] * 2
        ax.plot(x, y, 'b-', linewidth=width, alpha=0.6)
        
        # Beta 값 표시
        mid_x = (x_pos[i] + x_pos[i+1]) / 2
        ax.text(mid_x, 0.15, f'{betas[i]:.2f}', ha='center', fontsize=9)
    
    # 노드 그리기
    ax.scatter(x_pos, y_pos, s=500, c='lightblue', edgecolors='black', zorder=5)
    
    # 노드 라벨
    for i, tf in enumerate(path):
        ax.text(x_pos[i], 0, tf, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # 임계값 표시
    ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
    ax.text(0, -0.35, f'β threshold: {beta_threshold}', color='red', fontsize=9)
    
    ax.set_xlim(-0.5, len(path) - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(
        f"{cwg.pathway_name}: Greedy Max-Beta Path\n"
        f"Total β = {path_info['total_beta']:.2f}, Length = {path_info['length']}"
    )
    ax.axis('off')
    
    plt.tight_layout()
    return fig, ax


# =============================================================
# 예시 사용법
# =============================================================

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
        find_top_greedy_paths,
    )
    
    # 1. 단일 CWG 생성
    cwg = ClusterWeightedGraph(
        adata=adata,
        dorothea_df=dorothea,
        cluster_id='13',
        cluster_key='leiden',
        require_both_expressed=True,
    )
    print(cwg)
    
    # 2. 모든 세포 norm 계산
    norms = cwg.compute_all_norms(add_to_obs=True)
    
    # 3. 클러스터 세포만 norm 계산 (activity)
    cluster_norms = cwg.compute_cluster_norms()
    summary = cwg.get_cluster_activity_summary()
    
    # 4. Greedy Max-Beta Path 탐색
    path_info = cwg.find_greedy_max_beta_path(
        beta_threshold=0.5,
        max_length=20,
    )
    print(f"Best path: {' -> '.join(path_info['path'])}")
    print(f"Total beta: {path_info['total_beta']:.2f}")
    
    # 5. 모든 클러스터 CWG 생성
    cwg_dict = build_all_cluster_graphs(
        adata=adata,
        dorothea_df=dorothea,
        cluster_key='leiden',
    )
    
    # 6. 배치 norm 계산
    norm_df = compute_all_cluster_norms(adata, cwg_dict)
    
    # 7. 클러스터 activity 비교
    activity_df = compute_cluster_activity_norms(adata, cwg_dict)
    
    # 8. 전체 Top Greedy Paths
    paths_df = find_top_greedy_paths(cwg_dict, beta_threshold=0.5)
    """)
