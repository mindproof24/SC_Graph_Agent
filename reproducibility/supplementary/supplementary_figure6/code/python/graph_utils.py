"""
graph_utils.py
=================
MCP 서버와 독립적인 순수 계산 유틸리티 모음.

포함:
  A. A* / PhateGenePathFinder  (leiden_astar_pipeline)
  B. path 처리               (path_filter_process_)
  C. CWG + conservative graph (build_cluster_conservative_graphs)
  D. LIANA R-L map           (build_rl_gene_map)
  E. KEGG pathway 파싱/선택  (parse_all_kegg_xmls,
                              compute_and_select_top_kegg,
                              get_top_edges_per_pathway)
"""

# ── imports ──────────────────────────────────────────────────────
import re
import time
import heapq
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import scipy.sparse as sp
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

import cwg_rust
# ══════════════════════════════════════════════════════════════
# ── 위치 1: 기존 cwg_rust import 블록을 아래로 교체 ──
# ══════════════════════════════════════════════════════════════
 
from cwg_rust import (
    astar_all_pairs,         
    ClusterWeightedGraphRust,
    build_conservative_graph,
    KEGGPathway,
    make_kegg_edges_bidirectional,
    compute_all_kegg_norms_sparse,
    compute_all_kegg_norms_cluster_mean
)
 

warnings.filterwarnings("ignore")


# ================================================================
# 공통 헬퍼
# ================================================================

def _ensure_csr(X):
    """adata.X → scipy CSR float32 (Rust sparse 함수용).

    uint8/uint16/uint32/int* 등 정수형 행렬(ATAC-seq count 등)을
    float32로 변환하여 Rust 구현체의 numeric dtype 요구사항을 충족.
    """
    if sp.issparse(X):
        X = X.tocsr() if X.format != "csr" else X
    else:
        X = sp.csr_matrix(X)
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32)
    return X


# ================================================================
# A. PhateGenePathFinder
# ================================================================

class PhateGenePathFinder:
    """PHATE 좌표 기반 A* 경로 탐색기."""

    def __init__(self, adata, gene_col: str = "n_genes", gene_weight: float = 0.1):
        self.adata       = adata
        self.coords      = adata.obsm["X_phate"]
        self.gene_values = adata.obs[gene_col].values
        self.gene_weight = gene_weight
        self.tree        = KDTree(self.coords)
        self.n_obs       = adata.n_obs
        self.delta       = 0.0001  # 기본값; find_all_pairs에서 덮어씀

    def get_neighbors(self, point_idx: int, delta: float) -> list:
        return self.tree.query_ball_point(self.coords[point_idx], r=delta)


    
        # ── v2: noise 없음, L2 arc g, max_iter 적용 ──
    def find_all_pairs(
        self,
        low_indices:    list,
        high_indices:   list,
        delta:          float,
        cluster_count:  int,
        max_iter_ratio: float = 0.8,
    ) -> List[List[int]]:
        """
        astar_all_pairs_v2 호출.
        max_iter = cluster_count * max_iter_ratio (기본 0.8).
        noise 없음 — Rust heuristic_l2 사용.
        """
        self.delta  = delta
        low_int     = [self.adata.obs.index.get_loc(bc) for bc in low_indices]
        high_int    = [self.adata.obs.index.get_loc(bc) for bc in high_indices]
        coords      = np.ascontiguousarray(self.coords,      dtype=np.float64)
        gene_vals   = np.ascontiguousarray(self.gene_values, dtype=np.float64)
        max_iter    = int(cluster_count * max_iter_ratio)
        return astar_all_pairs(
            coords, gene_vals,
            low_int, high_int,
            float(delta), float(self.gene_weight),
            max_iter,
        )
 
    
######
#4월1~수정사항.py의 2가지 항목을 고려해서 - ATAC단의 작업이 다 끝내고 lv0 한쪽 나머지 만들때 즈음에 astar가 cb_rna-atac_not_re_mat 사이의 계산을
##### 할때 클러스터 세포수가 비슷한 둘에서 갑자기 비효율적으로 시간이 늘어나는 경우를 해소해야함. 4월1~수정사항.py참조!


# ══════════════════════════════════════════════════════════════
# ── 위치 3: get_local_params 함수 바로 아래에 추가 ──
# ══════════════════════════════════════════════════════════════
 
def _estimate_params_phate(
    adata,
    gene_col:    str   = "n_genes",
    cluster_mask = None,
    max_cells:   int   = 30_000,
    delta_pct:   float = 5.0,
    weight_pct:  float = 50.0,   # gene_weight는 median 스케일
) -> Tuple[float, float]:
    """
    PHATE pdist 기반 delta와 gene_weight 동시 추정.

    delta      = percentile(pdist(phate), delta_pct)
    gene_weight = percentile(pdist(phate), weight_pct)
                / percentile(pdist(gene),  weight_pct)

    Returns
    -------
    delta, gene_weight
    """
    coords     = adata.obsm["X_phate"]
    gene_vals  = adata.obs[gene_col].values
    n_total    = len(coords)

    # ── subset 선택 (delta와 동일 로직) ──
    if cluster_mask is not None:
        idx = np.where(cluster_mask)[0]
        if len(idx) <= max_cells:
            sel = idx
        else:
            sel = np.random.choice(idx, max_cells, replace=False)
    elif n_total <= max_cells:
        sel = np.arange(n_total)
    else:
        sel = np.random.choice(n_total, max_cells, replace=False)

    phate_sub = coords[sel]
    gene_sub  = gene_vals[sel].reshape(-1, 1)

    phate_dists = pdist(phate_sub)
    gene_dists  = pdist(gene_sub)

    delta       = float(np.percentile(phate_dists, delta_pct))
    phate_scale = float(np.percentile(phate_dists, weight_pct))
    gene_scale  = float(np.percentile(gene_dists,  weight_pct))

    gene_weight = phate_scale / gene_scale if gene_scale > 0 else 0.1

    return delta, gene_weight

def _select_candidates(sub_ad, gene_col, cell_count, q_low, q_high, verbose):
    values = sub_ad[gene_col].values

    if cell_count < 1000:
        q_lo_val   = np.quantile(values, q_low)
        q_hi_val   = np.quantile(values, q_high)
        low_cands  = sub_ad[sub_ad[gene_col] <= q_lo_val].index
        high_cands = sub_ad[sub_ad[gene_col] >= q_hi_val].index
        if verbose:
            print(f"  [candidates] quantile q{int(q_low*100)}={q_lo_val:.1f} / "
                  f"q{int(q_high*100)}={q_hi_val:.1f}  "
                  f"low={len(low_cands)}, high={len(high_cands)}")
    else:
        n         = min(10, cell_count // 1000)
        q_lo_ext  = (q_low  * 100 + n) / 100
        q_hi_ext  = (q_high * 100 - n) / 100
        q_lo_val  = np.quantile(values, q_lo_ext)
        q_hi_val  = np.quantile(values, q_hi_ext)
        lo_pool   = sub_ad[sub_ad[gene_col] <= q_lo_val].index
        hi_pool   = sub_ad[sub_ad[gene_col] >= q_hi_val].index
        low_cands  = np.random.choice(lo_pool, size=min(100, len(lo_pool)),  replace=False)
        high_cands = np.random.choice(hi_pool, size=min(100, len(hi_pool)), replace=False)
        if verbose:
            print(f"  [candidates] 확장 n={n}  lo={len(lo_pool)} hi={len(hi_pool)} "
                  f"→ 샘플 low={len(low_cands)} high={len(high_cands)}")
    return low_cands, high_cands


def _fallback_random_split(cell_indices, n_splits: int = 10, verbose: bool = True):
    shuffled = cell_indices.copy()
    np.random.shuffle(shuffled)
    splits = np.array_split(shuffled, n_splits)
    paths  = [list(map(int, s)) for s in splits if len(s) > 0]
    if verbose:
        sizes = [len(p) for p in paths]
        print(f"  [fallback] random {n_splits}등분 → {len(paths)}개 path "
              f"크기: {min(sizes)}~{max(sizes)}")
    return paths




def run_astar_for_cluster(
    adata,
    cluster_id:      str,
    leiden_key:      str   = "leiden",
    gene_col:        str   = "n_genes",
    q_low:           float = 0.1,
    q_high:          float = 0.9,
    delta:           float = None,
    gene_weight:     float = None,
    min_paths:       int   = 300,
    fallback_splits: int   = 10,
    max_iter_ratio:  float = 0.8,
    verbose:         bool  = True,
) -> List[List[int]]:
    """
    단일 cluster A* 경로 탐색 — v2 (개선판).
 
    기존 대비 변경점:
      1. delta 추정 : get_local_params 루프(30회) → PHATE pdist 5th percentile (1회)
      2. g(n)       : hop count → 누적 L2 arc length
      3. h(n)       : L2(current→goal) + gene_weight×|Δgene|  (noise 없음, /delta 없음)
      4. iter_count : pop 직후 → 노드 확정 시점 (stale pop 미카운트)
      5. max_iter   : cluster_count × max_iter_ratio (기본 0.8)
    """
    t0         = time.time()
    cluster_id = str(cluster_id)
    mask_bool  = adata.obs[leiden_key] == cluster_id
    sub_ad     = adata.obs[mask_bool]
    cell_count = len(sub_ad)
    cell_int   = np.where(mask_bool.values)[0]
 
    if verbose:
        print(f"\n{'='*55}\n[Cluster {cluster_id}] v2  cells = {cell_count}\n{'='*55}")
 
    if cell_count < 2:
        return []
 
    if cell_count <= min_paths:
        if verbose:
            print(f"  cells({cell_count}) ≤ {min_paths} → A* 생략, fallback만 반환")
        return _fallback_random_split(cell_int, fallback_splits, verbose)
 
    low_cands, high_cands = _select_candidates(
        sub_ad, gene_col, cell_count, q_low, q_high, verbose
    )
    if len(low_cands) == 0 or len(high_cands) == 0:
        return _fallback_random_split(cell_int, fallback_splits, verbose)
 
    # ── A* 탐색 (v2) ────────────────────────────────────────

    t_delta0 = time.time()
    if delta is None or gene_weight is None:
        delta_est, weight_est = _estimate_params_phate(
            adata, gene_col=gene_col, cluster_mask=mask_bool.values
        )
        delta_val       = delta       if delta       is not None else delta_est
        gene_weight_val = gene_weight if gene_weight is not None else weight_est
    else:
        delta_val, gene_weight_val = delta, gene_weight
    t_delta = time.time() - t_delta0
    max_iter = int(cell_count * min(max_iter_ratio, 1.0))  # ← 추가
    if verbose:
        print(f"  delta={delta_val:.6f}  gene_weight={gene_weight_val:.6f}  "
              f"( {t_delta:.3f}s)  max_iter={max_iter:,}")
              
    t_astar0 = time.time()  # ← 추가
    finder = PhateGenePathFinder(adata, gene_weight=gene_weight_val)  # 추정값 사용
    paths    = finder.find_all_pairs(
        low_cands, high_cands,
        delta_val,
        cluster_count  = cell_count,
        max_iter_ratio = max_iter_ratio,
    )
    t_astar = time.time() - t_astar0
 
    if verbose:
        print(f"  paths: {len(paths)}  "
              f"(delta {t_delta:.3f}s + astar {t_astar:.1f}s = {time.time()-t0:.1f}s)")
    return paths


# ================================================================
# B. path 처리
# ================================================================

def path_filter_process_(paths: List[List[int]]) -> List[List[int]]:
    """길이 필터 → 포함 관계 merge → 중앙값 근처 선택."""

    def filter_paths(ps):
        mean_l = np.mean([len(p) for p in ps])
        return [p for p in ps if len(p) >= mean_l]

    def merge_contained(ps):
        path_sets = [set(p) for p in ps]
        n      = len(path_sets)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for i in range(n):
            for j in range(n):
                if i != j and path_sets[i] <= path_sets[j]:
                    parent[find(i)] = find(j)

        groups = defaultdict(set)
        for i in range(n):
            groups[find(i)] |= path_sets[i]

        merged = list(groups.values())
        print(f"  병합 전: {n}개 → 병합 후: {len(merged)}개")
        return merged

    if len(paths) <= 30:
        return paths

    check_p = filter_paths(paths)
    merged  = [list(s) for s in merge_contained(check_p)]
    lengths = np.array([len(p) for p in merged])
    median  = np.median(lengths)
    close   = np.where(np.abs(lengths - median) <= 1)[0]
    return [merged[i] for i in close]


# ================================================================
# C. CWG + conservative graph + flatten
# ================================================================



def build_cluster_conservative_graphs(
    adata,
    dorothea_df:    pd.DataFrame,
    all_results:    dict,
    beta_threshold: float = 1.45,
    threshold:      float = 0.8,
    verbose:        bool  = True,
) -> dict:
    """
    cluster_id → paths 를 받아 cluster별로:
      path_filter → CWG → conservative graph → DiGraph → gene list
    """

    results = {}

    for cid, paths in all_results.items():
        if not paths:
            continue
        if verbose:
            print(f"\n── Cluster {cid} ──")

        processed = path_filter_process_(paths)
        if not processed:
            continue

        unique_cells = np.array(list({cell for path in processed for cell in path}))
        cluster_mask = np.zeros(adata.n_obs, dtype=bool)
        cluster_mask[unique_cells] = True

        X_csr = _ensure_csr(adata.X)   # uint32 등 정수형 → float32 변환

        cwg = ClusterWeightedGraphRust.new_sparse(
            sparse_matrix          = X_csr,
            gene_names             = list(adata.var_names),
            cluster_mask           = cluster_mask,
            dorothea_sources       = list(dorothea_df["source"]),
            dorothea_targets       = list(dorothea_df["target"]),
            dorothea_weights       = dorothea_df["weight"].values.astype(np.float64),
            dorothea_confidences   = list(dorothea_df["confidence"]),
            cluster_id             = str(cid),
            beta_mode              = "dynamic",
            tf_expr_threshold      = 0.001,
            target_expr_threshold  = 0.001,
            require_both_expressed = True,
        )

        edge_data = build_conservative_graph(
            cwg, processed, X_csr,
            use_greedy     = False,
            beta_threshold = beta_threshold,
            threshold      = threshold,
        )
#birkhoff 정리의 좌변과 우변, 우변의 조건부 by graphon의 어떤 부분?,  x. = - del f는 반례이다.. 흠..
#위 맥락을 필요한 만큼만 정리하고, 아래의 contrib * beta로 판단할지 어떻게할지.. - 이전의 rs함수에서 어떻게 mean를 취할지도 정리해야함.

        df = pd.DataFrame({
            "source":    edge_data["source"],
            "target":    edge_data["target"],
            "count":     edge_data["count"],
            "freq":      edge_data["freq"],
            "mean_beta": edge_data["mean_beta"],
            "mean_contribution": edge_data["mean_contribution"],
            "alpha_i": edge_data["mean_alpha_i"],
            "alpha_j": edge_data["mean_alpha_j"]
        })
        df["score"] = df["freq"] * df["mean_beta"]
        df.attrs    = {k: edge_data[k] for k in ("total_paths","threshold","cutoff","mode")}
        df          = df.sort_values("score", ascending=False).reset_index(drop=True)

        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row["source"], row["target"],
                       weight=row["score"], freq=row["freq"], mean_beta=row["mean_beta"])

        components = list(nx.weakly_connected_components(G))
        genes      = list(G.nodes())

        if verbose:
            print(f"  edges={len(df)}, components={len(components)}, genes={len(genes)}")

        results[cid] = {"df": df, "G": G, "components": components, "genes": genes}

    return results


# ================================================================
# D. LIANA R-L map
# ================================================================

# 종(organism) 자동 감지
_LIANA_RESOURCE = {
    "human": "consensus",
    "mouse": "mouseconsensus",
}

def detect_organism(adata) -> str:
    """
    adata.var_names 패턴으로 human / mouse 자동 감지.

    - Human: 전체 대문자 (CD8A, GAPDH, TP53)
    - Mouse: Title case (Cd8a, Gapdh, Trp53)

    Returns
    -------
    "human" 또는 "mouse"
    """
    _mouse_pat = re.compile(r'^[A-Z][a-z]')
    genes = [str(g) for g in adata.var_names[: min(5000, adata.n_vars)] if str(g)]
    human_n = sum(1 for g in genes if g.isupper() or (len(g) > 1 and g[0].isupper() and g[1:].isupper()))
    mouse_n = sum(1 for g in genes if _mouse_pat.match(g))
    detected = "mouse" if mouse_n > human_n else "human"
    print(f"[detect_organism] human-like={human_n}, mouse-like={mouse_n} → '{detected}'")
    return detected


def build_rl_gene_map(
    adata,
    liana_key:        str   = "liana_res",
    leiden_key:       str   = "leiden",
    specificity_rank: float = 0.015,
    lrscore_min:      float = None,
    top_n:            int   = 15,
    expr_threshold:   float = 0.0,
    organism:         str   = "human",   # "human" | "mouse"
    verbose:          bool  = True,
) -> dict:
    """
    LIANA 결과에서 cluster별 (ligand, receptor) edge 추출.

    organism 파라미터에 따라 LIANA resource를 자동 선택:
      human → "consensus"
      mouse → "mouseconsensus"

    Returns
    -------
    {
        "specificity_table": pd.DataFrame,
        "cluster_edge": {
            cid: {
                "ligand":   List[str],
                "receptor": List[str],
                "edges":    List[Tuple[str, str, str]],  # (ligand, receptor, receiver_cluster)
            }
        }
    }
    """
    try:
        import liana as li
    except ImportError:
        raise ImportError("liana 패키지가 필요합니다: pip install liana")

    resource_name = _LIANA_RESOURCE.get(organism, "consensus")
    if verbose:
        print(f"[build_rl_gene_map] organism='{organism}' → resource='{resource_name}'")

    if liana_key not in adata.uns or adata.uns[liana_key] is None:
        if verbose:
            print("LIANA 결과 없음 → 계산 시작...")
        adata.raw = adata
        li.mt.rank_aggregate(
            adata, groupby=leiden_key,
            resource_name=resource_name, expr_prop=0.1, verbose=verbose,
        )

    liana_res = adata.uns[liana_key].copy()
    mask      = liana_res["specificity_rank"] < specificity_rank
    if lrscore_min is not None:
        mask &= liana_res["lrscore"] > lrscore_min
    sig_res = liana_res[mask].copy()

    if verbose:
        print(f"\n필터링: {len(liana_res):,} → {len(sig_res):,}")

    cluster_RL = {}
    for cid in np.unique(adata.obs[leiden_key]):
        cid_str  = str(cid)
        mask_c   = adata.obs[leiden_key] == cid_str
        sub      = (
            sig_res[sig_res["source"] == cid_str]
            .sort_values("specificity_rank")
            .drop_duplicates(subset=["ligand_complex", "receptor_complex"])
            .head(top_n)
        )

        edges, ligands_set, receptors_set = [], set(), set()
        for _, row in sub.iterrows():
            ligand          = row["ligand_complex"]
            receptor        = row["receptor_complex"]
            receiver_cluster = str(row.get("target", cid_str))

            for gene, gene_set, col_name in [
                (ligand,   ligands_set,   "ligand"),
                (receptor, receptors_set, "receptor"),
            ]:
                if gene not in adata.var_names:
                    break
                col_idx  = adata.var_names.get_loc(gene)
                col_expr = adata.X[mask_c.values, col_idx]
                mean_val = (col_expr.toarray().ravel().mean()
                            if sp.issparse(col_expr) else col_expr.mean())
                if mean_val <= expr_threshold:
                    break
            else:
                edges.append((ligand, receptor, receiver_cluster))
                ligands_set.add(ligand)
                receptors_set.add(receptor)

        cluster_RL[cid_str] = {
            "ligand":   sorted(ligands_set),
            "receptor": sorted(receptors_set),
            "edges":    edges,
        }
        if verbose:
            print(f"  cluster {cid_str}: L={len(ligands_set)}, R={len(receptors_set)}, "
                  f"edges={len(edges)}")

    return {"specificity_table": sig_res, "cluster_edge": cluster_RL}


# ================================================================
# E. KEGG pathway 파싱 / 선택 / edge 추출
# ================================================================

def parse_all_kegg_xmls_raw(
    kegg_xml_dir: str,
) -> List[dict]:
    """
    KEGG XML → raw intermediate list (adata 필터 없음, 전역 캐시용).
    각 항목: {"name": str, "edge_df": DataFrame, "all_genes": set}
    compound 노드 제거 및 gene-gene edge만 유지하지만 adata 필터는 적용하지 않음.
    """
    from keggx import KEGG

    xml_files = list(Path(kegg_xml_dir).glob("*.kgml"))
    print(f"\n{'='*60}\nParsing {len(xml_files)} KEGG XML files (raw, no adata filter)\n{'='*60}")

    raw_pathways, skipped = [], 0
    for xml_file in tqdm(xml_files, desc="Parsing"):
        try:
            pathway = KEGG(KGML_file=str(xml_file))
            edge_df = pathway.edge_attributes_df
            node_df = pathway.node_attributes_df

            if edge_df is None or len(edge_df) == 0:
                skipped += 1; continue

            gene_ids = set(node_df[node_df["type"] == "gene"]["name"])
            edge_df  = edge_df[
                edge_df["source"].isin(gene_ids) & edge_df["target"].isin(gene_ids)
            ].copy()

            if len(edge_df) == 0:
                skipped += 1; continue

            raw_pathways.append({
                "name":      pathway.title,
                "edge_df":   edge_df,
                "all_genes": set(edge_df["source"]) | set(edge_df["target"]),
            })
        except Exception:
            continue

    print(f"  Skipped (no gene-gene edges): {skipped}")
    print(f"✅ Raw parsed {len(raw_pathways)} pathways")
    return raw_pathways


def filter_kegg_pathways(
    raw_pathways: List[dict],
    adata,
    min_genes: int = 5,
) -> List[KEGGPathway]:
    """
    raw_pathways (parse_all_kegg_xmls_raw 결과)에 adata.var_names 필터를 적용하여
    KEGGPathway 리스트 반환. sampleid별 캐시에 사용.
    """
    adata_genes = set(adata.var_names)
    kegg_pathways, skipped_low = [], 0

    for raw in raw_pathways:
        valid_genes = raw["all_genes"] & adata_genes
        if len(valid_genes) < min_genes:
            skipped_low += 1; continue

        edge_df = (
            raw["edge_df"][
                raw["edge_df"]["source"].isin(valid_genes) |
                raw["edge_df"]["target"].isin(valid_genes)
            ]
            .drop_duplicates(subset=["source", "target"], keep="first")
            .reset_index(drop=True)
        )
        if len(edge_df) == 0:
            skipped_low += 1; continue

        sources, targets, effects, types, indirects, mods = make_kegg_edges_bidirectional(
            sources       = edge_df["source"].tolist(),
            targets        = edge_df["target"].tolist(),
            effects       = edge_df["effect"].fillna(0).astype(int).tolist(),
            types         = edge_df["type"].fillna("").tolist(),
            indirects     = edge_df["indirect"].fillna(False).astype(bool).tolist(),
            modifications = edge_df["modification"].fillna("").tolist(),
        )

        kegg_pathways.append(KEGGPathway(
            name          = raw["name"],
            sources       = sources,
            targets       = targets,
            weights       = [1.0] * len(sources),
            modifications = mods,
            effects       = effects,
            types         = types,
            indirects     = indirects,
        ))

    print(f"  Filtered: {len(kegg_pathways)} pathways (skipped {skipped_low} low-gene)")
    return kegg_pathways


def parse_all_kegg_xmls(
    kegg_xml_dir: str,
    adata,
    min_genes: int = 5,
) -> List[KEGGPathway]:
    """KEGG XML → KEGGPathway 리스트 (compound 제거, adata 필터). 기존 인터페이스 유지."""
    raw = parse_all_kegg_xmls_raw(kegg_xml_dir)
    return filter_kegg_pathways(raw, adata, min_genes)


def compute_and_select_top_kegg(
    adata,
    kegg_pathways:  List[KEGGPathway],
    cluster_id:     str,
    cluster_key:    str = "leiden",
    top_n:          int = 10,
    mode:           str = "cluster_mean",   # "per_cell" | "cluster_mean"
) -> List[KEGGPathway]:

    sparse_X      = _ensure_csr(adata.X)
    adata_genes   = list(adata.var_names)
    cluster_mask  = (adata.obs[cluster_key].astype(str) == str(cluster_id)).values
    cluster_indices = np.where(cluster_mask)[0].tolist()

    if mode == "cluster_mean":
        # 새 함수: cluster mean α → G2 한 번 계산
        norm_dict = compute_all_kegg_norms_cluster_mean(
            sparse_X, adata_genes, kegg_pathways, cluster_indices
        )
        scores = [(pw, norm_dict.get(pw.name, 0.0)) for pw in kegg_pathways]
    else:
        # 기존: per-cell norms → cluster 평균
        all_norms = compute_all_kegg_norms_sparse(sparse_X, adata_genes, kegg_pathways)
        scores = [
            (pw, float(np.mean(np.array(all_norms[pw.name])[cluster_indices])))
            for pw in kegg_pathways if pw.name in all_norms
        ]

    scores.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scores[:top_n]]

def get_top_edges_per_pathway(
    adata,
    top_pathways: List,  # List[KEGGPathway]
    cluster_id: str,
    cluster_key: str = "leiden",
    top_n_edges: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    cluster 평균 발현량 기반 contribution 계산 → pathway별 top edge DataFrame 반환.
    
    alpha_G: cluster 내 모든 유전자의 평균 발현량 합
    
    Parameters
    ----------
    adata : AnnData
        gene expression data
    top_pathways : List[KEGGPathway]
        pathway 리스트
    cluster_id : str
        cluster ID
    cluster_key : str
        adata.obs의 cluster 컬럼명
    top_n_edges : int
        pathway별 상위 edge 수
    
    Returns
    -------
    results : Dict[str, pd.DataFrame]
        contribution 기준 상위 edges (pathway별)
        
        DataFrame 컬럼:
        - source, target: edge 노드
        - weight: edge 가중치
        - alpha_i, alpha_j: 각 유전자 발현량
        - alpha_i*alpha_j: 발현량 곱
        - beta: 상호작용 강도 (sqrt(|w + alpha_i + alpha_j|))f
        - contribution: pathway 내 상대적 중요도
    """
    
    # ──────────────────────────────────────────────────────
    # Step 1: cluster 데이터 추출
    # ──────────────────────────────────────────────────────
    mask = (adata.obs[cluster_key].astype(str) == str(cluster_id)).values
    X_cluster = adata.X[mask]
    
    if sp.issparse(X_cluster):
        X_cluster = X_cluster.toarray()
    
    gene_mean = X_cluster.mean(axis=0)  # (n_genes,)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    
    # ──────────────────────────────────────────────────────
    # Step 2: alpha_G 계산 (cluster 전체 유전자 합)
    # ──────────────────────────────────────────────────────
    # 방식 2: cluster 내 모든 유전자의 발현량 합
    alpha_G = gene_mean.sum()
    alpha_g_sq = alpha_G ** 2
    
    print(f"\n[alpha_G 계산]")
    print(f"  cluster: {cluster_id}")
    print(f"  alpha_G (cluster 전체 발현량): {alpha_G:.6f}")
    print(f"  alpha_G^2: {alpha_g_sq:.6f}")
    
    if alpha_G < 1e-10:
        print(f"  ⚠️ 경고: alpha_G가 너무 작음 ({alpha_G})")
        print(f"  → contribution을 계산할 수 없습니다")
    
    # ──────────────────────────────────────────────────────
    # Step 3: pathway별 처리
    # ──────────────────────────────────────────────────────
    results = {}
    
    for pathway in top_pathways:
        edges_data = pathway.get_edges_data()
        
        print(f"\n[{pathway.name}]")
        print(f"  edges: {len(edges_data['source'])}")
        
        # edge별 계산
        rows = []
        for src, tgt, w in zip(
            edges_data["source"],
            edges_data["target"],
            edges_data["weight"]
        ):
            # 발현량 추출
            alpha_i = (
                gene_mean[gene_to_idx[src]]
                if src in gene_to_idx
                else 0.0
            )
            alpha_j = (
                gene_mean[gene_to_idx[tgt]]
                if tgt in gene_to_idx
                else 0.0
            )
            
            # beta 계산 (제곱근)
            beta = abs(w + alpha_i + alpha_j) ** 0.5
            
            # contribution 계산 (정규화)
            # 분모: alpha_G^2 (cluster 전체)
            contribution = (
                (alpha_i * alpha_j / alpha_g_sq) * (beta ** 2)
                if alpha_G > 1e-10
                else 0.0
            )
            
            rows.append({
                "source": src,
                "target": tgt,
                "weight": w,
                "alpha_i": alpha_i,
                "alpha_j": alpha_j,
                "alpha_i*alpha_j": alpha_i * alpha_j,
                "beta": beta,
                "contribution": contribution,
            })

        # ──────────────────────────────────────────────────
        # Step 4: beta 기준으로 정렬 (edge interaction strength)
        # pathway activity는 이미 norm 기준으로 정렬됨
        # ──────────────────────────────────────────────────
        df = pd.DataFrame(rows)

        df_sorted = df.sort_values("contribution", ascending=False)
        results[pathway.name] = df_sorted.head(top_n_edges).reset_index(drop=True)

        print(f"  ✓ top {top_n_edges} edges (contribution 기준) 추출 완료")
    return results
