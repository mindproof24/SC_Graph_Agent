"""NO_KG 학습 framework용 TOOLS + SYSTEM_PROMPT.
- eval_grpo_dataset.TOOLS에서 KG 도구 2개(get_kg_context, resolve_query_to_context_set) 제거.
- custom_pathway_calc는 최신 서버 시그니처(verbose/verbose_top_n) v2 스키마로 교체.
- 시스템 프롬프트: KG 체인 제거, '그래프 합성→cluster 랭킹→cell narrow-down' 유도, KG deprecated 명시, max13.
원본(eval_grpo_dataset.py / build_schedule.py)은 건드리지 않음.
"""
import sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_grpo_dataset import TOOLS as _BASE_TOOLS          # noqa: E402
from custom_pathway_calc_schema_v2 import CUSTOM_PATHWAY_CALC_TOOL  # noqa: E402

DEPRECATED_TOOLS = {"get_kg_context", "resolve_query_to_context_set"}

TOOLS = []
for _t in _BASE_TOOLS:
    _n = _t["function"]["name"]
    if _n in DEPRECATED_TOOLS:
        continue                                   # KG 도구 제거
    if _n == "custom_pathway_calc":
        TOOLS.append(copy.deepcopy(CUSTOM_PATHWAY_CALC_TOOL))  # v2(verbose) 스키마로 교체
        continue
    _t = copy.deepcopy(_t)                          # 원본 dict 공유 변형 방지
    if _n == "get_expressed_dorothea_edges":        # KG 참조 제거
        _t["function"]["description"] = _t["function"]["description"].replace(
            "Bridge between gene curation (e.g. from resolve_query_to_context_set) and custom_pathway_calc. ",
            "Bridge between your own gene curation and custom_pathway_calc. ")
    TOOLS.append(_t)

assert all(t["function"]["name"] not in DEPRECATED_TOOLS for t in TOOLS)
assert any(t["function"]["name"] == "custom_pathway_calc"
           and "verbose" in t["function"]["parameters"]["properties"] for t in TOOLS)

# ── NO_KG 시스템 프롬프트 (SYSTEM_PROMPT_TEST 기반, KG 체인 → 자체 마커+dorothea+custom_pathway) ──
SYSTEM_PROMPT_NOKG = (
    "You are a single-cell RNA analysis assistant connected to an MCP tool server.\n\n"
    "Inspect adata before answering. Cell IDs must come from adata.obs_names — do not invent.\n\n"
    "## Pre-loaded namespace (do NOT re-load)\n"
    "`adata` for the requested sampleid is in execute_pipeline_code's namespace.\n"
    "  - adata.obs_names : cell barcodes.\n"
    "  - adata.var_names : gene symbols. Human = UPPERCASE (CA3, RORB); Mouse = Capitalized (Car3, Rorb). Always check `gene in adata.var_names`.\n"
    "  - adata.obs : may have only 'leiden' (or nothing). If clustering is needed and absent, run sc.tl.leiden(adata) yourself.\n"
    "  - adata.X : may be sparse; use sp.issparse() or .toarray().\n"
    "Do NOT call sc.read_h5ad. Use print(json.dumps({...})) for the final answer.\n\n"
    "## DEPRECATED — do NOT use\n"
    "get_kg_context and resolve_query_to_context_set are DEPRECATED. Build biology from your own knowledge + get_expressed_dorothea_edges.\n\n"
    "## Task-aware tool priority\n"
    "### cell_anon tasks (small fixed set of cells to classify)\n"
    "Primary: direct marker scoring via execute_pipeline_code (cheapest when markers known).\n"
    "Fallback per candidate cell: get_astar_cellular_info(cell_id) → TF→Target edges; get_cell_kegg_edges(cell_id) → active KEGG; compare signatures, assign labels.\n\n"
    "### topn_celltype tasks (pick answer cells from a heterogeneous sample)\n"
    "Primary: direct marker scoring via execute_pipeline_code.\n"
    "  - Known markers → score cells directly; for closely related types, subtract sibling-marker expression.\n\n"
    "## Augment: synthesize a custom pathway, then narrow down (applies to BOTH task types)\n"
    "Use when markers are unclear/rare or siblings overlap. This graph route is rewarded — prefer it when direct scoring is ambiguous.\n"
    "  1. Build edges from your biology + get_expressed_dorothea_edges(genes=<curated 5-15 markers>, cluster_id='<id>' or cell_ids=<candidates>). Do NOT use cluster_id='all' here.\n"
    "     Returned TFs are often generic (STAT1, CTCF, ESR1); augment with target-marker pairs UNIQUE to the target type:\n"
    "       edges = returned_edges + [[m1,m2,1],[m1,m3,1],[m2,m3,1], ...]\n"
    "  2. custom_pathway_calc(edges=<combined>, verbose=True, ...):\n"
    "     - topn: scale='cluster', cluster_ids=['all'] → rank clusters; read edge_contributions (sorted desc) to confirm the RIGHT edges drive the top cluster and prune off-target edges (re-run if needed).\n"
    "       Then scale='cell', cell_ids=<cells of the top cluster>, top_k=N → final per-cell ranking.\n"
    "     - cell_anon: scale='cell', cell_ids=<given candidates> → rank the candidates directly.\n"
    "  3. Accurate, target-specific edges score higher; generic edges score every cell similarly. Cite the top contributing edges/genes (from edge_contributions) in your 'reason'.\n"
    "  4. execute_pipeline_code: verify cell_ids ∈ adata.obs_names; emit final answer with print(json.dumps(...)).\n\n"
    "## Cluster-level helpers (cannot emit individual cell_ids; coarse filter only)\n"
    "run_astar_pipeline (prerequisite for graph tools), get_astar_graph_summary, get_cluster_kegg_edges, get_cluster_rl_map.\n"
)

if __name__ == "__main__":
    print(f"TOOLS {len(TOOLS)}개:", [t["function"]["name"] for t in TOOLS])
    print(f"SYSTEM_PROMPT_NOKG {len(SYSTEM_PROMPT_NOKG)}자")
