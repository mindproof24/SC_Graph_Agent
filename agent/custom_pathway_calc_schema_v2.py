# custom_pathway_calc 갱신 스키마 (sc_graph_mcp_server.py 최신, Jun6 기준)
# eval_grpo_dataset_NO_KG_Tool.py 의 TOOLS 리스트에서 기존 custom_pathway_calc 블록을 이걸로 교체.
# 변경점: verbose / verbose_top_n 추가, description에 edge_contributions(narrow-down 근거) 명시,
#         cluster_key='leiden' 가 반드시 존재해야 함을 명시(없으면 먼저 sc.tl.leiden 실행).

CUSTOM_PATHWAY_CALC_TOOL = {"type": "function", "function": {
    "name": "custom_pathway_calc",
    "description": (
        "Customize your own pathway: supply a tailored TF->Target edge signature and efficiently "
        "score it at cluster-scale OR cell-scale (edge_L2 norm). "
        "edges=[[src, tgt, weight], ...]: curate from get_expressed_dorothea_edges + your own markers, "
        "prune off-target TFs (aim for >=4 edges / >=5 unique vertices). "
        "Workflow to narrow down: (1) scale='cluster' + cluster_ids=['all'] ranks every cluster in one "
        "call; (2) then scale='cell' + cluster_id='<top cluster>' returns top_k cells within that cluster, "
        "or scale='cell' + cell_ids=<specific cells> scores only those cells. "
        "verbose=True attaches edge_contributions per target (per-edge [source,target,weight,beta,"
        "contribution] sorted desc, top verbose_top_n) so you can SEE which edges drive the score and "
        "prune/refine. ACCURATE, target-specific edges (true marker genes of the requested celltype) "
        "score higher and are rewarded; generic/off-target edges score every cell similarly. Use "
        "verbose=True to VERIFY the top-contributing edges really match the target signature before "
        "committing. Requires cluster_key to exist in adata.obs (run sc.tl.leiden first if absent)."
    ),
    "parameters": {"type": "object", "properties": {
        "sampleid":      {"type": "string",  "description": "Sample ID"},
        "edges":         {"type": "array",   "items": {"type": "array"},
                          "description": "Marker edges as [[src, tgt, weight], ...]. weight optional (default 1.0)."},
        "vertices":      {"type": "array",   "items": {"type": "string"},
                          "description": "Optional gene list (informational; edges already define endpoints)."},
        "scale":         {"type": "string",  "description": "'cluster' (default) or 'cell' (per-cell norm)."},
        "cluster_id":    {"type": "string",  "description": "Single cluster id. For scale='cluster', score that cluster; for scale='cell', restrict top_k cell ranking to cells in this cluster."},
        "cluster_ids":   {"type": "array",   "items": {"type": "string"},
                          "description": "Batch cluster ids. For scale='cluster', score clusters; for scale='cell', restrict top_k cell ranking to cells in these clusters. Use ['all'] only for cluster-scale all-cluster ranking."},
        "cell_ids":      {"type": "array",   "items": {"type": "string"},
                          "description": "Specific cell barcodes (scale='cell'). Empty -> top_k ranking over all cells, or over cluster_id/cluster_ids when provided."},
        "cluster_key":   {"type": "string",  "description": "obs column for clustering (default: 'leiden'). Must exist in adata.obs."},
        "top_k":         {"type": "integer", "description": "Top-K cells when scale='cell' and cell_ids empty (default 10)."},
        "top":           {"type": "integer", "description": "Deprecated alias for top_k; prefer top_k."},
        "top_n":         {"type": "integer", "description": "Deprecated alias for top_k; prefer top_k."},
        "name":          {"type": "string",  "description": "Pathway name label (informational)."},
        "verbose":       {"type": "boolean", "description": "If True, include per-edge contribution breakdown (sorted desc) for each target."},
        "verbose_top_n": {"type": "integer", "description": "Max edges per target in edge_contributions when verbose=True (default 10)."},
    }, "required": ["sampleid", "edges"]},
}}
