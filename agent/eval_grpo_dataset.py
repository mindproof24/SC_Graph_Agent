#!/usr/bin/env python3
"""
eval_grpo_dataset.py — model evaluation on the historical training/evaluation dataset

Usage:
    # Ollama using the default model (gpt-oss:20b)
    MCP_URL=http://localhost:8005/mcp python eval_grpo_dataset.py \
        --jsonl imm_grpo_no_context.jsonl --ollama

    # Ollama using another model
    python eval_grpo_dataset.py --jsonl imm_grpo_no_context.jsonl \
        --ollama --ollama-model llama3.1:8b

    # HF base model in BF16
    python eval_grpo_dataset.py --jsonl imm_grpo_no_context.jsonl --base-only

    # HF ckpt500
    python eval_grpo_dataset.py --jsonl imm_grpo_no_context.jsonl \
        --ckpt checkpoints_v2_moe/checkpoint-500

    # Evaluate only N samples
    python eval_grpo_dataset.py --jsonl imm_grpo_no_context.jsonl \
        --ollama --n-samples 10
"""
# NOTE :: Interactive runtime imports only TOOLS from this historical evaluation module; evaluation execution remains separate.
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

# ─── Config ───────────────────────────────────────────────────────────────────

HF_MODEL_NAME  = "unsloth/gpt-oss-20b-BF16"
OLLAMA_MODEL   = "gpt-oss:20b"
OLLAMA_API_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MCP_URL        = os.getenv("MCP_URL", "http://localhost:8005/mcp")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", os.getenv("NUM_CTX", "8192")))
_OLLAMA_SEED_ENV = os.getenv("OLLAMA_SEED", "").strip()
OLLAMA_SEED: int | None = int(_OLLAMA_SEED_ENV) if _OLLAMA_SEED_ENV else None
MAX_TURNS      = 20
LOG_DIR        = Path("logs/eval_grpo")
TEMPERATURE    = 0.3   # overridden by --temp CLI flag

# Sticky MCP workers (for example, ports 8005-8008), selected by a deterministic sample-ID hash.
import zlib
MCP_BASE_PORT = int(os.getenv("MCP_BASE_PORT", "8005"))
MCP_N_WORKERS = int(os.getenv("MCP_N_WORKERS", "1"))   # 1 uses MCP_URL; values >1 enable sticky distribution.

def _mcp_url_for(sampleid: str) -> str:
    """Kepp each sample ID in one server namesapce while distributing different samples across workers"""
    if MCP_N_WORKERS <= 1:
        return MCP_URL
    port = MCP_BASE_PORT + (zlib.crc32((sampleid or "").encode()) % MCP_N_WORKERS)
    return f"http://localhost:{port}/mcp"


def _seed_for_item(base_seed: int | None, item_id: str, rep: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed + (zlib.crc32((item_id or "").encode()) % 1_000_000) + rep)

# vLLM backend globals configured by main() when --vllm is enabled.
EXECUTE_ONLY = False   # Expose only execute_pipeline_code when --execute-only is enabled.
USE_VLLM    = False
VLLM_ENGINE = None
VLLM_TOK    = None
VLLM_LORA   = None

# MCP tools without a sampleid argument; automatic injection would fail Pydantic validation.
_NO_SAMPLEID_TOOLS = {
    "get_kg_context",
    "resolve_query_to_context_set",   # Global KG tool without sampleid; retained for historical evaluation.
    #"score_context_subgraph",
    #"synthesize_context_kg_paths",
}

TOOLS = [
    {"type": "function", "function": {
        "name": "run_astar_pipeline",
        "description": "Per-cluster A* pathfinding + conservative DoRothEA TF->Target graph construction. Must be run before graph tools.",
        "parameters": {"type": "object", "properties": {
            "sampleid":        {"type": "string",  "description": "Sample ID"},
            "cluster_ids":     {"type": "array",   "items": {"type": "string"}, "description": "Cluster IDs to process."},
            "organism":        {"type": "string",  "description": "'human', 'mouse', or 'auto'"},
            "force_recompute": {"type": "boolean", "description": "Force recomputation"},
        }, "required": ["sampleid"]},
    }},
    {"type": "function", "function": {
        "name": "get_astar_graph_summary",
        "description": (
            "Per-cluster top edges from the conservative DoRothEA TF→Target graph. Call after run_astar_pipeline. "
            "Edges are sorted by normalized score. mean_beta and score are 0-1 normalized; "
            "mean_beta_raw and score_raw preserve the original expression-scale values."
        ),
        "parameters": {"type": "object", "properties": {
            "sampleid":   {"type": "string",  "description": "Sample ID"},
            "cluster_id": {"type": "string",  "description": "Cluster ID"},
            "top_n":      {"type": "integer", "description": "Number of top edges (default: 20)"},
        }, "required": ["sampleid", "cluster_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_astar_cellular_info",
        "description": (
            "Cell-specific DoRothEA TF→Target edges for one cell, recomputed from that cell's expression. "
            "Returns normalized cellular_beta and normalized mean_beta on the same cluster scale, plus "
            "cellular_beta_raw and mean_beta_raw for original expression-scale values. Compare normalized "
            "cellular_beta > mean_beta to see whether the cell activates an edge above its cluster average."
        ),
        "parameters": {"type": "object", "properties": {
            "sampleid":    {"type": "string",  "description": "Sample ID"},
            "cell_id":     {"type": "string",  "description": "Cell barcode"},
            "leiden_key":  {"type": "string",  "description": "leiden obs column name (default: leiden)"},
            "top_n_edges": {"type": "integer", "description": "Number of top edges (default: 20)"},
        }, "required": ["sampleid", "cell_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_cluster_rl_map",
        "description": "Ligand-receptor pairs via LIANA (cluster level).",
        "parameters": {"type": "object", "properties": {
            "sampleid":         {"type": "string",  "description": "Sample ID"},
            "organism":         {"type": "string",  "description": "'human', 'mouse', 'auto'"},
            "specificity_rank": {"type": "number",  "description": "LIANA specificity_rank threshold"},
            "top_n":            {"type": "integer", "description": "Number of top edges per cluster"},
        }, "required": ["sampleid"]},
    }},
    {"type": "function", "function": {
        "name": "get_cluster_kegg_edges",
        "description": "KEGG top pathways + beta-sorted edges for a cluster.",
        "parameters": {"type": "object", "properties": {
            "sampleid":       {"type": "string",  "description": "Sample ID"},
            "cluster_id":     {"type": "string",  "description": "Cluster ID"},
            "organism":       {"type": "string",  "description": "'human', 'mouse', 'auto'"},
            "top_n_pathways": {"type": "integer", "description": "Number of top pathways (default 5)"},
            "top_n_edges":    {"type": "integer", "description": "Number of top edges per pathway (default 5)"},
        }, "required": ["sampleid", "cluster_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_cell_kegg_edges",
        "description": "KEGG edges based on single-cell expression (beta per cell).",
        "parameters": {"type": "object", "properties": {
            "sampleid":       {"type": "string",  "description": "Sample ID"},
            "cell_id":        {"type": "string",  "description": "Cell barcode"},
            "organism":       {"type": "string",  "description": "'human', 'mouse', 'auto'"},
            "top_n_pathways": {"type": "integer", "description": "Number of top pathways (default 5)"},
            "top_n_edges":    {"type": "integer", "description": "Number of top edges per pathway (default 7)"},
        }, "required": ["sampleid", "cell_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_kg_context",
        "description": "Retrieve biological context from LightRAG Knowledge Graph (genes + keywords).",
        "parameters": {"type": "object", "properties": {
            "genes":    {"type": "array",   "items": {"type": "string"}, "description": "Gene symbols to query"},
            "keywords": {"type": "string",  "description": "Keywords for KG entity search"},
            "top_n":    {"type": "integer", "description": "Number of top entities to return"},
        }},
    }},
    {"type": "function", "function": {
        "name": "execute_pipeline_code",
        "description": "Execute Python code with adata, cluster_graphs, kegg_pathways, etc. in namespace.",
        "parameters": {"type": "object", "properties": {
            "sampleid":    {"type": "string",  "description": "Sample ID"},
            "code":        {"type": "string",  "description": "Python code to execute"},
            "timeout_sec": {"type": "integer", "description": "Best-effort execution timeout in seconds (default 150; 0 disables)."},
        }, "required": ["sampleid", "code"]},
    }},
    {"type": "function", "function": {
        "name": "resolve_query_to_context_set",
        "description": (
            "Anchor seed_genes (gene nodes) + keywords (mechanism/celltype/tissuestate nodes) on the LightRAG KG, "
            "BFS-expand n_hop from anchors, and return a context_set: ranked context_genes "
            "(each with a short 'desc' from the KG node). Use the returned genes as input to "
            "get_expressed_dorothea_edges or as markers for direct execute_pipeline_code scoring."
        ),
        "parameters": {"type": "object", "properties": {
            "seed_genes":      {"type": "array",   "items": {"type": "string"}, "description": "Genes lifted from the question or biological knowledge."},
            "keywords":        {"type": "string",  "description": "Comma-separated phrases for non-gene KG nodes (mechanism / celltype / tissuestate)."},
            "entity_types":    {"type": "array",   "items": {"type": "string"}, "description": "Allowed non-gene types. Default: mechanism/celltype/tissuestate/other."},
            "n_hop":           {"type": "integer", "description": "BFS hop count from anchors (default 1)."},
            "min_edge_weight": {"type": "number",  "description": "Drop edges below this weight."},
            "decay":           {"type": "number",  "description": "Per-hop score decay (default 0.5)."},
            "top_n_genes":     {"type": "integer", "description": "Top N context genes (default 50)."},
            "max_anchors":     {"type": "integer", "description": "Cap anchors by score after matching (default 50, 0 disables). Lowers noise when broad keywords overmatch."},
            "min_keyword_hits":{"type": "integer", "description": "Non-gene anchor must match at least this many comma-split keyword tokens (default 1). Raise to 2+ for AND-like behavior."},
            "context_id":      {"type": "string",  "description": "Optional id. Auto-generated if empty."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_expressed_dorothea_edges",
        "description": (
            "Bridge between gene curation (e.g. from resolve_query_to_context_set) and custom_pathway_calc. "
            "Given an agent-curated gene list, return the top-N DoRothEA TF→Target edges where at least one "
            "endpoint is in the gene list AND both endpoints are expressed in the sample (specify cell_ids OR "
            "cluster_id for the expression context). Returns edge_details sorted by normalized activity "
            "[{source, target, weight, alpha_src, alpha_tgt, beta, activity_norm}] for agent review/pruning, plus "
            "[[src, tgt, w], ...] ready to feed custom_pathway_calc. Review edge_details and remove off-target "
            "TFs before passing to custom_pathway_calc. Aim for ≥4 edges / ≥5 vertices for a meaningful score."
        ),
        "parameters": {"type": "object", "properties": {
            "sampleid":    {"type": "string",  "description": "Sample ID"},
            "genes":       {"type": "array",   "items": {"type": "string"},
                            "description": "Agent-curated gene list. At least one endpoint of each returned edge will be in this list."},
            "top_n":       {"type": "integer", "description": "Max edges to return sorted by normalized activity desc (default 30)."},
            "cluster_id":  {"type": "string",  "description": "Cluster id for mean expression context."},
            "cell_ids":    {"type": "array",   "items": {"type": "string"},
                            "description": "Cell barcodes for per-cell expression context."},
            "cluster_key": {"type": "string",  "description": "obs column (default leiden)."},
            "min_alpha":   {"type": "number",  "description": "Min expression for both endpoints (default 0.0)."},
            "organism":    {"type": "string",  "description": "'human'/'mouse'/'auto'."},
        }, "required": ["sampleid", "genes"]},
    }},
    {"type": "function", "function": {
        "name": "custom_pathway_calc",
        "description": (
            "Customize your own pathway: supply a tailored TF→Target edge signature and efficiently "
            "score it at cell-scale OR cluster-scale (edge_L2 norm). Designing the edges lets you probe "
            "a specific pattern, then get per-cell or per-cluster scores in a single call. "
            "edges=[[src, tgt, weight], ...]: curate from get_expressed_dorothea_edges output after "
            "pruning off-target TFs (aim for ≥4 edges / ≥5 unique vertices). "
            "scale='cell' + cell_ids=[...] → per-cell {cell_id: score}. "
            "scale='cluster' + cluster_ids=['all'] → rank every cluster in one call."
        ),
        "parameters": {"type": "object", "properties": {
            "sampleid":    {"type": "string",  "description": "Sample ID"},
            "edges":       {"type": "array",   "items": {"type": "array"},
                            "description": "Marker edges as [[src, tgt, weight], ...]. weight optional (default 1.0)."},
            "vertices":    {"type": "array",   "items": {"type": "string"},
                            "description": "Optional gene list (informational)."},
            "scale":       {"type": "string",  "description": "'cluster' (default) or 'cell' (per-cell norm)."},
            "cluster_id":  {"type": "string",  "description": "Single cluster id (scale='cluster')."},
            "cluster_ids": {"type": "array",   "items": {"type": "string"},
                            "description": "Batch cluster ids. Use ['all'] to score every cluster."},
            "cell_ids":    {"type": "array",   "items": {"type": "string"},
                            "description": "Specific cell barcodes (scale='cell')."},
            "cluster_key": {"type": "string",  "description": "obs column for clustering (default: leiden)."},
            "top_k":       {"type": "integer", "description": "Top-K cells when scale='cell' and cell_ids empty."},
            "name":        {"type": "string",  "description": "Pathway name label (informational)."},
        }, "required": ["sampleid", "edges"]},
    }},
]

#SYSTEM_PROMPT = (
#    "You are a single-cell RNA analysis assistant connected to an MCP tool server.\n"
#    "Do NOT ask for clarification — call tools immediately with available information.\n\n"
#    "## Tool Guide\n"
#    "- run_astar_pipeline: A* + DoRothEA TF network (must run before graph tools)\n"
#    "- get_astar_graph_summary: cluster-level conservative graph edges\n"
#    "- get_astar_cellular_info: cell-specific DoRothEA edges (cellular_beta)\n"
#    "- get_cluster_kegg_edges: KEGG pathways ranked by G2 norm (cluster level)\n"
#    "- get_cell_kegg_edges: KEGG pathways using single-cell expression\n"
#    "- get_cluster_rl_map: Ligand-receptor pairs via LIANA\n"
#    "- get_kg_context: LightRAG KG — gene nodes, mechanism/celltype descriptions\n"
#    "- resolve_query_to_context_set (KG-A): seed_genes+keywords -> context_set (genes/pathways/DoRothEA), returns context_id. Pass sampleid for #per-cell scoring downstream.\n"
#    "- score_context_subgraph (KG-B): cluster/cell L2-norm against a context_id (DoRothEA + KEGG prior).\n"
#    "- synthesize_context_kg_paths (KG-C): top KG gene-gene paths re-ranked by sample β-L2.\n"
#    "- execute_pipeline_code: custom Python with adata, cluster_graphs, kegg_pathways\n\n"
#    "## KEGG metrics\n"
#    "  beta = sqrt(|w + alpha_i + alpha_j|)  — edge interaction strength\n"
#    "  contribution = alpha_i*alpha_j/alpha_G^2 * beta^2  — activation density\n"
#    "  Pathway selection: G2 norm (sum of contributions)\n"
#)

#SYSTEM_PROMPT = (
#    "You are a single-cell RNA analysis assistant connected to an MCP tool server.\n\n"
#    "Inspect adata before answering. Cell IDs must come from adata.obs_names — do not invent.\n\n"
#    "## Pre-loaded namespace (do NOT re-load)\n"
#    "`adata` for the requested sampleid is in execute_pipeline_code's namespace.\n"
#    "  - adata.obs_names : cell barcodes.\n"
#    "  - adata.var_names : gene symbols. Human = UPPERCASE (CA3, RORB, IGHG1). Mouse = Capitalized (Car3, Rorb). Always check `gene in adata.var_names`.\n"
#    "  - adata.obs : usually has only 'leiden'. Do NOT assume 'cell_type' / 'Subclass' / 'tissue' exist.\n"
#    "  - adata.X : may be sparse; use sp.issparse() or [...].toarray().\n"
#    "Do NOT call sc.read_h5ad — the h5ad is not on disk.\n"
#    "Do NOT end with a bare expression — its repr is capped at ~4 KB and silently truncates. Always print(json.dumps({...})) for final output.\n\n"
#    "## Approach that scores high\n"
#    "Single shared markers (e.g. MS4A1 for any B cell, RORB for any cortical excitatory) rarely discriminate. What works:\n"
#    "  1. Gather per-cell evidence FIRST:\n"
#    "     - get_astar_cellular_info(cell_id) for active TF→target edges per cell\n"
#    "     - get_cell_kegg_edges(cell_id) for active pathways per cell\n"
#    "     Compare these signatures between the candidate cells / types.\n"
#    "  2. If celltype context is unclear, call get_kg_context(keywords='<celltype>') — 1-2 calls suffice.\n"
#    "  3. Build a customized pathway only when you have observed a discriminating signal. The edges must connect markers UNIQUE to one option, not shared markers:\n"
#    "       custom_pathway_calc(\n"
#    "         edges=[[m1,m2,1],[m1,m3,1],[m2,m3,1], ...],   # ≥3 edges, all from one signature\n"
#    "         scale='cell',\n"
#    "         cell_ids=[...],\n"
#    "         top_k=10)\n"
#    "     Pitfall: edges built from generic lineage markers score every cell similarly and mislead.\n"
#    "  4. For top-N cell selection (topn_celltype), restrict cell_ids to the leiden cluster that concentrates the target.\n"
#    "  5. Emit the final answer via execute_pipeline_code with print(json.dumps(...)).\n\n"
#    "## Tool efficiency budget\n"
#    "- per-cell tools (astar_cellular_info, cell_kegg_edges): 1 call per candidate cell is usually enough.\n"
#    "- get_kg_context: ≤2 calls. More wastes budget.\n"
#    "- custom_pathway_calc: optional. If used, edges must encode a UNIQUE-to-one-type signature.\n"
#    "- cluster tools (kegg/rl/astar_graph): cluster-level only; cannot emit cell_ids. Useful only as a coarse filter for topn tasks.\n"
#    "- execute_pipeline_code: ALWAYS used as the final answer step. print(), don't bare-eval.\n"
#)

SYSTEM_PROMPT_TEST = (
    "You are a single-cell RNA analysis assistant connected to an MCP tool server.\n\n"
    "Inspect adata before answering. Cell IDs must come from adata.obs_names — do not invent.\n\n"
    "## Pre-loaded namespace (do NOT re-load)\n"
    "`adata` for the requested sampleid is in execute_pipeline_code's namespace.\n"
    "  - adata.obs_names : cell barcodes.\n"
    "  - adata.var_names : gene symbols (UPPERCASE for safety).\n"
    "  - adata.obs : usually has only 'leiden'.\n"
    "  - adata.X : may be sparse; use sp.issparse() or .toarray().\n"
    "Do NOT call sc.read_h5ad. Use print(json.dumps({...})) for the final answer.\n\n"
    "## Task-aware tool priority\n"
    "Use a per-task default workflow; mix-and-match tools as needed.\n\n"
    "### For cell_anon tasks (small fixed set of cells to classify)\n"
    "Primary: direct marker scoring via execute_pipeline_code (cheapest when markers known).\n"
    "Fallback: per-cell evidence collection — call once per candidate cell.\n"
    "  1. get_astar_cellular_info(cell_id) → active TF→Target edges for that cell (cellular_beta).\n"
    "  2. get_cell_kegg_edges(cell_id)     → active KEGG pathways for that cell.\n"
    "  3. Compare the per-cell TF / pathway signatures across the candidates and assign labels.\n"
    "  (optional) get_kg_context(keywords='<celltype>') for marker / mechanism context.\n\n"
    "### For topn_celltype tasks (picking answer cells from a heterogeneous sample)\n"
    "Primary: direct marker scoring via execute_pipeline_code.\n"
    "  - If the target celltype has known markers, score cells directly via marker expression.\n"
    "  - For closely related celltypes, subtract sibling-marker expression to discriminate.\n\n"
    "## Augment: KG → custom_pathway chain (applicable to BOTH task types)\n"
    "Use when markers are unclear/rare, sibling overlap is severe, or the primary approach is ambiguous.\n"
    "For cell_anon: rank the given candidate cells; for topn_celltype: rank all cells in a candidate cluster.\n"
    "  1. resolve_query_to_context_set(seed_genes=[3-5 known markers], keywords='<celltype>')\n"
    "     → context_genes with 'desc' field. Read desc to keep only biologically relevant genes.\n"
    "  2. get_expressed_dorothea_edges(genes=<curated>, cluster_id='<specific leiden id>' or cell_ids=<candidates>)\n"
    "     → expressed TF-Target edges. Do NOT use cluster_id='all'.\n"
    "     Returned TFs are often generic (STAT1, CTCF, ESR1, FOXP1). Augment with target marker pairs:\n"
    "       edges = returned_edges + [[m1, m2, 1], [m1, m3, 1], [m2, m3, 1], ...]\n"
    "  3. custom_pathway_calc(edges=<combined>, scale='cell', cell_ids=<candidates>, top_k=N)\n"
    "     → per-cell pathway L2 ranking. For cell_anon: use the given cells; for topn: cells from a candidate cluster.\n"
    "  4. execute_pipeline_code: verify cell_ids ∈ adata.obs_names, apply sibling subtraction if needed.\n\n"
    "## Cluster-level helpers (cannot emit individual cell_ids on their own; use as coarse filter only)\n"
    "run_astar_pipeline (prerequisite for graph tools), "
    "get_astar_graph_summary (per-cluster conservative TF→Target graph top edges), "
    "get_cluster_kegg_edges (per-cluster KEGG pathways + edges), "
    "get_cluster_rl_map (per-cluster ligand-receptor pairs via LIANA).\n"
)

TOOL_CALL_LIMIT  = 10
ITEM_TIMEOUT_SEC = 720  # 12 minutes


# ─── Scoring ──────────────────────────────────────────────────────────────────

_CELLTYPE_STOP = {
    "the", "of", "a", "and", "or",
    "cell", "cells",
    "positive", "negative",
    "alpha", "beta", "alphabeta",
    "derived", "thymus", "thymusderived",
    "subset", "subtype",
    "with", "high", "low",
    "expressing",
}


def _norm_celltype(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\s\-,+/]+", " ", s)
    s = re.sub(r"[^\w ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _key_tokens(s: str) -> set:
    """Return identifier-like tokens after removing stopwords"""
    return {t for t in _norm_celltype(s).split() if t and t not in _CELLTYPE_STOP}


def _resolve_option(pred: str, options: list[str]) -> str | None:
    """Return the option with the largest token overlap, or None for a tie or zero overlap."""
    p_keys = _key_tokens(pred)
    if not p_keys:
        return None
    scores = [(opt, len(p_keys & _key_tokens(opt))) for opt in options]
    scores.sort(key=lambda x: -x[1])
    if scores[0][1] == 0:
        return None
    if len(scores) > 1 and scores[0][1] == scores[1][1]:
        return None  # tie → ambiguous
    return scores[0][0]


def _celltype_match(pred: str, gt: str) -> bool:
    """Fallback matching by substring or identical key tokens."""
    p, g = _norm_celltype(pred), _norm_celltype(gt)
    if not p or not g:
        return False
    if p == g or p in g or g in p:
        return True
    pk, gk = _key_tokens(pred), _key_tokens(gt)
    return bool(pk) and pk == gk


def score_prediction(pred_json: dict, item: dict) -> dict:
    level = item["level"]

    if level == "cluster":
        correct = int(pred_json.get("answer", "").strip().upper() == item["answer"])
        return {
            "score":           float(correct),
            "cluster_correct": bool(correct),
            "n_correct":       correct,
            "n_total":         1,
        }

    if level == "topn_celltype":
        picked = (pred_json.get("cell_ids")
                  or pred_json.get("cells")
                  or pred_json.get("answer")
                  or [])
        if isinstance(picked, str):
            picked = [picked]
        n_select  = int(item.get("n_to_select", 10))
        picked    = [str(c) for c in picked][:n_select]
        answer_pool = set(str(c) for c in item.get("answer_cells", []))
        s_hit     = float(item.get("score_per_hit", 0.1))

        n_hit, n_miss = 0, 0
        score = 0.0
        for cid in picked:
            if cid in answer_pool:
                score += s_hit; n_hit += 1
            else:
                n_miss += 1
        score = min(round(score, 4), 1.0)
        return {
            "score":        score,
            "cluster_correct": False,
            "n_correct":    n_hit,
            "n_total":      n_select,
            "n_hit":        n_hit,
            "n_miss":       n_miss,
            "n_pool":       len(answer_pool),
            "picked_cells": picked,
            "n_picked":     len(picked),
        }

    if level == "gbm_neural_topn":
        picked = (pred_json.get("cell_ids")
                  or pred_json.get("cells")
                  or pred_json.get("answer")
                  or [])
        if isinstance(picked, str):
            picked = [picked]
        n_select = int(item.get("n_to_select", 10))
        picked = [str(c) for c in picked][:n_select]

        top30    = set(item.get("gt_top30_cells", []))
        positive = set(item.get("gt_positive_cells", []))
        s_top    = float(item.get("score_per_top30",    0.1))
        s_pos    = float(item.get("score_per_positive", 0.05))

        n_top, n_pos, n_zero = 0, 0, 0
        score = 0.0
        for cid in picked:
            if cid in top30:
                score += s_top; n_top += 1
            elif cid in positive:
                score += s_pos; n_pos += 1
            else:
                n_zero += 1
        score = min(round(score, 4), 1.0)
        return {
            "score":           score,
            "cluster_correct": False,
            "n_correct":       n_top,
            "n_total":         n_select,
            "n_top30":         n_top,
            "n_positive":      n_pos,
            "n_zero":          n_zero,
            "picked_cells":    picked,
            "n_picked":        len(picked),
        }

    if level == "cell_anon":
        gt_labels   = item.get("answer_labels") or item.get("celltype_labels", [])
        pred_labels = pred_json.get("labels", []) or pred_json.get("cell_labels", [])
        n_labels    = len(gt_labels)
        options     = [item.get("celltype_a", ""), item.get("celltype_b", "")]
        options     = [o for o in options if o]

        resolved_preds = []
        n_correct = 0
        for p, g in zip(pred_labels, gt_labels):
            resolved = _resolve_option(p, options) if options else None
            if resolved is None:
                # Fall back to the existing fuzzy matcher when option mapping fails.
                ok = _celltype_match(p, g)
            else:
                ok = (resolved == g)
            resolved_preds.append(resolved or p)
            if ok:
                n_correct += 1
        score = n_correct / n_labels if n_labels else 0.0
        return {
            "score":           round(score, 4),
            "cluster_correct": False,
            "label_correct":   n_correct,
            "n_correct":       n_correct,
            "n_total":         n_labels,
            "pred_labels":     pred_labels,
            "resolved_preds":  resolved_preds,
            "gt_labels":       gt_labels,
        }

    cluster_ok  = int(pred_json.get("cluster", "").strip().upper() == item["answer"])
    pred_labels = [l.strip().upper() for l in pred_json.get("cell_labels", [])]
    gt_labels   = item.get("cell_labels", [])
    n_labels    = len(gt_labels)

    label_correct = sum(
        1 for p, g in zip(pred_labels, gt_labels) if p == g
    ) if pred_labels and n_labels else 0

    n_correct = cluster_ok + label_correct
    n_total   = 1 + n_labels
    score     = n_correct * (1.0 / n_total)

    return {
        "score":           round(score, 4),
        "cluster_correct": bool(cluster_ok),
        "label_correct":   label_correct,
        "n_correct":       n_correct,
        "n_total":         n_total,
        "pred_labels":     pred_labels,
        "gt_labels":       gt_labels,
    }


def apply_tool_penalty(score: float, n_tool_calls: int) -> float:
    if n_tool_calls <= TOOL_CALL_LIMIT:
        return score
    import math
    return round(score * math.exp(-0.04 * (n_tool_calls - TOOL_CALL_LIMIT)), 4)


def parse_final_answer(text: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue
    return {}


# ─── MCP ──────────────────────────────────────────────────────────────────────

async def _mcp_session_init(http: httpx.AsyncClient, url: str | None = None) -> str:
    url = url or MCP_URL
    resp = await http.post(
        url,
        json={
            "jsonrpc": "2.0", "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "eval_grpo", "version": "1.0"},
            },
            "id": 0,
        },
        headers={"Accept": "application/json, text/event-stream"},
    )
    session_id = resp.headers.get("mcp-session-id", "")
    await http.post(
        url,
        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        headers={"Accept": "application/json, text/event-stream", "mcp-session-id": session_id},
    )
    return session_id


async def call_mcp_tool(tool_name: str, args: dict, mcp_url: str | None = None) -> str:
    url = mcp_url or MCP_URL
    try:
        async with httpx.AsyncClient(timeout=300) as http:
            session_id = await _mcp_session_init(http, url)
            resp = await http.post(
                url,
                json={
                    "jsonrpc": "2.0", "method": "tools/call",
                    "params": {"name": tool_name, "arguments": args},
                    "id": 1,
                },
                headers={"Accept": "application/json, text/event-stream", "mcp-session-id": session_id},
            )
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
                    return data["result"]["content"][0]["text"]
            return json.dumps({"success": False, "error": f"no SSE data: {resp.text[:200]}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ─── Ollama generate ──────────────────────────────────────────────────────────

async def ollama_generate(messages: list, model: str, max_tokens: int, seed: int | None = None) -> dict:
    """
    Call Ollama /api/chat and return a message containing role, content and optional tool_calls.
    Return an empty message after server errors such as context overflow so the loop can stop cleanly.
    """
    async with httpx.AsyncClient(timeout=300) as http:
        options = {
            "temperature":    TEMPERATURE,
            "top_p":          0.9,
            "repeat_penalty": 1.05,
            "num_ctx":        OLLAMA_NUM_CTX,
            "num_predict":    max_tokens,
        }
        if seed is not None:
            options["seed"] = int(seed)
        resp = await http.post(
            f"{OLLAMA_API_URL}/api/chat",
            json={
                "model":    model,
                "messages": messages,
                "tools":    TOOLS,
                "stream":   False,
                "think":    True,
                "options":  options,
            },
        )
        if resp.status_code >= 500:
            print(f"  [Ollama {resp.status_code}] context limit or server error; stopping early")
            return {"role": "assistant", "content": "", "tool_calls": []}
        resp.raise_for_status()
        return resp.json()["message"]


async def ollama_generate_no_tools(messages: list, model: str, max_tokens: int, seed: int | None = None) -> dict:
    """Generate a forced final response without exposing tools to the model."""
    async with httpx.AsyncClient(timeout=300) as http:
        options = {
            "temperature":    TEMPERATURE,
            "top_p":          0.9,
            "repeat_penalty": 1.05,
            "num_ctx":        OLLAMA_NUM_CTX,
            "num_predict":    max_tokens,
        }
        if seed is not None:
            options["seed"] = int(seed)
        resp = await http.post(
            f"{OLLAMA_API_URL}/api/chat",
            json={
                "model":    model,
                "messages": messages,
                "stream":   False,
                "think":    True,
                "options":  options,
            },
        )
        if resp.status_code >= 500:
            return {"role": "assistant", "content": "", "tool_calls": []}
        resp.raise_for_status()
        return resp.json()["message"]


def parse_ollama_tool_calls(message: dict) -> list[dict]:
    """Ollama tool_calls → [{"name": str, "arguments": dict}]"""
    calls = []
    for tc in message.get("tool_calls") or []:
        fn   = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            calls.append({"name": name, "arguments": args})
    return calls


# ───Parse Harmony-style tool calls in Hugging Face mode ──────────────────────────────────────

def _coerce(v: str):
    s = v.strip()
    if s in ("True", "true"):  return True
    if s in ("False", "false"): return False
    if s in ("None", "null"):  return None
    try: return int(s)
    except Exception: pass
    try: return float(s)
    except Exception: pass
    if s[:1] in "[{":
        try: return json.loads(s)
        except Exception: pass
    return s


def parse_harmony_tool_calls(text: str) -> list[dict]:
    """Qwen3.5 XML tool calls expected: <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>
    (gpt-oss Harmony / Hermes JSON fallback included)."""
    calls = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        body = m.group(1)
        fm = re.search(r'<function=([^>\s]+)\s*>', body)
        if fm:  # XML function format
            name = fm.group(1).strip()
            args = {}
            for pm in re.finditer(r'<parameter=([^>]+?)>\s*(.*?)\s*</parameter>', body, re.DOTALL):
                args[pm.group(1).strip()] = _coerce(pm.group(2))
            if name:
                calls.append({"name": name, "arguments": args})
            continue
        jm = re.search(r'(\{.*\})', body, re.DOTALL)   # JSON fallback inside <tool_call>, <tool_call>{json}</tool_call>
        if jm:
            obj = None
            for cand in (jm.group(1), jm.group(1).replace('\\"', '"')):
                try: obj = json.loads(cand); break
                except Exception: obj = None
            if obj and obj.get("name"):
                a = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(a, str):
                    try: a = json.loads(a)
                    except Exception: a = {}
                calls.append({"name": obj["name"], "arguments": a})
    # gpt-oss Harmony fallback
    if not calls:
        for m in re.finditer(r'to=functions\.(\w+).*?<\|message\|>(.*?)(?:<\|call\|>|$)', text, re.DOTALL):
            try: a = json.loads(m.group(2).strip().strip('"'))
            except Exception: a = {}
            calls.append({"name": m.group(1), "arguments": a})
    return calls


def _strip_think(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Handle output that closes </think> without emitting the opening tag already present in the prompt.
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    return text


def extract_thinking(text: str) -> str:
    """Extract the reasoning segment from raw Qwen3.5 output for logging.

    The vLLM prompt already ends with '<think>\n', so generated output commonly
    omits the opening tag and returns '... </think><tool_call>...'.
    """
    if not text:
        return ""
    m = re.search(r'<think>\s*(.*?)\s*</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if '</think>' in text:
        return text.split('</think>', 1)[0].strip()
    if '<tool_call>' in text:
        # Text before a direct tool call is reasoning even when </think> is omitted.
        return text.split('<tool_call>', 1)[0].strip()
    return ""


def extract_text_content(text: str) -> str:
    """Remove Qwen3.5 thinking and tool-call blocks and return the remaining final-answer text.
    A gpt-oss Harmony fallback is also supported."""
    if "<|message|>" in text and "<|start|>" in text:  # gpt-oss Harmony fallback.
        parts = []
        for seg in text.split("<|start|>"):
            if not seg.strip() or "to=functions." in seg:
                continue
            m = re.search(r'<\|message\|>(.*?)(?:<\|end\|>|$)', seg, re.DOTALL)
            if m:
                parts.append(m.group(1).strip())
        return "\n".join(parts) if parts else text.strip()
    t = _strip_think(text)
    t = re.sub(r'<tool_call>.*?</tool_call>', '', t, flags=re.DOTALL)
    return t.strip()


# ─── HF Generate ──────────────────────────────────────────────────────────────

def hf_generate(model, tok, messages, max_new_tokens=512):
    import torch
    input_text = tok.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True,
    )
    inputs    = tok(input_text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    return tok.decode(output[0][input_len:], skip_special_tokens=False), input_len


async def vllm_generate(prompt_text: str, max_tokens) -> str:
    """Generate one turn with vLLM AsyncLLMEngine and an optional LoRA adapter.
    Append raw model output without rerendering to preserve the training-time reasoning sequence."""
    import uuid
    from vllm import SamplingParams
    sp = SamplingParams(temperature=TEMPERATURE, top_p=0.9,
                        max_tokens=max_tokens)   # 학습 rollout과 일치 (rep penalty 없음)
    final = None
    async for o in VLLM_ENGINE.generate(prompt_text, sp, uuid.uuid4().hex,
                                        lora_request=VLLM_LORA):
        final = o
    return final.outputs[0].text


# ─── Run one item ─────────────────────────────────────────────────────────────

_RESET_CODE = """\
# Reset columns added by earlier runs and remove evaluation-leakage columns.
import json as _json
_KEEP_PREFIX = ('n_genes', 'n_counts', 'total_counts', 'pct_counts', 'doublet')
_KEGG_MARKER = (' signaling', ' cancer', ' disease', ' infection',
                'pathway', 'metabolism', 'Amyo', 'Kaposi', 'FoxO',
                'Aldosterone', 'Colorectal', 'Viral', 'Inflammatory',
                'Rheumatoid', 'Adipocytokine', 'Chemical')
def _is_original(col):
    if col.startswith(_KEEP_PREFIX):
        return True
    for m in _KEGG_MARKER:
        if m in col:
            return True
    return False
_drop = [c for c in adata.obs.columns if not _is_original(c)]
if _drop:
    adata.obs = adata.obs.drop(columns=_drop)
    print(f'[reset] dropped {len(_drop)} computed columns: {_drop[:8]}')
else:
    print('[reset] obs clean — no computed columns found')
_UNS_LEAK = ('cell_type_colors', 'leiden', 'leiden_colors', 'louvain',
             'louvain_colors', 'class_colors', 'annotation')
_uns_drop = [k for k in _UNS_LEAK if k in adata.uns]
for _k in _uns_drop:
    adata.uns.pop(_k, None)
if _uns_drop:
    print(f'[reset] dropped uns leak keys: {_uns_drop[:8]}')
print(f'[reset] obs columns now: {list(adata.obs.columns)[:10]}')
"""


async def _run_item_inner(
    item: dict, max_turns: int, max_tokens: int,
    use_ollama: bool, ollama_model: str,
    hf_model=None, hf_tok=None,
    system_prompt: str | None = None,
    ollama_seed: int | None = None,
) -> dict:
    sampleid = item["sampleid"]
    mcp_url  = _mcp_url_for(sampleid)   # Keep the same sample ID on the same server.

    # Reset adata.obs before the session to prevent leakage from earlier runs.
    reset_out = await call_mcp_tool("execute_pipeline_code",
                                    {"sampleid": sampleid, "code": _RESET_CODE},
                                    mcp_url=mcp_url)
    try:
        reset_msg = json.loads(reset_out).get("stdout", reset_out)
    except Exception:
        reset_msg = reset_out
    print(f"  [obs-reset] {reset_msg[:200]}")

    sys_prompt = system_prompt or SYSTEM_PROMPT_TEST
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": item["question"]},
    ]
    tools_called = []
    trajectory   = []
    final_text   = ""
    t_start      = time.time()

    # vLLM appends raw output without rerendering, preserving the training token sequence.
    # Keep vllm_P without an assistant header and add _ASST_HDR only immediately before generation.
    _ASST_HDR = "<|im_start|>assistant\n<think>\n"   # Qwen3.5 add_generation_prompt format.
    vllm_P = ""
    if USE_VLLM:
        _tools = ([t for t in TOOLS if t["function"]["name"] == "execute_pipeline_code"]
                  if EXECUTE_ONLY else TOOLS)
        vllm_P = VLLM_TOK.apply_chat_template(
            messages, tools=_tools, tokenize=False, add_generation_prompt=False)
    def _vp_turn(raw_out, tail):
        # Append one assistant turn and its user/tool tail without another assistant header.
        return _ASST_HDR + raw_out + "<|im_end|>\n" + tail

    for turn in range(max_turns):
        t_gen = time.time()

        if USE_VLLM:
            raw          = await vllm_generate(vllm_P + _ASST_HDR, max_tokens)
            elapsed_gen  = time.time() - t_gen
            tool_calls   = parse_harmony_tool_calls(raw)
            text_content = extract_text_content(raw)
            thinking     = extract_thinking(raw)
        elif use_ollama:
            message      = await ollama_generate(messages, ollama_model, max_tokens, seed=ollama_seed)
            elapsed_gen  = time.time() - t_gen
            tool_calls   = parse_ollama_tool_calls(message)
            text_content = message.get("content", "") or ""
            thinking     = message.get("thinking", "") or ""
        else:
            raw, _       = hf_generate(hf_model, hf_tok, messages, max_tokens)
            elapsed_gen  = time.time() - t_gen
            tool_calls   = parse_harmony_tool_calls(raw)
            text_content = extract_text_content(raw)
            thinking     = extract_thinking(raw)

        if not use_ollama:
            print(f"  [RAW turn={turn+1} len={len(raw)}] {raw[:400]!r}")   # Optional raw-output diagnostic.
        if thinking:
            print(f"  [thinking turn={turn+1}, len={len(thinking)}] {thinking[:160]}...")

        # Reject disallowed calls in execute-only mode; schema restriction alone may not override learned behavior.
        if EXECUTE_ONLY and tool_calls:
            _bad = [tc["name"] for tc in tool_calls if tc["name"] != "execute_pipeline_code"]
            tool_calls = [tc for tc in tool_calls if tc["name"] == "execute_pipeline_code"]
            if _bad and not tool_calls:
                print(f"  [turn={turn+1}] execute-only: rejected {_bad}")
                _rej = "Only execute_pipeline_code is available. Use execute_pipeline_code, or give your final JSON answer."
                if USE_VLLM:
                    vllm_P += _vp_turn(raw, f"<|im_start|>user\n{_rej}<|im_end|>\n")
                else:
                    messages.append({"role": "user", "content": _rej})
                continue

        if not tool_calls:
            if text_content.strip():
                # Normal termination after the model supplies a final answer.
                final_text = text_content
                print(f"  [turn={turn+1}] no tool call ({elapsed_gen:.1f}s) → {text_content[:120]}")
                trajectory.append({"turn": turn+1, "type": "text", "content": text_content, "thinking": thinking})
                break
            else:
                # Nudge another turn when the model emits reasoning but neither a final answer nor a tool call.
                print(f"  [turn={turn+1}] empty (no tool, no text) — nudging to continue")
                trajectory.append({"turn": turn+1, "type": "text", "content": "", "thinking": thinking})
                _nudge = "Continue investigating with tool calls, or provide your final JSON answer now."
                if USE_VLLM:
                    vllm_P += _vp_turn(raw, f"<|im_start|>user\n{_nudge}<|im_end|>\n")
                else:
                    if thinking:
                        messages.append({"role": "assistant", "content": "", "thinking": thinking})
                    messages.append({"role": "user", "content": _nudge})
                continue
        # Stop executing tools after the budget and force a final response, matched behaviour of agent_func.
        if len(tools_called) >= TOOL_CALL_LIMIT:
            print(f"  [turn={turn+1}] budget exhausted ({len(tools_called)}) — forcing final")
            _bud = "Tool budget exhausted. Provide your final answer now as a single JSON object — no more tool calls."
            if USE_VLLM:
                vllm_P += _vp_turn(raw, f"<|im_start|>user\n{_bud}<|im_end|>\n")
            else:
                messages.append({"role": "user", "content": _bud})
            continue

        # Process tool calls
        if use_ollama:
            asst_msg = {
                "role":       "assistant",
                "content":    text_content,
                "tool_calls": message.get("tool_calls", []),
            }
            if thinking:
                # Harmony reasoning passthrough — 다음 턴에 analysis channel 복원
                asst_msg["thinking"] = thinking
            messages.append(asst_msg)

        # In plain HF mode, preserve raw assistant content and tool history.
        if not use_ollama and not USE_VLLM:
            messages.append({"role": "assistant", "content": raw})

        _vp_tool_blocks = []   # Accumulate vLLM tool responses for this turn.
        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            # Do not inject sampleid into tools whose schemas do not accept it.
            if isinstance(args, dict) and "sampleid" not in args and name not in _NO_SAMPLEID_TOOLS:
                args["sampleid"] = sampleid

            print(f"  [tool_call turn={turn+1}] {name}({json.dumps(args, ensure_ascii=False)}) ({elapsed_gen:.1f}s)")

            tool_result = await call_mcp_tool(name, args, mcp_url=mcp_url)

            try:
                d = json.loads(tool_result)
                if "pathways" in d and len(d["pathways"]) > 3:
                    d["pathways"] = d["pathways"][:3]
                    tool_result = json.dumps(d)
            except Exception:
                pass

            print(f"  [tool_resp] {tool_result}")
            # Truncate long tool responses before adding them to the model context.
            content_for_msg = tool_result if len(tool_result) <= 4000 else \
                tool_result[:4000] + f"\n...({len(tool_result)-4000} chars truncated)"
            if USE_VLLM:
                _vp_tool_blocks.append(f"<|im_start|>user\n<tool_response>\n{content_for_msg}\n</tool_response><|im_end|>\n")
            else:
                messages.append({"role": "tool", "content": content_for_msg})

            tools_called.append(name)
            trajectory.append({
                "turn":     turn + 1,
                "type":     "function_call",
                "tool":     name,
                "args":     args,
                "response": tool_result,
                "thinking": thinking,
            })

        # Append the assistant output and all tool responses for this vLLM turn together.
        if USE_VLLM and tool_calls:
            vllm_P += _vp_turn(raw, "".join(_vp_tool_blocks))

        if text_content and not text_content.startswith("to="):
            final_text = text_content
            trajectory.append({"turn": turn+1, "type": "text", "content": text_content, "thinking": thinking})

    if not final_text:
        _force = "Tool budget exhausted. Provide your final answer now as a single JSON object — no more tool calls."
        if USE_VLLM:
            vllm_P    += f"<|im_start|>user\n{_force}<|im_end|>\n"
            raw_f      = await vllm_generate(vllm_P + _ASST_HDR, max_tokens)
            final_text = extract_text_content(raw_f)
            final_think = extract_thinking(raw_f)
            print(f"  [force-final/vllm] → {final_text[:120]}")
            trajectory.append({"turn": max_turns+1, "type": "text", "content": final_text, "thinking": final_think})
        elif use_ollama:
            messages.append({"role": "user", "content": _force})
            final_msg     = await ollama_generate_no_tools(messages, ollama_model, max_tokens, seed=ollama_seed)
            final_text    = final_msg.get("content", "") or ""
            final_think   = final_msg.get("thinking", "") or ""
            print(f"  [force-final] → {final_text[:120]}")
            if final_think:
                print(f"  [force-final thinking len={len(final_think)}] {final_think[:160]}...")
            trajectory.append({"turn": max_turns+1, "type": "text", "content": final_text, "thinking": final_think})

    elapsed   = time.time() - t_start
    pred_json = parse_final_answer(final_text)
    scoring   = score_prediction(pred_json, item)

    raw_score   = scoring["score"]
    final_score = apply_tool_penalty(raw_score, len(tools_called))
    scoring["score"]           = final_score
    scoring["score_raw"]       = raw_score
    scoring["penalty_applied"] = final_score < raw_score

    return {
        "id":           item["id"],
        "level":        item["level"],
        "variant":      item.get("variant", ""),
        "answer_gt":    (item.get("answer")
                         or item.get("answer_labels")
                         or item.get("celltype_labels")
                         or item.get("answer_cells", "")),
        "pred_json":    pred_json,
        "tools_called": tools_called,
        "n_tool_calls": len(tools_called),
        "elapsed_sec":  round(elapsed, 1),
        "timed_out":    False,
        "ollama_seed":  ollama_seed,
        "trajectory":   trajectory,
        **scoring,
    }


async def run_item(item: dict, max_turns: int, max_tokens: int,
                   use_ollama: bool, ollama_model: str,
                   hf_model=None, hf_tok=None,
                   system_prompt: str | None = None,
                   ollama_seed: int | None = None) -> dict:
    try:
        return await asyncio.wait_for(
            _run_item_inner(item, max_turns, max_tokens, use_ollama, ollama_model,
                            hf_model, hf_tok, system_prompt, ollama_seed),
            timeout=ITEM_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print(f"  [TIMEOUT] {item['id']} — exceeded {ITEM_TIMEOUT_SEC//60} minutes; score=0")
        return {
            "id":              item["id"],
            "level":           item["level"],
            "variant":         item.get("variant", ""),
            "answer_gt":       (item.get("answer")
                                or item.get("answer_labels")
                                or item.get("celltype_labels")
                                or item.get("answer_cells", "")),
            "pred_json":       {},
            "tools_called":    [],
            "n_tool_calls":    0,
            "elapsed_sec":     float(ITEM_TIMEOUT_SEC),
            "timed_out":       True,
            "ollama_seed":     ollama_seed,
            "score":           0.0,
            "score_raw":       0.0,
            "penalty_applied": False,
            "cluster_correct": False,
            "trajectory":      [],
        }


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], label: str):
    cluster_res  = [r for r in results if r["level"] == "cluster"]
    cell_res     = [r for r in results if r["level"] == "cell"]
    cell_anon_res = [r for r in results if r["level"] == "cell_anon"]
    topn_res     = [r for r in results if r["level"] == "topn_celltype"]

    def _avg(lst, key):
        vals = [r[key] for r in lst if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'='*65}")
    print(f"  SUMMARY — {label}")
    print(f"{'='*65}")
    print(f"  Total items : {len(results)}")

    n_timeout = sum(1 for r in results if r.get("timed_out"))
    n_penalty = sum(1 for r in results if r.get("penalty_applied"))
    if n_timeout or n_penalty:
        print(f"  timeouts : {n_timeout}  |  tool-penalty : {n_penalty}")

    if cluster_res:
        print(f"\n  [cluster]  n={len(cluster_res)}")
        print(f"    accuracy       : {_avg(cluster_res, 'cluster_correct')*100:.1f}%")
        print(f"    avg score      : {_avg(cluster_res, 'score'):.3f}")
        print(f"    avg tool calls : {_avg(cluster_res, 'n_tool_calls'):.1f}")
        print(f"    avg time (s)   : {_avg(cluster_res, 'elapsed_sec'):.1f}")

    if cell_res:
        lc_vals   = [r.get("label_correct", 0) for r in cell_res]
        label_acc = sum(lc_vals) / (len(lc_vals) * 4) if lc_vals else 0
        print(f"\n  [cell]  n={len(cell_res)}")
        print(f"    mean score     : {_avg(cell_res, 'score'):.3f}")
        print(f"    cluster acc    : {_avg(cell_res, 'cluster_correct')*100:.1f}%")
        print(f"    cell label acc : {label_acc*100:.1f}%")
        print(f"    avg tool calls : {_avg(cell_res, 'n_tool_calls'):.1f}")
        print(f"    avg time (s)   : {_avg(cell_res, 'elapsed_sec'):.1f}")

    if cell_anon_res:
        n_cells = sum(r.get("n_total", 4) for r in cell_anon_res)
        n_correct = sum(r.get("n_correct", 0) for r in cell_anon_res)
        print(f"\n  [cell_anon]  n={len(cell_anon_res)}")
        print(f"    mean score     : {_avg(cell_anon_res, 'score'):.3f}")
        print(f"    per-cell acc   : {n_correct/max(n_cells,1)*100:.1f}%  ({n_correct}/{n_cells})")
        print(f"    perfect (all)  : {sum(1 for r in cell_anon_res if r.get('n_correct')==r.get('n_total'))}/{len(cell_anon_res)}")
        print(f"    avg tool calls : {_avg(cell_anon_res, 'n_tool_calls'):.1f}")
        print(f"    avg time (s)   : {_avg(cell_anon_res, 'elapsed_sec'):.1f}")

    if topn_res:
        print(f"\n  [topn_celltype]  n={len(topn_res)}")
        print(f"    mean score     : {_avg(topn_res, 'score'):.3f}")
        print(f"    avg hit/10     : {_avg(topn_res, 'n_hit'):.1f}")
        print(f"    avg tool calls : {_avg(topn_res, 'n_tool_calls'):.1f}")
        print(f"    avg time (s)   : {_avg(topn_res, 'elapsed_sec'):.1f}")

    print(f"{'='*65}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    global TEMPERATURE
    p = argparse.ArgumentParser(description="GRPO dataset evaluation runner")
    p.add_argument("--jsonl",         required=True,
                   help="JSONL file to evaluate")
    p.add_argument("--ollama",        action="store_true",        help="Ollama 모드 (gpt-oss:20b)")
    p.add_argument("--ollama-model",  default=OLLAMA_MODEL,       help=f"Ollama 모델명 (default: {OLLAMA_MODEL})")
    p.add_argument("--base-only",     action="store_true",        help="HF base 모델")
    p.add_argument("--ckpt",          default=None,               help="HF LoRA checkpoint path")
    p.add_argument("--vllm",          action="store_true",        help="vLLM AsyncLLM 백엔드 (배칭+동시 채점). --ckpt와 함께 사용")
    p.add_argument("--base-model",    default=None,               help="vLLM base 모델 override (예: unsloth/Qwen3.5-27B, LoRA 없이 원모델 평가용)")
    p.add_argument("--concurrency",   type=int,   default=8,      help="--vllm 동시 item 수 (MCP_N_WORKERS와 함께 조율)")
    p.add_argument("--tool-cutoff",   type=int,   default=None,   help="tool budget cap (기본 TOOL_CALL_LIMIT=10). 도달 시 강제 final + penalty 임계")
    p.add_argument("--execute-only",  action="store_true",        help="execute_pipeline_code 도구만 모델에 노출")
    p.add_argument("--adk",           action="store_true",        help="ADK runner (placeholder)")
    p.add_argument("--n-samples",     type=int,   default=None,   help="평가할 최대 item 수")
    p.add_argument("--skip",          type=int,   default=0,      help="앞의 N개 item 건너뜀 (이어서 실행 시)")
    p.add_argument("--filter-id",     default=None,               help="특정 id만 실행 (쉼표 구분으로 다중 지정 가능)")
    p.add_argument("--n-repeats",     type=int,   default=1,      help="각 item을 N번 반복 실행 (sampling 변동성 측정용)")
    p.add_argument("--max-turns",     type=int,   default=MAX_TURNS)
    p.add_argument("--max-tokens",    type=int,   default=2048)
    p.add_argument("--max-model-len", type=int,   default=16384,
                   help="vLLM max_model_len (default 16384)")
    p.add_argument("--temp",          type=float, default=TEMPERATURE,
                   help=f"sampling temperature for Ollama/HF (default {TEMPERATURE})")
    p.add_argument("--seed",          type=int,   default=OLLAMA_SEED,
                   help="Base Ollama seed. If set, each item/repeat uses deterministic seed = base + crc32(item_id) + repeat.")
    p.add_argument("--kg-guided",     action="store_true",
                   help="KG 워크플로우 강제 (gbm_neural_topn 전용 system prompt 사용)")
    p.add_argument("--test-prompt",   action="store_true",
                   help="resolve_query_to_context_set → get_expressed_dorothea_edges → custom_pathway_calc 흐름 권장 테스트용 system prompt")
    p.add_argument("--out",           default=None,               help="결과 JSONL 저장 경로")
    args = p.parse_args()

    if not args.ollama and not args.base_only and not args.ckpt and not args.adk and not args.vllm:
        p.error("--ollama, --base-only, --ckpt, --adk, --vllm 중 하나 필요")


    TEMPERATURE = args.temp
    print(f"[temp] temperature={TEMPERATURE}")
    if args.seed is not None:
        print(f"[seed] base={args.seed} mode=item_id+repeat")

    items = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if args.filter_id:
        keep = {x.strip() for x in args.filter_id.split(",") if x.strip()}
        before = len(items)
        items = [it for it in items if it["id"] in keep]
        matched = {it["id"] for it in items}
        missing = keep - matched
        print(f"[filter-id] {before} → {len(items)}개 (요청 {len(keep)}개 중 {len(matched)}개 매칭)")
        if missing:
            print(f"[filter-id] 미매칭 id: {sorted(missing)}")
        if not items:
            print("[filter-id] 매칭된 item이 없어 종료")
            return
    if args.skip:
        items = items[args.skip:]
        print(f"[skip] 앞의 {args.skip}개 건너뜀")
    if args.n_samples:
        items = items[:args.n_samples]
    print(f"평가 대상: {len(items)}개  ({args.jsonl})")

    if args.adk:
        print("[ADK] placeholder — 미구현")
        return

    use_ollama   = args.ollama
    ollama_model = args.ollama_model
    hf_model     = None
    hf_tok       = None
    if args.kg_guided:
        active_system_prompt = SYSTEM_PROMPT_KG_GUIDED
        print("[kg-guided] KG 워크플로우 강제 system prompt 적용")
    elif args.test_prompt:
        active_system_prompt = SYSTEM_PROMPT_TEST
        print("[test-prompt] SYSTEM_PROMPT_TEST (resolve_query→expressed_dorothea→custom_pathway 흐름) 적용")
    else:
        active_system_prompt = None

    if use_ollama:
        label = f"ollama:{ollama_model}" + ("_kgguided" if args.kg_guided else "")
        print(f"\n[Ollama] model={ollama_model}  url={OLLAMA_API_URL}")
        # Ollama 연결 확인
        async with httpx.AsyncClient(timeout=10) as http:
            try:
                r = await http.get(f"{OLLAMA_API_URL}/api/tags")
                names = [m["name"] for m in r.json().get("models", [])]
                if ollama_model not in names:
                    print(f"[Ollama] 경고: '{ollama_model}' 목록에 없음. 사용 가능: {names}")
                else:
                    print(f"[Ollama] OK — {ollama_model} 확인")
            except Exception as e:
                print(f"[Ollama] 연결 실패: {e}")
                return
    elif args.vllm:
        global USE_VLLM, VLLM_ENGINE, VLLM_TOK, VLLM_LORA, TOOL_CALL_LIMIT, EXECUTE_ONLY
        if args.tool_cutoff:
            TOOL_CALL_LIMIT = args.tool_cutoff
        EXECUTE_ONLY = args.execute_only
        print(f"[cfg] TOOL_CALL_LIMIT={TOOL_CALL_LIMIT} | execute_only={EXECUTE_ONLY}")
        USE_VLLM = True
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer
        adapter = args.ckpt
        base_model = args.base_model or HF_MODEL_NAME
        if adapter and not args.base_model:
            _cfg = os.path.join(adapter, "adapter_config.json")
            if os.path.exists(_cfg):
                _b = json.load(open(_cfg)).get("base_model_name_or_path")
                if _b:
                    base_model = _b
        print(f"\n[vLLM] base={base_model}  adapter={adapter}  N_MCP={MCP_N_WORKERS}  conc={args.concurrency}  max_model_len={args.max_model_len}")
        VLLM_ENGINE = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=base_model, enable_lora=bool(adapter), max_lora_rank=64,
            max_model_len=args.max_model_len, dtype="bfloat16",
            gpu_memory_utilization=0.85, trust_remote_code=True,
        ))
        VLLM_TOK  = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        VLLM_LORA = LoRARequest("grpo", 1, adapter) if adapter else None
        label = (adapter or "vllm-base") + "_vllm"
    else:
        from unsloth import FastLanguageModel
        # --ckpt가 있으면 adapter_config.json의 base_model_name_or_path를 따라감
        # (HF_MODEL_NAME=gpt-oss 하드코딩이 Qwen3.5 adapter와 불일치하던 버그 수정)
        base_model = HF_MODEL_NAME
        if args.ckpt:
            _cfg = os.path.join(args.ckpt, "adapter_config.json")
            if os.path.exists(_cfg):
                _b = json.load(open(_cfg)).get("base_model_name_or_path")
                if _b:
                    base_model = _b
        print(f"\nLoading base: {base_model}")
        hf_model, hf_tok = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        if args.ckpt:
            print(f"Loading LoRA: {args.ckpt}")
            from peft import PeftModel
            hf_model = PeftModel.from_pretrained(hf_model, args.ckpt)
        FastLanguageModel.for_inference(hf_model)
        hf_model.eval()
        label = ("BASE" if args.base_only else args.ckpt) + ("_kgguided" if args.kg_guided else "")

    # MCP 연결 확인 (NO_KG: KG 도구 대신 execute_pipeline_code로 핑 — KG graphml 의존 제거)
    print("\n[MCP] Testing connection...")
    ping = await call_mcp_tool("execute_pipeline_code", {"sampleid": items[0]["sampleid"], "code": "print('ping')"})
    try:
        if json.loads(ping).get("success") is False:
            print(f"[MCP] 연결 실패: {ping[:200]}")
            return
    except Exception:
        pass
    print("[MCP] OK")

    # 평가 루프
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else \
        LOG_DIR / f"eval_{Path(args.jsonl).stem}_{label.replace('/', '_').replace(':', '_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_repeats = max(1, args.n_repeats)
    total_runs = len(items) * n_repeats
    # (item, rep) 작업 목록
    jobs = [(item, rep) for item in items for rep in range(n_repeats)]
    results = []

    async def _run_one(item, rep):
        mcp_url = _mcp_url_for(item["sampleid"])
        await call_mcp_tool("reset_pipeline_namespace", {"sampleid": item["sampleid"]}, mcp_url=mcp_url)
        ollama_seed = _seed_for_item(args.seed, item.get("id", item["sampleid"]), rep)
        result = await run_item(
            item, args.max_turns, args.max_tokens,
            use_ollama, ollama_model, hf_model, hf_tok,
            system_prompt=active_system_prompt,
            ollama_seed=ollama_seed,
        )
        if n_repeats > 1:
            result["repeat"] = rep
            result["id"] = f"{result.get('id', item['id'])}__rep{rep}"
        return result

    with out_path.open("w") as f:
        if USE_VLLM:
            # vLLM: 동시 실행 (continuous batching + MCP sticky 분산)
            # 증분 저장 + 작업별 예외격리 — 한 item 실패/엔진 이상이 전체를 날리지 않게
            sem = asyncio.Semaphore(max(1, args.concurrency))
            done = 0
            async def _guarded(item, rep):
                nonlocal done
                async with sem:
                    print(f"  [start] {item['id']} level={item['level']}", flush=True)
                    try:
                        r = await _run_one(item, rep)
                    except Exception as e:
                        import traceback as _tb
                        print(f"  [ERR] {item['id']}: {e}\n{_tb.format_exc()[:500]}", flush=True)
                        r = {"id": item["id"], "level": item["level"], "variant": item.get("variant",""),
                             "pred_json": {}, "tools_called": [], "n_tool_calls": 0,
                             "elapsed_sec": 0.0, "timed_out": False, "trajectory": [],
                             "score": 0.0, "score_raw": 0.0, "penalty_applied": False,
                             "error": str(e)[:300]}
                    done += 1
                    print(f"  [{done}/{total_runs}] {r['id']} score={r.get('score')} tools={r.get('n_tool_calls')} t={r.get('elapsed_sec')}s", flush=True)
                    return r
            tasks = [asyncio.create_task(_guarded(it, rep)) for it, rep in jobs]
            for fut in asyncio.as_completed(tasks):
                r = await fut
                results.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()   # 완료 즉시 기록 → 중간 크래시에도 보존
        else:
            # 기존 직렬 (ollama/unsloth)
            for run_idx, (item, rep) in enumerate(jobs, 1):
                tag = f"  rep={rep+1}/{n_repeats}" if n_repeats > 1 else ""
                print(f"\n[{run_idx}/{total_runs}] {item['id']}  level={item['level']}{tag}")
                result = await _run_one(item, rep)
                results.append(result)
                print(f"  score={result['score']}  tools={result['tools_called']}  t={result['elapsed_sec']}s")
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    print_summary(results, label)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
