"""
SC Graph MCP Server (3_31 — human/mouse organism branch support)
============================================================
Version of sc_graph_mcp_server.py with added human / mouse support.

Changes:
  + detect_organism(adata)           : auto-detect species from var_names pattern
  + state["organism_cache"]          : sampleid → detected organism cache
  + state["dorothea_dfs"]            : per-organism DoRothEA cache (dict)
  + _get_organism(sampleid, adata)   : organism lookup/detection/cache helper
  + _kegg_dir(organism, base)        : returns KEGG path matching organism
  + run_astar_pipeline               : added organism parameter
  + get_cluster_rl_map               : added organism parameter
  + get_cluster_kegg_edges           : added organism parameter
  + get_cell_kegg_edges              : added organism parameter

Port: 8006  (to avoid conflict with existing port 8005)

Tool structure (2-layer):
  ── cluster level ─────────────────────────────────────────────
  run_astar_pipeline          cluster A* search + graph construction (expensive)
  get_astar_graph_summary     conservative graph result query
  get_cluster_rl_map          LIANA R-L map query
  get_cluster_kegg_edges      KEGG top pathway + top edge query
  ── cell level ────────────────────────────────────────────────
  get_astar_cellular_info     cell → conservative graph edges recomputed with cell's expression (cellular_beta)
  get_cell_kegg_edges         top edges based on single-cell beta for a given cell
"""

import sys
import io
import os
import re
import json
import hashlib
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from importlib.resources import files

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

# Global JSON patch: make json.dumps numpy-safe even when models re-import json
# inside execute_pipeline_code. Without this, `import json` in user code shadows
# the namespace-injected safe json and causes "float32 not serializable" errors.
_orig_json_default = json.JSONEncoder.default
def _np_safe_default(self, o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, set):         return list(o)
    return _orig_json_default(self, o)
json.JSONEncoder.default = _np_safe_default
from fastmcp import FastMCP
from pydantic import Field
from typing import Dict, List, Optional
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmcp_server import AnndataBackend

# Pure computation utilities (3_31 version)
from graph_utils import (
    parse_all_kegg_xmls_raw, filter_kegg_pathways,
    detect_organism,
    run_astar_for_cluster,
    build_cluster_conservative_graphs,
    build_rl_gene_map,
    parse_all_kegg_xmls,
    compute_and_select_top_kegg,
    get_top_edges_per_pathway,
    _ensure_csr,
)

# Rust kernels for custom V/E scoring (Tool D path)
from cwg_rust import (
    KEGGPathway,
    compute_all_kegg_norms_sparse,
    compute_all_kegg_norms_cluster_mean,
)

# ================================================================
# FastMCP app + server state
# ================================================================

mcp = FastMCP("SC Graph Analysis")


def _format_rl_map(raw: dict) -> dict:
    """Convert raw build_rl_gene_map() result to the same structure as the get_cluster_rl_map tool response.
    Used when storing in namespace['rl_map'] to align with the tool response schema."""
    if not raw or "cluster_edge" not in raw:
        return {}
    return {
        "clusters": {
            cid: {
                "n_ligand":   len(v["ligand"]),
                "n_receptor": len(v["receptor"]),
                "n_edges":    len(v["edges"]),
                "edges": [
                    {"sender": cid, "ligand": e[0], "receptor": e[1], "receiver": e[2]}
                    for e in v["edges"]
                ],
            }
            for cid, v in raw["cluster_edge"].items()
        }
    }

state = {
    "backend":         None,          # AnndataBackend (lazy initialization)
    "dorothea_dfs":    {},            # {organism: DataFrame} — per-organism cache
    "organism_cache":  {},            # {sampleid: "human" | "mouse"}
    # sampleid → result cache
    "astar_results":   {},            # {sampleid: {cid: List[List[int]]}}
    "cluster_graphs":  {},            # {sampleid: cluster_results dict}
    "rl_maps":         {},            # {sampleid: rl_map dict}
    "kegg_pathways":   {},            # {sampleid or (sampleid,kegg_dir,organism): List[KEGGPathway]}
    "kegg_parsed":     {},            # {(kegg_dir, organism): List[KEGGPathway]} — global XML cache
    "python_envs":     {},            # {sampleid: ScGraphPythonEnvironment}
    # ── adata snapshot for cross-run isolation ────────────────────────────
    "adata_obs_snap":  {},            # {sampleid: pd.DataFrame} — deep copy at first access
    "adata_uns_snap":  {},            # {sampleid: dict}
    # ── Full call log for GSPO training ──────────────────────────────────
    "call_log":        [],
    # ── KnowledgeGraph cache ──────────────────────────────────────────────
    "kg_graph":        None,          # networkx.Graph loaded from graphml
    "kg_path":         None,          # path used for cached kg_graph
    "context_sets":     {},           # {context_id: result dict}
}


def _log_call(tool_name: str, inputs: dict, result: dict, elapsed_ms: int) -> None:
    """Append all tool calls to call_log in chronological order."""
    def _safe(v):
        if isinstance(v, dict):
            return {k: _safe(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(i) for i in v]
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        try:
            import json; json.dumps(v)
            return v
        except (TypeError, ValueError):
            return str(v)

    success = result.get("success", True) if isinstance(result, dict) else True
    state["call_log"].append({
        "timestamp":  datetime.now().isoformat(),
        "elapsed_ms": elapsed_ms,
        "tool":       tool_name,
        "input":      _safe(inputs),
        "success":    success,
        "result":     _safe(result),
    })


DATA_DIR      = os.getenv("MCP_DATA_DIR", str(Path.cwd() / "data"))

def _default_kegg_dir() -> str:
    local_kegg = Path(DATA_DIR) / "KEGG_Graph_processing"
    if local_kegg.exists():
        return str(local_kegg)
    try:
        return str(files("keggx").joinpath("data", "KEGG_Graph_processing"))
    except Exception:
        return str(local_kegg)

KEGG_DIR      = os.getenv("KEGG_DIR", _default_kegg_dir())
_ALIASES_PATH = Path(__file__).parent / ".sc_graph_aliases.json"


def _resolve_sampleid(sampleid: str) -> str:
    """Convert alias → actual sampleid."""
    if not _ALIASES_PATH.exists():
        return sampleid
    try:
        aliases = json.loads(_ALIASES_PATH.read_text())
        return aliases.get(sampleid, sampleid)
    except Exception:
        return sampleid


def _get_backend() -> AnndataBackend:
    if state["backend"] is None:
        state["backend"] = AnndataBackend(DATA_DIR)
    return state["backend"]


def _get_adata(sampleid: str):
    real  = _resolve_sampleid(sampleid)
    adata = _get_backend().get_adata(real)
    if adata is None:
        raise ValueError(f"Sample '{sampleid}' not found in backend")
    if real not in state["adata_obs_snap"]:
        state["adata_obs_snap"][real] = adata.obs.copy(deep=True)
        state["adata_uns_snap"][real] = dict(adata.uns)
    return adata


# ── organism helper ─────────────────────────────────────────────────

def _get_organism(sampleid: str, adata=None, hint: str = "auto") -> str:
    """
    Return the organism for a given sampleid.

    Priority:
      1. If hint is "human" or "mouse", use that value (update cache)
      2. Return from state["organism_cache"][sampleid] if cached
      3. Auto-detect via detect_organism() from adata.var_names and cache
    """
    sid = _resolve_sampleid(sampleid)

    if hint in ("human", "mouse"):
        state["organism_cache"][sid] = hint
        return hint

    if sid in state["organism_cache"]:
        return state["organism_cache"][sid]

    # Auto-detect
    if adata is None:
        adata = _get_adata(sid)
    org = detect_organism(adata)
    state["organism_cache"][sid] = org
    return org


_DOROTHEA_PARQUET = {
    "human": os.getenv("DOROTHEA_HUMAN_PARQUET", str(Path(DATA_DIR) / "dorothea_ABC_human.parquet")),
    "mouse": os.getenv("DOROTHEA_MOUSE_PARQUET", str(Path(DATA_DIR) / "dorothea_ABC_mouse.parquet")),
}

def _get_dorothea(organism: str = "human") -> pd.DataFrame:
    """Load DoRothEA network — parquet cache first, fallback to decoupler."""
    if organism not in state["dorothea_dfs"]:
        parquet_path = _DOROTHEA_PARQUET.get(organism)
        if parquet_path and Path(parquet_path).exists():
            print(f"[_get_dorothea] loading from parquet: {parquet_path}")
            state["dorothea_dfs"][organism] = pd.read_parquet(parquet_path)
        else:
            import decoupler as dc
            print(f"[_get_dorothea] organism='{organism}' loading via decoupler...")
            df = dc.op.dorothea(organism=organism, levels=["A", "B", "C"])
            if parquet_path:
                df.to_parquet(parquet_path, index=False)
            state["dorothea_dfs"][organism] = df
        print(f"[_get_dorothea] done: {len(state['dorothea_dfs'][organism]):,} edges")
    return state["dorothea_dfs"][organism]


def _kegg_dir(organism: str, base: str = KEGG_DIR) -> str:
    """
    Return the KEGG KGML directory matching the organism.

      human → base/          (hsa .kgml files located directly)
      mouse → base/mmu/      (mmu .kgml files location)
    """
    if organism == "mouse":
        return str(Path(base) / "mmu")
    return base

def _make_safe_json():
    """Return a json-like module where dumps() auto-converts numpy types."""
    import json as _json

    class _NumpyEncoder(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.ndarray):  return o.tolist()
            return super().default(o)

    class _SafeJson:
        @staticmethod
        def dumps(*args, **kwargs):
            kwargs.setdefault("cls", _NumpyEncoder)
            return _json.dumps(*args, **kwargs)
        loads = staticmethod(_json.loads)

    return _SafeJson()


# ================================================================
# Persistent Python Environment
# ================================================================

class ScGraphPythonEnvironment:
    """Persistent Python execution environment per sampleid."""

    _DANGEROUS = [
        (r'\bopen\s*\([^)]*["\']w',    "File writing not allowed"),
        (r'\bopen\s*\([^)]*["\']a',    "File appending not allowed"),
        (r'\bos\.remove\b',            "File deletion not allowed"),
        (r'\bos\.unlink\b',            "File deletion not allowed"),
        (r'\bshutil\.rmtree\b',        "Directory deletion not allowed"),
        (r'\bos\.rmdir\b',             "Directory deletion not allowed"),
        (r'\bos\.mkdir\b',             "Directory creation not allowed"),
        (r'\bos\.system\b',            "System command not allowed"),
        (r'\bsubprocess\b',            "Subprocess not allowed"),
        (r'\b__import__\b',            "Dynamic import not allowed"),
        (r'\bimportlib\b',             "Import manipulation not allowed"),
        (r'\brequests\b',              "Network requests not allowed"),
        (r'\burllib\b',                "Network requests not allowed"),
        (r'\bsocket\b',                "Socket operations not allowed"),
        (r'\bcompile\b',               "Code compilation not allowed"),
        (r'\bos\.environ\[',           "Env var modification not allowed"),
        (r'\bos\.chdir\b',             "Directory change not allowed"),
        (r'\bsetattr\s*\(\s*adata',    "adata attribute modification not allowed"),
        (r'\bdelattr\b',               "Attribute deletion not allowed"),
        (r'\bexit\s*\(',               "exit() not allowed — use return or raise"),
        (r'\bquit\s*\(',               "quit() not allowed — use return or raise"),
        (r'\bsys\.exit\s*\(',          "sys.exit() not allowed — use return or raise"),
        (r'\bad\.read_h5ad\b',         "Do not reload adata — it is pre-loaded in namespace"),
        (r'\banndata\.read_h5ad\b',    "Do not reload adata — it is pre-loaded in namespace"),
    ]

    def __init__(self, sampleid: str, adata, state_ref: dict):
        self.sampleid   = sampleid
        self._state_ref = state_ref
        self.history    = []

        self.namespace = {
            "adata":    adata,
            "sc":       sc,
            "np":       np,
            "pd":       pd,
            "sp":       sp,
            "nx":       nx,
            "issparse": sp.issparse,
            "run_astar_for_cluster":       run_astar_for_cluster,
            "build_cluster_conservative_graphs":      build_cluster_conservative_graphs,
            "build_rl_gene_map":           build_rl_gene_map,
            "parse_all_kegg_xmls":         parse_all_kegg_xmls,
            "compute_and_select_top_kegg": compute_and_select_top_kegg,
            "get_top_edges_per_pathway":   get_top_edges_per_pathway,
            "cluster_graphs":  state_ref["cluster_graphs"].get(sampleid, {}),
            "rl_map":          _format_rl_map(state_ref["rl_maps"].get(sampleid, {})),
            "kegg_pathways":   state_ref["kegg_pathways"].get(sampleid, []),
            "astar_results":   state_ref["astar_results"].get(sampleid, {}),
            "json":            _make_safe_json(),
        }

    def sync_state(self):
        sid = self.sampleid
        self.namespace["cluster_graphs"]    = self._state_ref["cluster_graphs"].get(sid, {})
        self.namespace["rl_map"]            = _format_rl_map(self._state_ref["rl_maps"].get(sid, {}))
        self.namespace["kegg_pathways"]     = self._state_ref["kegg_pathways"].get(sid, [])
        self.namespace["astar_results"]     = self._state_ref["astar_results"].get(sid, {})
        self.namespace["execution_history"] = self.history

    def _check_safety(self, code: str):
        for pattern, msg in self._DANGEROUS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, msg
        return True, ""

    def execute(self, code: str) -> dict:
        ok, safety_err = self._check_safety(code)
        if not ok:
            return {"success": False, "error": f"SecurityError: {safety_err}"}

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace)

            MAX_STDOUT = 4000
            stdout_raw = stdout_buf.getvalue()
            if len(stdout_raw) > MAX_STDOUT:
                stdout = stdout_raw[:MAX_STDOUT] + f"\n... [truncated, {len(stdout_raw)} chars total]"
            else:
                stdout = stdout_raw
            stderr = stderr_buf.getvalue()

            result = None
            try:
                last = code.strip().split("\n")[-1]
                if "=" not in last or last.strip().startswith("result"):
                    result = eval(last, self.namespace)
            except Exception:
                pass

            self.history.append({"code": code, "success": True, "stdout": stdout})
            return {
                "success": True,
                "stdout":  stdout,
                "stderr":  stderr if stderr else None,
                "result":  str(result) if result is not None else None,
            }

        except SystemExit as e:
            # exit() / sys.exit() in generated code must NOT crash the server
            stdout_raw = stdout_buf.getvalue()
            stdout = stdout_raw[:4000] + f"\n... [truncated, {len(stdout_raw)} chars total]" if len(stdout_raw) > 4000 else stdout_raw
            err = f"SystemExit({e.code}) — use 'raise RuntimeError(...)' instead of exit()"
            self.history.append({"code": code, "success": False, "error": err})
            return {
                "success":   False,
                "error":     err,
                "stdout":    stdout,
                "hint":      "Do not call exit() or sys.exit(). Raise an exception or use early return.",
            }

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb  = traceback.format_exc()
            self.history.append({"code": code, "success": False, "error": err})
            return {
                "success":   False,
                "error":     err,
                "traceback": tb,
                "hint":      self._suggest_fix(e, code),
            }

    def _suggest_fix(self, error, code: str):
        s = str(error)
        if isinstance(error, KeyError):
            if "cluster_graphs" in code:
                return "cluster_graphs is populated after run_astar_pipeline is executed."
            if "kegg_pathways" in code:
                return "kegg_pathways is populated after get_cluster_kegg_edges is executed."
            return "Available keys: list(cluster_graphs.keys())"
        if isinstance(error, AttributeError):
            return "Use dir() or help() to inspect attributes"
        if "sparse" in s.lower():
            return "Check sp.issparse(X) then use .toarray()"
        if "not in index" in s:
            return "Check adata.obs_names or adata.var_names"
        return None


# ================================================================
# Tool 1: run_astar_pipeline
# ================================================================

@mcp.tool()
async def run_astar_pipeline(
    sampleid:        str        = Field(...,    description="Sample ID"),
    cluster_ids:     List[str]  = Field([],     description='List of cluster IDs to process (e.g. ["2"] or ["1","3"]). Empty list processes all clusters.'),
    organism:        str        = Field("auto", description="Species: 'human', 'mouse', or 'auto' (auto-detected from var_names pattern)"),
    leiden_key:      str        = Field("leiden", description="leiden obs column"),
    beta_threshold:  float      = Field(1.45,   description="conservative graph beta threshold"),
    threshold:       float      = Field(0.8,    description="edge frequency threshold"),
    force_recompute: bool       = Field(False,  description="ignore cache and recompute"),
) -> dict:
    """
    ⚠️ EXPENSIVE — A* pathfinding per cluster + conservative graph construction.

    **Runtime**: number of clusters × ~1–3 min

    **Organism auto-detection**:
    - 'auto' (default): infers human/mouse from adata.var_names pattern
    - Human: all-uppercase symbols (CD8A, GAPDH)
    - Mouse: Titlecase symbols (Cd8a, Gapdh)
    - Detection result is cached per sampleid and applied to subsequent tool calls

    **Results stored in**:
    - state["astar_results"][sampleid]
    - state["cluster_graphs"][sampleid]
    """
    _t0  = time.time()
    _inp = {"sampleid": sampleid, "cluster_ids": cluster_ids, "organism": organism,
            "leiden_key": leiden_key, "beta_threshold": beta_threshold,
            "threshold": threshold, "force_recompute": force_recompute}
    try:
        adata = _get_adata(sampleid)
    except ValueError as e:
        result = {"success": False, "error": str(e)}
        _log_call("run_astar_pipeline", _inp, result, int((time.time()-_t0)*1000))
        return result

    # Determine organism (detect → cache)
    org = _get_organism(sampleid, adata=adata, hint=organism)

    # Load DoRothEA (per-organism cache)
    try:
        dorothea_df = _get_dorothea(org)
    except Exception as e:
        return {"success": False, "error": f"DoRothEA load failed: {e}"}

    all_cids = list(adata.obs[leiden_key].astype(str).unique())
    target   = [str(c) for c in cluster_ids] if cluster_ids else all_cids

    sid = _resolve_sampleid(sampleid)

    if not force_recompute and sid in state["cluster_graphs"]:
        cached = list(state["cluster_graphs"][sid].keys())
        skip   = [c for c in target if c in cached]
        target = [c for c in target if c not in cached]
        if not target:
            return {
                "success":  True,
                "organism": org,
                "message":  "returned from cache (use force_recompute=True to recompute)",
                "clusters": cached,
            }
        print(f"[run_astar_pipeline] cache hit: {skip} / newly computing: {target}")

    astar_results = state["astar_results"].setdefault(sid, {})
    for cid in target:
        print(f"\n[A*] cluster {cid} ...")
        astar_results[cid] = run_astar_for_cluster(
            adata, cid, leiden_key=leiden_key
        )

    new_graphs = build_cluster_conservative_graphs(
        adata          = adata,
        dorothea_df    = dorothea_df,
        all_results    = {c: astar_results[c] for c in target},
        beta_threshold = beta_threshold,
        threshold      = threshold,
    )

    state["cluster_graphs"].setdefault(sid, {}).update(new_graphs)

    if sid in state["python_envs"]:
        state["python_envs"][sid].sync_state()

    summary = {
        cid: {
            "n_edges":      len(v["df"]),
            "n_components": len(v["components"]),
            "n_genes":      len(v["genes"]),
        }
        for cid, v in new_graphs.items()
    }

    result = {
        "success":           True,
        "organism":          org,
        "clusters_computed": list(new_graphs.keys()),
        "summary":           summary,
    }
    _log_call("run_astar_pipeline", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 2: get_astar_graph_summary
# ================================================================

@mcp.tool()
async def get_astar_graph_summary(
    sampleid:   str = Field(..., description="Sample ID"),
    cluster_id: str = Field(..., description="Cluster ID"),
    top_n:      int = Field(20, description="Number of top edges to return (sorted by score desc)"),
) -> dict:
    """
    Query conservative graph results at the cluster level.

    **Prerequisite**: must run run_astar_pipeline first.

    **Returns**:
    - top_edges    : top N edges by score (source, target, freq, mean_beta, score)
    - n_components : number of weakly connected components
    """
    _t0  = time.time()
    _inp = {"sampleid": sampleid, "cluster_id": cluster_id, "top_n": top_n}
    sid  = _resolve_sampleid(sampleid)
    cid  = str(cluster_id)

    graphs = state["cluster_graphs"].get(sid, {})
    if cid not in graphs:
        result = {
            "success": False,
            "error":   f"Cluster {cid} result not found. Run run_astar_pipeline first.",
            "available_clusters": list(graphs.keys()),
        }
        _log_call("get_astar_graph_summary", _inp, result, int((time.time()-_t0)*1000))
        return result

    v   = graphs[cid]
    df  = v["df"]

    top_edges_raw = df.head(top_n)[["source","target","freq","mean_beta","score"]].to_dict("records")
    top_edges = [
        {k: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
         for k, val in row.items()}
        for row in top_edges_raw
    ]
    attrs = {k: (float(v) if isinstance(v, np.floating) else
                 int(v)   if isinstance(v, np.integer)  else v)
             for k, v in df.attrs.items()}

    result = {
        "success":      True,
        "cluster_id":   cid,
        "organism":     state["organism_cache"].get(sid, "unknown"),
        "n_edges":      int(len(df)),
        "n_components": int(len(v["components"])),
        "top_edges":    top_edges,
        "attrs":        attrs,
    }
    _log_call("get_astar_graph_summary", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 3: get_astar_cellular_info
# ================================================================

@mcp.tool()
async def get_astar_cellular_info(
    sampleid:    str = Field(..., description="Sample ID"),
    cell_id:     str = Field(..., description="adata.obs.index value"),
    leiden_key:  str = Field("leiden"),
    top_n_edges: int = Field(20, description="Number of top edges to return (sorted by cellular_beta desc), max 100"),
) -> dict:
    """
    Recompute conservative graph edges for a specific cell using its own expression values.

    Takes all edges from the cluster-level A* conservative graph and recomputes beta
    using this cell's expression: cellular_beta = sqrt(|w + alpha_i_cell + alpha_j_cell|).
    Edges sorted by cellular_beta descending.

    Returns per-edge fields: source, target, freq, mean_beta (cluster ref), cellular_beta, alpha_i_cell, alpha_j_cell.

    Available after run_astar_pipeline has been executed.
    """
    _t0         = time.time()
    sampleid    = _resolve_sampleid(sampleid)
    top_n_edges = min(top_n_edges, 100)
    _inp        = {"sampleid": sampleid, "cell_id": cell_id, "leiden_key": leiden_key, "top_n_edges": top_n_edges}
    try:
        adata = _get_adata(sampleid)
    except ValueError as e:
        result = {"success": False, "error": str(e)}
        _log_call("get_astar_cellular_info", _inp, result, int((time.time()-_t0)*1000))
        return result

    if cell_id not in adata.obs.index:
        result = {"success": False, "error": f"cell_id '{cell_id}' not found"}
        _log_call("get_astar_cellular_info", _inp, result, int((time.time()-_t0)*1000))
        return result

    # Cell expression vector
    cell_loc = adata.obs.index.get_loc(cell_id)
    X_cell   = adata.X[cell_loc]
    if sp.issparse(X_cell):
        X_cell = X_cell.toarray().ravel()
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    # DoRothEA weight lookup: (source, target) -> w
    org         = state["organism_cache"].get(sampleid, "human")
    dorothea_df = _get_dorothea(org)
    w_lookup    = {(r.source, r.target): r.weight for r in dorothea_df.itertuples(index=False)}

    # ── Determine baseline edges (cluster astar if available, else full DoRothEA) ──
    cid     = None
    df      = None
    anon    = False
    if leiden_key in adata.obs.columns:
        try:
            cid    = str(adata.obs.loc[cell_id, leiden_key])
            graphs = state["cluster_graphs"].get(sampleid, {})
            if cid in graphs:
                df = graphs[cid]["df"]
        except Exception:
            cid = None
    if df is None:
        # anon / single-cell mode: cluster context unavailable → use full DoRothEA
        anon = True

    rows = []
    if df is not None:
        # Cluster baseline: keep freq/mean_beta from astar
        for _, edge in df.iterrows():
            src, tgt = edge["source"], edge["target"]
            w        = w_lookup.get((src, tgt), 0.0)
            alpha_i  = float(X_cell[gene_to_idx[src]]) if src in gene_to_idx else 0.0
            alpha_j  = float(X_cell[gene_to_idx[tgt]]) if tgt in gene_to_idx else 0.0
            cellular_beta = abs(w + alpha_i + alpha_j) ** 0.5
            rows.append({
                "source":        src,
                "target":        tgt,
                "freq":          round(float(edge["freq"]), 4),
                "mean_beta":     round(float(edge["mean_beta"]), 4),
                "cellular_beta": round(cellular_beta, 4),
            })
    else:
        # Anon: full DoRothEA edges, drop ones with no expression in this cell
        for r in dorothea_df.itertuples(index=False):
            src, tgt = r.source, r.target
            w        = float(r.weight)
            alpha_i  = float(X_cell[gene_to_idx[src]]) if src in gene_to_idx else 0.0
            alpha_j  = float(X_cell[gene_to_idx[tgt]]) if tgt in gene_to_idx else 0.0
            if alpha_i == 0.0 and alpha_j == 0.0:
                continue
            cellular_beta = abs(w + alpha_i + alpha_j) ** 0.5
            rows.append({
                "source":        src,
                "target":        tgt,
                "weight":        round(w, 4),
                "cellular_beta": round(cellular_beta, 4),
            })

    rows.sort(key=lambda x: x["cellular_beta"], reverse=True)

    result = {
        "success":   True,
        "cell_id":   cell_id,
        "organism":  org,
        "n_edges":   len(rows),
        "top_edges": rows[:top_n_edges],
        "mode":      "anon_dorothea" if anon else "cluster_astar",
    }
    if not anon and cid is not None:
        result["cluster_id"] = cid
    _log_call("get_astar_cellular_info", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 4: get_cluster_rl_map
# ================================================================

@mcp.tool()
async def get_cluster_rl_map(
    sampleid:         str   = Field(...,    description="Sample ID"),
    organism:         str   = Field("auto", description="Species: 'human', 'mouse', or 'auto'"),
    specificity_rank: float = Field(0.015),
    top_n:            int   = Field(20),
    top_n_edges:      int   = Field(5,     description="maximum number of edges to return per cluster"),
    expr_threshold:   float = Field(0.0),
    force_recompute:  bool  = Field(False),
) -> dict:
    """
    Query the LIANA R-L map at the cluster level.

    LIANA resource is auto-selected by organism:
      human → consensus (default)
      mouse → mouseconsensus (bundled, no extra download needed)

    Runs LIANA rank_aggregate automatically on first call.
    Results are cached in state and returned quickly on subsequent calls.

    Return structure:
      clusters[cid] = R-L interactions where cluster cid is the **sender**
        edges: [{"sender": cid, "ligand": ..., "receptor": ..., "receiver": ...}, ...]
          - sender  : cluster expressing the ligand (= cid itself)
          - receiver: cluster expressing the receptor (signal target)
          - signal flows ligand → receptor
    """
    _t0  = time.time()
    _inp = {"sampleid": sampleid, "organism": organism,
            "specificity_rank": specificity_rank, "top_n": top_n,
            "top_n_edges": top_n_edges, "expr_threshold": expr_threshold,
            "force_recompute": force_recompute}
    sid  = _resolve_sampleid(sampleid)

    if not force_recompute and sid in state["rl_maps"]:
        rl_map = state["rl_maps"][sid]
        result = {
            "success": True,
            "cached":  True,
            "organism": state["organism_cache"].get(sid, "unknown"),
            "clusters": {
                cid: {
                    "n_ligand":   len(v["ligand"]),
                    "n_receptor": len(v["receptor"]),
                    "n_edges":    len(v["edges"]),
                    "edges": [
                        {"sender": cid, "ligand": e[0], "receptor": e[1], "receiver": e[2]}
                        for e in v["edges"][:top_n_edges]
                    ],
                }
                for cid, v in rl_map["cluster_edge"].items()
            },
        }
        _log_call("get_cluster_rl_map", _inp, result, int((time.time()-_t0)*1000))
        return result

    try:
        adata = _get_adata(sid)
    except ValueError as e:
        result = {"success": False, "error": str(e)}
        _log_call("get_cluster_rl_map", _inp, result, int((time.time()-_t0)*1000))
        return result

    org = _get_organism(sampleid, adata=adata, hint=organism)

    rl_map = build_rl_gene_map(
        adata,
        specificity_rank = specificity_rank,
        top_n            = top_n,
        expr_threshold   = expr_threshold,
        organism         = org,
    )
    state["rl_maps"][sid] = rl_map

    if sid in state["python_envs"]:
        state["python_envs"][sid].sync_state()

    result = {
        "success":  True,
        "cached":   False,
        "organism": org,
        "clusters": {
            cid: {
                "n_ligand":   len(v["ligand"]),
                "n_receptor": len(v["receptor"]),
                "n_edges":    len(v["edges"]),
                "edges": [
                    {"sender": cid, "ligand": e[0], "receptor": e[1], "receiver": e[2]}
                    for e in v["edges"][:top_n_edges]
                ],
            }
            for cid, v in rl_map["cluster_edge"].items()
        },
    }
    _log_call("get_cluster_rl_map", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 5: get_cluster_kegg_edges
# ================================================================

@mcp.tool()
async def get_cluster_kegg_edges(
    sampleid:       str   = Field(...,       description="Sample ID"),
    cluster_id:     str   = Field(...,       description="Value in the obs column specified by cluster_key (e.g. '0', 'O', 'X')"),
    cluster_key:    str   = Field("leiden",  description="obs column to group cells by (e.g. 'leiden', 'OX', 'cell_type'). Use 'OX' to analyse O vs X labels directly."),
    organism:       str   = Field("auto",    description="Species: 'human', 'mouse', or 'auto'"),
    kegg_base_dir:  str   = Field(KEGG_DIR,  description="KGML base directory (sub-path determined automatically by organism)"),
    top_n_pathways: int   = Field(5,         description="number of top pathways to return"),
    top_n_edges:    int   = Field(6,         description="number of top edges per pathway"),
    force_reparse:  bool  = Field(False,     description="force re-parsing of KEGG XMLs"),
) -> dict:
    """
    Query KEGG top pathways + top edges for a group of cells.

    By default groups by leiden cluster (cluster_key='leiden').
    To analyse by OX label: cluster_key='OX', cluster_id='O' or 'X'.

    KEGG path is auto-selected by organism:
      human → kegg_base_dir/       (hsa .kgml)
      mouse → kegg_base_dir/mmu/   (mmu .kgml, 365 files locally cached)
    """
    _t0  = time.time()
    _inp = {"sampleid": sampleid, "cluster_id": cluster_id, "cluster_key": cluster_key,
            "organism": organism, "kegg_base_dir": kegg_base_dir,
            "top_n_pathways": top_n_pathways, "top_n_edges": top_n_edges,
            "force_reparse": force_reparse}
    sid  = _resolve_sampleid(sampleid)
    cid  = str(cluster_id)

    try:
        adata = _get_adata(sid)
    except ValueError as e:
        result = {"success": False, "error": str(e)}
        _log_call("get_cluster_kegg_edges", _inp, result, int((time.time()-_t0)*1000))
        return result

    org      = _get_organism(sampleid, adata=adata, hint=organism)
    kegg_dir = _kegg_dir(org, kegg_base_dir)

    _cache_key = (kegg_dir, org)
    if force_reparse or _cache_key not in state["kegg_parsed"]:
        print(f"[KEGG] parsing XML ({kegg_dir}) ...")
        state["kegg_parsed"][_cache_key] = parse_all_kegg_xmls_raw(kegg_dir)
    _pathway_key = (sid, kegg_dir, org)
    if force_reparse or _pathway_key not in state["kegg_pathways"]:
        state["kegg_pathways"][_pathway_key] = filter_kegg_pathways(
            state["kegg_parsed"][_cache_key], adata
        )
        state["kegg_pathways"][sid] = state["kegg_pathways"][_pathway_key]
        if sid in state["python_envs"]:
            state["python_envs"][sid].sync_state()
    pathways = state["kegg_pathways"][_pathway_key]

    if not pathways:
        result = {"success": False, "error": "No valid KEGG pathways found"}
        _log_call("get_cluster_kegg_edges", _inp, result, int((time.time()-_t0)*1000))
        return result

    top_pathways = compute_and_select_top_kegg(
        adata, pathways, cid, top_n=top_n_pathways, cluster_key=cluster_key,
    )
    edge_results = get_top_edges_per_pathway(
        adata, top_pathways, cid, top_n_edges=top_n_edges, cluster_key=cluster_key,
    )

    result = {
        "success":    True,
        "cluster_id": cid,
        "cluster_key": cluster_key,
        "organism":   org,
        "kegg_dir":   kegg_dir,
        "n_pathways": len(top_pathways),
        "pathways": [
            {"name": pw_name, "edges": df.to_dict("records")}
            for pw_name, df in edge_results.items()
        ],
    }
    _log_call("get_cluster_kegg_edges", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 6: get_cell_kegg_edges
# ================================================================

@mcp.tool()
async def get_cell_kegg_edges(
    sampleid:       str = Field(...,      description="Sample ID"),
    cell_id:        str = Field(...,      description="adata.obs.index value"),
    organism:       str = Field("auto",   description="Species: 'human', 'mouse', or 'auto'"),
    leiden_key:     str = Field("leiden"),
    kegg_base_dir:  str = Field(KEGG_DIR),
    top_n_pathways: int = Field(5),
    top_n_edges:    int = Field(7),
) -> dict:
    """
    Query KEGG top edges at the single-cell level (cell level).

    Computes beta using the cell's own expression values → returns personalized pathway-edge activity.
    KEGG path is auto-selected by organism (human/mouse).
    """
    _t0  = time.time()
    _inp = {"sampleid": sampleid, "cell_id": cell_id, "organism": organism,
            "leiden_key": leiden_key, "kegg_base_dir": kegg_base_dir,
            "top_n_pathways": top_n_pathways, "top_n_edges": top_n_edges}
    sid  = _resolve_sampleid(sampleid)

    try:
        adata = _get_adata(sid)
    except ValueError as e:
        result = {"success": False, "error": str(e)}
        _log_call("get_cell_kegg_edges", _inp, result, int((time.time()-_t0)*1000))
        return result

    if cell_id not in adata.obs.index:
        result = {"success": False, "error": f"cell_id '{cell_id}' not found"}
        _log_call("get_cell_kegg_edges", _inp, result, int((time.time()-_t0)*1000))
        return result

    org      = _get_organism(sampleid, adata=adata, hint=organism)
    kegg_dir = _kegg_dir(org, kegg_base_dir)

    # ── Determine cluster context (or single-cell anon fallback) ──
    cid  = None
    anon = False
    if leiden_key in adata.obs.columns:
        try:
            cid = str(adata.obs.loc[cell_id, leiden_key])
        except Exception:
            cid = None
    if cid is None:
        anon = True
        # 단일 cell 만으로 pathway score 계산 — leiden을 일시 컬럼으로 부착
        adata.obs["__anon_single_cell__"] = pd.Series(
            ["X" if bc == cell_id else "Y" for bc in adata.obs.index],
            index=adata.obs.index,
            dtype="category",
        )
        cluster_key_for_kegg = "__anon_single_cell__"
        cid_for_kegg         = "X"
    else:
        cluster_key_for_kegg = leiden_key
        cid_for_kegg         = cid

    _cache_key = (kegg_dir, org)
    if _cache_key not in state["kegg_parsed"]:
        print(f"[KEGG] parsing XML ({kegg_dir}) ...")
        state["kegg_parsed"][_cache_key] = parse_all_kegg_xmls_raw(kegg_dir)
    _pathway_key = (sid, kegg_dir, org)
    if _pathway_key not in state["kegg_pathways"]:
        state["kegg_pathways"][_pathway_key] = filter_kegg_pathways(
            state["kegg_parsed"][_cache_key], adata
        )
        state["kegg_pathways"][sid] = state["kegg_pathways"][_pathway_key]
    pathways = state["kegg_pathways"][_pathway_key]

    top_pathways = compute_and_select_top_kegg(
        adata, pathways, cid_for_kegg,
        cluster_key=cluster_key_for_kegg,
        top_n=top_n_pathways,
        mode="per_cell" if anon else "cluster_mean",
    )

    # Temporary Column Removal
    if anon and "__anon_single_cell__" in adata.obs.columns:
        del adata.obs["__anon_single_cell__"]

    cell_loc = adata.obs.index.get_loc(cell_id)
    X_cell   = adata.X[cell_loc]
    if sp.issparse(X_cell):
        X_cell = X_cell.toarray().ravel()
    gene_to_idx = {g.upper(): i for i, g in enumerate(adata.var_names)}

    alpha_G    = float(X_cell.sum())
    alpha_g_sq = alpha_G ** 2

    result_edges = []
    for pathway in top_pathways:
        edges_data = pathway.get_edges_data()
        rows = []
        for src, tgt, w in zip(edges_data["source"], edges_data["target"], edges_data["weight"]):
            alpha_i = float(X_cell[gene_to_idx[src.upper()]]) if src.upper() in gene_to_idx else 0.0
            alpha_j = float(X_cell[gene_to_idx[tgt.upper()]]) if tgt.upper() in gene_to_idx else 0.0
            beta    = abs(w + alpha_i + alpha_j) ** 0.5
            contribution = (alpha_i * alpha_j / alpha_g_sq) * (beta ** 2) if alpha_G > 1e-10 else 0.0
            rows.append({"source": src, "target": tgt,
                         "weight": round(w, 4),
                         "beta": round(beta, 4),
                         "contribution": round(contribution, 6)})
        rows.sort(key=lambda x: x["contribution"], reverse=True)
        result_edges.append({"name": pathway.name, "edges": rows[:top_n_edges]})

    result = {
        "success":    True,
        "cell_id":    cell_id,
        "organism":   org,
        "n_pathways": len(result_edges),
        "pathways":   result_edges,
        "mode":       "anon_single_cell" if anon else "cluster_mean",
    }
    if not anon and cid is not None:
        result["cluster_id"] = cid
    _log_call("get_cell_kegg_edges", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 7: get_kg_context
# ================================================================

KG_DEFAULT_PATH = str(Path(__file__).parent / "KG_v2" / "graph_chunk_entity_relation.trimmed.graphml")


def _load_kg(kg_path: str):
    """Load and cache the LightRAG KG graphml."""
    import networkx as nx
    if state["kg_graph"] is None or state["kg_path"] != kg_path:
        print(f"[KG] loading graphml: {kg_path}")
        state["kg_graph"] = nx.read_graphml(kg_path)
        state["kg_path"]  = kg_path
    return state["kg_graph"]


def _kg_query(G, genes: list[str], keywords: str, top_n: int) -> dict:
    """
    Query KG for a set of gene names + optional free-text keywords.

    Strategy:
    1. Exact match gene names against node entity_id (case-insensitive).
    2. Keyword substring search in entity_id and description for non-gene nodes.
    3. For each matched node, collect 1-hop edges and neighbour descriptions.
    4. Return de-duplicated, top_n-limited context.
    """
    genes_upper = {g.upper() for g in genes if g}
    kw_lower    = [k.strip().lower() for k in keywords.split(",") if k.strip()] if keywords else []

    matched_nodes = set()

    for node, data in G.nodes(data=True):
        etype = data.get("entity_type", "")
        eid   = (data.get("entity_id") or node or "").strip()
        desc  = (data.get("description") or "").lower()
        label = eid.upper()

        if etype == "gene" and label in genes_upper:
            matched_nodes.add(node)
        elif kw_lower and any(kw in eid.lower() or kw in desc for kw in kw_lower):
            matched_nodes.add(node)

    node_ctx  = []
    edge_ctx  = []
    seen_edges = set()

    for node in matched_nodes:
        data = G.nodes[node]
        eid  = data.get("entity_id") or node
        desc = (data.get("description") or "").split("<SEP>")[0].strip()[:300]
        node_ctx.append({
            "entity":      eid,
            "type":        data.get("entity_type", "?"),
            "description": desc,
        })

        for nbr in (list(G.neighbors(node)) if not G.is_directed() else list(G.predecessors(node)) + list(G.successors(node))):
            edge_key = tuple(sorted([node, nbr]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edata = G.get_edge_data(node, nbr) or G.get_edge_data(nbr, node) or {}
            nbr_data = G.nodes[nbr]
            nbr_eid  = nbr_data.get("entity_id") or nbr
            nbr_desc  = (nbr_data.get("description") or "").split("<SEP>")[0].strip()[:300]
            edge_desc = (edata.get("description") or "").strip()[:200]

            src_eid = (G.nodes[node].get("entity_id") or node)
            entry = {
                "from":      src_eid,
                "to":        nbr_eid,
                "edge_desc": edge_desc,
            }
            if nbr_desc:
                entry["to_desc"] = nbr_desc
            edge_ctx.append(entry)

    edge_ctx = edge_ctx[:top_n]

    return {
        "matched_nodes": len(matched_nodes),
        "nodes":         node_ctx,
        "edges":         edge_ctx,
    }


@mcp.tool()
async def get_kg_context(
    genes:    list[str] = Field(default_factory=list, description="Gene names to look up in the KG (e.g. ['APOE','PTEN'])"),
    keywords: str       = Field("",                   description="Comma-separated keywords to search in mechanism/celltype/tissuestate nodes (e.g. 'axon regeneration,corticospinal')"),
    top_n:    int       = Field(15,                   description="Max number of edge contexts to return (max 30)"),
    kg_path:  str       = Field(KG_DEFAULT_PATH,      description="Path to graph_chunk_entity_relation.graphml"),
) -> dict:
    """
    Retrieve biological context from the LightRAG Knowledge Graph.

    Matches input genes against gene nodes and keywords against all node descriptions.
    For each matched node returns:
    - node description (entity type, biological role)
    - 1-hop neighbours with edge descriptions (mechanism, celltype, tissuestate links)

    Use this after identifying DE genes or KEGG pathway hits to ground interpretation
    in curated biological knowledge beyond gene-level statistics.

    Example usage:
    - After DGE: get_kg_context(genes=['Apoe','Gap43','Pten'], keywords='axon regeneration')
    - After KEGG: get_kg_context(genes=['Calm1','Gsk3b'], keywords='neurodegeneration')
    """
    _t0    = time.time()
    top_n  = min(top_n, 30)
    _inp   = {"genes": genes, "keywords": keywords, "top_n": top_n, "kg_path": kg_path}
    try:
        G      = _load_kg(kg_path)
        result = _kg_query(G, genes, keywords, top_n)
        result["success"] = True
    except Exception as e:
        result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    _log_call("get_kg_context", _inp, result, int((time.time() - _t0) * 1000))
    return result


# ================================================================
# Tool 8: resolve_query_to_context_set  (Tool A — anchor expansion)
# ================================================================

def _resolve_anchors(G, seed_genes: list, keywords: str, entity_types: list,
                     min_keyword_hits: int = 1, max_anchors: int = 0) -> dict:
    """Match anchor nodes by exact gene id and/or keyword substring.

    Guards against anchor explosion when keywords are broad:
      - min_keyword_hits: a non-gene node must match at least this many
        keyword tokens (raise from 1 to 2+ for AND-like behavior).
      - max_anchors: if non-zero, cap the result to the top-N by score.
        seed_gene anchors carry score=2.0 so they survive small caps naturally.

    Returns {node_id: {"score","reason","type"}}.
    """
    genes_upper = {g.strip().upper() for g in seed_genes if g and g.strip()}
    kw_lower    = [k.strip().lower() for k in keywords.split(",") if k.strip()] if keywords else []
    types_set   = set(entity_types) if entity_types else None
    min_hits    = max(1, int(min_keyword_hits))

    anchors = {}
    for node, data in G.nodes(data=True):
        etype = data.get("entity_type", "")
        eid   = (data.get("entity_id") or node or "").strip()
        desc  = (data.get("description") or "").lower()

        if etype == "gene" and eid.upper() in genes_upper:
            anchors[node] = {"score": 2.0, "reason": "seed_gene", "type": etype}
            continue
        if kw_lower and (types_set is None or etype in types_set):
            el = eid.lower()
            matched_kws = [kw for kw in kw_lower if kw in el or kw in desc]
            n_hits = len(matched_kws)
            if n_hits >= min_hits:
                anchors[node] = {
                    "score":          1.0 * n_hits,
                    "reason":         f"keyword_hits={n_hits}: {matched_kws}",
                    "matched_kws":    matched_kws,
                    "type":           etype,
                }

    if max_anchors and len(anchors) > max_anchors:
        kept = sorted(anchors.items(), key=lambda kv: -kv[1]["score"])[:max_anchors]
        anchors = dict(kept)

    return anchors


def _expand_with_weights(G, anchors: dict, n_hop: int, min_edge_weight: float, decay: float):
    """BFS up to n_hop with weighted score propagation.

    contrib(nbr) += score(node) * edge_weight * decay**hop
    """
    scores   = {n: a["score"] for n, a in anchors.items()}
    distance = {n: 0 for n in anchors}
    sources  = {n: n for n in anchors}

    frontier = list(anchors.keys())
    for hop in range(1, n_hop + 1):
        nxt = set()
        for node in frontier:
            base = scores[node]
            if G.is_directed():
                neighbors = set(G.predecessors(node)) | set(G.successors(node))
            else:
                neighbors = set(G.neighbors(node))
            for nbr in neighbors:
                edata = G.get_edge_data(node, nbr) or G.get_edge_data(nbr, node) or {}
                w = float(edata.get("weight", 1.0) or 1.0)
                if w < min_edge_weight:
                    continue
                contrib = base * w * (decay ** hop)
                if nbr not in distance:
                    distance[nbr] = hop
                    sources[nbr]  = sources.get(node, node)
                    nxt.add(nbr)
                scores[nbr] = scores.get(nbr, 0.0) + contrib
        frontier = list(nxt)
        if not frontier:
            break
    return scores, distance, sources


def _extract_context_genes(G, scores, distance, sources, top_n: int) -> list:
    rows = []
    for node, s in scores.items():
        data = G.nodes[node]
        if data.get("entity_type") != "gene":
            continue
        eid  = data.get("entity_id") or node
        src  = sources.get(node, node)
        desc = (data.get("description") or "").strip()
        rows.append({
            "gene":          eid,
            "score":         round(float(s), 4),
            "hop":           distance.get(node, 0),
            "anchor_source": G.nodes[src].get("entity_id") or src,
            "desc":          desc[:200] if desc else "",
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_n]


def _pathway_gene_set(pw) -> set:
    """Best-effort gene-set extraction across KEGGPathway implementations."""
    if hasattr(pw, "gene_set"):
        try:
            return {str(g).upper() for g in pw.gene_set if g}
        except Exception:
            pass
    if hasattr(pw, "get_edges_data"):
        try:
            ed = pw.get_edges_data()
            return {str(g).upper()
                    for g in (list(ed.get("source", [])) + list(ed.get("target", [])))
                    if g}
        except Exception:
            pass
    return set()



def _gen_context_id(seed_genes, keywords, kg_path) -> str:
    payload = json.dumps({
        "g": sorted([s.strip() for s in seed_genes if s and s.strip()]),
        "k": sorted([k.strip() for k in (keywords or "").split(",") if k.strip()]),
        "p": kg_path,
    }, sort_keys=True)
    return "ctx_" + hashlib.md5(payload.encode()).hexdigest()[:12]


@mcp.tool()
async def resolve_query_to_context_set(
    seed_genes:        List[str] = Field(default_factory=list,
        description="Genes lifted directly from the question (e.g. ['APOE','TREM2','GFAP'])."),
    keywords:          str       = Field("",
        description="Comma-separated phrases to match against mechanism/celltype/tissuestate descriptions (e.g. 'microglia activation,amyloid plaque,neuroinflammation')."),
    entity_types:      List[str] = Field(
        default_factory=lambda: ["mechanism", "celltype", "tissuestate", "other"],
        description="Node types eligible for keyword matching. Gene anchors via seed_genes are always honored."),
    n_hop:             int       = Field(1,
        description="BFS hop count from anchors. 1 is safe; 2 risks noise on small KGs."),
    min_edge_weight:   float     = Field(0.0,
        description="Drop edges below this weight during expansion."),
    decay:             float     = Field(0.5,
        description="Score decay per hop: contrib = anchor_score * edge_weight * decay**hop."),
    top_n_genes:       int       = Field(50),
    max_anchors:       int       = Field(50,
        description="Cap anchors by score after matching. 0 disables. Guards against keyword overmatch (set to 30–50)."),
    min_keyword_hits:  int       = Field(1,
        description="Non-gene anchors must match at least this many keyword tokens. Raise to 2+ for AND-like behavior on broad keyword lists."),
    kg_path:           str       = Field(KG_DEFAULT_PATH,
        description="Path to the LightRAG graphml. Single graph; expand by appending nodes/edges to the same file."),
    context_id:        str       = Field("",
        description="Deterministic id; auto-generated from inputs if empty. Reuse to retrieve cached context_set."),
) -> dict:
    """
    Anchor seed_genes (gene nodes) + keywords (mechanism/celltype/tissuestate nodes)
    on the LightRAG KG, BFS-expand n_hop from anchors, and return a context_set:
    ranked context_genes (each with a short 'desc' from the KG node).

    context_id is deterministic: same (seed_genes, keywords, kg_path) → same id.
    """
    _t0  = time.time()
    if not isinstance(max_anchors,      int): max_anchors      = 50
    if not isinstance(min_keyword_hits, int): min_keyword_hits = 1
    _inp = {"seed_genes": seed_genes, "keywords": keywords,
            "entity_types": entity_types, "n_hop": n_hop,
            "min_edge_weight": min_edge_weight, "decay": decay,
            "top_n_genes": top_n_genes,
            "max_anchors": max_anchors,
            "min_keyword_hits": min_keyword_hits,
            "kg_path": kg_path,
            "context_id": context_id}
    try:
        G = _load_kg(kg_path)

        anchors = _resolve_anchors(
            G, seed_genes, keywords, entity_types,
            min_keyword_hits=min_keyword_hits,
            max_anchors=max_anchors,
        )
        if not anchors:
            result = {
                "success":     False,
                "error":       "no anchors matched (check seed_genes / keywords / entity_types)",
                "n_anchors":   0,
                "kg_path":     kg_path,
                "kg_n_nodes":  G.number_of_nodes(),
                "kg_n_edges":  G.number_of_edges(),
            }
            _log_call("resolve_query_to_context_set", _inp, result, int((time.time()-_t0)*1000))
            return result

        scores, distance, sources = _expand_with_weights(
            G, anchors,
            n_hop           = n_hop,
            min_edge_weight = min_edge_weight,
            decay           = decay,
        )
        context_genes = _extract_context_genes(G, scores, distance, sources, top_n_genes)

        anchor_list = sorted(
            [
                {
                    "entity": G.nodes[n].get("entity_id") or n,
                    "type":   a["type"],
                    "score":  round(float(a["score"]), 4),
                    "reason": a["reason"],
                }
                for n, a in anchors.items()
            ],
            key=lambda x: -x["score"],
        )

        cid = context_id or _gen_context_id(seed_genes, keywords, kg_path)

        result = {
            "success":    True,
            "context_id": cid,
            "kg_path":    kg_path,
            "kg_n_nodes": G.number_of_nodes(),
            "kg_n_edges": G.number_of_edges(),
            "n_anchors":  len(anchors),
            "anchors":    anchor_list,
            "n_genes":    len(context_genes),
            "genes":      context_genes,
        }

        from collections import Counter as _Counter
        _anchor_types = _Counter(a["type"] for a in anchor_list)
        response = {k: v for k, v in result.items() if k != "anchors"}
        response["anchor_type_counts"] = dict(_anchor_types)
        _log_call("resolve_query_to_context_set", _inp, response, int((time.time()-_t0)*1000))
        return response

    except Exception as e:
        result = {
            "success":   False,
            "error":     str(e),
            "traceback": traceback.format_exc(),
        }

    _log_call("resolve_query_to_context_set", _inp, result, int((time.time()-_t0)*1000))
    return result




@mcp.tool()
async def get_expressed_dorothea_edges(
    sampleid:    str       = Field(...,   description="Sample ID"),
    genes:       List[str] = Field(...,   description="Agent-curated gene list. At least one endpoint of each returned edge will be in this list."),
    top_n:       int       = Field(30,    description="Max edges to return, sorted by normalized activity desc."),
    cluster_id:  Optional[str]       = Field(None,
        description="Cluster id to compute mean expression. Null/empty + cell_ids null/empty → whole-sample mean. Takes priority over cell_ids."),
    cell_ids:    Optional[List[str]] = Field(None,
        description="Cell barcodes for per-cell expression context. Null/empty AND cluster_id null/empty → whole-sample mean."),
    cluster_key: str       = Field("leiden"),
    min_alpha:   float     = Field(0.0,   description="Min expression required for BOTH source and target gene to be included."),
    organism:    str       = Field("auto"),
) -> dict:
    """
    Given an agent-curated gene list, return expressed DoRothEA TF→Target edges
    where at least one endpoint is in the gene list and both are expressed above
    min_alpha. Intended as a preparation step before custom_pathway_calc.

    Workflow:
      1. Curate a gene list (e.g. from resolve_query_to_context_set or marker knowledge)
      2. Call this tool with the gene list + sampleid + expression context
      3. Review edge_details, remove off-target TFs
      4. Pass edges + vertices to custom_pathway_calc

    Returns:
      - edges        : [[src, tgt, weight], ...] — ready for custom_pathway_calc
      - vertices     : unique genes in returned edges
      - edge_details : [{source, target, weight, alpha_src, alpha_tgt, beta, activity_norm}]
      - n_edges / n_vertices: summary counts

    Aim for ≥4 edges / ≥5 vertices before calling custom_pathway_calc.
    If too few edges returned, lower min_alpha or expand gene list.
    """
    _t0 = time.time()
    # Normalize None to defaults
    if cluster_id is None:             cluster_id = ""
    if cell_ids   is None:             cell_ids   = []
    if not isinstance(genes,    list): genes      = []
    if not isinstance(cell_ids, list): cell_ids   = []

    _inp = {"sampleid": sampleid, "n_genes_in": len(genes),
            "top_n": top_n, "cluster_id": cluster_id,
            "cell_ids_count": len(cell_ids), "min_alpha": min_alpha}

    if not genes:
        result = {"success": False, "error": "genes must be a non-empty list"}
        _log_call("get_expressed_dorothea_edges", _inp, result, int((time.time()-_t0)*1000))
        return result

    try:
        sid   = _resolve_sampleid(sampleid)
        adata = _get_adata(sid)
        org   = _get_organism(sid, adata=adata, hint=organism)
        df    = _get_dorothea(org)
    except ValueError as e:
        result = {"success": False, "error": str(e)}
        _log_call("get_expressed_dorothea_edges", _inp, result, int((time.time()-_t0)*1000))
        return result

    var_to_idx  = {str(g).upper(): i for i, g in enumerate(adata.var_names)}
    genes_upper = {str(g).upper() for g in genes if g}

    if cluster_id:
        if cluster_key not in adata.obs.columns:
            result = {"success": False, "error": f"cluster_key '{cluster_key}' not in adata.obs"}
            _log_call("get_expressed_dorothea_edges", _inp, result, int((time.time()-_t0)*1000))
            return result
        mask = adata.obs[cluster_key].astype(str) == str(cluster_id)
        if int(mask.sum()) == 0:
            valid_ids = sorted(adata.obs[cluster_key].astype(str).unique().tolist()[:20])
            result = {"success": False, "error": f"cluster_id '{cluster_id}' has 0 cells. Valid ids: {valid_ids}. Use a specific cluster id or pass cell_ids instead."}
            _log_call("get_expressed_dorothea_edges", _inp, result, int((time.time()-_t0)*1000))
            return result
        X = adata[mask].X
        alpha = (np.asarray(X.mean(axis=0)).ravel() if sp.issparse(X) else np.asarray(X).mean(axis=0))
        expr_ctx = f"cluster_{cluster_id}"
    elif cell_ids:
        valid = [c for c in cell_ids if c in adata.obs.index]
        if not valid:
            result = {"success": False, "error": "none of the cell_ids found in adata.obs.index"}
            _log_call("get_expressed_dorothea_edges", _inp, result, int((time.time()-_t0)*1000))
            return result
        loc   = adata.obs.index.get_loc(valid[0])
        X     = adata.X[loc]
        alpha = (X.toarray().ravel() if sp.issparse(X) else np.asarray(X).ravel())
        expr_ctx = f"cell_{valid[0][:20]}"
    else:
        alpha = (np.asarray(adata.X.mean(axis=0)).ravel()
                 if sp.issparse(adata.X) else np.asarray(adata.X).mean(axis=0))
        expr_ctx = "sample_mean"

    alpha_arr = np.asarray(alpha).ravel()
    alpha_G = float(alpha_arr.sum())
    alpha_g_sq = alpha_G * alpha_G if alpha_G > 1e-10 else 1.0

    rows = []
    for r in df.itertuples(index=False):
        su, tu = str(r.source).upper(), str(r.target).upper()
        if su not in genes_upper and tu not in genes_upper:
            continue
        if su not in var_to_idx or tu not in var_to_idx:
            continue
        ai = float(alpha[var_to_idx[su]])
        aj = float(alpha[var_to_idx[tu]])
        if ai <= min_alpha or aj <= min_alpha:
            continue
        w        = float(r.weight)
        beta     = abs(w + ai + aj) ** 0.5
        activity_norm = (ai * aj / alpha_g_sq) * (beta * beta) if alpha_G > 1e-10 else 0.0
        rows.append({
            "source":    r.source,
            "target":    r.target,
            "weight":    round(w, 4),
            "alpha_src": round(ai, 4),
            "alpha_tgt": round(aj, 4),
            "beta":      round(beta, 4),
            "activity_norm": round(activity_norm, 10),
        })

    rows.sort(key=lambda x: x["activity_norm"], reverse=True)
    rows = rows[:top_n]

    edges = [[e["source"], e["target"], e["weight"]] for e in rows]
    vertices = sorted({e["source"] for e in rows} | {e["target"] for e in rows})

    result = {
        "success":      True,
        "expr_context": expr_ctx,
        "n_edges":      len(rows),
        "n_vertices":   len(vertices),
        "edges":        edges,
        "edge_details": rows,
        "hint": (
            "Review edge_details: remove off-target TFs. "
            f"Currently: {len(rows)} edges / {len(vertices)} vertices. "
            "Aim for ≥4 edges / ≥5 vertices before custom_pathway_calc."
        ),
    }
    _log_call("get_expressed_dorothea_edges", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 10: custom_pathway_calc  (KG-free, custom pathway calculation)
# ================================================================

def _decompose_pathway_for_cell(pathway, alpha_vec, var_to_idx: dict, top_n: int = 10) -> list:
    """Per-edge contribution breakdown for one cell.

    Same formula as graph_utils.get_top_edges_per_pathway (cluster version):
        beta         = sqrt(|w + alpha_i + alpha_j|)
        contribution = (alpha_i * alpha_j / alpha_G^2) * beta^2
    alpha_vec is this cell's expression vector; alpha_G is its sum.
    Returns rows sorted by contribution desc, capped at top_n.
    """
    edges_data = pathway.get_edges_data()
    alpha_arr  = np.asarray(alpha_vec).ravel()
    alpha_G    = float(alpha_arr.sum())
    alpha_g_sq = alpha_G * alpha_G if alpha_G > 1e-10 else 1.0
    rows = []
    for src, tgt, w in zip(edges_data["source"], edges_data["target"], edges_data["weight"]):
        ai = float(alpha_arr[var_to_idx[src]]) if src in var_to_idx else 0.0
        aj = float(alpha_arr[var_to_idx[tgt]]) if tgt in var_to_idx else 0.0
        beta         = abs(float(w) + ai + aj) ** 0.5
        contribution = (ai * aj / alpha_g_sq) * (beta * beta)
        rows.append({
            "source":       src,
            "target":       tgt,
            "weight":       round(float(w), 4),
            "beta":         round(beta, 4),
            "contribution": round(contribution, 6),
        })
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return rows[:top_n]


def _build_custom_pathway(edges: list, name: str = "custom_pathway") -> "KEGGPathway":
    """User edge list -> KEGGPathway container for Rust kernels.

    Each edge: [src_gene, tgt_gene, weight (optional, default 1.0)].
    KEGGPathway carries (src, tgt, weight) plus metadata fields filled with dummies.
    Note: KEGG's bidirectional expansion only applies to types[i]=="PComplex",
    so for user-supplied edges with empty type we skip that call entirely.
    """
    srcs    = [str(e[0]) for e in edges]
    tgts    = [str(e[1]) for e in edges]
    weights = [float(e[2]) if len(e) >= 3 else 1.0 for e in edges]
    n       = len(srcs)
    return KEGGPathway(
        name=name, sources=srcs, targets=tgts, weights=weights,
        modifications=[""]*n, effects=[0]*n, types=[""]*n, indirects=[False]*n,
    )


@mcp.tool()
async def custom_pathway_calc(
    sampleid:    str        = Field(..., description="Sample ID"),
    edges:       List[List] = Field(...,
        description="User edges as list of [src, tgt, weight]. weight optional (default 1.0)."),
    vertices:    List[str]  = Field(default_factory=list,
        description="Optional gene list (informational; edges already define endpoints)."),
    scale:       str        = Field("cluster",
        description="'cluster' (cluster_mean) or 'cell' (per-cell norm)."),
    cluster_id:  Optional[str]       = Field(None,
        description="Single cluster id for scale='cluster'. Use null/omit for scale='cell'."),
    cluster_ids: Optional[List[str]] = Field(None,
        description="Batch cluster ids for scale='cluster'. Use ['all'] for all clusters. Use null/omit for scale='cell'."),
    cell_ids:    Optional[List[str]] = Field(None,
        description="Specific cells for scale='cell'. Null/empty -> return top-K ranking over all cells or over cluster_id/cluster_ids when provided."),
    cluster_key: str        = Field("leiden"),
    top_k:       int        = Field(10,
        description="When scale='cell' and cell_ids empty, return top_k cells by score."),
    top:         Optional[int] = Field(None,
        description="Deprecated alias for top_k."),
    top_n:       Optional[int] = Field(None,
        description="Deprecated alias for top_k."),
    name:        str        = Field("custom_pathway", description="Pathway name (informational)."),
    verbose:     bool       = Field(False,
        description="If True, include per-edge contribution breakdown (sorted desc) for each target cluster/cell."),
    verbose_top_n: int      = Field(10,
        description="When verbose=True, max edges per target to include in edge_contributions."),
) -> dict:
    """
    KG-free custom edge-L2 scoring via Rust kernels.

    Formula (mathematically equivalent to Python edge_L2;
            verified by verify_rust_custom_pathway.py, Spearman rho = 0.999999):
        beta          = sqrt(|w + alpha_i + alpha_j|)
        contribution  = (alpha_i * alpha_j / alpha_G^2) * beta^2
        edge_L2       = sqrt(sum(contribution))

    Scale:
      - 'cluster' : compute_all_kegg_norms_cluster_mean  (cluster mean alpha)
      - 'cell'    : compute_all_kegg_norms_sparse        (per-cell alpha)

    Returns differ by scale:
      cluster + cluster_id        -> single score
      cluster + cluster_ids       -> {cid: score, ...} + ranked list
      cell    + cell_ids          -> {cell_id: score, ...}
      cell    + cluster_id(s)     -> top_k cells within the requested cluster scope
      cell    + (no filters)      -> top_k cells by score over all cells

    verbose:
      When True, each target entry (cluster or cell) gets `edge_contributions`:
      per-edge [source, target, weight, beta, contribution] sorted by contribution
      desc (top `verbose_top_n` edges). Useful to see which edges drive the score.
      - cluster: alpha = cluster mean expression
      - cell:    alpha = that single cell's expression vector
    """
    _t0  = time.time()
    if not isinstance(edges, list) or len(edges) == 0:
        return {"success": False, "error": "edges must be a non-empty list"}
    # Normalize None to defaults
    if cluster_id  is None:               cluster_id  = ""
    if cluster_ids is None:               cluster_ids = []
    if cell_ids    is None:               cell_ids    = []
    if not isinstance(cluster_ids, list): cluster_ids = []
    if not isinstance(cell_ids,    list): cell_ids    = []
    if top is not None:
        top_k = int(top)
    if top_n is not None:
        top_k = int(top_n)
    # Scale-aware actionable validation
    if scale == "cluster" and not cluster_id and not cluster_ids:
        return {"success": False,
                "error": ("scale='cluster' requires cluster_id (single id) or "
                          "cluster_ids (batch, use ['all'] for every cluster).")}
    if scale not in ("cluster", "cell"):
        return {"success": False,
                "error": f"scale must be 'cluster' or 'cell', got '{scale}'"}

    _inp = {"sampleid": sampleid, "n_vertices": len(vertices),
            "n_edges": len(edges), "scale": scale,
            "cluster_id": cluster_id, "cluster_ids": cluster_ids,
            "cell_ids_count": len(cell_ids), "cluster_key": cluster_key,
            "top_k": top_k, "name": name}

    try:
        adata = _get_adata(sampleid)
    except ValueError as e:
        result = {"success": False, "error": str(e)}
        _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
        return result

    # Build pathway + CSR once
    try:
        custom_pw = _build_custom_pathway(edges, name=name)
    except Exception as e:
        result = {"success": False, "error": f"failed to build custom pathway: {e}"}
        _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
        return result

    X_csr = _ensure_csr(adata.X)
    var_names = list(adata.var_names)

    # ----- Cluster scale --------------------------------------------------
    if scale == "cluster":
        if cluster_key not in adata.obs.columns:
            result = {"success": False,
                      "error": f"cluster_key '{cluster_key}' not in adata.obs"}
            _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
            return result

        # 'all' expansion
        use_cids = [str(c) for c in cluster_ids] if cluster_ids else []
        if use_cids and any(c.lower() == "all" for c in use_cids):
            use_cids = list(adata.obs[cluster_key].astype(str).unique())

        # Single cluster_id fallback
        if not use_cids and cluster_id:
            use_cids = [str(cluster_id)]

        if not use_cids:
            result = {"success": False,
                      "error": "scale='cluster' requires cluster_id or cluster_ids"}
            _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
            return result

        scores = {}
        for cid in use_cids:
            mask = (adata.obs[cluster_key].astype(str) == cid).values
            cluster_idx = np.where(mask)[0].tolist()
            if not cluster_idx:
                scores[cid] = {"error": "0 cells in cluster"}
                continue
            norm_d = compute_all_kegg_norms_cluster_mean(
                X_csr, var_names, [custom_pw], cluster_idx,
            )
            entry = {
                "score":   float(norm_d.get(name, 0.0)),
                "n_cells": len(cluster_idx),
            }
            if verbose:
                try:
                    edge_df_map = get_top_edges_per_pathway(
                        adata, [custom_pw], str(cid),
                        cluster_key=cluster_key, top_n_edges=verbose_top_n,
                    )
                    edf = edge_df_map.get(name)
                    if edf is not None and len(edf) > 0:
                        entry["edge_contributions"] = [
                            {
                                "source":       r["source"],
                                "target":       r["target"],
                                "weight":       round(float(r["weight"]), 4),
                                "beta":         round(float(r["beta"]), 4),
                                "contribution": round(float(r["contribution"]), 6),
                            }
                            for r in edf.to_dict("records")
                        ]
                except Exception as e:
                    entry["edge_contributions_error"] = f"{type(e).__name__}: {e}"
            scores[cid] = entry

        ranked = sorted(
            [(c, s["score"]) for c, s in scores.items() if "score" in s],
            key=lambda x: -x[1],
        )
        result = {
            "success":   True,
            "scale":     "cluster",
            "sampleid":  sampleid,
            "name":      name,
            "n_edges":   len(edges),
            "scores":    scores,
            "ranked":    ranked,
        }
        _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
        return result

    # ----- Cell scale -----------------------------------------------------
    if scale == "cell":
        all_norms = compute_all_kegg_norms_sparse(X_csr, var_names, [custom_pw])
        norms = np.asarray(all_norms[name], dtype=np.float64)

        var_to_idx = {g: i for i, g in enumerate(var_names)} if verbose else None

        def _alpha_for_cell(loc: int):
            row = X_csr[loc]
            return np.asarray(row.todense()).ravel() if sp.issparse(row) else np.asarray(row).ravel()

        cell_scope = None
        scope_description = None
        if cluster_id or cluster_ids:
            if cluster_key not in adata.obs.columns:
                result = {"success": False,
                          "error": f"cluster_key '{cluster_key}' not in adata.obs"}
                _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
                return result
            use_cids = [str(c) for c in cluster_ids] if cluster_ids else []
            if use_cids and any(c.lower() == "all" for c in use_cids):
                use_cids = list(adata.obs[cluster_key].astype(str).unique())
            if not use_cids and cluster_id:
                use_cids = [str(cluster_id)]
            mask = adata.obs[cluster_key].astype(str).isin(use_cids).values
            cell_scope = np.where(mask)[0]
            if len(cell_scope) == 0:
                result = {"success": False,
                          "error": f"0 cells matched {cluster_key} in {use_cids}"}
                _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
                return result
            scope_description = {
                "cluster_key": cluster_key,
                "cluster_ids": use_cids,
                "n_scope": int(len(cell_scope)),
            }

        if cell_ids:
            scores = {}
            scope_set = set(cell_scope.tolist()) if cell_scope is not None else None
            for cid in cell_ids:
                if cid in adata.obs.index:
                    loc = adata.obs.index.get_loc(cid)
                    if scope_set is not None and loc not in scope_set:
                        scores[cid] = {"error": "cell outside requested cluster scope"}
                        continue
                    if verbose:
                        scores[cid] = {
                            "score": float(norms[loc]),
                            "edge_contributions": _decompose_pathway_for_cell(
                                custom_pw, _alpha_for_cell(loc), var_to_idx, verbose_top_n,
                            ),
                        }
                    else:
                        scores[cid] = float(norms[loc])
                else:
                    scores[cid] = None
            result = {
                "success":  True,
                "scale":    "cell",
                "sampleid": sampleid,
                "name":     name,
                "n_edges":  len(edges),
                "scores":   scores,
            }
            if scope_description is not None:
                result["scope"] = scope_description
            _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
            return result

        # No cell_ids -> top_k over all cells, or over requested cluster scope.
        if cell_scope is not None:
            scoped_order = cell_scope[np.argsort(-norms[cell_scope])[:max(1, top_k)]]
            order = scoped_order
        else:
            order = np.argsort(-norms)[:max(1, top_k)]
        barcodes = [str(adata.obs.index[i]) for i in order]
        top_cells = []
        for b, i in zip(barcodes, order):
            entry = {"cell_id": b, "score": float(norms[i])}
            if verbose:
                entry["edge_contributions"] = _decompose_pathway_for_cell(
                    custom_pw, _alpha_for_cell(int(i)), var_to_idx, verbose_top_n,
                )
            top_cells.append(entry)
        result = {
            "success":  True,
            "scale":    "cell",
            "sampleid": sampleid,
            "name":     name,
            "n_edges":  len(edges),
            "n_total":  int(adata.n_obs),
            "top_k":    int(top_k),
            "top_cells": top_cells,
        }
        if scope_description is not None:
            result["scope"] = scope_description
        _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
        return result

    result = {"success": False, "error": f"unknown scale '{scale}' (use 'cluster' or 'cell')"}
    _log_call("custom_pathway_calc", _inp, result, int((time.time()-_t0)*1000))
    return result


# ================================================================
# Tool 11: reset_pipeline_namespace
# ================================================================

@mcp.tool()
async def reset_pipeline_namespace(
    sampleid: str = Field(..., description="Sample ID whose Python env and adata.obs mutation to reset"),
) -> dict:
    """
    Reset a sample's Python execution environment and restore adata.obs/uns to
    the snapshot taken at first access.

    Call this between eval episodes or GRPO rollouts to prevent cross-run
    contamination (e.g. score columns added by a previous agent run leaking
    into the next run's adata.obs).
    """
    sid     = _resolve_sampleid(sampleid)
    env_was = sid in state["python_envs"]
    if env_was:
        del state["python_envs"][sid]

    obs_restored = False
    if sid in state["adata_obs_snap"]:
        try:
            adata = _get_backend().get_adata(sid)
            if adata is not None:
                adata.obs = state["adata_obs_snap"][sid].copy(deep=True)
                adata.uns = dict(state["adata_uns_snap"][sid])
                obs_restored = True
        except Exception as e:
            return {"success": False, "error": str(e), "sampleid": sid}

    return {
        "success":       True,
        "sampleid":      sid,
        "env_removed":   env_was,
        "obs_restored":  obs_restored,
    }


# ================================================================
# Tool 12: execute_pipeline_code
# ================================================================

@mcp.tool()
async def execute_pipeline_code(
    sampleid: str = Field(..., description="Sample ID"),
    code:     str = Field(..., description="Python code to execute"),
) -> dict:
    """
    Execute code directly in a persistent Python environment (supports both cell and cluster analysis).

    ## IMPORTANT — do NOT reload adata
    `adata` for the requested sampleid is already loaded and injected into the namespace.
    Never call `ad.read_h5ad(...)` or `anndata.read_h5ad(...)` — it will fail (relative path,
    wrong CWD) and crash the server if followed by exit().
    Never call `exit()`, `quit()`, or `sys.exit()` — use `raise RuntimeError(msg)` instead.

    ## Objects automatically available in the namespace
    - **adata** — AnnData for `sampleid`, already loaded (do NOT re-load)
    - sc, np, pd, sp, nx, issparse
    - run_astar_for_cluster, build_cluster_conservative_graphs
    - build_rl_gene_map, parse_all_kegg_xmls
    - compute_and_select_top_kegg, get_top_edges_per_pathway
    - cluster_graphs, rl_map, kegg_pathways, astar_results

    ## Key variable structures

    ### cluster_graphs
    Dict populated after run_astar_pipeline. Access directly by cluster_id — sampleid is NOT a key.
    ```
    cluster_graphs[cluster_id]  # e.g. cluster_graphs["2"]
    → {
        "n_edges": int,
        "n_components": int,
        "top_edges": [
            {"source": str, "target": str, "freq": float, "mean_beta": float, "score": float},
            ...  # sorted by score descending, top 20
        ],
        "attrs": {"total_paths": int, "threshold": float, "cutoff": int, "mode": str}
      }
    ```
    Example — extract top genes:
    ```python
    from collections import Counter
    cnt = Counter()
    for e in cluster_graphs["2"]["top_edges"]:
        cnt[e["source"]] += 1
        cnt[e["target"]] += 1
    top_genes = [g for g, _ in cnt.most_common(5)]
    ```

    ### rl_map
    Dict populated after get_cluster_rl_map. Same structure as the tool response.
    ```
    rl_map["clusters"][cluster_id]  # e.g. rl_map["clusters"]["2"]
    → {
        "n_ligand": int,
        "n_receptor": int,
        "n_edges": int,
        "edges": [
            {"sender": str, "ligand": str, "receptor": str, "receiver": str},
            ...
        ]
      }
    ```
    Example — find common LR pairs between clusters (receiver-agnostic, ligand-receptor pair only):
    ```python
    def lr_pairs(cid):
        return set(f"{e['ligand']}-{e['receptor']}"
                   for e in rl_map["clusters"][cid]["edges"])
    common = lr_pairs("2") & lr_pairs("3")
    ```

    ## Returns
    - success, stdout, result, error/traceback, namespace_vars
    """
    sampleid = _resolve_sampleid(sampleid)
    if sampleid not in state["python_envs"]:
        try:
            adata = _get_adata(sampleid)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        print(f"\n[execute_pipeline_code] creating new environment: '{sampleid}'")
        state["python_envs"][sampleid] = ScGraphPythonEnvironment(
            sampleid, adata, state
        )

    env = state["python_envs"][sampleid]
    env.sync_state()

    print(f"\n[execute_pipeline_code] '{sampleid}'")
    print("--- CODE ---")
    print(code)
    print("--- END  ---")

    _t0    = time.time()
    result = env.execute(code)
    _log_call(
        "execute_pipeline_code",
        {"sampleid": sampleid, "code": code},
        result,
        int((time.time() - _t0) * 1000),
    )

    print(f"  success={result.get('success')}  "
          f"result={result.get('result')}  "
          f"stdout={result.get('stdout', '')[:120]}")

    return result


# ================================================================
# Server startup
# ================================================================

if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8005"))
    print("=" * 60)
    print(f"SC Graph MCP Server  (port: {port})")
    print("human / mouse organism branch support")
    print("=" * 60)
    print("\nOrganism detection:")
    print("  Human → uppercase symbols (CD8A, GAPDH, etc.)")
    print("  Mouse → Titlecase symbols (Cd8a, Gapdh, etc.)")
    print("\nPer-organism resources:")
    print("  DoRothEA : dc.op.dorothea(organism=org)")
    print("  LIANA    : consensus | mouseconsensus")
    print("  KEGG     : KEGG_DIR | KEGG_DIR/mmu")
    print("\nTools:")
    print("  [cluster]  run_astar_pipeline        ⚠️  EXPENSIVE")
    print("  [cluster]  get_astar_graph_summary")
    print("  [cluster]  get_cluster_rl_map")
    print("  [cluster]  get_cluster_kegg_edges")
    print("  [cell]     get_astar_cellular_info")
    print("  [cell]     get_cell_kegg_edges")
    print("  DEPRECATED [kg]       get_kg_context")
    print("  DEPRECATED [kg/ctx]   resolve_query_to_context_set")
    print("  [custom]   custom_pathway_calc")
    print("  [flex]     execute_pipeline_code")
    print("=" * 60)
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port)
