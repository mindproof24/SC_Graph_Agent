# =============================================================================
# bench_kegg_full.py
#
# WHAT THIS BENCHMARKS (axes)
# --------------------------------------------------------------------------
# Per-cell KEGG-pathway ACTIVITY, evaluated as: does the expected pathway's
# activity discriminate the matching lineage (one-vs-rest AUROC)?
#
# It automates, for every immune lineage, three CURATION STAGES x methods:
#   generic   : the raw KEGG pathway (all edges / all member genes)
#   curated   : top-k lineage-specific edges (data-driven, dilution removed)
#   augment   : curated edges + a clique of canonical lineage marker genes
#               (= the agent's custom_pathway_calc "augment unique markers" step)
#
# METHODS
#   Edge_*      : edge-L2 scoring variants of the graph-based pathway tool
#                 (contribution = (ai*aj/aG^2)*|ai+aj+w|)
#   AUCell_*    : decoupler dc.mt.aucell on the SAME gene sets (fair baseline)
#
# IMPORTANT (honest framing): on identical genes, edge-L2 is ~on par with (often
# marginally below) AUCell. The gain from generic->augment is CURATION, which is
# method-agnostic. TOOL's unique value is edge-level interpretability, not a
# higher scalar. This script shows all of that side by side.
#
# Also reports COMPUTATION TIME per method (wall-clock).
#
# INPUTS (repository defaults can be overridden on the command line):
#   data/Lung_cancer_Imm_filtered.h5ad
#   obs["cell_type_predicted"]      benchmark labels
#   var["feature_name"]             human gene symbols
#
# OUTPUTS (--outdir):
#   auroc_table.csv               lineage x method AUROC
#   lineage_metadata.csv          cell counts, KEGG mapping, and edge counts
#   input_metadata.json           AnnData provenance and benchmark parameters
#   timing_table.csv              method -> seconds
#   active_edges.csv              per-lineage top active edges (TOOL's unique output)
#   fig_auroc_stages.png          grouped bar: mean AUROC by method/stage
#   fig_auroc_heatmap.png         lineage x method AUROC heatmap
#   fig_timing.png                computation time per method
#   fig_edge_interpretability.png per-lineage top lineage-specific active edges (TOOL only)
#
# USAGE FROM THE REPOSITORY ROOT
#   python figure2/bench_kegg_full.py --outdir figure2
# =============================================================================
from __future__ import annotations
import argparse, itertools, json, os, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# keep text as editable objects (not outlines) when opened in Illustrator
matplotlib.rcParams["pdf.fonttype"] = 42     # TrueType -> editable text in PDF
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none" # SVG keeps <text> elements

def _save(fig, path):
    """Save PNG (preview) + PDF & SVG (vector, decomposable in Illustrator)."""
    from pathlib import Path as _P
    p = _P(path)
    fig.savefig(p, dpi=150)                # .png preview
    fig.savefig(p.with_suffix(".pdf"))     # vector, editable text
    fig.savefig(p.with_suffix(".svg"))     # vector backup
    plt.close(fig)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5AD = REPO_ROOT / "data" / "Lung_cancer_Imm_filtered.h5ad"
DEFAULT_SERVER_DIR = REPO_ROOT / "server"
DEFAULT_KEGG_DIR = REPO_ROOT / "vendor" / "keggx" / "keggx" / "data" / "KEGG_Graph_processing"
sys.path.insert(0, str(DEFAULT_SERVER_DIR))
os.environ.setdefault("KEGG_DIR", str(DEFAULT_KEGG_DIR))

# ---- per-lineage ground truth: KEGG pathway (title fragment) + marker genes ----
LINEAGES = {
    "T cell CD4": dict(
        kegg="T cell receptor signaling",
        markers=["CD4","IL7R","CD40LG","CCR7","TCF7","LEF1","MAL"]),
    "T cell CD8": dict(
        kegg="T cell receptor signaling",
        markers=["CD8A","CD8B","GZMK","GZMA","NKG7","CCL5","DUSP2"]),
    "B cell": dict(
        kegg="B cell receptor signaling",
        markers=["CD79A","CD79B","MS4A1","CD19","BANK1","BLK","CD22","TCL1A","VPREB3","FCRL1"]),
    "NK cell": dict(
        kegg="Natural killer cell mediated cytotoxicity",
        markers=["NKG7","GNLY","KLRD1","NCAM1","NCR1","KLRF1","PRF1","GZMB","KLRC1","FCGR3A"]),
    "Macrophage": dict(
        kegg="Fc gamma R-mediated phagocytosis",
        markers=["CD68","CD14","LYZ","CSF1R","MRC1","MARCO","C1QA","C1QB","APOE"]),
    "cDC": dict(
        kegg="Antigen processing and presentation",
        markers=["CLEC9A","FCER1A","CD1C","XCR1","BATF3","CLEC10A","ITGAX"]),
}
TOP_K_EDGES = 10            # curated edge budget
TIMING_REPEATS = 1


def portable_path(path: Path) -> str:
    """Prefer repository-relative provenance paths when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def match(frag, names):
    h = [n for n in names if frag.lower() in n.lower()]
    return h[0] if h else None


def safe_auroc(score, is_type):
    if is_type.sum() == 0 or is_type.sum() == len(is_type):
        return np.nan
    try:
        return float(roc_auc_score(is_type.astype(int), score))
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--h5ad",
        type=Path,
        default=DEFAULT_H5AD,
        help="Analysis-ready LuCA AnnData input (default: data/Lung_cancer_Imm_filtered.h5ad).",
    )
    ap.add_argument(
        "--celltype-col",
        default="cell_type_predicted",
        help="obs column containing the benchmark lineage labels.",
    )
    ap.add_argument("--kegg-dir", default=os.environ["KEGG_DIR"])
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "figure2")
    ap.add_argument("--min-genes", type=int, default=5)
    ap.add_argument(
        "--feature-col",
        default="feature_name",
        help="var column containing gene symbols; use an empty string when var_names already contain symbols.",
    )
    a = ap.parse_args()

    import anndata as ad, decoupler as dc
    from graph_utils import parse_all_kegg_xmls, compute_all_kegg_norms_sparse

    if not a.h5ad.is_file():
        ap.error(f"AnnData input not found: {a.h5ad}")
    if not Path(a.kegg_dir).is_dir():
        ap.error(f"KEGG directory not found: {a.kegg_dir}")

    outdir = a.outdir; outdir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(a.h5ad)
    if a.celltype_col not in adata.obs:
        ap.error(f"obs column not found: {a.celltype_col}")
    if a.feature_col and a.feature_col not in adata.var:
        ap.error(f"var column not found: {a.feature_col}")
    if a.feature_col:
        adata.var_names = adata.var[a.feature_col].astype(str)
        if not adata.var_names.is_unique:
            adata.var_names_make_unique()
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    labels = adata.obs[a.celltype_col].astype(str)
    print(f"[data] {adata.n_obs} cells x {adata.n_vars} genes | {labels.nunique()} types")

    input_metadata = {
        "h5ad": portable_path(a.h5ad),
        "file_size_bytes": a.h5ad.stat().st_size,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "celltype_column": a.celltype_col,
        "feature_column": a.feature_col or None,
        "label_matching": "case-insensitive literal substring",
        "dataset_title": str(adata.uns.get("title", "")),
        "dataset_citation": str(adata.uns.get("citation", "")),
        "kegg_directory": portable_path(Path(a.kegg_dir)),
        "minimum_pathway_genes": int(a.min_genes),
        "curated_edge_budget": int(TOP_K_EDGES),
        "requested_lineages": list(LINEAGES),
    }
    (outdir / "input_metadata.json").write_text(
        json.dumps(input_metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    pathways = parse_all_kegg_xmls(a.kegg_dir, adata, min_genes=a.min_genes)
    pw_by_name = {pw.name: pw for pw in pathways}
    titles = list(pw_by_name)
    print(f"[kegg] {len(pathways)} pathways")

    # ---------- shared expression helpers ----------
    g2i = {g: i for i, g in enumerate(adata.var_names)}
    Xc = adata.X.tocsc()
    aG = np.asarray(Xc.sum(1)).ravel(); aG2 = aG * aG
    _col_cache = {}
    def col(gi):
        if gi not in _col_cache:
            _col_cache[gi] = np.asarray(Xc[:, gi].todense()).ravel()
        return _col_cache[gi]
    def edge_l2(edges):
        nsq = np.zeros(adata.n_obs)
        for s, t, w in edges:
            if s in g2i and t in g2i:
                ai, aj = col(g2i[s]), col(g2i[t])
                nsq += np.divide(ai*aj, aG2, out=np.zeros_like(aG2), where=aG2 > 1e-10) * np.abs(ai+aj+w)
        return np.sqrt(nsq)

    def edges_of(pw):
        return list(zip(pw.sources, pw.targets, [float(x) for x in pw.weights]))

    def edge_spec_topk(pw, is_L, k):                # top-k lineage-specific edges
        edges = edges_of(pw)
        aL = np.asarray(adata[is_L].X.mean(0)).ravel();  AL = aL.sum(); AL2 = AL*AL
        aR = np.asarray(adata[~is_L].X.mean(0)).ravel(); AR = aR.sum(); AR2 = AR*AR
        scored = []
        for s, t, w in edges:
            if s in g2i and t in g2i:
                i, j = g2i[s], g2i[t]
                cL = (aL[i]*aL[j]/AL2)*abs(aL[i]+aL[j]+w) if AL2 > 0 else 0
                cR = (aR[i]*aR[j]/AR2)*abs(aR[i]+aR[j]+w) if AR2 > 0 else 0
                scored.append((s, t, w, cL - cR))
        scored.sort(key=lambda x: x[3], reverse=True)
        top = scored[:k]                                # (s, t, w, specificity)
        return [(s, t, w) for s, t, w, _ in top], top, edges

    # ---------- generic activity matrices (timed) ----------
    timing = {}
    t0 = time.perf_counter()
    norms = compute_all_kegg_norms_sparse(adata.X, list(adata.var_names), pathways)
    timing["Edge_generic (all pw)"] = time.perf_counter() - t0
    A_tool = pd.DataFrame({pw.name: np.asarray(norms[pw.name]) for pw in pathways if pw.name in norms},
                          index=adata.obs_names)

    net_kegg = pd.DataFrame([(pw.name, g, 1.0) for pw in pathways
                             for g in set(pw.sources) | set(pw.targets)],
                            columns=["source", "target", "weight"])
    t0 = time.perf_counter()
    dc.mt.aucell(adata, net_kegg, tmin=5)
    timing["AUCell_generic (all pw)"] = time.perf_counter() - t0
    A_auc = adata.obsm["score_aucell"]

    # ---------- per-lineage stages ----------
    rows = []
    aug_net_rows = []          # for one AUCell_augment call
    edge_records = {}          # lineage -> [(src, tgt, w, specificity)]  (TOOL interpretability)
    t_tool_cur = t_tool_aug = 0.0
    for L, cfg in LINEAGES.items():
        is_L = labels.str.contains(L, case=False, regex=False).values
        if is_L.sum() < 10:
            print(f"[skip] {L}: only {is_L.sum()} cells"); continue
        pw_title = match(cfg["kegg"], titles)
        if pw_title is None:
            print(f"[warn] {L}: KEGG '{cfg['kegg']}' not found"); continue
        pw = pw_by_name[pw_title]
        markers = [g for g in cfg["markers"] if g in g2i]
        curated, curated_scored, _ = edge_spec_topk(pw, is_L, TOP_K_EDGES)
        edge_records[L] = curated_scored
        aug = curated + [(x, y, 1.0) for x, y in itertools.combinations(markers, 2)]

        tt = time.perf_counter(); s_cur = edge_l2(curated); t_tool_cur += time.perf_counter() - tt
        tt = time.perf_counter(); s_aug = edge_l2(aug);     t_tool_aug += time.perf_counter() - tt

        aug_genes = sorted({g for e in aug for g in e[:2]})
        for g in aug_genes:
            aug_net_rows.append((L, g, 1.0))

        rows.append(dict(
            lineage=L, n_cells=int(is_L.sum()), kegg=pw_title,
            n_edges_generic=len(edges_of(pw)), n_edges_curated=len(curated), n_edges_aug=len(aug),
            Edge_generic=safe_auroc(A_tool[pw_title].values, is_L),
            Edge_curated=safe_auroc(s_cur, is_L),
            Edge_augment=safe_auroc(s_aug, is_L),
            AUCell_generic=safe_auroc(A_auc[pw_title].values, is_L),
        ))
    n_lineages = len(rows)
    timing[f"Edge_curated ({n_lineages} lin)"] = t_tool_cur
    timing[f"Edge_augment ({n_lineages} lin)"] = t_tool_aug

    # ---------- AUCell on the SAME augmented sets (fair comparison), timed ----------
    aug_net = pd.DataFrame(aug_net_rows, columns=["source", "target", "weight"])
    t0 = time.perf_counter()
    dc.mt.aucell(adata, aug_net, tmin=3)
    timing[f"AUCell_augment ({n_lineages} lin)"] = time.perf_counter() - t0
    A_auc_aug = adata.obsm["score_aucell"]
    for r in rows:
        L = r["lineage"]; is_L = labels.str.contains(L, case=False, regex=False).values
        r["AUCell_augment"] = safe_auroc(A_auc_aug[L].values, is_L) if L in A_auc_aug.columns else np.nan

    df = pd.DataFrame(rows).set_index("lineage")
    method_cols = ["Edge_generic","AUCell_generic","Edge_curated",
                   "Edge_augment","AUCell_augment"]
    auroc = df[method_cols]
    auroc.to_csv(outdir / "auroc_table.csv")
    df[["n_cells", "kegg", "n_edges_generic", "n_edges_curated", "n_edges_aug"]].to_csv(
        outdir / "lineage_metadata.csv"
    )
    pd.Series(timing).round(3).to_csv(outdir / "timing_table.csv", header=["seconds"])

    print("\n===== AUROC (lineage x method) =====")
    print(auroc.round(3).to_string())
    print("\nmean:\n", auroc.mean().round(3).to_string())
    print("\n===== timing (s) =====\n", pd.Series(timing).round(3).to_string())

    # ---------- plots ----------
    _bar_stages(auroc, outdir / "fig_auroc_stages.png")
    _heatmap(auroc, outdir / "fig_auroc_heatmap.png")
    _timing(timing, outdir / "fig_timing.png")
    _edge_interpretability(edge_records, outdir / "fig_edge_interpretability.png")
    # also dump the active edges as a table (TOOL's unique output)
    pd.DataFrame([(L, f"{s}->{t}", w, sc) for L, recs in edge_records.items()
                  for s, t, w, sc in recs],
                 columns=["lineage", "edge", "weight", "specificity"]).to_csv(
        outdir / "active_edges.csv", index=False)
    print(f"\n[saved] tables + figures in {outdir}/")


def _bar_stages(auroc, path):
    m = auroc.mean()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#bbb","#88c","#5B8FF9","#2b6cb0","#aa6"]
    ax.bar(range(len(m)), m.values, color=colors[:len(m)])
    for i, c in enumerate(auroc.columns):           # per-lineage points
        ys = auroc[c].dropna().values
        ax.scatter(np.full(len(ys), i)+np.random.uniform(-.12,.12,len(ys)), ys, c="k", s=16, zorder=3)
    ax.axhline(0.5, ls="--", c="grey"); ax.set_ylim(0,1.02)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m.index, rotation=20, ha="right")
    ax.set_ylabel("mean AUROC (points = lineages)")
    ax.set_title("Pathway-recovery AUROC: generic → curated → augment (TOOL vs AUCell)")
    fig.tight_layout(); _save(fig, path)


def _heatmap(auroc, path):
    fig, ax = plt.subplots(figsize=(1.4*auroc.shape[1]+3, 0.55*auroc.shape[0]+2))
    im = ax.imshow(auroc.values, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(auroc.shape[1])); ax.set_xticklabels(auroc.columns, rotation=25, ha="right")
    ax.set_yticks(range(auroc.shape[0])); ax.set_yticklabels(auroc.index)
    for i in range(auroc.shape[0]):
        for j in range(auroc.shape[1]):
            v = auroc.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="w" if v < 0.8 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="AUROC"); ax.set_title("AUROC: lineage × method")
    fig.tight_layout(); _save(fig, path)


def _edge_interpretability(edge_records, path, topn=8):
    """TOOL-only panel: per lineage, the top lineage-specific ACTIVE edges it names.
    AUCell/PROGENy return only a scalar and cannot produce this."""
    Ls = list(edge_records)
    if not Ls:
        return
    ncol = 3; nrow = int(np.ceil(len(Ls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.7, nrow * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, L in zip(axes, Ls):
        recs = edge_records[L][:topn][::-1]            # ascending for barh
        names = [f"{s}→{t}" for s, t, w, sc in recs]
        vals = [sc for s, t, w, sc in recs]
        ax.barh(range(len(recs)), vals, color="#e07b39")
        ax.set_yticks(range(len(recs))); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(L, fontsize=9); ax.set_xlabel("lineage specificity", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
    for ax in axes[len(Ls):]:
        ax.axis("off")
    fig.suptitle("TOOL-only: top lineage-specific active edges (interpretability)", fontsize=11)
    fig.tight_layout(); _save(fig, path)


def _timing(timing, path):
    s = pd.Series(timing).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.5*len(s)+2.25))
    ax.barh(range(len(s)), s.values, color="#5B8FF9")
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel("seconds"); ax.set_title("Computation time per method")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    note = (
        "Timing note: generic runs score all retained KEGG pathways, whereas curated/augment TOOL runs "
        "score only six compact lineage-specific edge sets. Thus the within-TOOL speed-up mainly reflects "
        "a smaller evaluated graph. AUCell is slower because it performs per-cell rank-based gene-set "
        "recovery, while TOOL directly evaluates sparse edge-L2 terms."
    )
    fig.text(0.02, 0.02, note, ha="left", va="bottom", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.15, 1, 1)); _save(fig, path)


if __name__ == "__main__":
    main()
