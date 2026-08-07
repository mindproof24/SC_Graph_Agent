#!/usr/bin/env python
"""figure_edge_axis_pdf.py  (Step B+C)

Plot edge activity along the maturation (interpretation) axis and export an
Illustrator-editable vector PDF.

Readout = target-program score (control-referenced z, sign-weighted; no alpha_g^2
normalization). This readout measures signed target-program changes along the
maturation axis. It complements the edge-count-normalized CWG activity analysis,
which measures non-negative per-edge intensity rather than traversal coherence.

Tracks (all control-referenced z):
  A*      : mean target program over A*-selected top-K targets of cardiac TFs
  prior   : mean target program over the full DoRothEA target sets of the same TFs
            (baseline without traversal)
  random  : random targets matched to the A* edge count (empirical null)
  IRX4    : injected-edge program = z(IRX4)      (mouse literature, independent source)
  NRG1    : held-out = z(NRG1)                   (NKX2-5 -| NRG1, novel in the paper;
                                                  not injected)

Layout: left (maturation-axis profile, 5 lines + 95% interval + direct labels)
        right (two PHATE panels: IRX4 z, NRG1 z).
PDF: pdf.fonttype=42 (editable text); only the point cloud is rasterized, while
axes, labels and legends remain vector.
"""
import argparse, os, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42     # keep text editable rather than converting it to outlines
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 9
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Generate Supplementary Figure 4 and its source values.")
parser.add_argument("--h5ad", type=Path, required=True)
parser.add_argument("--dorothea", type=Path, default=ROOT.parent / "shared_inputs" / "dorothea_ABC_human.parquet")
parser.add_argument("--astar-edges", type=Path, default=ROOT / "inputs" / "dorothea_astar_ranked_edges.csv")
parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
H5AD, DOROTHEA, ASTAR = args.h5ad, args.dorothea, args.astar_edges
OUT_PDF = args.out_dir / "supplementary_figure4_edge_activity_axis.pdf"

HEART_TFS = ["GATA4","GATA6","MEF2A","MEF2C","SRF","NR2F2","TEAD1","NKX2-5","TBX5"]
IRX4_SRC  = ["TBX5","NKX2-5","TBX20","GATA4","NKX2-6"]     # injected IRX4 activators
TOPK      = 20
NBIN      = 20
SEED      = 0
COL = {"A*":"#2a78d6", "prior":"#eda100", "injected IRX4":"#008300",
       "gold NRG1":"#4a3aa7", "random":"#898781"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
# diverging colormap: blue <-> red with a gray midpoint (palette.md diverging)
DIVERGE = LinearSegmentedColormap.from_list("bwr_pal", ["#2a78d6", "#f0efec", "#e34948"])

# ================= load =================
print("[load]", H5AD, flush=True)
adata = sc.read_h5ad(H5AD)
isc = adata.obs.is_control.values
tgt = adata.obs.target_gene.astype(str).values
axis = adata.obs["score_maturation"].values                 # interpretation axis (corr with PHATE1 = -0.89)
ph = np.asarray(adata.obsm["X_phate"])
var_pos = {g: i for i, g in enumerate(adata.var_names)}
doro = pd.read_parquet(DOROTHEA).rename(columns={"source":"tf","weight":"mor"})
doro["ms"] = np.sign(doro["mor"]).replace(0, 1)

astar = pd.read_csv(ASTAR)
astar = astar[astar["method"] == "astar_path"] if "method" in astar else astar   # filtering is required
tfs = [t for t in HEART_TFS if t in set(astar.source)]
print("[cardiac TFs covered by A*]", tfs, flush=True)

# ============ program target sets (A*/prior/random, per TF) ============
rng = np.random.default_rng(SEED)
genes_arr = np.array(adata.var_names)
prog_targets = {}     # (track, tf) -> [(gene, sign), ...]
for tf in tfs:
    a_sub = (astar[astar.source == tf].sort_values("rank").head(TOPK)
             .merge(doro[doro.tf == tf][["target","ms"]], left_on="target", right_on="target", how="inner"))
    a_t = [(g, s) for g, s in zip(a_sub.target, a_sub.ms) if g in var_pos]
    p_sub = doro[doro.tf == tf]
    p_t = [(g, s) for g, s in zip(p_sub.target, p_sub.ms) if g in var_pos]
    if len(a_t) < 2:              # skip when too few A*-selected targets remain
        continue
    k = len(a_t)
    r_g = rng.choice(genes_arr, size=k, replace=False); r_s = rng.choice([1, -1], size=k)
    prog_targets[("A*", tf)]     = a_t
    prog_targets[("prior", tf)]  = p_t
    prog_targets[("random", tf)] = [(g, s) for g, s in zip(r_g, r_s)]
used_tfs = sorted({tf for (_, tf) in prog_targets})
print("[TFs used for program scoring]", used_tfs, flush=True)

# ========= control-referenced z (load only the union of genes) =========
union = sorted({g for tt in prog_targets.values() for (g, _) in tt} | {"IRX4", "NRG1"} & set(var_pos))
union = [g for g in union if g in var_pos]
col = {g: i for i, g in enumerate(union)}
Xu = adata[:, union].X; Xu = Xu.toarray() if sp.issparse(Xu) else np.asarray(Xu)
cmean = Xu[isc].mean(0); cstd = Xu[isc].std(0) + 1e-9
Z = (Xu - cmean) / cstd                                     # control-referenced z (control program is centred near 0)

def program(targets):                                       # mean over targets( sign × z )
    idx = [col[g] for (g, _) in targets if g in col]
    sgn = np.array([s for (g, s) in targets if g in col], float)
    return (Z[:, idx] * sgn).mean(1) if idx else np.zeros(Z.shape[0])

# per-cell program per track (A*/prior/random are averaged across TFs)
tracks = {}
for name in ["A*", "prior", "random"]:
    mats = [program(prog_targets[(name, tf)]) for tf in used_tfs if (name, tf) in prog_targets]
    tracks[f"{name}"] = np.mean(mats, axis=0)
tracks["injected IRX4"] = Z[:, col["IRX4"]] if "IRX4" in col else None
tracks["gold NRG1"]     = Z[:, col["NRG1"]] if "NRG1" in col else None
tracks = {k: v for k, v in tracks.items() if v is not None}

# ========= maturation-axis binning (mean + 95% interval) =========
edges_bin = np.quantile(axis, np.linspace(0, 1, NBIN + 1))
edges_bin[-1] += 1e-9
binid = np.clip(np.digitize(axis, edges_bin) - 1, 0, NBIN - 1)
xc = 0.5 * (edges_bin[:-1] + edges_bin[1:])
prof = {}                                                   # name -> (mean[NBIN], ci[NBIN])
for name, v in tracks.items():
    m = np.full(NBIN, np.nan); ci = np.full(NBIN, np.nan)
    for b in range(NBIN):
        s = v[binid == b]
        if len(s):
            m[b] = s.mean(); ci[b] = 1.96 * s.std() / np.sqrt(len(s))
    prof[name] = (m, ci)

# ====== axis tracking: |corr(program, maturation axis)| summary ======
print("\n=== axis tracking |corr(program, maturation axis)| (A* vs prior vs random) ===")
for name in ["A*", "prior", "random"]:
    if name in tracks:
        print(f"  {name:14s}: {abs(np.corrcoef(tracks[name], axis)[0,1]):.3f}")

# ================= figure =================
LINE_ORDER = ["A*", "prior", "random", "injected IRX4", "gold NRG1"]
fig = plt.figure(figsize=(9.5, 5.2))
gs = GridSpec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1],
              wspace=0.28, hspace=0.35, left=0.08, right=0.97, top=0.92, bottom=0.12)

# left: maturation-axis profile (spans both rows)
axp = fig.add_subplot(gs[:, 0])
for name in [n for n in LINE_ORDER if n in prof]:
    m, ci = prof[name]; c = COL[name]
    ls = "--" if name == "random" else "-"
    axp.fill_between(xc, m - ci, m + ci, color=c, alpha=0.15, linewidth=0)
    axp.plot(xc, m, color=c, lw=2, ls=ls, zorder=3)
    j = np.where(~np.isnan(m))[0][-1]
    axp.annotate(name, (xc[j], m[j]), xytext=(4, 0), textcoords="offset points",
                 color=c, fontsize=8, va="center", fontweight="bold")
axp.axhline(0, color=MUTED, lw=0.8, ls=":", zorder=1)
axp.set_xlabel("maturation axis  (score_maturation ->  mature CM)", color=INK)
axp.set_ylabel("target-program activity  (control-ref z)", color=INK)
axp.set_title("Edge activity along the cardiac maturation axis", color=INK, fontsize=10)
axp.spines[["top", "right"]].set_visible(False)
axp.spines[["left", "bottom"]].set_color("#c3c2b7")
axp.tick_params(colors=MUTED)

# right: two PHATE panels (point cloud rasterized, everything else vector)
def phate_panel(ax, val, title):
    vmax = np.nanpercentile(np.abs(val), 98) or 1.0
    o = np.argsort(np.abs(val))                              # draw strong signals on top
    scp = ax.scatter(ph[o, 0], ph[o, 1], c=val[o], cmap=DIVERGE,
                     norm=TwoSlopeNorm(0, -vmax, vmax), s=1.5, linewidths=0,
                     rasterized=True)                        # rasterize the points only
    ax.set_title(title, color=INK, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_color("#c3c2b7")
    cb = fig.colorbar(scp, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_edgecolor("#c3c2b7")
    cb.set_label("z", fontsize=7, color=MUTED)

if "injected IRX4" in tracks:
    phate_panel(fig.add_subplot(gs[0, 1]), tracks["injected IRX4"], "PHATE — injected IRX4 (z)")
if "gold NRG1" in tracks:
    phate_panel(fig.add_subplot(gs[1, 1]), tracks["gold NRG1"], "PHATE — held-out NRG1 (z)")

source_rows = []
for name, (mean, ci95) in prof.items():
    for b in range(NBIN):
        source_rows.append({
            "bin": b,
            "score_maturation_center": xc[b],
            "track": name,
            "mean": mean[b],
            "ci95": ci95[b],
            "n_cells": int((binid == b).sum()),
        })
pd.DataFrame(source_rows).to_csv(args.out_dir / "supplementary_figure4_source_values.csv", index=False)

fig.savefig(OUT_PDF)                                         # vector PDF (fonttype 42)
fig.savefig(OUT_PDF.with_suffix(".png"), dpi=200)             # quick preview
plt.close(fig)
print("\n[saved]", OUT_PDF, "  (+ .png preview and source-values CSV)")
print("-> In Illustrator: axes, labels, legends and lines stay editable text/paths; "
      "only the PHATE point cloud is an embedded image")
