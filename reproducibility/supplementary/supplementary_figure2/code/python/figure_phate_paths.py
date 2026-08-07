#!/usr/bin/env python
"""
fig_supple/scripts/02_figure_phate_paths.py
===========================================
'정확히 계산된 PHATE path'(378k 정제 manifold 단일그룹 A*)를 Illustrator-editable PDF 로.

3 패널 (동일 PHATE 레이아웃, 배경 378k 회색 + A* path cell 오버레이):
  A  path cell colored by score_maturation   (검증된 성숙축, corr(PHATE1)=-0.80)
  B  path cell colored by n_genes            (A* 구동변수/끝점축, ⊥ maturation)
  C  path cell colored by pert_status        (DE=perturbed 세포가 path 어디에 있나)
+ A 패널에 대표 path 궤적선(subsample) 오버레이.

editable: pdf.fonttype=42(텍스트 벡터), 점구름만 rasterized(축/라벨/선은 벡터).

실행:  source ~/venv-grpo/bin/activate
       python fig_supple/scripts/02_figure_phate_paths.py
필요:  00 산출 astar_path_cells.parquet + cardio_perturb_phate.h5ad (배경)
"""
import argparse
import json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})

BUNDLE = Path(__file__).resolve().parents[2]
DATA, FIG = BUNDLE / "data", BUNDLE / "results"; FIG.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Plot sampled A* paths on the cardiac PHATE embedding.")
parser.add_argument("--h5ad", type=Path, required=True)
args = parser.parse_args()

cells = pd.read_parquet(DATA / "astar_path_cells.parquet")
prov  = json.loads((DATA / "path_provenance.json").read_text())
print(f"[load] path cells={len(cells)} unique={cells.cell_idx.nunique()} paths={cells.path_id.nunique()}")

# 배경 = 전체 378k PHATE (backed, X 미로드)
A = ad.read_h5ad(args.h5ad, backed="r")
bg = np.asarray(A.obsm["X_phate"])[:, :2]
print(f"[bg] {bg.shape[0]:,} cells")

px, py = cells.phate1.to_numpy(), cells.phate2.to_numpy()

def base(ax, title):
    ax.scatter(bg[:, 0], bg[:, 1], s=1.2, c="#dcdcdc", linewidths=0, rasterized=True, zorder=0)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("PHATE 1"); ax.set_ylabel("PHATE 2")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True)

fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)

# ── A: score_maturation (diverging) ────────────────────────────────
ax = axs[0]; base(ax, "A  A* path cells — score_maturation")
vmax = np.nanpercentile(np.abs(cells.score_maturation), 98)
sc = ax.scatter(px, py, c=cells.score_maturation, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                s=5, linewidths=0, rasterized=True, zorder=2)
# 대표 path 궤적선 (벡터, subsample)
pids = cells.path_id.unique()
rng = np.random.default_rng(0)
for pid in rng.choice(pids, size=min(40, len(pids)), replace=False):
    seg = cells[cells.path_id == pid].sort_values("step")
    ax.plot(seg.phate1, seg.phate2, "-", color="#333333", lw=0.3, alpha=0.25, zorder=1)
cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02); cb.set_label("maturation", fontsize=7)

# ── B: n_genes (sequential, A* 구동축) ─────────────────────────────
ax = axs[1]; base(ax, "B  A* path cells — n_genes (driver)")
sc = ax.scatter(px, py, c=cells.n_genes, cmap="viridis",
                vmin=np.nanpercentile(cells.n_genes, 2), vmax=np.nanpercentile(cells.n_genes, 98),
                s=5, linewidths=0, rasterized=True, zorder=2)
cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02); cb.set_label("n_genes", fontsize=7)

# ── C: pert_status (DE 세포 위치) ──────────────────────────────────
ax = axs[2]; base(ax, "C  A* path cells — perturbation")
pal = {"perturb": "#d62728", "perturb_recovered": "#ff9896",
       "control": "#1f77b4", "control_recovered": "#aec7e8", "ambiguous": "#bbbbbb"}
st = cells.pert_status.astype(str)
for k, col in pal.items():
    m = (st == k).to_numpy()
    if m.any():
        ax.scatter(px[m], py[m], c=col, s=5, linewidths=0, rasterized=True, zorder=2,
                   label=f"{k} ({m.sum()})")
handles = [Line2D([0], [0], marker="o", ls="", mfc=pal[k], mec="none", ms=5,
                  label=k) for k in pal if (st == k).any()]
ax.legend(handles=handles, fontsize=6, loc="best", frameon=False)

# ── 캡션 (축-정합 정직 명시) ───────────────────────────────────────
cap = (f"378k refined manifold (cardio_perturb_phate.h5ad), single-group A* traversal | "
       f"paths(reduced)={prov['n_paths_reduced']}, unique path cells={prov['n_unique_path_cells']}, "
       f"seed={prov['seed']}, git={prov['git_hash']}\n"
       f"axis check: corr(PHATE1,maturation)={prov['corr_phate1_maturation']:+.3f} (validated) ; "
       f"corr(n_genes,PHATE1)={prov['corr_phate1_ngenes']:+.3f}, "
       f"corr(n_genes,maturation)={prov['corr_ngenes_maturation']:+.3f}  "
       f"→ A* driver (n_genes) ⊥ evaluation axis (maturation).")
fig.suptitle("Correctly-computed PHATE A* paths in Perturb-seq", fontsize=11, y=1.06)
fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=6.3, color="#333333")

for ext in ("pdf", "png"):
    fig.savefig(FIG / f"supplementary_figure2_phate_paths.{ext}", bbox_inches="tight",
                dpi=(200 if ext == "png" else 300))
print(f"[done] -> {FIG}/supplementary_figure2_phate_paths.pdf (editable) + .png")
