#!/usr/bin/env python
"""figure_edge_axis_cwg_peredge.py

Correction of figure_edge_axis_cwg.py using **per-edge normalization**.

Problem: the CWG norm accumulates over pathway edges, so the prior track (many
    edges) outranks A* (few edges) purely by edge count (= count confound).
Correction: the norm is an **L2 accumulation**, norm = sqrt(sum over edges), and
    therefore scales with the square root of the edge count
    (selective_integrated.rs:884-893). Accordingly we divide by **sqrt(n)**, not by
    n (= RMS per edge). Dividing by n would over-correct edge-rich sets by a factor
    of sqrt(n).
    -> A* and prior are compared fairly on a per-edge activity basis.

Tracks: A*/prior (per-edge, averaged across TFs) + injected IRX4 / held-out NRG1
(per-edge).
"""
import argparse, os, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["font.size"]=9
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, LinearSegmentedColormap
import anndata as ad
from cwg_rust import KEGGPathway, compute_all_kegg_norms_sparse

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description="Generate per-edge CWG activity profiles and Supplementary Figure 5.")
parser.add_argument("--h5ad",type=Path,required=True)
parser.add_argument("--dorothea",type=Path,default=ROOT.parent/"shared_inputs"/"dorothea_ABC_human.parquet")
parser.add_argument("--astar-edges",type=Path,default=ROOT/"inputs"/"dorothea_astar_ranked_edges.csv")
parser.add_argument("--out-dir",type=Path,default=ROOT/"results")
args=parser.parse_args()
args.out_dir.mkdir(parents=True,exist_ok=True)
SOURCE=args.out_dir/"source_data"; SOURCE.mkdir(parents=True,exist_ok=True)
H5AD,DORO,ASTAR=args.h5ad,args.dorothea,args.astar_edges
OUT_PDF=args.out_dir/"supplementary_figure5_edge_activity_per_edge.pdf"
HEART=["GATA4","GATA6","MEF2A","MEF2C","SRF","NR2F2","TEAD1","NKX2-5","TBX5"]
IRX4_SRC=["TBX5","NKX2-5","TBX20","GATA4","NKX2-6"]; TOPK,NBIN,SEED=20,20,0
COL={"A*":"#2a78d6","prior":"#eda100","injected IRX4":"#008300","gold NRG1":"#4a3aa7","random":"#898781"}
INK,MUTED="#0b0b0b","#898781"
SEQ=LinearSegmentedColormap.from_list("seq",["#f0efec","#8bb9e8","#2a78d6","#0b3b6b"])

def pw(edges,name):
    s=[str(e[0]) for e in edges];t=[str(e[1]) for e in edges];w=[float(e[2]) for e in edges];n=len(s)
    return KEGGPathway(name=name,sources=s,targets=t,weights=w,modifications=[""]*n,effects=[0]*n,types=[""]*n,indirects=[False]*n)

print("[load]",H5AD,flush=True)
A=ad.read_h5ad(H5AD,backed="r"); obs=A.obs
isc=obs["is_control"].to_numpy().astype(bool) if "is_control" in obs else obs["is_control_use"].to_numpy().astype(bool)
axis=pd.to_numeric(obs["score_maturation"],errors="coerce").to_numpy()
ph=np.asarray(A.obsm["X_phate"]); vp={g:i for i,g in enumerate(A.var_names)}
doro=pd.read_parquet(DORO).rename(columns={"source":"tf","weight":"mor"}); doro["ms"]=np.sign(doro["mor"]).replace(0,1)
astar=pd.read_csv(ASTAR); astar=astar[astar["method"]=="astar_path"]
tfs=[t for t in HEART if t in set(astar.source)]

rng=np.random.default_rng(SEED); genes=np.array(A.var_names)
pathways=[]; nedge={}; agg={"A*":[],"prior":[],"random":[]}
erec=[]   # source data: one row per edge that actually entered the computation
for tf in tfs:
    a=(astar[astar.source==tf].sort_values("rank").head(TOPK)
       .merge(doro[doro.tf==tf][["target","ms"]],on="target",how="inner"))
    a_t=[(g,s) for g,s in zip(a.target,a.ms) if g in vp]
    if len(a_t)<2: continue
    arank=dict(zip(a.target,a["rank"])) if "rank" in a else {}
    ascore=dict(zip(a.target,a["score"])) if "score" in a else {}
    aai=dict(zip(a.target,a["alpha_i"])) if "alpha_i" in a else {}
    aaj=dict(zip(a.target,a["alpha_j"])) if "alpha_j" in a else {}
    abeta=dict(zip(a.target,a["mean_beta"])) if "mean_beta" in a else {}
    p=doro[doro.tf==tf]; p_t=[(g,s) for g,s in zip(p.target,p.ms) if g in vp]
    k=len(a_t); rg=rng.choice(genes,size=k,replace=False); rs=rng.choice([1,-1],size=k)
    for trk,tset in [("A*",a_t),("prior",p_t),("random",list(zip(rg,rs)))]:
        nm=f"{trk}__{tf}"; pathways.append(pw([[tf,g,s] for g,s in tset],nm))
        nedge[nm]=len(tset); agg[trk].append(nm)
        erec+= [dict(track=trk,pathway=nm,tf_group=tf,source=tf,target=str(g),mor_sign=int(s),
                     astar_rank=arank.get(g,np.nan),astar_score=ascore.get(g,np.nan),
                     alpha_i=aai.get(g,np.nan),alpha_j=aaj.get(g,np.nan),mean_beta=abeta.get(g,np.nan))
                for g,s in tset]
if "IRX4" in vp:
    e=[[s,"IRX4",1] for s in IRX4_SRC if s in vp]; pathways.append(pw(e,"IRX4")); nedge["IRX4"]=len(e)
    erec+=[dict(track="injected IRX4",pathway="IRX4",tf_group="IRX4",source=str(s),target="IRX4",
                mor_sign=1,astar_rank=np.nan,astar_score=np.nan) for s,_,_ in e]
if "NRG1" in vp and "NKX2-5" in vp:
    pathways.append(pw([["NKX2-5","NRG1",-1]],"NRG1")); nedge["NRG1"]=1
    erec.append(dict(track="gold NRG1",pathway="NRG1",tf_group="NRG1",source="NKX2-5",target="NRG1",
                     mor_sign=-1,astar_rank=np.nan,astar_score=np.nan))

union=sorted({s for p in pathways for s in p.sources}|{t for p in pathways for t in p.targets}); union=[g for g in union if g in vp]
X=A[:,union].to_memory().X; X=X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X)); X.sort_indices()
raw={k:np.asarray(v,float) for k,v in compute_all_kegg_norms_sparse(X,union,pathways).items()}
# per-edge normalization: the CWG norm is an L2 accumulation over edges,
#   sqrt(sum(...)), so it scales with the square root of the edge count
#   (selective_integrated.rs:884-893  norm = sqrt(sum_edges alpha_i*alpha_j/alpha_g^2 * beta^2))
#   -> to make it count-invariant we divide by sqrt(n), not by n (= RMS per edge).
norms={k:(raw[k]/np.sqrt(max(nedge[k],1))) for k in raw}

tracks={}
for trk,names in agg.items():
    mats=[norms[n] for n in names if n in norms]
    if mats: tracks[trk]=np.mean(mats,0)
if "IRX4" in norms: tracks["injected IRX4"]=norms["IRX4"]
if "NRG1" in norms: tracks["gold NRG1"]=norms["NRG1"]

print("\n=== per-edge CWG / sqrt(n) (RMS per edge) by track: (mean edge count, per-cell mean, corr with axis) ===")
for name in ["A*","prior","random","injected IRX4","gold NRG1"]:
    if name in tracks:
        ne=np.mean([nedge[n] for n in agg.get(name,[])]) if name in agg else nedge.get({"injected IRX4":"IRX4","gold NRG1":"NRG1"}.get(name,name),1)
        print(f"  {name:16s}: n_edges~{ne:5.1f}  mean={tracks[name].mean():.5f}  corr={np.corrcoef(tracks[name],axis)[0,1]:+.3f}")

eb=np.quantile(axis,np.linspace(0,1,NBIN+1)); eb[-1]+=1e-9
binid=np.clip(np.digitize(axis,eb)-1,0,NBIN-1); xc=0.5*(eb[:-1]+eb[1:])
prof={}
for name,v in tracks.items():
    m=np.full(NBIN,np.nan); ci=np.full(NBIN,np.nan)
    for b in range(NBIN):
        s=v[binid==b]
        if len(s): m[b]=s.mean(); ci[b]=1.96*s.std()/np.sqrt(len(s))
    prof[name]=(m,ci)

ORDER=["A*","prior","random","injected IRX4","gold NRG1"]

# ================= source data export (edges / bins / per-cell) =================
NM={"injected IRX4":"IRX4","gold NRG1":"NRG1"}
pd.DataFrame(erec).to_csv(SOURCE/"edges_used.csv",index=False)

cnt=np.bincount(binid,minlength=NBIN)
brows=[]
for name in [n for n in ORDER if n in prof]:
    m,ci=prof[name]
    for b in range(NBIN):
        s=tracks[name][binid==b]
        brows.append(dict(track=name,bin=b+1,axis_lo=eb[b],axis_hi=eb[b+1],axis_center=xc[b],
                          n_cells=int(cnt[b]),mean=m[b],sd=float(s.std()) if len(s) else np.nan,
                          sem=float(s.std()/np.sqrt(len(s))) if len(s) else np.nan,
                          ci95_lo=m[b]-ci[b],ci95_hi=m[b]+ci[b]))
pd.DataFrame(brows).to_csv(SOURCE/"bin_profile.csv",index=False)

cell=pd.DataFrame({"cell_id":np.asarray(obs.index),"score_maturation":axis,"bin":binid+1,
                   "is_control":isc,"target_gene":obs["target_gene"].astype(str).to_numpy(),
                   "phate1":ph[:,0],"phate2":ph[:,1]})
for name in [n for n in ORDER if n in tracks]:
    cell[f"activity__{name.replace(' ','_').replace('*','star')}"]=tracks[name]
cell.to_parquet(SOURCE/"percell_activity.parquet",index=False)

srows=[]
for name in [n for n in ORDER if n in tracks]:
    ne=[nedge[n] for n in agg[name]] if name in agg else [nedge[NM.get(name,name)]]
    v=tracks[name]
    srows.append(dict(track=name,n_pathways=len(ne),n_edges_total=int(np.sum(ne)),n_edges_mean=float(np.mean(ne)),
                      cell_mean=float(v.mean()),cell_sd=float(v.std()),
                      pearson_r_axis=float(np.corrcoef(v,axis)[0,1]),
                      spearman_r_axis=float(pd.Series(v).corr(pd.Series(axis),method="spearman"))))
pd.DataFrame(srows).to_csv(SOURCE/"track_summary.csv",index=False)
print("[sourcedata]",SOURCE,"edges_used / bin_profile / percell_activity / track_summary",flush=True)

fig=plt.figure(figsize=(9.5,5.2))
gs=GridSpec(2,2,width_ratios=[1.55,1.0],height_ratios=[1,1],wspace=0.28,hspace=0.35,left=0.08,right=0.97,top=0.92,bottom=0.12)
axp=fig.add_subplot(gs[:,0])
for name in [n for n in ORDER if n in prof]:
    m,ci=prof[name]; c=COL[name]; ls="--" if name=="random" else "-"
    axp.fill_between(xc,m-ci,m+ci,color=c,alpha=0.15,linewidth=0)
    axp.plot(xc,m,color=c,lw=2,ls=ls,zorder=3)
    j=np.where(~np.isnan(m))[0][-1]
    axp.annotate(name,(xc[j],m[j]),xytext=(4,0),textcoords="offset points",color=c,fontsize=8,va="center",fontweight="bold")
axp.set_xlabel("maturation axis  (score_maturation -> mature CM)",color=INK)
axp.set_ylabel("edge activity  (CWG G2 norm / sqrt(#edges) = RMS per edge)",color=INK)
axp.set_title("CWG per-edge (L2-consistent, /sqrt n) — A* vs prior, count confound removed",color=INK,fontsize=10)
axp.spines[["top","right"]].set_visible(False); axp.spines[["left","bottom"]].set_color("#c3c2b7"); axp.tick_params(colors=MUTED)

def panel(ax,val,title):
    vmax=np.nanpercentile(val,99) or 1.0; o=np.argsort(val)
    scp=ax.scatter(ph[o,0],ph[o,1],c=val[o],cmap=SEQ,norm=Normalize(0,vmax),s=1.5,linewidths=0,rasterized=True)
    ax.set_title(title,color=INK,fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_color("#c3c2b7")
    cb=fig.colorbar(scp,ax=ax,fraction=0.046,pad=0.02); cb.ax.tick_params(labelsize=7,colors=MUTED); cb.outline.set_edgecolor("#c3c2b7"); cb.set_label("norm/sqrt(n)",fontsize=7,color=MUTED)
if "injected IRX4" in tracks: panel(fig.add_subplot(gs[0,1]),tracks["injected IRX4"],"PHATE — injected IRX4 (per-edge)")
if "gold NRG1" in tracks: panel(fig.add_subplot(gs[1,1]),tracks["gold NRG1"],"PHATE — held-out NRG1 (per-edge)")
fig.savefig(OUT_PDF); fig.savefig(OUT_PDF.with_suffix(".png"),dpi=200); plt.close(fig)
print("\n[saved]",OUT_PDF,"(+ .png)")
