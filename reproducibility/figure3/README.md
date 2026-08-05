# Figure 3 reproducibility

This directory contains the core calculation scripts and numerical records
used in Figure 3. The copies are organized under `scripts/` and `results/` so
that the reported comparison can be inspected independently of plotting code.

## Analysis implemented

Figure 3 compares ranked transcription factor (TF)-target edges produced for
seven clusters (`0`, `1`, `2`, `3`, `6`, `9` and `10`) from the
`imm_atlas_TT_p01` T-cell atlas subset. Three workflows are evaluated:

- `astar_path`: trajectory-aware ranking of prior TF-target edges along A*
  paths in the PHATE neighborhood geometry;
- `scenic`: GRNBoost2 network inference followed by cisTarget motif and
  regulatory-context pruning;
- `celloracle`: cluster-specific link estimation after PCA, KNN imputation and
  ridge-regression modeling with a promoter base GRN.

The three outputs are restricted to the same directed human CollecTRI
TF-target pair set before ranking. Comparisons use top-k directed-edge overlap,
pairwise Jaccard similarity, cluster specificity and Venn regions. This common
candidate set controls for differences in regulatory-reference coverage; it
does not make the three inference procedures algorithmically equivalent.

## Canonical source code

The core calculation scripts are:

- `reproducibility/figure3/scripts/benchmark_tf_target_tool.py`: A* path construction and
  trajectory-aware TF-target edge ranking;
- `reproducibility/figure3/scripts/run_scenic_for_astar_clusters.py`: cluster expression
  export, GRNBoost2 execution and cisTarget context pruning;
- `reproducibility/figure3/scripts/run_celloracle_for_astar_clusters.py`: promoter-GRN
  restriction, PCA/KNN imputation and cluster-specific CellOracle link
  estimation;
- `reproducibility/figure3/scripts/benchmark_collectri_methods.py`: schema normalization,
  directed CollecTRI restriction, within-method ranking, top-k Jaccard
  calculation and edge-specificity summaries.

The plotting-only scripts are intentionally separate:

- `figure3/scripts/make_figure3_editable_pdfs.py`;
- `figure3/scripts/plot_topk_edge_venn_v2.py`;
- `figure3/scripts/make_supplementary_figure1_venn.py`;
- `figure3/reproduce/run_figure3.sh`.

`run_figure3.sh` regenerates vector figures from deposited processed tables. It
does not rerun A*, SCENIC or CellOracle.

## Input data and external resources

The expression input is:

```text
data/imm_atlas_TT_p01.h5ad
```

The AnnData object must contain at least:

```text
adata.X
adata.obs["leiden"]
adata.obs["n_genes"]
adata.obsm["X_phate"]
```

Raw-method recomputation also requires:

- a fixed human CollecTRI source-target table;
- a working Rust A* graph extension from `rust/cwg_rust/`;
- pySCENIC and its human hg38 ranking databases and motif-annotation table;
- CellOracle and the `hg38_gimmemotifsv5_fpr2` promoter base GRN.

The SCENIC ranking databases and motif annotations are external resources and
are not stored in this repository. Runtime and inferred-edge values can vary
with package versions, thread counts and resource snapshots; these should be
recorded for every raw recomputation.

## Core calculation sequence

Run commands from the repository root. The paths below use `work/figure3/` for
new outputs so that deposited results are not overwritten.

### 1. A* TF-target ranking

```bash
python reproducibility/figure3/scripts/benchmark_tf_target_tool.py \
  --data-dir data \
  --samples imm_atlas_TT_p01 \
  --clusters 0,1,2,3,6,9,10 \
  --methods astar_path \
  --leiden-key leiden \
  --gene-col n_genes \
  --embedding-key X_phate \
  --min-paths 300 \
  --seed 13 \
  --out-dir work/figure3/astar
```

This writes one `*__edges.csv` file per cluster and
`work/figure3/astar/benchmark_summary.csv`.

### 2. SCENIC inference and context pruning

```bash
python reproducibility/figure3/scripts/run_scenic_for_astar_clusters.py \
  --h5ad data/imm_atlas_TT_p01.h5ad \
  --sampleid imm_atlas_TT_p01 \
  --benchmark-summary work/figure3/astar/benchmark_summary.csv \
  --collectri path/to/collectri_human.csv \
  --clusters 0,1,2,3,6,9,10 \
  --rankings path/to/hg38_ranking_database.feather \
  --annotations path/to/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
  --num-workers 4 \
  --seed 13 \
  --out-dir work/figure3/scenic
```

The runner records per-cluster `adjacencies.csv`, `ctx.csv`, logs and timing
summaries. The cisTarget regulons must then be exported as a long directed-edge
table with `sampleid`, `cluster_id`, `source`, `target` and `score` columns
before the common comparison step. The deposited harmonized edge table is the
authoritative record for the reported figure because this raw-to-long export
adapter is not currently packaged as a standalone script.

### 3. CellOracle link estimation

```bash
python reproducibility/figure3/scripts/run_celloracle_for_astar_clusters.py \
  --h5ad data/imm_atlas_TT_p01.h5ad \
  --sampleid imm_atlas_TT_p01 \
  --collectri path/to/collectri_human.csv \
  --clusters 0,1,2,3,6,9,10 \
  --base-grn-version hg38_gimmemotifsv5_fpr2 \
  --alpha 10 \
  --bagging-number 5 \
  --n-jobs 24 \
  --seed 0 \
  --out-dir work/figure3/celloracle
```

The combined compatible edge table is written to
`work/figure3/celloracle/celloracle_edges_combined.csv`.

### 4. Shared CollecTRI restriction and comparison

```bash
python reproducibility/figure3/scripts/benchmark_collectri_methods.py \
  --collectri path/to/collectri_human.csv \
  --astar-dir work/figure3/astar \
  --scenic work/figure3/scenic/scenic_edges_long.csv \
  --celloracle work/figure3/celloracle/celloracle_edges_combined.csv \
  --samples imm_atlas_TT_p01 \
  --topk 10,20,50,100,200 \
  --out-dir work/figure3/comparison
```

Column-name overrides documented by `--help` should be supplied if a SCENIC
or CellOracle table uses nonstandard field names.

## Primary numerical records

The reported Figure 3 calculations are copied to
`reproducibility/figure3/results/`. The main
verification files are:

- `normalized_edges_collectri_ranked.csv`: all harmonized directed edges,
  scores and within-cluster ranks;
- `pairwise_topk_jaccard.csv`: pairwise method overlap at each top-k cutoff;
- `method_cluster_summary.csv`: edge counts before and after CollecTRI
  restriction;
- `topk_edge_cluster_specificity.csv`: cluster recurrence and specificity of
  ranked edges;
- `runtime_by_cluster_method.csv`: timed core stages by cluster and method;
- `runtime_method_summary.csv`: method-level runtime summaries;
- `top200_venn_region_edges.csv`: directed edges assigned to each top-200 Venn
  region;
- `cluster_identity_top10_edges_3methods.csv`: representative high-ranking
  edges used for biological interpretation;
- `run_metadata.json`: benchmark scope and record counts.

The deposited metadata records 42,990 CollecTRI pairs, 938,943 raw method
edges and 23,112 retained method edges across the three workflows.

## Regenerate vector figures

After installing the packages in
`figure3/reproduce/requirements-plotting.txt`, run:

```bash
figure3/reproduce/run_figure3.sh
```

The generated PDF and SVG files are derived visualizations. The CSV and JSON
files above are the primary numerical records.

## Runtime interpretation

The runtime panel reports the timed core stages used to produce ranked edges
under each workflow. For A*, this includes graph construction and A* search;
for SCENIC, GRNBoost2 and cisTarget context pruning; and for CellOracle,
PCA/KNN preprocessing and promoter-GRN-restricted link estimation. It is a
comparison of practical workflow latency under the reported settings, not a
comparison of identical de novo GRN inference tasks. Absolute runtimes are
hardware-, thread- and software-version-dependent.
