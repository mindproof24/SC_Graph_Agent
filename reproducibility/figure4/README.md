# Figure 4 reproducibility

This directory contains the primary records and calculations supporting the
Figure 4 glioblastoma case study. It focuses on the interactive trajectory,
the external C2S reference and the custom B-cell edge activity calculation.


## Contents

```text
reproducibility/figure4/
├── README.md
├── trajectories/
│   ├── README.md
│   ├── main_cluster18_lymphoid_bcell_apc_dissection_20260716_111025.jsonl
│   ├── explicit_cluster18_t_nk_nkt_bapc_dissection_20260715_174514.jsonl
│   └── supporting_cluster18_t_nk_nkt_apc_dissection_20260716_003656.jsonl
├── scripts/
│   └── score_custom_edge_activity.py
├── inputs/
│   ├── b_cell_apc_edges.csv
│   └── b_cell_no_CD74_edges.csv
└── results/
    ├── cluster18_C2S_composition_full.csv
    ├── cluster18_C2S_composition_minor5_grouped.csv
    ├── custom_b_cell_apc_score_all_umap_top10.csv
    └── custom_b_cell_no_CD74_score_all_umap_top10.csv
```

## Case-study scope

The trajectories record how the agent identified Leiden cluster 18 as a
lymphocyte-enriched cluster and was then guided to test heterogeneous T-,
NK/NKT- and B-cell/APC-like programs at single-cell resolution. The JSONL files
retain the user questions, assistant messages, tool calls, tool responses and
Ctrl-G guiding messages used in the analysis.

The C2S cell-type annotations were not supplied to the agent. They were added
after the interaction and used only as an external reference for evaluating
the composition of cluster 18.

## Required AnnData object

The calculation and C2S summaries were derived from:

```text
figure_with_C2S_predicted.h5ad
```

The deposited source object used for this analysis has:

```text
shape: 4,889 cells x 15,188 genes
SHA-256: 4f9e6af6d8d02a34d04473076e96ffe2c4c4f75b4a3196483352c2484b8c2358
```

The object must contain:

```text
adata.X
adata.obs["leiden"]
adata.obs["C2S_pred_raw"]
adata.obs["C2S_celltype_final"]
adata.obsm["X_umap"]
```


## Edge definitions

`inputs/b_cell_apc_edges.csv` defines the broad module:

```text
BANK1 -> CD22       weight 1.0
EBF1  -> BANK1      weight 0.9
CD37  -> CD22       weight 0.8
CD74  -> HLA-DRA    weight 0.7
IGKC  -> CD74       weight 0.6
```

`inputs/b_cell_no_CD74_edges.csv` defines the refined module:

```text
BANK1 -> CD22       weight 1.0
EBF1  -> BANK1      weight 0.9
CD37  -> CD22       weight 0.8
```

The refined module removes CD74-linked antigen-presentation activity. These
edges are custom scoring relations used for hypothesis testing and should not
all be interpreted as independently validated directed regulatory relations.

## Edge activity calculation

`scripts/score_custom_edge_activity.py` calculates the per-cell edge-L2 value

```text
score(c) = sqrt(sum[(alpha_i alpha_j / alpha_G^2)
                    abs(w_ij + alpha_i + alpha_j)] over (i,j) in E)
```

where `alpha_i` and `alpha_j` are the source and target expression values and
`alpha_G` is the total expression value used for normalization. The script
requires an explicit CSV or JSON edge set

From the repository root, run:

```bash
python reproducibility/figure4/scripts/score_custom_edge_activity.py \
  --h5ad path/to/figure_with_C2S_predicted.h5ad \
  --edges-csv reproducibility/figure4/inputs/b_cell_apc_edges.csv \
  --cluster-key leiden \
  --cluster-id 18 \
  --score-col custom_b_cell_apc_score_all \
  --color-all-cells \
  --out-h5ad work/figure4/b_cell_apc_scored.h5ad \
  --out-prefix work/figure4/custom_b_cell_apc_score_all_umap

python reproducibility/figure4/scripts/score_custom_edge_activity.py \
  --h5ad path/to/figure_with_C2S_predicted.h5ad \
  --edges-csv reproducibility/figure4/inputs/b_cell_no_CD74_edges.csv \
  --cluster-key leiden \
  --cluster-id 18 \
  --score-col custom_b_cell_no_CD74_score_all \
  --color-all-cells \
  --out-h5ad work/figure4/b_cell_no_CD74_scored.h5ad \
  --out-prefix work/figure4/custom_b_cell_no_CD74_score_all_umap
```

The commands write a scored h5ad, a UMAP preview and a top-10 cell table. The
scored h5ad files are temporary derived objects and do not need to be deposited.

## Primary results

- `cluster18_C2S_composition_full.csv` contains all C2S labels, cell counts and
  percentages in cluster 18;
- `cluster18_C2S_composition_minor5_grouped.csv` groups labels representing no
  more than 5% of cluster 18 as minor cell types;
- `custom_b_cell_apc_score_all_umap_top10.csv` records the ten highest-scoring
  cells under the broad module;
- `custom_b_cell_no_CD74_score_all_umap_top10.csv` records the ten
  highest-scoring cells under the refined module.

Cluster 18 contains 420 cells in the external C2S reference. The largest
groups are T cells (47.86%), mature NK T cells (17.86%), progenitor cells
(7.14%), mature T cells (6.43%) and B cells (5.48%). These summaries support
the use of single-cell-level dissection rather than assigning one label to the
entire cluster.
