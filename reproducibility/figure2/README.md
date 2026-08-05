# Figure 2 reproducibility

This directory documents how to recompute the numerical results used in
Figure 2. The benchmark implementation and graph kernels remain in their
canonical repository locations and are not duplicated here.

## Analysis implemented

Figure 2 evaluates per-cell KEGG pathway activity for six immune lineages:
CD4 T cells, CD8 T cells, B cells, NK cells, macrophages and conventional
dendritic cells. For each lineage, the expected pathway activity is evaluated
by one-vs-rest AUROC.

The benchmark compares:

- `Edge_generic`: edge-L2 activity on the full retained KEGG pathway;
- `AUCell_generic`: AUCell activity on the genes from the same pathway;
- `Edge_curated`: edge-L2 activity after selecting the ten KEGG edges with the
  largest target-lineage versus remaining-cell contribution difference;
- `Edge_augment`: the curated edge set supplemented with synthetic pairwise
  edges among canonical lineage-marker genes;
- `AUCell_augment`: AUCell applied to the genes represented in the augmented
  edge set.

The marker-pair edges are scoring constructs and should not be interpreted as
individually validated or directed biological interactions. Edge selection and
AUROC evaluation use the same annotated cells, so the curated and augmented
results are within-dataset refinement diagnostics rather than held-out
classification estimates.

## Canonical source code

Run the benchmark directly from its canonical location:

- `figure2/bench_kegg_full.py`: benchmark configuration, edge refinement,
  AUCell comparison, AUROC calculation, timing and result export;
- `server/graph_utils.py`: KEGG parsing and graph-computation interface;
- `rust/cwg_rust/`: Rust implementation of sparse edge-L2 calculations;
- `vendor/keggx/`: KEGG parser and the pathway resources used by the benchmark.

All paths above are relative to the repository root.

## Input data

The analysis expects:

```text
data/Lung_cancer_Imm_filtered.h5ad
```

The analysis-ready object used for the reported run contains 50,000 cells and
17,764 genes and was derived from the immune compartment of the LuCA core
atlas. The object must provide:

```text
adata.X
adata.obs["cell_type_predicted"]
adata.var["feature_name"]
```

The source object records the LuCA publication DOI
`10.1016/j.ccell.2022.10.008` and the CELLxGENE-distributed dataset URL:

```text
https://datasets.cellxgene.cziscience.com/c1870f1f-ca36-4d96-b03b-7dc0e96d83ee.h5ad
```

The analysis-ready file is approximately 2.1 GB and is therefore not stored in
the standard Git repository. Place it at the path above or pass another path
with `--h5ad`.

## Environment

Install the repository dependencies and build the Rust extension from the
repository root:

```bash
python -m pip install -r requirements.txt
cd rust/cwg_rust
maturin develop --release
cd ../..
```

The benchmark requires Python 3.10 or later and a working Rust toolchain.
Relevant Python packages include `anndata`, `numpy`, `pandas`, `scipy`,
`scikit-learn`, `decoupler`, `matplotlib` and `maturin`. Exact versions are
recorded in the repository `requirements.txt`.

## Recompute Figure 2 results

From the repository root, run:

```bash
python figure2/bench_kegg_full.py \
  --h5ad data/Lung_cancer_Imm_filtered.h5ad \
  --celltype-col cell_type_predicted \
  --feature-col feature_name \
  --kegg-dir vendor/keggx/keggx/data/KEGG_Graph_processing \
  --outdir reproducibility/figure2/results
```

The default benchmark parameters are:

```text
minimum pathway genes: 5
curated edge budget: 10
label matching: case-insensitive literal substring
```

## Numerical outputs

The command writes the following calculation outputs:

- `input_metadata.json`: input provenance, dimensions and benchmark settings;
- `lineage_metadata.csv`: lineage cell counts, matched KEGG pathways and edge
  counts;
- `auroc_table.csv`: lineage-by-method one-vs-rest AUROC values;
- `timing_table.csv`: measured wall-clock time for each scoring stage;
- `active_edges.csv`: curated KEGG edges and their lineage-specificity values.

The same command also produces PDF, SVG and PNG visualizations. Those files are
derived outputs; the CSV and JSON files above are the primary numerical records
for verification.

## Runtime interpretation

Runtime values are machine-dependent and should not be expected to match
exactly across hardware. The timing panel reports the measured core stages in
this benchmark workflow. Generic runs score all retained KEGG pathways, whereas
curated and augmented runs score smaller lineage-specific edge sets. The
runtime comparison should therefore be interpreted as practical workflow
latency under the stated benchmark configuration, not as a resource-matched
comparison of equivalent algorithms.

