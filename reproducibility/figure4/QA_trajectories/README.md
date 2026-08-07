# Figure 4 Key Interactive Trajectories

This folder preserves the main JSONL trajectories used to interpret the Figure 4
GBM cluster 18 case study. The original files remain in `analysis_logs/`; these
copies are renamed by role so the relevant trajectories can be found without
searching through all logs again.

## Files

| File | Role in Figure 4 interpretation |
|---|---|
| `main_cluster18_lymphoid_bcell_apc_dissection_20260716_111025.jsonl` | Main trajectory for the cluster 18 lymphoid dissection. It identifies cluster 18 as lymphocyte enriched, tests memory-like T, NK, NKT and APC-like modules, then follows the B-cell/APC question using observed genes only. |
| `explicit_cluster18_t_nk_nkt_bapc_dissection_20260715_174514.jsonl` | Explicit trajectory where the guidance asks not to assign a single cell type to cluster 18 and to test separable T-cell, NK/cytotoxic, NKT and B-cell/APC subprograms. This is the clearest log for the heterogeneous-lymphoid framing. |
| `supporting_cluster18_t_nk_nkt_apc_dissection_20260716_003656.jsonl` | Supporting trajectory that also redirects analysis to cluster 18 and builds T-cell, NK/cytotoxic, NKT and APC custom edge modules. |

## Key Anchors

- Main Fig. 4 trajectory:
  - `figure_20260716_111025.jsonl`, lines 53-54:
    Ctrl-G guidance restricts the analysis to cluster 18 and asks for
    `custom_pathway_calc(scale='cell', cluster_id='18', cluster_key='leiden')`
    modules for memory T cells, NK cells, NKT cells and
    APC signal.
  - `figure_20260716_111025.jsonl`, lines 99-100:
    Ctrl-G guidance refines the B-cell/APC question by avoiding the generic
    `CD74 -> HLA-DRA` edge and testing more B-cell edges.

- Explicit heterogeneous-lymphoid trajectory:
  - `figure_20260715_174514.jsonl`, lines 86-87:
    Ctrl-G guidance states that cluster 18 is lymphoid enriched and asks to
    test separable T-cell, NK/cytotoxic, NKT and B-cell/APC
    subprograms at single-cell resolution.
  - `figure_20260715_174514.jsonl`, line 109:
    The trajectory records a working subdivision of cluster 18 into NK,
    T-cell, NKT and APC candidate groups.

- Supporting trajectory:
  - `figure_20260716_003656.jsonl`, lines 60-61:
    Ctrl-G guidance stops analysis drifting to cluster 0 and redirects the
    agent to cluster 18 T-cell/lymphoid heterogeneity.
  - `figure_20260716_003656.jsonl`, lines 68-74:
    The trajectory describes observed T-cell, NK/cytotoxic, NKT and
    APC marker/edge programs before running custom pathway scoring.

## Interpretation Boundary

These logs support the claim that the interactive agent can move from a broad
cluster-level annotation to a more focused, single-cell-level hypothesis test
within a heterogeneous lymphoid cluster. They should not be described as proving
that cluster 18 is purely NKT, purely B cell or purely APC. The correct
description is that cluster 18 is lymphoid enriched, dominated by T-cell signal,
with smaller NK, APC and B-cell hypotheses tested through
observed marker expression and custom TF-target/pathway edge activity.

