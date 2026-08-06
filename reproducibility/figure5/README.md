# Figure 5 Reproducibility

This directory reproduces the Figure 5 comparison of identity assignment and biological-context retention in interactive glioblastoma scRNA-seq analysis.

## Design

The same cluster-annotation question was evaluated in four cohorts:

| Context window | Base | Trained |
|---|---:|---:|
| 16k (`num_ctx=16384`) | 15 | 15 |
| 32k (`num_ctx=32768`) | 15 | 15 |

The benchmark therefore contains 60 independent completed trajectories. The exact question is in `question/QUESTION.md`, and each JSONL trajectory is stored separately under `trajectories/{16k,32k}/{base,trained}/`.

## Scoring

`scoring/score_figure5_c2s.py` calculates identity scores against the C2S majority reference in `results/figure5_truth_by_cluster.csv`.

- `visible_strict` and `visible_lineage` use user-visible intermediate reasoning and assistant messages.
- `final_strict` and `final_lineage` use only the last assistant response.
- Strict scoring requires the parsed prediction to match the C2S majority reference class.
- Lineage-compatible scoring accepts predefined biologically related labels described in `scoring/SCORING.md`.

Clusters 0, 3, and 12 were microglial-cell-majority populations. Cluster 18 was a mixed lymphoid population whose two largest C2S annotations were T cells (47.86%) and mature NK T cells (17.86%). Accordingly, T-, NK-, NKT-, cytotoxic- and general lymphoid-associated expressions are mapped to the cluster-18 T/NK-associated reference class. This score indicates recovery of the dominant lymphoid identity, not homogeneous conventional T-cell identity.

`scoring/score_context_retention.py` calculates a lexical context score from the last assistant response:

- `2`: both GBM/cancer and immune-context terms are present.
- `1`: immune-context terms are present without a GBM/cancer term.
- `0`: immune-context terms are absent.

The context score is automatic and independent of identity correctness. It is not a human biological assessment.

## Results

The two published result tables are:

- `results/figure5_all_run_identity_context_scores.csv`: identity and context scores for all 60 trajectories.
- `results/figure5_cohort_summary.csv`: means for the four model/context cohorts.

The complete 60-run table is also reproduced in `scoring/SCORING.md` for direct inspection.

The run-level table contains visible and final identity scores alongside the context score. Cohort means are:

| Context | Model | n | Visible strict | Visible lineage | Final strict | Final lineage | Context |
|---|---|---:|---:|---:|---:|---:|---:|
| 16k | Base | 15 | 0.317 | 0.533 | 0.217 | 0.383 | 1.133 |
| 16k | Trained | 15 | 0.333 | 0.467 | 0.250 | 0.467 | 1.133 |
| 32k | Base | 15 | 0.633 | 0.800 | 0.533 | 0.733 | 1.733 |
| 32k | Trained | 15 | 0.633 | 0.783 | 0.483 | 0.700 | 1.867 |

## Recompute

Run from the repository root:

```bash
python reproducibility/figure5/scoring/score_figure5_c2s.py
python reproducibility/figure5/scoring/score_context_retention.py
```

These commands regenerate detailed component tables under `recomputed/`. This directory contains temporary verification outputs and does not need to be committed. The compact verified outputs used for Figure 5 are stored under `results/`.

## Input data

The trajectories used sample ID `figure`, a four-cluster glioblastoma AnnData object containing 4,889 cells and 15,188 genes. The C2S-annotated object is shared with the Figure 4 validation bundle and is not duplicated here.
