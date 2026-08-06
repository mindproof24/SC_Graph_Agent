# Figure 5 Scoring

## Reference labels

The majority `C2S_celltype_final` label in each Leiden cluster was used as the external reference. Clusters `0`, `3`, and `12` were microglial-cell-majority populations. Cluster `18` was a mixed lymphoid cluster whose two largest C2S populations were T cells (201 of 420 cells; 47.86%) and mature NK T cells (75 of 420 cells; 17.86%). Its reference label was assigned from the largest population, T cell. Exact majority counts and fractions are in `results/figure5_truth_by_cluster.csv`.

## Answer scopes

- **Visible**: all recorded assistant and thinking events in a trajectory. This measures whether a biologically relevant interpretation appeared during the analysis.
- **Final**: recorded assistant messages only. This measures whether the interpretation was retained in the answer presented to the user.

The JSONL files were generated with `SHOW_THINKING=1`, which is why thinking events are available for visible-trajectory scoring.

## Cell-lineage scores

Each run received one prediction for each of the four clusters. The run score is the mean of the four binary cluster scores.

- **Strict**: the parsed prediction must match the C2S majority reference class: `microglial cell` for clusters `0`, `3`, and `12`, and `T cell` for cluster `18`.
- **Lineage-compatible**: a microglial-cell reference also accepts `myeloid cell`, `macrophage`, or `monocyte`. Dendritic-cell predictions receive zero for a microglial-cell reference.

For cluster `18`, the deterministic parser maps explicit T-cell terms and T-lineage markers to the `T cell` reference class. It also maps NK, NKT, cytotoxic and general lymphoid terms to that class because C2S identifies cluster `18` as a mixed T/NK-lineage population rather than a homogeneous T-cell population. Thus, a cluster-18 match should be interpreted as recovery of the dominant T/NK-associated lymphoid identity, not as proof that every cell in the cluster is a conventional T cell.

`scoring/score_figure5_c2s.py` implements the deterministic text-window extraction, label mapping, strict scoring, and lineage-compatible scoring used for reproducibility.

## Biological-context retention

Context retention was scored automatically from the last recorded `assistant` response. The final Figure 5 comparison used the same ordinal 0-2 keyword rule for both context-window settings:

- **2**: at least one GBM/cancer term and at least one immune-context term are detected.
- **1**: an immune-context term is detected without a GBM/cancer term.
- **0**: no immune-context term is detected.

This was not a human audit. The original calculation was executed as session-local Python and was not initially retained as a standalone script. Its exact logic has been restored in `scoring/score_context_retention.py` from the recorded command history.

The GBM/cancer expression was `glioblastoma|gbm|tumor/tumour|cancer|malignan|neoplasm`. The immune expression was `inflamm|immune|microgl|macrophage|myeloid|t cell|cytotoxic|antigen|hla|interferon|apc`. This is a deterministic lexical diagnostic, not biological adjudication of the response.

For combined metrics, `context_norm = context_score / 2`. The reported products are calculated per run and then averaged:

```text
visible lineage x context = visible_lineage * context_norm
final lineage x context   = final_lineage * context_norm
```

Tool calls and tool errors are descriptive trajectory-burden measures and do not contribute bonus points to biological accuracy.

## Complete 60-run score table

`VS`, `VL`, `FS`, `FL`, and `C` denote visible strict, visible lineage, final strict, final lineage, and context score, respectively.

| Context | Model | Run | VS | VL | FS | FL | C |
|---|---|---|---:|---:|---:|---:|---:|
| 16k | base | `base_r01` | 0.25 | 0.50 | 0.25 | 0.25 | 1 |
| 16k | base | `base_r02` | 0.25 | 0.75 | 0.00 | 0.75 | 1 |
| 16k | base | `base_r03` | 0.25 | 0.75 | 0.25 | 0.25 | 2 |
| 16k | base | `base_r04` | 0.75 | 1.00 | 0.00 | 0.00 | 1 |
| 16k | base | `base_r05` | 0.00 | 0.00 | 0.00 | 0.00 | 2 |
| 16k | base | `base_r06` | 0.25 | 0.25 | 0.25 | 0.25 | 0 |
| 16k | base | `base_r07` | 0.75 | 1.00 | 0.50 | 0.75 | 1 |
| 16k | base | `base_r08` | 0.25 | 0.50 | 0.25 | 0.50 | 2 |
| 16k | base | `base_r09` | 0.25 | 0.25 | 0.25 | 0.25 | 1 |
| 16k | base | `base_r10` | 0.25 | 0.25 | 0.25 | 0.25 | 1 |
| 16k | base | `base_r11` | 0.75 | 0.75 | 0.50 | 0.75 | 1 |
| 16k | base | `base_r12` | 0.25 | 0.75 | 0.25 | 0.75 | 1 |
| 16k | base | `base_r13` | 0.25 | 0.25 | 0.25 | 0.25 | 2 |
| 16k | base | `base_r14` | 0.00 | 0.25 | 0.00 | 0.00 | 0 |
| 16k | base | `base_r15` | 0.25 | 0.75 | 0.25 | 0.75 | 1 |
| 16k | trained | `trained_r01` | 0.25 | 0.75 | 0.25 | 0.75 | 1 |
| 16k | trained | `trained_r02` | 0.50 | 0.50 | 0.25 | 0.25 | 2 |
| 16k | trained | `trained_r03` | 0.25 | 1.00 | 0.00 | 0.00 | 0 |
| 16k | trained | `trained_r04` | 0.25 | 0.25 | 0.00 | 0.00 | 1 |
| 16k | trained | `trained_r05` | 0.50 | 0.75 | 0.25 | 1.00 | 1 |
| 16k | trained | `trained_r06` | 0.50 | 0.50 | 0.50 | 0.50 | 1 |
| 16k | trained | `trained_r07` | 0.25 | 0.25 | 0.25 | 0.50 | 1 |
| 16k | trained | `trained_r08` | 0.25 | 0.25 | 0.00 | 0.00 | 1 |
| 16k | trained | `trained_r09` | 1.00 | 1.00 | 0.75 | 0.75 | 1 |
| 16k | trained | `trained_r10` | 0.00 | 0.00 | 0.00 | 0.00 | 1 |
| 16k | trained | `trained_r14` | 0.25 | 0.50 | 0.25 | 0.50 | 2 |
| 16k | trained | `trained_r15` | 0.00 | 0.00 | 0.00 | 0.25 | 2 |
| 16k | trained | `trained_suppC_r01` | 0.25 | 0.25 | 0.50 | 0.75 | 1 |
| 16k | trained | `trained_suppC_r02` | 0.25 | 0.25 | 0.25 | 1.00 | 1 |
| 16k | trained | `trained_suppC_r03` | 0.50 | 0.75 | 0.50 | 0.75 | 1 |
| 32k | base | `base32_r01` | 0.75 | 1.00 | 0.50 | 1.00 | 2 |
| 32k | base | `base32_r02` | 0.75 | 0.75 | 0.75 | 0.75 | 2 |
| 32k | base | `base32_r03` | 0.75 | 1.00 | 0.75 | 1.00 | 2 |
| 32k | base | `base32_r04` | 0.25 | 0.25 | 0.25 | 0.25 | 2 |
| 32k | base | `base32_r05` | 0.75 | 0.75 | 0.75 | 0.75 | 1 |
| 32k | base | `base32_r06` | 0.25 | 0.75 | 0.25 | 0.75 | 1 |
| 32k | base | `base32_r07` | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| 32k | base | `base32_r08` | 0.25 | 0.50 | 0.00 | 0.00 | 1 |
| 32k | base | `base32_r09` | 0.75 | 1.00 | 0.75 | 0.75 | 2 |
| 32k | base | `base32_r10` | 0.50 | 0.75 | 0.50 | 1.00 | 2 |
| 32k | base | `base32_r11` | 1.00 | 1.00 | 0.00 | 0.50 | 2 |
| 32k | base | `base32_r12` | 0.25 | 0.75 | 0.25 | 0.75 | 1 |
| 32k | base | `base32_r13` | 0.75 | 0.75 | 0.75 | 0.75 | 2 |
| 32k | base | `base32_r14` | 0.75 | 1.00 | 0.75 | 1.00 | 2 |
| 32k | base | `base32_r15` | 0.75 | 0.75 | 0.75 | 0.75 | 2 |
| 32k | trained | `trained32_extra_r01` | 0.75 | 1.00 | 0.50 | 0.75 | 2 |
| 32k | trained | `trained32_extra_r02` | 0.75 | 0.75 | 1.00 | 1.00 | 2 |
| 32k | trained | `trained32_extra_r03` | 0.75 | 0.75 | 0.50 | 0.75 | 2 |
| 32k | trained | `trained32_extra_r04` | 0.50 | 0.75 | 0.25 | 0.50 | 2 |
| 32k | trained | `trained32_extra_r05` | 1.00 | 1.00 | 0.50 | 0.75 | 2 |
| 32k | trained | `trained32_r01` | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| 32k | trained | `trained32_r02` | 0.50 | 0.75 | 0.25 | 0.50 | 2 |
| 32k | trained | `trained32_r03` | 0.25 | 0.25 | 0.25 | 0.50 | 2 |
| 32k | trained | `trained32_r04` | 0.25 | 0.25 | 0.50 | 0.75 | 2 |
| 32k | trained | `trained32_r05` | 0.50 | 0.75 | 0.50 | 0.75 | 2 |
| 32k | trained | `trained32_r06` | 0.50 | 0.75 | 0.50 | 0.75 | 2 |
| 32k | trained | `trained32_r07` | 0.50 | 0.75 | 0.50 | 0.75 | 2 |
| 32k | trained | `trained32_r08` | 0.75 | 1.00 | 0.75 | 1.00 | 2 |
| 32k | trained | `trained32_r09` | 1.00 | 1.00 | 0.00 | 0.00 | 0 |
| 32k | trained | `trained32_r10` | 0.50 | 1.00 | 0.25 | 0.75 | 2 |
```

Tool calls and tool errors are descriptive trajectory-burden measures and do not contribute bonus points to biological accuracy.
