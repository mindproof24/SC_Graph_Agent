# UNSEEN Evaluation Summary

Models:
- `base`: `qwen35-base-8k` for the T-cell run, `qwen35-base-16k` for the B-cell run.
- `step9`: `qwen35-grpo-step9-16k`.

Scoring:
- Strict uses the original automatic `{"labels": [...]}` parser.
- Format-corrected audited scoring rescues explicit cell-wise classifications from malformed JSON/tables.
- One correct cell contributes 0.25.

The scoring function and audited regrade rules are documented in `SCORING.md`.
The exact user-facing prompts and answer labels are recorded in `QUESTIONS.md`.
Those prompts contain a stale `get_kg_context` mention from the task text template, but the actual runs used the NO_KG 9-tool harness; no result trajectory called `get_kg_context`.

## T-cell UNSEEN, Strict

| Model | UNSEEN overall | CD8 memory vs eff.mem CD8 | CD4 helper vs eff.mem CD4 | gamma-delta T vs Treg |
|---|---|---|---|---|
| base | 0.675 | 0.873 | 0.156 | 0.998 |
| step9 | 0.729 | 1.000 | 0.188 | 1.000 |
| delta | 0.054 | 0.127 | 0.031 | 0.002 |

## T-cell UNSEEN, Format-Corrected Audited

| Model | UNSEEN overall | CD8 memory vs eff.mem CD8 | CD4 helper vs eff.mem CD4 | gamma-delta T vs Treg |
|---|---|---|---|---|
| base | 0.733 | 0.935 | 0.266 | 0.998 |
| step9 | 0.734 | 1.000 | 0.203 | 1.000 |
| delta | 0.002 | 0.065 | -0.062 | 0.002 |

## B-cell UNSEEN, Strict

| Model | B-cell overall | memory B vs naive B | memory B vs plasmablast |
|---|---|---|---|
| base | 0.375 | 0.062 | 0.688 |
| step9 | 0.484 | 0.281 | 0.688 |
| delta | 0.109 | 0.219 | 0.000 |

## B-cell UNSEEN, Format-Corrected Audited

| Model | B-cell overall | memory B vs naive B | memory B vs plasmablast |
|---|---|---|---|
| base | 0.688 | 0.516 | 0.859 |
| step9 | 0.781 | 0.656 | 0.906 |
| delta | 0.094 | 0.141 | 0.047 |

## Combined UNSEEN

| Scoring | base | step9 | delta |
|---|---|---|---|
| Strict | 0.555 | 0.631 | 0.076 |
| Format-corrected audited | 0.715 | 0.753 | 0.038 |

## Data Files

Compressed h5ad files are stored in `data/h5ad_gz/`.
Decompress with `gzip -dk data/h5ad_gz/*.h5ad.gz` if local h5ad files are needed.
