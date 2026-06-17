# Scoring

This snapshot was scored by the `score_prediction` logic in `agent/eval_grpo_dataset.py`, invoked through the NO_KG wrapper used for these runs.

## Strict Scoring Used for the Main JSONL Results

The five tasks in this snapshot are `cell_anon` 2-vs-2 tasks: each question contains four cells and two possible cell types.

The strict scorer expects the final answer to parse as JSON with a `labels` array:

```json
{"labels": ["<celltype for cell 1>", "<celltype for cell 2>", "<celltype for cell 3>", "<celltype for cell 4>"], "reason": "..."}
```

For each prediction:

1. `parse_final_answer` first tries to parse a fenced JSON block, then any flat JSON object in the final text.
2. The scorer reads `labels`; if absent, it falls back to `cell_labels`.
3. Each predicted label is compared against the ground-truth label at the same cell position.
4. Because labels can be verbose, each predicted label is first mapped to one of the two task options by key-token overlap against `celltype_a` and `celltype_b`.
5. If that option mapping fails, the scorer falls back to fuzzy matching:
   - normalized strings equal, or
   - one normalized string is a substring of the other, or
   - non-stopword key-token sets are identical.
6. Score is `n_correct / 4`, rounded to four decimals.

For these tasks, one correctly labeled cell contributes `0.25`.

The output fields therefore mean:

- `score_raw`: unpenalized strict score from parsed labels.
- `score`: final strict score after any tool-call penalty.
- `label_correct` / `n_correct`: number of correctly labeled cells.
- `n_total`: number of cells, normally `4`.
- `pred_labels`: parsed labels before option resolution.
- `resolved_preds`: labels after mapping to the two task options when possible.
- `gt_labels`: ground-truth labels in cell order.

## Tool-Call Penalty

If the run exceeds `TOOL_CALL_LIMIT`, the raw score is multiplied by:

```text
exp(-0.04 * (n_tool_calls - TOOL_CALL_LIMIT))
```

and rounded to four decimals.

For these runs the task prompt says "Tool call limit: 10"; the harness also records whether a penalty was applied in `penalty_applied`.

## Timeout

If an item exceeds the configured item timeout, the scorer returns:

```text
score = 0.0
score_raw = 0.0
timed_out = true
```

## Format-Corrected Audited Regrade

The audited regrade files are separate from strict scoring.

They were used because some model outputs contained a biologically explicit cell-wise answer but did not match the strict `{"labels": [...]}` JSON format. The audit recovered labels from structured or near-structured final answers such as cell-keyed JSON, nested cell type fields, or final text lines.

The audited score still uses the same per-cell rule:

```text
adjusted_score = number of correctly labeled cells / 4
```

In the audited files:

- `orig_score`: strict score from the automatic parser.
- `adjusted_score`: audited format-corrected score.
- `delta`: `adjusted_score - orig_score`.
- `star`: `true` when the audit changed the score.
- `source`: extraction route used by the audit, e.g. `strict`, `cell_keyed_json`, `nested_celltypes`, or `final_text_lines`.

For B-cell audited scoring, ambiguous subclasses such as germinal-center, marginal-zone, or mixed/activated B cell were not automatically collapsed to memory B cell unless the answer explicitly supported the target task label.

## Other Scorer Modes in the Shared Evaluator

The shared evaluator also supports task types not used by the five committed UNSEEN tasks:

- `cluster`: exact answer match, score `0` or `1`.
- `topn_celltype`: score accumulates `score_per_hit` for selected cells found in the answer pool, capped at `1`.
- `gbm_neural_topn`: gives separate credit for top-30 and positive-cell hits, capped at `1`.
- cluster-plus-cell tasks: combine exact cluster correctness with per-cell label correctness.
