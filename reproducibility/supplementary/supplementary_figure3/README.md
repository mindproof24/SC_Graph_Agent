# Supplementary Figure 3

This figure displays expression of eight cardiac transcription factors across the fixed PHATE representation.

## Files

- `scripts/figure_cardiac_tf_phate.py`: expression extraction and figure generator.
- `results/cardiac_tf_phate_correlations.csv`: expression prevalence and correlations with PHATE1 and the cardiomyocyte-associated state score.
- `results/supplementary_figure3_cardiac_tf_phate.pdf`: reported editable PDF.

## Reproduce

```bash
python reproducibility/supplementary/supplementary_figure3/scripts/figure_cardiac_tf_phate.py \
  --h5ad /path/to/cardio_perturb_phate.h5ad
```

## Figure-text check

The archived code, correlation table, and figure use `HAND2`. The current supplementary-document legend states `HAND1`. This discrepancy must be resolved in the manuscript before release; the files here preserve the analysis that generated the archived figure.
