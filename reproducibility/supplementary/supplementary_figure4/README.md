# Supplementary Figure 4

This figure compares control-referenced target-program activity across quantile bins of the cardiomyocyte-associated state score. Tracks include A*-ranked targets, the corresponding full DoRothEA prior, a size-matched signed random program, and IRX4 and NRG1 expression references.

## Files

- `inputs/dorothea_astar_ranked_edges.csv`: A* and comparison rankings for the cardiac analysis population.
- `../shared_inputs/dorothea_ABC_human.parquet`: signed DoRothEA A-C prior.
- `scripts/figure_edge_axis_pdf.py`: figure and source-values generator.
- `results/supplementary_figure4_edge_activity_axis.pdf`: archived reported PDF.
- `results/supplementary_figure4_source_values.csv`: generated when the script is rerun with the external H5AD.

## Reproduce

```bash
python reproducibility/supplementary/supplementary_figure4/scripts/figure_edge_axis_pdf.py \
  --h5ad /path/to/cardio_perturb_phate.h5ad
```

The A* input contains no IRX4 edge. In this figure, IRX4 and NRG1 are calculated directly from their control-referenced expression in the H5AD as independent reference tracks; they are not read from the A* edge CSV. The historical `_injIRX4_astar` path name in the source environment was therefore replaced by an explicit input argument.
