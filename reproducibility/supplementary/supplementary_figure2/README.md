# Supplementary Figure 2

This figure projects cells encountered along sampled A* paths onto the fixed cardiac PHATE representation. Panels show the cardiomyocyte-associated state score, `n_genes`, and perturbation status.

## Files

- `data/astar_path_cells.parquet`: cells and path positions used in the figure.
- `data/path_provenance.json`: A* parameters, input dimensions, seed, and representation checks.
- `code/python/figure_phate_paths.py`: figure generator.
- `results/supplementary_figure2_phate_paths.pdf`: reported editable PDF.

## Reproduce

```bash
python reproducibility/supplementary/supplementary_figure2/code/python/figure_phate_paths.py \
  --h5ad /path/to/cardio_perturb_phate.h5ad
```

The bundled path-cell table is sufficient to reproduce the plotted paths once the external analysis-ready H5AD is available.
