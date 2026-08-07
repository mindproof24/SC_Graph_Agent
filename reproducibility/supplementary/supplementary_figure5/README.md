# Supplementary Figure 5

This figure evaluates graph edge activity after correcting the cumulative edge-L2 value for edge-set size. Because the CWG value is proportional to the square root of the summed edge contributions, each pathway value is divided by `sqrt(number of edges)` to obtain an edge-level RMS quantity. The figure compares A*-selected, full-prior, size-matched random, injected IRX4, and held-out NRG1 tracks across the cardiomyocyte-associated state axis.

## Files

- `inputs/dorothea_astar_ranked_edges.csv`: cardiac A* and comparison edge rankings used to select TF-specific programs.
- `../shared_inputs/dorothea_ABC_human.parquet`: signed DoRothEA A-C prior.
- `scripts/figure_edge_axis_cwg_peredge.py`: per-cell CWG calculation, edge-count correction, source-data export, and figure generation.
- `results/supplementary_figure5_edge_activity_per_edge.pdf`: reported editable PDF.
- `results/supplementary_figure5_edge_activity_per_edge.png`: preview image.
- `results/source_data/edges_used.csv`: all edges passed to the activity calculation.
- `results/source_data/bin_profile.csv`: 20-bin means, variation, and 95% confidence intervals for all five tracks.
- `results/source_data/track_summary.csv`: edge counts and correlations with the state axis.
- `results/source_data/percell_activity.parquet`: activity values for all 378,802 cells.

## Reproduce

```bash
python reproducibility/supplementary/supplementary_figure5/scripts/figure_edge_axis_cwg_peredge.py \
  --h5ad /path/to/cardio_perturb_phate.h5ad
```

The Rust `cwg_rust` extension must be installed. IRX4 and NRG1 edges are explicitly constructed in the script and are not selected from the A* edge CSV.
