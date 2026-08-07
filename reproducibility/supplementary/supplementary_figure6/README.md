# Supplementary Figure 6

This figure evaluates control-referenced TF target-program scores along computational A* paths for MEF2C, MEF2A, and HIF1A. Fixed-path program shuffles and random paths provide complementary null comparisons.

## Files

- `code/python/trajectory_figure_cardiac.py`: seeded path search, target-program scoring, null comparisons, and figure generation.
- `code/python/summarize_trajectory_seeds.py`: aggregation across seeds 0-4.
- `code/python/graph_utils.py` and `code/rust/`: archived A* and graph-scoring implementation.
- `results/trajectory_cardiac_metrics.csv`: archived seed-0 detailed metrics.
- `results/trajectory_5seed_per_seed.csv`: TF-level results for all five seeds.
- `results/trajectory_5seed_summary.csv`: means and sample standard deviations reported in Table S4.
- `results/supplementary_figure6_astar_paths.pdf`: archived reported PDF.

## Reproduce one seed

```bash
python reproducibility/supplementary/supplementary_figure6/code/python/trajectory_figure_cardiac.py \
  --h5ad /path/to/cardio_perturb_phate.h5ad \
  --seed 0
```

Repeat with seeds 1-4 in separate output directories, then summarize:

```bash
python reproducibility/supplementary/supplementary_figure6/code/python/summarize_trajectory_seeds.py \
  --root /path/to/seed_outputs
```

The Rust extension used by `graph_utils.py` must be built or installed before rerunning A* path search.
