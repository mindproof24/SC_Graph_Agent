# Supplementary Figure 1

This figure shows overlap among the top 200 directed TF-target edges prioritized by A*, SCENIC, and CellOracle for T-cell atlas clusters 0, 1, 2, 3, 6, 9, and 10. All methods were restricted to the shared CollecTRI candidate set.

## Files

- `data/normalized_edges_collectri_ranked.csv`: harmonized ranked edges used to construct the Venn regions.
- `data/cluster_identity_for_venn.csv`: cluster identity labels displayed in panel titles.
- `scripts/make_supplementary_figure1_venn.py`: seven-panel figure generator.
- `scripts/plot_topk_edge_venn.py`: Venn-region helper functions.
- `results/supplementary_figure1_top200_venn.pdf`: reported editable PDF.

## Reproduce

```bash
python reproducibility/supplementary/supplementary_figure1/scripts/make_supplementary_figure1_venn.py
```

The compact files reproduce the plotted overlap figure. Regenerating the upstream A*, SCENIC, and CellOracle rankings additionally requires the T-cell atlas AnnData object and the method-specific workflows documented for Main Figure 3.
