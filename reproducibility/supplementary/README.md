# Supplementary Figure Reproducibility

This directory contains compact inputs, generating code, and reported outputs for Supplementary Figures 1-6. The per-edge CWG analysis was added as Supplementary Figure 5, and the former path-coherence Supplementary Figure 5 was renumbered as Supplementary Figure 6.

## Structure

- `supplementary_figure1`: top-200 A*/SCENIC/CellOracle TF-target overlap across seven T-cell clusters.
- `supplementary_figure2`: sampled A* paths on the cardiac PHATE representation.
- `supplementary_figure3`: selected cardiac TF expression on the PHATE representation.
- `supplementary_figure4`: target-program activity across the cardiomyocyte-associated state axis.
- `supplementary_figure5`: edge-count-corrected per-edge CWG activity across the cardiac state axis.
- `supplementary_figure6`: TF target-program coherence along computational A* paths.
- `shared_inputs`: compact inputs used by more than one supplementary figure.

## External analysis object

Supplementary Figures 2-6 require the exact analysis-ready `cardio_perturb_phate.h5ad` object (378,802 cells x 25,396 genes). It is not duplicated in this repository and should be obtained from the accompanying Zenodo record.

| File | Used by | Zenodo URL | SHA-256 |
|---|---|---|---|
| `cardio_perturb_phate.h5ad` | Supplementary Figures 2-6 | `TO_BE_ADDED` | `TO_BE_ADDED` |

All commands below are intended to be run from the repository root. PDF files are the reported figure outputs; PNG files are included as previews.
