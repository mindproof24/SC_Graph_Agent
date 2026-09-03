// src/integrated2.rs
// ============================================================
// KEGG pathway edge representation and activity scoring
// ============================================================
//
// This module stores dataset-represented KEGG edges, optionally inserts the
// reverse direction for protein-complex relations and computes edge-L2
// activity at either cell or cluster-mean resolution.
//
// ============================================================

use hashbrown::{HashMap, HashSet};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::sparse_utils::csr_to_rust;

// ============================================================
// 1. Edge coefficient
// ============================================================

#[derive(Clone, Copy)]
struct BetaCalculator;

impl BetaCalculator {
    #[inline]
    fn compute(&self, alpha_i: f64, alpha_j: f64, weight: f64) -> f64 {
        (weight + alpha_i + alpha_j).abs().sqrt()
    }
}

// ============================================================
// 2. KEGG pathway representation
// ============================================================

#[pyclass]
#[derive(Clone, Debug)]
pub struct KEGGPathway {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub sources: Vec<String>,
    #[pyo3(get)]
    pub targets: Vec<String>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub modifications: Vec<String>,
    #[pyo3(get)]
    pub effects: Vec<i32>,
    #[pyo3(get)]
    pub types: Vec<String>,
    #[pyo3(get)]
    pub indirects: Vec<bool>,
}

#[pymethods]
impl KEGGPathway {
    #[new]
    fn new(
        name: String,
        sources: Vec<String>,
        targets: Vec<String>,
        weights: Vec<f64>,
        modifications: Vec<String>,
        effects: Vec<i32>,
        types: Vec<String>,
        indirects: Vec<bool>,
    ) -> Self {
        Self {
            name,
            sources,
            targets,
            weights,
            modifications,
            effects,
            types,
            indirects,
        }
    }

    fn compute_base_weights(&self) -> Vec<f64> {
        self.compute_base_weights_internal()
    }

    fn compute_all_norms(
        &self,
        py: Python<'_>,
        expr_matrix: PyReadonlyArray2<f64>,
        adata_genes: Vec<String>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let expr = expr_matrix.as_array();
        let n_cells = expr.nrows();

        let gene_to_global: HashMap<String, usize> = adata_genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let mut genes_set = HashSet::new();
        for g in self.sources.iter().chain(self.targets.iter()) {
            if gene_to_global.contains_key(g) {
                genes_set.insert(g.clone());
            }
        }

        if genes_set.len() < 2 {
            return Ok(PyArray1::from_vec_bound(py, vec![0.0; n_cells]).into());
        }

        let mut genes: Vec<String> = genes_set.into_iter().collect();
        genes.sort();

        let gene_to_local: HashMap<String, usize> = genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let global_indices: Vec<usize> = genes
            .iter()
            .map(|g| *gene_to_global.get(g).unwrap())
            .collect();

        let weights = self.compute_base_weights_internal();
        let edges: Vec<(usize, usize, f64)> = (0..self.sources.len())
            .filter_map(|i| {
                let src_l = gene_to_local.get(&self.sources[i])?;
                let tgt_l = gene_to_local.get(&self.targets[i])?;
                Some((*src_l, *tgt_l, weights[i]))
            })
            .collect();

        if edges.is_empty() {
            return Ok(PyArray1::from_vec_bound(py, vec![0.0; n_cells]).into());
        }

        let norms: Vec<f64> = py.allow_threads(|| {
            let calc = BetaCalculator;
            (0..n_cells)
                .into_par_iter()
                .map(|cell| {
                    let alpha: Vec<f64> = global_indices.iter().map(|&g| expr[[cell, g]]).collect();
                    // Corrected: total expression across all genes.
                    let alpha_g: f64 = expr.row(cell).iter().sum();
                    if alpha_g < 1e-10 {
                        return 0.0;
                    }

                    let alpha_g_sq = alpha_g * alpha_g;
                    let norm_sq: f64 = edges
                        .iter()
                        .map(|&(i, j, w)| {
                            let beta = calc.compute(alpha[i], alpha[j], w);
                            (alpha[i] * alpha[j] / alpha_g_sq) * beta * beta
                        })
                        .sum();

                    norm_sq.sqrt()
                })
                .collect()
        });

        Ok(PyArray1::from_vec_bound(py, norms).into())
    }

    /// G2 norm (sparse matrix version) — no toarray() needed
    fn compute_all_norms_sparse(
        &self,
        py: Python<'_>,
        sparse_matrix: &Bound<'_, PyAny>,
        adata_genes: Vec<String>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let csr = csr_to_rust(sparse_matrix)?;
        let n_cells = csr.nrows();

        let gene_to_global: HashMap<String, usize> = adata_genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let mut genes_set = HashSet::new();
        for g in self.sources.iter().chain(self.targets.iter()) {
            if gene_to_global.contains_key(g) {
                genes_set.insert(g.clone());
            }
        }

        if genes_set.len() < 2 {
            return Ok(PyArray1::from_vec_bound(py, vec![0.0; n_cells]).into());
        }

        let mut genes: Vec<String> = genes_set.into_iter().collect();
        genes.sort();

        let gene_to_local: HashMap<String, usize> = genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let global_indices: Vec<usize> = genes
            .iter()
            .map(|g| *gene_to_global.get(g).unwrap())
            .collect();

        // global_col → local_gene mapping for sparse row lookup
        let global_to_local: HashMap<usize, usize> = global_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        let n_local = genes.len();

        let weights = self.compute_base_weights_internal();
        let edges: Vec<(usize, usize, f64)> = (0..self.sources.len())
            .filter_map(|i| {
                let src_l = gene_to_local.get(&self.sources[i])?;
                let tgt_l = gene_to_local.get(&self.targets[i])?;
                Some((*src_l, *tgt_l, weights[i]))
            })
            .collect();

        if edges.is_empty() {
            return Ok(PyArray1::from_vec_bound(py, vec![0.0; n_cells]).into());
        }

        let norms: Vec<f64> = py.allow_threads(|| {
            let calc = BetaCalculator;
            (0..n_cells)
                .into_par_iter()
                .map(|cell| {
                    let mut alpha = vec![0.0f64; n_local];

                    // Sparse row access: only non-zero entries
                    let row = csr.row(cell);
                    // Corrected: the full sparse-row sum equals total expression across all genes.
                    let alpha_g: f64 = row.values().iter().sum();
                    for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                        if let Some(&local) = global_to_local.get(&col) {
                            alpha[local] = val;
                        }
                    }

                    if alpha_g < 1e-10 {
                        return 0.0;
                    }

                    let alpha_g_sq = alpha_g * alpha_g;
                    let norm_sq: f64 = edges
                        .iter()
                        .map(|&(i, j, w)| {
                            let beta = calc.compute(alpha[i], alpha[j], w);
                            (alpha[i] * alpha[j] / alpha_g_sq) * beta * beta
                        })
                        .sum();

                    norm_sq.sqrt()
                })
                .collect()
        });

        Ok(PyArray1::from_vec_bound(py, norms).into())
    }

    fn get_edges_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);

        dict.set_item("source", self.sources.clone())?;
        dict.set_item("target", self.targets.clone())?;
        dict.set_item("weight", self.compute_base_weights_internal())?;
        dict.set_item("modification", self.modifications.clone())?;
        dict.set_item("effect", self.effects.clone())?;
        dict.set_item("type", self.types.clone())?;
        dict.set_item("indirect", self.indirects.clone())?;

        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!("KEGGPathway('{}', edges={})", self.name, self.sources.len())
    }
}

impl KEGGPathway {
    pub fn compute_base_weights_internal(&self) -> Vec<f64> {
        self.effects
            .iter()
            .zip(self.types.iter())
            .zip(self.indirects.iter())
            .map(|((&eff, typ), &ind)| {
                let e = match eff {
                    1 => 1.0,
                    2 => 0.8,
                    -1 => -0.8,
                    _ => 0.5,
                };
                let t = match typ.as_str() {
                    "PPrel" | "GErel" => 1.0,
                    "PCrel" => 0.8,
                    "PComplex" => 0.5,
                    _ => 0.5,
                };
                let i = if ind { 0.7 } else { 1.0 };
                e * t * i
            })
            .collect()
    }
}

// ============================================================
// 3. KEGG edge helper
// ============================================================

#[pyfunction]
pub fn make_kegg_edges_bidirectional(
    sources: Vec<String>,
    targets: Vec<String>,
    effects: Vec<i32>,
    types: Vec<String>,
    indirects: Vec<bool>,
    modifications: Vec<String>,
) -> PyResult<(
    Vec<String>,
    Vec<String>,
    Vec<i32>,
    Vec<String>,
    Vec<bool>,
    Vec<String>,
)> {
    let mut r_src = sources.clone();
    let mut r_tgt = targets.clone();
    let mut r_eff = effects.clone();
    let mut r_typ = types.clone();
    let mut r_ind = indirects.clone();
    let mut r_mod = modifications.clone();

    let mut existing: HashSet<(String, String)> = sources
        .iter()
        .zip(targets.iter())
        .map(|(s, t)| (s.clone(), t.clone()))
        .collect();

    for i in 0..sources.len() {
        if types[i] == "PComplex" {
            let rev = (targets[i].clone(), sources[i].clone());
            if !existing.contains(&rev) {
                r_src.push(targets[i].clone());
                r_tgt.push(sources[i].clone());
                r_eff.push(effects[i]);
                r_typ.push(types[i].clone());
                r_ind.push(indirects[i]);
                r_mod.push(modifications.get(i).cloned().unwrap_or_default());
                existing.insert(rev);
            }
        }
    }

    Ok((r_src, r_tgt, r_eff, r_typ, r_ind, r_mod))
}

// ============================================================
// 4. Sparse batch norm functions with one CSR conversion
// ============================================================

#[pyfunction]
pub fn compute_all_kegg_norms_cluster_mean(
    py: Python<'_>,
    sparse_matrix: &Bound<'_, PyAny>,
    adata_genes: Vec<String>,
    pathways: Vec<PyRef<KEGGPathway>>,
    cluster_indices: Vec<usize>,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;

    let csr = csr_to_rust(sparse_matrix)?;
    let n_cluster = cluster_indices.len();
    let n_genes = adata_genes.len();

    let gene_to_global: HashMap<String, usize> = adata_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    // ── Calculate cluster mean α across all genes once ──────
    let mut gene_sum = vec![0.0f64; n_genes];
    let mut alpha_g_sum = 0.0f64;

    for &cell in &cluster_indices {
        let row = csr.row(cell);
        alpha_g_sum += row.values().iter().sum::<f64>();
        for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
            if col < n_genes {
                gene_sum[col] += val;
            }
        }
    }

    let n_c = n_cluster as f64;
    let alpha_g = alpha_g_sum / n_c;
    let gene_mean: Vec<f64> = gene_sum.iter().map(|&s| s / n_c).collect();

    let result_dict = PyDict::new_bound(py);

    if alpha_g < 1e-10 {
        for pw in pathways.iter() {
            result_dict.set_item(&pw.name, 0.0f64)?;
        }
        return Ok(result_dict.into());
    }

    let alpha_g_sq = alpha_g * alpha_g;
    let calc = BetaCalculator;

    // ── G2(mean_α) for each pathway ─────────────────────────
    for pw in pathways.iter() {
        let weights = pw.compute_base_weights_internal();
        let norm_sq: f64 = (0..pw.sources.len())
            .filter_map(|i| {
                let gi = *gene_to_global.get(&pw.sources[i])?;
                let gj = *gene_to_global.get(&pw.targets[i])?;
                let ai = gene_mean[gi];
                let aj = gene_mean[gj];
                let beta = calc.compute(ai, aj, weights[i]);
                Some((ai * aj / alpha_g_sq) * beta * beta)
            })
            .sum();
        result_dict.set_item(&pw.name, norm_sq.sqrt())?;
    }

    Ok(result_dict.into())
}

/// Calculate norms for all KEGGPathways with one CSR conversion.
/// This remains expensive because cells are evaluated individually.
#[pyfunction]
pub fn compute_all_kegg_norms_sparse(
    py: Python<'_>,
    sparse_matrix: &Bound<'_, PyAny>,
    adata_genes: Vec<String>,
    kegg_pathways: Vec<PyRef<KEGGPathway>>,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;

    let csr = csr_to_rust(sparse_matrix)?;
    let n_cells = csr.nrows();

    let gene_to_global: HashMap<String, usize> = adata_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    let result_dict = PyDict::new_bound(py);

    for pathway in kegg_pathways.iter() {
        // Collect genes for this pathway
        let mut genes_set = HashSet::new();
        for g in pathway.sources.iter().chain(pathway.targets.iter()) {
            if gene_to_global.contains_key(g) {
                genes_set.insert(g.clone());
            }
        }

        if genes_set.len() < 2 {
            result_dict.set_item(
                &pathway.name,
                PyArray1::from_vec_bound(py, vec![0.0; n_cells]),
            )?;
            continue;
        }

        let mut genes: Vec<String> = genes_set.into_iter().collect();
        genes.sort();

        let gene_to_local: HashMap<String, usize> = genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let global_indices: Vec<usize> = genes
            .iter()
            .map(|g| *gene_to_global.get(g).unwrap())
            .collect();

        let global_to_local: HashMap<usize, usize> = global_indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        let n_local = genes.len();

        let weights = pathway.compute_base_weights_internal();
        let edges: Vec<(usize, usize, f64)> = (0..pathway.sources.len())
            .filter_map(|i| {
                let src_l = gene_to_local.get(&pathway.sources[i])?;
                let tgt_l = gene_to_local.get(&pathway.targets[i])?;
                Some((*src_l, *tgt_l, weights[i]))
            })
            .collect();

        if edges.is_empty() {
            result_dict.set_item(
                &pathway.name,
                PyArray1::from_vec_bound(py, vec![0.0; n_cells]),
            )?;
            continue;
        }

        let norms: Vec<f64> = py.allow_threads(|| {
            let calc = BetaCalculator;
            (0..n_cells)
                .into_par_iter()
                .map(|cell| {
                    let mut alpha = vec![0.0f64; n_local];

                    let row = csr.row(cell);
                    // Corrected: the full sparse-row sum equals total expression across all genes.
                    let alpha_g: f64 = row.values().iter().sum();
                    for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                        if let Some(&local) = global_to_local.get(&col) {
                            alpha[local] = val;
                        }
                    }

                    if alpha_g < 1e-10 {
                        return 0.0;
                    }

                    let alpha_g_sq = alpha_g * alpha_g;
                    let norm_sq: f64 = edges
                        .iter()
                        .map(|&(i, j, w)| {
                            let beta = calc.compute(alpha[i], alpha[j], w);
                            (alpha[i] * alpha[j] / alpha_g_sq) * beta * beta
                        })
                        .sum();

                    norm_sq.sqrt()
                })
                .collect()
        });

        result_dict.set_item(&pathway.name, PyArray1::from_vec_bound(py, norms))?;
    }

    Ok(result_dict.into())
}
