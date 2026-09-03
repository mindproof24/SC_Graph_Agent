
mod sparse_utils;
mod selective_integrated;
mod astar_phate;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use hashbrown::{HashMap, HashSet};

use crate::sparse_utils::csr_to_rust;
// =============================================================
// Core data structures
// =============================================================

/// Edge information
#[derive(Clone, Debug)]
struct Edge {
    source_local: usize,
    target_local: usize,
    dorothea_weight: f64,
    beta: f64,
    confidence: String,
}

/// ClusterWeightedGraph (Rust Computation)
#[derive(Clone)]
#[allow(dead_code)]  // These fields are accessed indirectly through methods
struct CWGData {
    pathway_name: String,
    cluster_id: String,

    // Gene
    pathway_genes: Vec<String>,
    gene_to_local_idx: HashMap<String, usize>,
    gene_global_indices: Vec<usize>,

    // Edge
    edges: Vec<Edge>,
    edge_src_local: Vec<usize>,
    edge_tgt_local: Vec<usize>,
    dorothea_weights: Vec<f64>,
    betas: Vec<f64>,
    
    
    // Expression
    source_expr: HashMap<String, f64>,
    target_expr: HashMap<String, f64>,
    
    // Cluster Index
    cluster_cells: Vec<usize>,
    

    beta_mode_dynamic: bool,
}

// =============================================================
// CWG class exposed to Python
// =============================================================

#[pyclass]
#[derive(Clone)]
pub struct ClusterWeightedGraphRust {
    data: CWGData,
    
    #[pyo3(get)]
    pathway_name: String,
    #[pyo3(get)]
    cluster_id: String,
    #[pyo3(get)]
    n_genes: usize,
    #[pyo3(get)]
    n_edges: usize,
}

#[pymethods]
impl ClusterWeightedGraphRust {
    /// Dense constructor exposed to Python.
	/// OPTIONAL_PUBLIC API: the current server uses new_sparse() instead.
    #[new]
    #[pyo3(signature = (
        expr_matrix,
        gene_names,
        cluster_mask,
        dorothea_sources,
        dorothea_targets,
        dorothea_weights,
        dorothea_confidences,
        cluster_id,
        cluster_key = "leiden",
        confidence_levels = None,
        tf_expr_threshold = 0.0,
        target_expr_threshold = 0.0,
        require_both_expressed = true,
        beta_mode = "dynamic"
    ))]
    fn new(
        _py: Python<'_>,
        expr_matrix: PyReadonlyArray2<f64>,
        gene_names: Vec<String>,
        cluster_mask: PyReadonlyArray1<bool>,
        dorothea_sources: Vec<String>,
        dorothea_targets: Vec<String>,
        dorothea_weights: PyReadonlyArray1<f64>,
        dorothea_confidences: Vec<String>,
        cluster_id: &str,
        cluster_key: &str,
        confidence_levels: Option<Vec<String>>,
        tf_expr_threshold: f64,
        target_expr_threshold: f64,
        require_both_expressed: bool,
        beta_mode: &str,
    ) -> PyResult<Self> {
        
        let expr = expr_matrix.as_array();
        let mask = cluster_mask.as_array();
        let d_weights = dorothea_weights.as_array();
        
        // Map gene names to indices.
        let gene_to_global_idx: HashMap<String, usize> = gene_names
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();
        
        // Cluster cell indices
        let cluster_cells: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect();
        
        let n_cluster_cells = cluster_cells.len();
        
        println!("============================================================");
        println!("Building ClusterWeightedGraphRust: adata_{}{}", cluster_key, cluster_id);
        println!("============================================================");
        println!("Cluster cells: {}", n_cluster_cells);
        
        //Confidence filtering
        let conf_set: HashSet<String> = confidence_levels
            .unwrap_or_else(|| vec!["A".into(), "B".into(), "C".into()])
            .into_iter()
            .collect();
        
        let all_tfs: HashSet<String> = dorothea_sources.iter().cloned().collect();
        let mut expressed_tfs: HashMap<String, f64> = HashMap::new();
        
        for tf in &all_tfs {
            if let Some(&global_idx) = gene_to_global_idx.get(tf) {
                let tf_mean: f64 = cluster_cells
                    .iter()
                    .map(|&cell_idx| expr[[cell_idx, global_idx]])
                    .sum::<f64>() / n_cluster_cells as f64;
                // Only expressed TFs 
                if tf_mean > tf_expr_threshold {
                    expressed_tfs.insert(tf.clone(), tf_mean);
                }
            }
        }
        
        println!("Expressed TFs: {}/{}", expressed_tfs.len(), all_tfs.len());
        
        // Edge Collection
	let mut pathway_genes_set: HashSet<String> = HashSet::new();
        let mut edges_data: Vec<(String, String, f64, String, f64, f64)> = Vec::new();
        let mut source_expr: HashMap<String, f64> = HashMap::new();
        let mut target_expr: HashMap<String, f64> = HashMap::new();
        let mut skipped_count = 0usize;
        
        for (idx, (src, tgt)) in dorothea_sources.iter().zip(dorothea_targets.iter()).enumerate() {
            let tf_mean = match expressed_tfs.get(src) {
                Some(&v) => v,
                None => continue,
            };
            
            let conf = &dorothea_confidences[idx];
            if !conf_set.contains(conf) {
                continue;
            }
            
            let tgt_global_idx = match gene_to_global_idx.get(tgt) {
                Some(&v) => v,
                None => continue,
            };
            
            let tgt_mean: f64 = cluster_cells
                .iter()
                .map(|&cell_idx| expr[[cell_idx, tgt_global_idx]])
                .sum::<f64>() / n_cluster_cells as f64;
            
            if require_both_expressed {
                if tgt_mean <= 0.0 || tf_mean <= 0.0 {
                    skipped_count += 1;
                    continue;
                }
            } else if tgt_mean < target_expr_threshold {
                skipped_count += 1;
                continue;
            }
            
            let d_weight = d_weights[idx];
            
            let beta = if beta_mode == "dynamic" {
                (tf_mean + tgt_mean + d_weight).abs().sqrt()
            } else {
                d_weight
            };
            
            edges_data.push((
                src.clone(), tgt.clone(), d_weight, conf.clone(), beta, tgt_mean,
            ));
            
            pathway_genes_set.insert(src.clone());
            pathway_genes_set.insert(tgt.clone());
            source_expr.insert(src.clone(), tf_mean);
            target_expr.insert(tgt.clone(), tgt_mean);
        }
        
        println!("Skipped (zero/low target expr): {}", skipped_count);
        
        // Ordering of Pathway genes
        let mut pathway_genes: Vec<String> = pathway_genes_set.into_iter().collect();
        pathway_genes.sort();
        
        let gene_to_local_idx: HashMap<String, usize> = pathway_genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();
        
        let gene_global_indices: Vec<usize> = pathway_genes
            .iter()
            .map(|g| *gene_to_global_idx.get(g)
                .unwrap_or_else(|| panic!("Gene '{}' not found in adata.var_names!", g)))
            .collect();
        
        println!("Pathway genes: {}", pathway_genes.len());
        println!("Edges: {}", edges_data.len());
        
        // Edge Structure
        let mut edges: Vec<Edge> = Vec::with_capacity(edges_data.len());
        let mut edge_src_local: Vec<usize> = Vec::with_capacity(edges_data.len());
        let mut edge_tgt_local: Vec<usize> = Vec::with_capacity(edges_data.len());
        let mut dorothea_w: Vec<f64> = Vec::with_capacity(edges_data.len());
        let mut betas: Vec<f64> = Vec::with_capacity(edges_data.len());
        
        for (src, tgt, d_weight, conf, beta, _) in edges_data {
            let src_local = *gene_to_local_idx.get(&src).unwrap();
            let tgt_local = *gene_to_local_idx.get(&tgt).unwrap();
            
            edges.push(Edge {
                source_local: src_local,
                target_local: tgt_local,
                dorothea_weight: d_weight,
                beta,
                confidence: conf,
            });
            
            edge_src_local.push(src_local);
            edge_tgt_local.push(tgt_local);
            dorothea_w.push(d_weight);
            betas.push(beta);
      
        let pathway_name = format!("adata_{}{}", cluster_key, cluster_id);
        let n_genes = pathway_genes.len();
        let n_edges = edges.len();

        
        println!("✓ Network built: {}", pathway_name);
        
        let data = CWGData {
            pathway_name: pathway_name.clone(),
            cluster_id: cluster_id.to_string(),
            pathway_genes,
            gene_to_local_idx,
            gene_global_indices,
            edges,
            edge_src_local,
            edge_tgt_local,
            dorothea_weights: dorothea_w,
            betas,
            source_expr,
            target_expr,
            cluster_cells,
            beta_mode_dynamic: beta_mode == "dynamic",
        };
        
        Ok(ClusterWeightedGraphRust {
            data,
            pathway_name,
            cluster_id: cluster_id.to_string(),
            n_genes,
            n_edges,
        })
    }
    
    /// Sparse matrix constructor (no toarray() needed)
    /// Python: ClusterWeightedGraphRust.new_sparse(adata.X, ...)
    #[staticmethod]
    #[pyo3(signature = (
        sparse_matrix,
        gene_names,
        cluster_mask,
        dorothea_sources,
        dorothea_targets,
        dorothea_weights,
        dorothea_confidences,
        cluster_id,
        cluster_key = "leiden",
        confidence_levels = None,
        tf_expr_threshold = 0.0,
        target_expr_threshold = 0.0,
        require_both_expressed = true,
        beta_mode = "dynamic"
    ))]
    fn new_sparse(
        _py: Python<'_>,
        sparse_matrix: &Bound<'_, PyAny>,
        gene_names: Vec<String>,
        cluster_mask: PyReadonlyArray1<bool>,
        dorothea_sources: Vec<String>,
        dorothea_targets: Vec<String>,
        dorothea_weights: PyReadonlyArray1<f64>,
        dorothea_confidences: Vec<String>,
        cluster_id: &str,
        cluster_key: &str,
        confidence_levels: Option<Vec<String>>,
        tf_expr_threshold: f64,
        target_expr_threshold: f64,
        require_both_expressed: bool,
        beta_mode: &str,
    ) -> PyResult<Self> {
        let csr = csr_to_rust(sparse_matrix)?;
        let mask = cluster_mask.as_array();
        let d_weights = dorothea_weights.as_array();

        // Gene name → global index
        let gene_to_global_idx: HashMap<String, usize> = gene_names
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        // Cluster cell indices
        let cluster_cells: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect();

        let n_cluster_cells = cluster_cells.len();

        println!("============================================================");
        println!("Building ClusterWeightedGraphRust (sparse): adata_{}{}", cluster_key, cluster_id);
        println!("============================================================");
        println!("Cluster cells: {}", n_cluster_cells);

        // Precompute column sums for cluster cells in one sparse pass
        // Replaces all expr[[cell_idx, global_idx]] lookups
        let mut col_sums: HashMap<usize, f64> = HashMap::new();
        for &cell_idx in &cluster_cells {
            let row = csr.row(cell_idx);
            for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                *col_sums.entry(col).or_insert(0.0) += val;
            }
        }
		
        // Confidence filtering
        let conf_set: HashSet<String> = confidence_levels
            .unwrap_or_else(|| vec!["A".into(), "B".into(), "C".into()])
            .into_iter()
            .collect();

        let all_tfs: HashSet<String> = dorothea_sources.iter().cloned().collect();
        let mut expressed_tfs: HashMap<String, f64> = HashMap::new();

        for tf in &all_tfs {
            if let Some(&global_idx) = gene_to_global_idx.get(tf) {
                let tf_mean = col_sums.get(&global_idx).copied().unwrap_or(0.0)
                    / n_cluster_cells as f64;
                if tf_mean > tf_expr_threshold {
                    expressed_tfs.insert(tf.clone(), tf_mean);
                }
            }
        }

        println!("Expressed TFs: {}/{}", expressed_tfs.len(), all_tfs.len());

        // Edge Collection
        let mut pathway_genes_set: HashSet<String> = HashSet::new();
        let mut edges_data: Vec<(String, String, f64, String, f64, f64)> = Vec::new();
        let mut source_expr: HashMap<String, f64> = HashMap::new();
        let mut target_expr: HashMap<String, f64> = HashMap::new();
        let mut skipped_count = 0usize;

        for (idx, (src, tgt)) in dorothea_sources.iter().zip(dorothea_targets.iter()).enumerate() {
            let tf_mean = match expressed_tfs.get(src) {
                Some(&v) => v,
                None => continue,
            };

            let conf = &dorothea_confidences[idx];
            if !conf_set.contains(conf) {
                continue;
            }

            let tgt_global_idx = match gene_to_global_idx.get(tgt) {
                Some(&v) => v,
                None => continue,
            };

            let tgt_mean = col_sums.get(&tgt_global_idx).copied().unwrap_or(0.0)
                / n_cluster_cells as f64;

            if require_both_expressed {
                if tgt_mean <= 0.0 || tf_mean <= 0.0 {
                    skipped_count += 1;
                    continue;
                }
            } else if tgt_mean < target_expr_threshold {
                skipped_count += 1;
                continue;
            }

            let d_weight = d_weights[idx];

            let beta = if beta_mode == "dynamic" {
                (tf_mean + tgt_mean + d_weight).abs().sqrt()
            } else {
                d_weight
            };

            edges_data.push((
                src.clone(), tgt.clone(), d_weight, conf.clone(), beta, tgt_mean,
            ));

            pathway_genes_set.insert(src.clone());
            pathway_genes_set.insert(tgt.clone());
            source_expr.insert(src.clone(), tf_mean);
            target_expr.insert(tgt.clone(), tgt_mean);
        }

        println!("Skipped (zero/low target expr): {}", skipped_count);

        // Ordering of Pathway genes
        let mut pathway_genes: Vec<String> = pathway_genes_set.into_iter().collect();
        pathway_genes.sort();

        let gene_to_local_idx: HashMap<String, usize> = pathway_genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let gene_global_indices: Vec<usize> = pathway_genes
            .iter()
            .map(|g| *gene_to_global_idx.get(g)
                .unwrap_or_else(|| panic!("Gene '{}' not found in adata.var_names!", g)))
            .collect();

        println!("Pathway genes: {}", pathway_genes.len());
        println!("Edges: {}", edges_data.len());

        // Edge Structure
        let mut edges: Vec<Edge> = Vec::with_capacity(edges_data.len());
        let mut edge_src_local: Vec<usize> = Vec::with_capacity(edges_data.len());
        let mut edge_tgt_local: Vec<usize> = Vec::with_capacity(edges_data.len());
        let mut dorothea_w: Vec<f64> = Vec::with_capacity(edges_data.len());
        let mut betas: Vec<f64> = Vec::with_capacity(edges_data.len());

        for (src, tgt, d_weight, conf, beta, _) in edges_data {
            let src_local = *gene_to_local_idx.get(&src).unwrap();
            let tgt_local = *gene_to_local_idx.get(&tgt).unwrap();

            edges.push(Edge {
                source_local: src_local,
                target_local: tgt_local,
                dorothea_weight: d_weight,
                beta,
                confidence: conf,
            });

            edge_src_local.push(src_local);
            edge_tgt_local.push(tgt_local);
            dorothea_w.push(d_weight);
            betas.push(beta);
        }

        println!("TF-TF edges: {}", tf_tf_edges.len());

        let pathway_name = format!("adata_{}{}", cluster_key, cluster_id);
        let n_genes = pathway_genes.len();
        let n_edges = edges.len();
        let n_tf_tf_edges = tf_tf_edges.len();

        println!("✓ Network built (sparse): {}", pathway_name);

        let data = CWGData {
            pathway_name: pathway_name.clone(),
            cluster_id: cluster_id.to_string(),
            pathway_genes,
            gene_to_local_idx,
            gene_global_indices,
            edges,
            edge_src_local,
            edge_tgt_local,
            dorothea_weights: dorothea_w,
            betas,
            tf_tf_edges,
            tf_graph,
            source_expr,
            target_expr,
            cluster_cells,
            beta_mode_dynamic: beta_mode == "dynamic",
        };

        Ok(ClusterWeightedGraphRust {
            data,
            pathway_name,
            cluster_id: cluster_id.to_string(),
            n_genes,
            n_edges,
            n_tf_tf_edges,
        })
    }

    /// G2 norm calculation for a single cell.
    fn compute_graph_norm(
        &self,
        _py: Python<'_>,
        expr_matrix: PyReadonlyArray2<f64>,
        cell_idx: usize,
    ) -> PyResult<f64> {
        let expr = expr_matrix.as_array();
        
        let alpha: Vec<f64> = self.data.gene_global_indices
            .iter()
            .map(|&global_idx| expr[[cell_idx, global_idx]])
            .collect();
		
        let total_alpha_g: f64 = expr.row(cell_idx).iter().sum();

        let norm = compute_single_norm_internal(
            &alpha,
            &self.data.edge_src_local,
            &self.data.edge_tgt_local,
            &self.data.dorothea_weights,
            self.data.beta_mode_dynamic,
            total_alpha_g,
        );
        
        Ok(norm)
    }
    
    /// Calculate per-cell edge-L2 values from a dense matrix in parallel.
	/// OPTIONAL PUBLIC API: not called by the current server.
    fn compute_all_norms(
        &self,
        py: Python<'_>,
        expr_matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let expr = expr_matrix.as_array();
        let n_cells = expr.shape()[0];
        
        let gene_indices = &self.data.gene_global_indices;
        let edge_src = &self.data.edge_src_local;
        let edge_tgt = &self.data.edge_tgt_local;
        let dorothea_w = &self.data.dorothea_weights;
        let beta_dynamic = self.data.beta_mode_dynamic;
        
        let norms: Vec<f64> = py.allow_threads(|| {
            (0..n_cells)
                .into_par_iter()
                .map(|cell_idx| {
                    let alpha: Vec<f64> = gene_indices
                        .iter()
                        .map(|&global_idx| expr[[cell_idx, global_idx]])
                        .collect();


                    let total_alpha_g: f64 = expr.row(cell_idx).iter().sum();

                    compute_single_norm_internal(
                        &alpha,
                        edge_src,
                        edge_tgt,
                        dorothea_w,
                        beta_dynamic,
                        total_alpha_g,
                    )
                })
                .collect()
        });
        
        Ok(PyArray1::from_vec_bound(py, norms).into())
    }
	/// Calculate per-cell edge-L2 values from a sparse matrix in parallel.
    /// OPTIONAL PUBLIC API: not called by the current server.
    fn compute_all_norms_sparse(
        &self,
        py: Python<'_>,
        sparse_matrix: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let csr = csr_to_rust(sparse_matrix)?;
        let n_cells = csr.nrows();

        let gene_indices = &self.data.gene_global_indices;
        let edge_src = &self.data.edge_src_local;
        let edge_tgt = &self.data.edge_tgt_local;
        let dorothea_w = &self.data.dorothea_weights;
        let beta_dynamic = self.data.beta_mode_dynamic;

        let global_to_local: HashMap<usize, usize> = gene_indices.iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let n_local = gene_indices.len();

        let norms: Vec<f64> = py.allow_threads(|| {
            (0..n_cells)
                .into_par_iter()
                .map(|cell_idx| {
                    let mut alpha = vec![0.0f64; n_local];
                    let row = csr.row(cell_idx);
                    let total_alpha_g: f64 = row.values().iter().sum();
                    for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                        if let Some(&local) = global_to_local.get(&col) {
                            alpha[local] = val;
                        }
                    }
                    compute_single_norm_internal(
                        &alpha, edge_src, edge_tgt, dorothea_w, beta_dynamic,
                        total_alpha_g,
                    )
                })
                .collect()
        });

        Ok(PyArray1::from_vec_bound(py, norms).into())
    }
	/// Calculate dense-matrix edge-L2 values for stored cluster cells.
    /// OPTIONAL PUBLIC API: not called by the current server.
    fn compute_cluster_norms(
        &self,
        py: Python<'_>,
        expr_matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let expr = expr_matrix.as_array();
        
        let gene_indices = &self.data.gene_global_indices;
        let edge_src = &self.data.edge_src_local;
        let edge_tgt = &self.data.edge_tgt_local;
        let dorothea_w = &self.data.dorothea_weights;
        let beta_dynamic = self.data.beta_mode_dynamic;
        let cluster_cells = &self.data.cluster_cells;
        
        let norms: Vec<f64> = py.allow_threads(|| {
            cluster_cells
                .par_iter()
                .map(|&cell_idx| {
                    let alpha: Vec<f64> = gene_indices
                        .iter()
                        .map(|&global_idx| expr[[cell_idx, global_idx]])
                        .collect();


                    let total_alpha_g: f64 = expr.row(cell_idx).iter().sum();

                    compute_single_norm_internal(
                        &alpha,
                        edge_src,
                        edge_tgt,
                        dorothea_w,
                        beta_dynamic,
                        total_alpha_g,
                    )
                })
                .collect()
        });
        
        Ok(PyArray1::from_vec_bound(py, norms).into())
    }
    /// Calculate sparse-matrix edge-L2 values for stored cluster cells.
    /// OPTIONAL PUBLIC API: not called by the current server.
    fn compute_cluster_norms_sparse(
        &self,
        py: Python<'_>,
        sparse_matrix: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let csr = csr_to_rust(sparse_matrix)?;

        let gene_indices = &self.data.gene_global_indices;
        let edge_src = &self.data.edge_src_local;
        let edge_tgt = &self.data.edge_tgt_local;
        let dorothea_w = &self.data.dorothea_weights;
        let beta_dynamic = self.data.beta_mode_dynamic;
        let cluster_cells = &self.data.cluster_cells;

        let global_to_local: HashMap<usize, usize> = gene_indices.iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let n_local = gene_indices.len();

        let norms: Vec<f64> = py.allow_threads(|| {
            cluster_cells
                .par_iter()
                .map(|&cell_idx| {
                    let mut alpha = vec![0.0f64; n_local];
                    let row = csr.row(cell_idx);
                    // Corrected: the full sparse-row sum equals total expression across all genes.
                    let total_alpha_g: f64 = row.values().iter().sum();
                    for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                        if let Some(&local) = global_to_local.get(&col) {
                            alpha[local] = val;
                        }
                    }
                    compute_single_norm_internal(
                        &alpha, edge_src, edge_tgt, dorothea_w, beta_dynamic,
                        total_alpha_g,
                    )
                })
                .collect()
        });

        Ok(PyArray1::from_vec_bound(py, norms).into())
    }

    /// Return cluster cell indices
	/// OPTIONAL PUBLIC API: not called by the current server.
    fn get_cluster_cells(&self, _py: Python<'_>) -> PyResult<Vec<usize>> {
        Ok(self.data.cluster_cells.clone())
    }
 
    /// Return Pathway genes
    fn get_pathway_genes(&self, _py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(self.data.pathway_genes.clone())
    }
    
    /// Return Edge information for DataFrame
    fn get_edges_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        
        let sources: Vec<String> = self.data.edges
            .iter()
            .map(|e| self.data.pathway_genes[e.source_local].clone())
            .collect();
        
        let targets: Vec<String> = self.data.edges
            .iter()
            .map(|e| self.data.pathway_genes[e.target_local].clone())
            .collect();
        
        let dorothea_w: Vec<f64> = self.data.edges
            .iter()
            .map(|e| e.dorothea_weight)
            .collect();
        
        let betas: Vec<f64> = self.data.edges
            .iter()
            .map(|e| e.beta)
            .collect();
        
        let confidences: Vec<String> = self.data.edges
            .iter()
            .map(|e| e.confidence.clone())
            .collect();
        
        dict.set_item("source", sources)?;
        dict.set_item("target", targets)?;
        dict.set_item("dorothea_weight", dorothea_w)?;
        dict.set_item("beta", betas)?;
        dict.set_item("confidence", confidences)?;
        
        Ok(dict.into())
    }
    
    /// Return Source expression
    fn get_source_expr(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.data.source_expr {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }
    
    /// Return Target expression
    fn get_target_expr(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.data.target_expr {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }
    

    
    fn __repr__(&self) -> String {
        format!(
            "ClusterWeightedGraphRust('{}', genes={}, edges={})",
            self.pathway_name, self.n_genes, self.n_edges
        )
    }
}
	
// =============================================================
// Shared edge-L2 calculation used by optional dense and sparse CWG norm APIs.
// This is distinct from build_conservative_graph(), which aggregates path means.
// =============================================================

#[inline]
fn compute_single_norm_internal(
    alpha: &[f64],
    edge_src: &[usize],
    edge_tgt: &[usize],
    edge_dorothea_weights: &[f64],
    beta_mode_dynamic: bool,
    total_alpha_g: f64, // Total expression across all genes, not restricted to the pathway.
) -> f64 {
    let alpha_g = total_alpha_g;

    if alpha_g < 1e-10 {
        return 0.0;
    }
    
    let alpha_g_sq = alpha_g * alpha_g;
    let mut norm_sq = 0.0;
    
    for idx in 0..edge_src.len() {
        let i = edge_src[idx];
        let j = edge_tgt[idx];
        let alpha_i = alpha[i];
        let alpha_j = alpha[j];
        
        let beta_ij = if beta_mode_dynamic {
            (alpha_i + alpha_j + edge_dorothea_weights[idx]).abs().sqrt()
        } else {
            edge_dorothea_weights[idx]
        };
        
        let coeff = (alpha_i * alpha_j) / alpha_g_sq;
        norm_sq += coeff * beta_ij * beta_ij;
    }
    
    norm_sq.sqrt()
}

// =============================================================
// Optional batch CWG APIs retained for wrapper.py compatibility.
// These functions are exported through PyO3 but are not called by the current server.
// =============================================================

#[pyfunction]
fn compute_all_cwg_norms(
    py: Python<'_>,
    expr_matrix: PyReadonlyArray2<f64>,
    cwg_list: Vec<PyRef<ClusterWeightedGraphRust>>,
) -> PyResult<PyObject> {
    let expr = expr_matrix.as_array();
    let n_cells = expr.shape()[0];
    
    let result_dict = PyDict::new_bound(py);
    
    for cwg in cwg_list.iter() {
        let gene_indices = &cwg.data.gene_global_indices;
        let edge_src = &cwg.data.edge_src_local;
        let edge_tgt = &cwg.data.edge_tgt_local;
        let dorothea_w = &cwg.data.dorothea_weights;
        let beta_dynamic = cwg.data.beta_mode_dynamic;
        let col_name = format!("{}_G2", cwg.pathway_name);
        
        let norms: Vec<f64> = py.allow_threads(|| {
            (0..n_cells)
                .into_par_iter()
                .map(|cell_idx| {
                    let alpha: Vec<f64> = gene_indices
                        .iter()
                        .map(|&global_idx| expr[[cell_idx, global_idx]])
                        .collect();

                    let total_alpha_g: f64 = expr.row(cell_idx).iter().sum();

                    compute_single_norm_internal(
                        &alpha,
                        edge_src,
                        edge_tgt,
                        dorothea_w,
                        beta_dynamic,
                        total_alpha_g,
                    )
                })
                .collect()
        });
        
        result_dict.set_item(col_name, PyArray1::from_vec_bound(py, norms))?;
    }
    
    Ok(result_dict.into())
}

/// Batch norm calculation for cluster cells only
#[pyfunction]
fn compute_cluster_cwg_norms(
    py: Python<'_>,
    expr_matrix: PyReadonlyArray2<f64>,
    cwg_list: Vec<PyRef<ClusterWeightedGraphRust>>,
) -> PyResult<PyObject> {
    let expr = expr_matrix.as_array();
    let result_dict = PyDict::new_bound(py);
    
    for cwg in cwg_list.iter() {
        let gene_indices = &cwg.data.gene_global_indices;
        let edge_src = &cwg.data.edge_src_local;
        let edge_tgt = &cwg.data.edge_tgt_local;
        let dorothea_w = &cwg.data.dorothea_weights;
        let beta_dynamic = cwg.data.beta_mode_dynamic;
        let cluster_cells = &cwg.data.cluster_cells;
        let col_name = format!("{}_cluster_G2", cwg.pathway_name);
        
        let norms: Vec<f64> = py.allow_threads(|| {
            cluster_cells.par_iter().map(|&cell_idx| {
                let alpha: Vec<f64> = gene_indices.iter().map(|&g| expr[[cell_idx, g]]).collect();
                // [수정] 전체 유전자 발현 총합
                let total_alpha_g: f64 = expr.row(cell_idx).iter().sum();
                compute_single_norm_internal(&alpha, edge_src, edge_tgt, dorothea_w, beta_dynamic,
                    total_alpha_g)
            }).collect()
        });
        
        result_dict.set_item(col_name, PyArray1::from_vec_bound(py, norms))?;
    }
    Ok(result_dict.into())
}

// =============================================================
// Sparse batch processing functions
// =============================================================

#[pyfunction]
fn compute_all_cwg_norms_sparse(
    py: Python<'_>,
    sparse_matrix: &Bound<'_, PyAny>,
    cwg_list: Vec<PyRef<ClusterWeightedGraphRust>>,
) -> PyResult<PyObject> {
    let csr = csr_to_rust(sparse_matrix)?;
    let n_cells = csr.nrows();

    let result_dict = PyDict::new_bound(py);

    for cwg in cwg_list.iter() {
        let gene_indices = &cwg.data.gene_global_indices;
        let edge_src = &cwg.data.edge_src_local;
        let edge_tgt = &cwg.data.edge_tgt_local;
        let dorothea_w = &cwg.data.dorothea_weights;
        let beta_dynamic = cwg.data.beta_mode_dynamic;
        let col_name = format!("{}_G2", cwg.pathway_name);

        let global_to_local: HashMap<usize, usize> = gene_indices.iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let n_local = gene_indices.len();

        let norms: Vec<f64> = py.allow_threads(|| {
            (0..n_cells)
                .into_par_iter()
                .map(|cell_idx| {
                    let mut alpha = vec![0.0f64; n_local];
                    let row = csr.row(cell_idx);
                    // [수정] sparse row 전체 합 = 전체 유전자 발현 총합
                    let total_alpha_g: f64 = row.values().iter().sum();
                    for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                        if let Some(&local) = global_to_local.get(&col) {
                            alpha[local] = val;
                        }
                    }
                    compute_single_norm_internal(
                        &alpha, edge_src, edge_tgt, dorothea_w, beta_dynamic,
                        total_alpha_g,
                    )
                })
                .collect()
        });

        result_dict.set_item(col_name, PyArray1::from_vec_bound(py, norms))?;
    }

    Ok(result_dict.into())
}

/// Sparse batch norm calculation for cluster cells only
#[pyfunction]
fn compute_cluster_cwg_norms_sparse(
    py: Python<'_>,
    sparse_matrix: &Bound<'_, PyAny>,
    cwg_list: Vec<PyRef<ClusterWeightedGraphRust>>,
) -> PyResult<PyObject> {
    let csr = csr_to_rust(sparse_matrix)?;
    let result_dict = PyDict::new_bound(py);

    for cwg in cwg_list.iter() {
        let gene_indices = &cwg.data.gene_global_indices;
        let edge_src = &cwg.data.edge_src_local;
        let edge_tgt = &cwg.data.edge_tgt_local;
        let dorothea_w = &cwg.data.dorothea_weights;
        let beta_dynamic = cwg.data.beta_mode_dynamic;
        let cluster_cells = &cwg.data.cluster_cells;
        let col_name = format!("{}_cluster_G2", cwg.pathway_name);

        let global_to_local: HashMap<usize, usize> = gene_indices.iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let n_local = gene_indices.len();

        let norms: Vec<f64> = py.allow_threads(|| {
            cluster_cells.par_iter().map(|&cell_idx| {
                let mut alpha = vec![0.0f64; n_local];
                let row = csr.row(cell_idx);
                // Corrected: the full sparse-row sum equals total expression across all genes.
                let total_alpha_g: f64 = row.values().iter().sum();
                for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                    if let Some(&local) = global_to_local.get(&col) {
                        alpha[local] = val;
                    }
                }
                compute_single_norm_internal(&alpha, edge_src, edge_tgt, dorothea_w, beta_dynamic,
                    total_alpha_g)
            }).collect()
        });

        result_dict.set_item(col_name, PyArray1::from_vec_bound(py, norms))?;
    }
    Ok(result_dict.into())
}
/// Aggregate prior TF-target edges over path-mean expression profiles.
/// CURRENT SERVER PATH: graph_utils.py calls this.
/// beta_threshold filters support within each path and threshold specifies the
/// minimum fraction of input paths that must support an edge.
///
/// In all-edge mode, for path Γ and edge (i,j):
///   beta_ij(Γ) = sqrt(abs(alpha_i(Γ) + alpha_j(Γ) + prior_weight_ij))
///   contribution_ij(Γ) = alpha_i(Γ) * alpha_j(Γ) * beta_ij(Γ)^2
///                        / alpha_G(Γ)^2
/// where alpha values are means over cells visited by Γ and alpha_G is the
/// corresponding mean total expression across all genes. Each edge can add at
/// most one support count per input path in this mode.
///
/// Rust returns count, freq, mean_beta, mean_contribution and mean alpha values.
/// graph_utils.py subsequently computes score = freq * mean_beta and performs
/// score-based ranking.
#[pyfunction]
#[pyo3(signature = (
    cwg,
    reduced_paths,
    sparse_matrix,
    use_greedy      = false,
    beta_threshold  = 0.0,
    min_path_length = 2,
    max_length      = 50,
    threshold       = 0.5,
))]
fn build_conservative_graph(
    py: Python<'_>,
    mut cwg: PyRefMut<'_, ClusterWeightedGraphRust>,
    reduced_paths: Vec<Vec<usize>>,
    sparse_matrix: &Bound<'_, PyAny>,
    beta_threshold: f64,
    threshold: f64,
) -> PyResult<PyObject> {

    // ── Convert to CSR exactly once here ───────────────────────
    let csr = csr_to_rust(sparse_matrix)?;

    let total = reduced_paths.len();
    // (source_local, target_local) → (count, beta_sum)
    // beta_sum: 누적해서 마지막에 mean_beta = beta_sum / count 계산
    let mut edge_counter: HashMap<(usize, usize), (usize, f64, f64, f64, f64)> = HashMap::new();
    // (count, beta_sum, contrib_sum, alpha_i_sum, alpha_j_sum)
    
    for (path_idx, path) in reduced_paths.iter().enumerate() {
        let n_cells = path.len();
        if n_cells == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                format!("reduced_paths[{}] is empty. All paths must contain at least one cell index.", path_idx)
            ));
        }
        let n = n_cells as f64;

        // ── 1. col_sums: sum expression by gene across path cells ─
        // Borrow &csr without copying.
        let mut col_sums: HashMap<usize, f64> = HashMap::new();
        for &cell_idx in path {
            let row = csr.row(cell_idx);
            for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                *col_sums.entry(col).or_insert(0.0) += val;
            }
        }
	
	let total_sum: f64 = col_sums.values().sum();
	let alpha_g_mean = total_sum / n;
	let alpha_g_sq = alpha_g_mean * alpha_g_mean;
	
        if !use_greedy {
            // ════════════════════════════════════════════════════
            // 모든 edge 방식 (main)
            // edges 전체(11994개) beta 갱신 후
            // beta >= beta_threshold인 edge를 바로 카운트
            // TF→non-TF 포함, 분기 표현 가능
            // ════════════════════════════════════════════════════
            // 읽기 전용 값을 루프 전에 추출
            // → edges[i] mutable borrow와 gene_global_indices immutable borrow 충돌 방지
            let beta_dynamic = cwg.data.beta_mode_dynamic;
            let n_edges      = cwg.data.edges.len();

            for i in 0..n_edges {
                // ① 읽기: source_local, target_local, dorothea_weight
                let src_local  = cwg.data.edges[i].source_local;
                let tgt_local  = cwg.data.edges[i].target_local;
                let dorothea_w = cwg.data.edges[i].dorothea_weight;

                // ② 읽기: global index 변환 (edges[i] borrow 종료 후)
                let src_g = cwg.data.gene_global_indices[src_local];
                let tgt_g = cwg.data.gene_global_indices[tgt_local];

                let tf_m  = col_sums.get(&src_g).copied().unwrap_or(0.0) / n;
                let tgt_m = col_sums.get(&tgt_g).copied().unwrap_or(0.0) / n;

		let new_beta   = (tf_m + tgt_m + dorothea_w).abs().sqrt();
		let contrib    = if alpha_g_mean > 1e-10 {
		    (tf_m * tgt_m / alpha_g_sq) * (new_beta * new_beta)
		    // beta² = tf_m + tgt_m + dorothea_w 의 절댓값 이므로 .abs() 필요
		} else { 0.0 };

		if new_beta >= beta_threshold {
		    let entry = edge_counter
		        .entry((src_local, tgt_local))
		        .or_insert((0, 0.0, 0.0, 0.0, 0.0));
		    entry.0 += 1;
		    entry.1 += new_beta;
		    entry.2 += contrib;
		    entry.3 += tf_m;   // alpha_i 누적
		    entry.4 += tgt_m;  // alpha_j 누적
		}
            }

        } else {
            // ════════════════════════════════════════════════════
            // greedy path 방식 (option)
            // tf_tf_edges(427개) beta 갱신 → tf_graph 재구성
            // → greedy path 상의 edge만 카운트
            // TF→TF cascade 방향 선호
            // ════════════════════════════════════════════════════

            // 3-a. tf_tf_edges beta 갱신
            // 읽기 전용 값을 루프 전에 추출
            // → tf_tf_edges[i] mutable borrow와 gene_to_local_idx/gene_global_indices borrow 충돌 방지
            let beta_dynamic  = cwg.data.beta_mode_dynamic;
            let n_tf_edges    = cwg.data.tf_tf_edges.len();

            for i in 0..n_tf_edges {
                let src_name   = cwg.data.tf_tf_edges[i].source.clone();
                let tgt_name   = cwg.data.tf_tf_edges[i].target.clone();
                let dorothea_w = cwg.data.tf_tf_edges[i].dorothea_weight;

                // tf_tf_edges[i] borrow 종료 후 다른 필드 접근
                let src_local  = cwg.data.gene_to_local_idx[&src_name];
                let tgt_local  = cwg.data.gene_to_local_idx[&tgt_name];
                let src_g      = cwg.data.gene_global_indices[src_local];
                let tgt_g      = cwg.data.gene_global_indices[tgt_local];

                let tf_m  = col_sums.get(&src_g).copied().unwrap_or(0.0) / n;
                let tgt_m = col_sums.get(&tgt_g).copied().unwrap_or(0.0) / n;

                // 쓰기: 앞의 모든 읽기 borrow 종료 후 안전
                cwg.data.tf_tf_edges[i].beta = if beta_dynamic {
                    (tf_m + tgt_m + dorothea_w).abs().sqrt()
                } else {
                    dorothea_w
                };
            }

            // 3-b. tf_graph 재구성 (427개)
            // tf_tf_edges를 먼저 Vec으로 수집 후 tf_graph에 삽입
            // → &tf_tf_edges(immutable)와 &mut tf_graph(mutable) 동시 borrow 방지
            let tf_graph_entries: Vec<(String, String, f64, f64)> = cwg.data.tf_tf_edges
                .iter()
                .map(|e| (e.source.clone(), e.target.clone(), e.beta, e.dorothea_weight))
                .collect();

            cwg.data.tf_graph.clear();
            for (src, tgt, beta, w) in tf_graph_entries {
                cwg.data.tf_graph
                    .entry(src)
                    .or_insert_with(Vec::new)
                    .push((tgt, beta, w));
            }

            // 3-c. 모든 expressed TF에서 greedy path 탐색 → edge 카운트
            for start_tf in cwg.data.source_expr.keys() {
                if !cwg.data.tf_graph.contains_key(start_tf) {
                    continue;
                }

                let mut path_nodes: Vec<String> = vec![start_tf.clone()];
                let mut visited: HashSet<&str> = HashSet::new();
                visited.insert(start_tf.as_str());
                let mut current = start_tf.as_str();

                while path_nodes.len() < max_length {
                    let next_edges = match cwg.data.tf_graph.get(current) {
                        Some(e) => e,
                        None => break,
                    };
                    let best = next_edges
                        .iter()
                        .filter(|(tgt, beta, _)| {
                            !visited.contains(tgt.as_str()) && *beta >= beta_threshold
                        })
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    match best {
                        Some((next_tf, _, _)) => {
                            visited.insert(next_tf.as_str());
                            path_nodes.push(next_tf.clone());
                            current = path_nodes.last().unwrap().as_str();
                        }
                        None => break,
                    }
                }

                if path_nodes.len() < min_path_length {
                    continue;
                }

                for w in path_nodes.windows(2) {
                    if let (Some(&s), Some(&t)) = (
                        cwg.data.gene_to_local_idx.get(&w[0]),
                        cwg.data.gene_to_local_idx.get(&w[1]),
                    ) {
                        // tf_graph에서 해당 edge의 beta 조회
                        let edge_beta = cwg.data.tf_graph
                            .get(&w[0])
                            .and_then(|edges| edges.iter().find(|(tgt, _, _)| tgt == &w[1]))
                            .map(|(_, b, _)| *b)
                            .unwrap_or(0.0);
                        let entry = edge_counter.entry((s, t)).or_insert((0, 0.0, 0.0, 0.0, 0.0));
                        entry.0 += 1;
                        entry.1 += edge_beta;
			 //entry.2 += contrib			
                        // TODO:: If beta-greedy is needed
                    }
                }
            }
        }
    }

    // ── threshold 적용 → Python dict 반환 ────────────────────────
    let cutoff = ((total as f64) * threshold).ceil() as usize;

    let mut filtered: Vec<((usize, usize), (usize, f64, f64, f64, f64))> = edge_counter
        .into_iter()
        .filter(|(_, (cnt, _, _, _, _))| *cnt >= cutoff)
        .collect();
    // count 내림차순 정렬
    filtered.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());

    let mut sources:    Vec<String> = Vec::new();
    let mut targets:    Vec<String> = Vec::new();
    let mut counts:     Vec<usize>  = Vec::new();
    let mut freqs:      Vec<f64>    = Vec::new();
    let mut mean_betas: Vec<f64>    = Vec::new();
    let mut mean_contribs: Vec<f64> = Vec::new();
    let mut mean_alpha_is: Vec<f64> = Vec::new();
    let mut mean_alpha_js: Vec<f64> = Vec::new(); 
    for ((s_local, t_local), (cnt, beta_sum, contrib_sum, alpha_i_sum, alpha_j_sum)) in filtered {
        sources.push(cwg.data.pathway_genes[s_local].clone());
        targets.push(cwg.data.pathway_genes[t_local].clone());
        counts.push(cnt);
        freqs.push(cnt as f64 / total as f64);
        mean_betas.push(beta_sum / cnt as f64);
        mean_contribs.push(contrib_sum / cnt as f64);
        mean_alpha_is.push(alpha_i_sum / cnt as f64);
        mean_alpha_js.push(alpha_j_sum / cnt as f64);
    }

    let dict = PyDict::new_bound(py);
    dict.set_item("source",      sources)?;
    dict.set_item("target",      targets)?;
    dict.set_item("count",       counts)?;
    dict.set_item("freq",        freqs)?;
    dict.set_item("mean_beta",   mean_betas)?;
    dict.set_item("mean_contribution", mean_contribs)?;
    dict.set_item("mean_alpha_i", mean_alpha_is)?;
    dict.set_item("mean_alpha_j", mean_alpha_js)?;
    dict.set_item("total_paths", total)?;
    dict.set_item("threshold",   threshold)?;
    dict.set_item("cutoff",      cutoff)?;
    dict.set_item("mode",        if use_greedy { "greedy" } else { "all_edges" })?;


    Ok(dict.into())
}
// =============================================================
// 모듈 등록
// =============================================================

#[pymodule]
fn _cwg_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // lib.rs — CWG (DoRothEA)
    m.add_class::<ClusterWeightedGraphRust>()?;
    m.add_function(wrap_pyfunction!(compute_all_cwg_norms, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cluster_cwg_norms, m)?)?;
    m.add_function(wrap_pyfunction!(compute_all_cwg_norms_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cluster_cwg_norms_sparse, m)?)?;
    
    // selective_integrated.rs
    m.add_class::<selective_integrated::CascadePath>()?;
    m.add_class::<selective_integrated::KEGGPathway>()?;
    m.add_class::<selective_integrated::SelectiveIntegratedGraph>()?;
    m.add_function(wrap_pyfunction!(selective_integrated::make_kegg_edges_bidirectional, m)?)?;
    m.add_function(wrap_pyfunction!(selective_integrated::build_all_integrated_graphs, m)?)?;
    m.add_function(wrap_pyfunction!(selective_integrated::compute_all_kegg_norms_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(selective_integrated::compute_all_kegg_norms_cluster_mean, m)?)?;
    m.add_function(wrap_pyfunction!(selective_integrated::compute_all_integrated_norms_sparse, m)?)?;
    
    // astar_phate.rs
    m.add_function(wrap_pyfunction!(astar_phate::astar_all_pairs, m)?)?;
    // test done :: m.add_function(wrap_pyfunction!(astar_phate::astar_all_pairs_legacy, m)?)?;
    // conservative graph
    m.add_function(wrap_pyfunction!(build_conservative_graph, m)?)?;
 
    Ok(())
}
