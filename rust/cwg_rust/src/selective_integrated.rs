// src/selective_integrated.rs
// ============================================================
// Selective Integrated Graph: KEGG + DoRothEA Cascade 통합
// ============================================================
//
// 목적:
//   KEGG pathway와 DoRothEA TF cascade를 연결하여 소형 그래프 생성
//   (KEGG, Cascade) 쌍마다 하나의 그래프
//
// 연결 조건:
//   KEGG edge의 modification='e' source TF가
//   cascade의 어떤 TF와 일치하면 연결
//
// ============================================================

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use hashbrown::{HashMap, HashSet};

use crate::sparse_utils::csr_to_rust;

// ============================================================
// 1. Edge 구조
// ============================================================

#[derive(Clone, Debug)]
enum EdgeSource {
    KEGG { pathway: String },
    DoRothEA { cascade_id: Option<usize> },
}

#[derive(Clone, Debug)]
struct Edge {
    src: usize,
    tgt: usize,
    weight: f64,
    source: EdgeSource,
}

// ============================================================
// 2. Beta Calculator
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
// 3. CascadePath
// ============================================================

#[pyclass]
#[derive(Clone, Debug)]
pub struct CascadePath {
    #[pyo3(get)]
    pub tfs: Vec<String>,
    #[pyo3(get)]
    pub betas: Vec<f64>,
    #[pyo3(get)]
    pub total_beta: f64,
    #[pyo3(get)]
    pub path_id: usize,
}

#[pymethods]
impl CascadePath {
    #[new]
    #[pyo3(signature = (tfs, betas, total_beta, path_id = 0))]
    fn new(tfs: Vec<String>, betas: Vec<f64>, total_beta: f64, path_id: usize) -> Self {
        Self { tfs, betas, total_beta, path_id }
    }
    
    fn start_tf(&self) -> Option<String> {
        self.tfs.first().cloned()
    }
    
    fn __repr__(&self) -> String {
        format!("CascadePath({}, β={:.2}, len={})", 
            self.tfs.join("→"), self.total_beta, self.tfs.len())
    }
    
    fn __len__(&self) -> usize {
        self.tfs.len()
    }
}

// ============================================================
// 4. KEGGPathway
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
        Self { name, sources, targets, weights, modifications, effects, types, indirects }
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
            .iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();
        
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
            .iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();
        
        let global_indices: Vec<usize> = genes
            .iter().map(|g| *gene_to_global.get(g).unwrap()).collect();
        
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
            (0..n_cells).into_par_iter().map(|cell| {
                let alpha: Vec<f64> = global_indices.iter().map(|&g| expr[[cell, g]]).collect();
                // [수정] 전체 유전자 발현 총합
                let alpha_g: f64 = expr.row(cell).iter().sum();
                if alpha_g < 1e-10 { return 0.0; }

                let alpha_g_sq = alpha_g * alpha_g;
                let norm_sq: f64 = edges.iter().map(|&(i, j, w)| {
                    let beta = calc.compute(alpha[i], alpha[j], w);
                    (alpha[i] * alpha[j] / alpha_g_sq) * beta * beta
                }).sum();

                norm_sq.sqrt()
            }).collect()
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
            .iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();

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
            .iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();

        let global_indices: Vec<usize> = genes
            .iter().map(|g| *gene_to_global.get(g).unwrap()).collect();

        // global_col → local_gene mapping for sparse row lookup
        let global_to_local: HashMap<usize, usize> = global_indices.iter()
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
            (0..n_cells).into_par_iter().map(|cell| {
                let mut alpha = vec![0.0f64; n_local];

                // Sparse row access: only non-zero entries
                let row = csr.row(cell);
                // [수정] sparse row 전체 합 = 전체 유전자 발현 총합
                let alpha_g: f64 = row.values().iter().sum();
                for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                    if let Some(&local) = global_to_local.get(&col) {
                        alpha[local] = val;
                    }
                }

                if alpha_g < 1e-10 { return 0.0; }

                let alpha_g_sq = alpha_g * alpha_g;
                let norm_sq: f64 = edges.iter().map(|&(i, j, w)| {
                    let beta = calc.compute(alpha[i], alpha[j], w);
                    (alpha[i] * alpha[j] / alpha_g_sq) * beta * beta
                }).sum();

                norm_sq.sqrt()
            }).collect()
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
        self.effects.iter()
            .zip(self.types.iter())
            .zip(self.indirects.iter())
            .map(|((&eff, typ), &ind)| {
                let e = match eff { 1 => 1.0, 2 => 0.8, -1 => -0.8, _ => 0.5 };
                let t = match typ.as_str() { "PPrel"|"GErel" => 1.0, "PCrel" => 0.8, "PComplex" => 0.5, _ => 0.5 };
                let i = if ind { 0.7 } else { 1.0 };
                e * t * i
            })
            .collect()
    }
}

// ============================================================
// 5. KEGG Helper
// ============================================================

#[pyfunction]
pub fn make_kegg_edges_bidirectional(
    sources: Vec<String>,
    targets: Vec<String>,
    effects: Vec<i32>,
    types: Vec<String>,
    indirects: Vec<bool>,
    modifications: Vec<String>,
) -> PyResult<(Vec<String>, Vec<String>, Vec<i32>, Vec<String>, Vec<bool>, Vec<String>)> {
    let mut r_src = sources.clone();
    let mut r_tgt = targets.clone();
    let mut r_eff = effects.clone();
    let mut r_typ = types.clone();
    let mut r_ind = indirects.clone();
    let mut r_mod = modifications.clone();
    
    let mut existing: HashSet<(String, String)> = sources.iter()
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
// 6. SelectiveIntegratedGraph
// ============================================================

#[pyclass]
#[derive(Clone)]
pub struct SelectiveIntegratedGraph {
    pathway_genes: Vec<String>,
    gene_global_indices: Vec<usize>,
    edges: Vec<Edge>,
    calc: BetaCalculator,
    
    #[pyo3(get)]
    pub kegg_name: String,
    #[pyo3(get)]
    pub cascade_id: usize,
    #[pyo3(get)]
    pub connection_tf: String,
    #[pyo3(get)]
    n_genes: usize,
    #[pyo3(get)]
    n_kegg_edges: usize,
    #[pyo3(get)]
    n_dorothea_edges: usize,
    #[pyo3(get)]
    n_total_edges: usize,
}

#[pymethods]
impl SelectiveIntegratedGraph {
    /// 단일 (KEGG, Cascade) 쌍으로 그래프 생성
    /// 연결점이 없으면 None 반환
    #[staticmethod]
    #[pyo3(signature = (adata_genes, kegg_pathway, cascade_path, doro_sources, doro_targets, doro_weights, beta_threshold = 0.0))]
    fn try_new(
        adata_genes: Vec<String>,
        kegg_pathway: KEGGPathway,
        cascade_path: CascadePath,
        doro_sources: Vec<String>,
        doro_targets: Vec<String>,
        doro_weights: Vec<f64>,
        beta_threshold: f64,
    ) -> Option<Self> {
        // KEGG mod='e' source TFs
        let kegg_e_tfs: HashSet<String> = kegg_pathway.sources.iter()
            .zip(kegg_pathway.modifications.iter())
            .filter(|(_, m)| m.as_str() == "e")
            .map(|(s, _)| s.clone())
            .collect();
        
        // 연결점 찾기: cascade TF 중 KEGG mod='e' TF와 겹치는 것
        let connection_tf = cascade_path.tfs.iter()
            .find(|tf| kegg_e_tfs.contains(*tf))?
            .clone();
        
        Some(Self::build_internal(
            adata_genes, kegg_pathway, cascade_path,
            doro_sources, doro_targets, doro_weights,
            beta_threshold, connection_tf,
        ))
    }
    
    /// G2 norm 계산 (모든 cell, 병렬)
    fn compute_all_norms(
        &self,
        py: Python<'_>,
        expr_matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let expr = expr_matrix.as_array();
        let n_cells = expr.nrows();
        let genes = &self.gene_global_indices;
        let edges = &self.edges;
        let calc = self.calc;
        
        let norms: Vec<f64> = py.allow_threads(|| {
            (0..n_cells).into_par_iter().map(|cell| {
                let alpha: Vec<f64> = genes.iter().map(|&g| expr[[cell, g]]).collect();
                // [수정] 전체 유전자 발현 총합
                let alpha_g: f64 = expr.row(cell).iter().sum();
                if alpha_g < 1e-10 { return 0.0; }

                let alpha_g_sq = alpha_g * alpha_g;
                let norm_sq: f64 = edges.iter().map(|e| {
                    let beta = calc.compute(alpha[e.src], alpha[e.tgt], e.weight);
                    (alpha[e.src] * alpha[e.tgt] / alpha_g_sq) * beta * beta
                }).sum();

                norm_sq.sqrt()
            }).collect()
        });

        Ok(PyArray1::from_vec_bound(py, norms).into())
    }

    /// G2 norm (sparse matrix version) — no toarray() needed
    fn compute_all_norms_sparse(
        &self,
        py: Python<'_>,
        sparse_matrix: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let csr = csr_to_rust(sparse_matrix)?;
        let n_cells = csr.nrows();
        let edges = &self.edges;
        let calc = self.calc;

        // global_col → local_gene mapping
        let global_to_local: HashMap<usize, usize> = self.gene_global_indices.iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        let n_local = self.gene_global_indices.len();

        let norms: Vec<f64> = py.allow_threads(|| {
            (0..n_cells).into_par_iter().map(|cell| {
                let mut alpha = vec![0.0f64; n_local];

                let row = csr.row(cell);
                // [수정] sparse row 전체 합 = 전체 유전자 발현 총합
                let alpha_g: f64 = row.values().iter().sum();
                for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                    if let Some(&local) = global_to_local.get(&col) {
                        alpha[local] = val;
                    }
                }

                if alpha_g < 1e-10 { return 0.0; }

                let alpha_g_sq = alpha_g * alpha_g;
                let norm_sq: f64 = edges.iter().map(|e| {
                    let beta = calc.compute(alpha[e.src], alpha[e.tgt], e.weight);
                    (alpha[e.src] * alpha[e.tgt] / alpha_g_sq) * beta * beta
                }).sum();

                norm_sq.sqrt()
            }).collect()
        });

        Ok(PyArray1::from_vec_bound(py, norms).into())
    }

    fn get_pathway_genes(&self) -> Vec<String> {
        self.pathway_genes.clone()
    }
    
    fn get_edges_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        let mut edge_types = Vec::new();
        let mut pathways = Vec::new();
        let mut cascade_ids = Vec::new();
        
        for e in &self.edges {
            sources.push(self.pathway_genes[e.src].clone());
            targets.push(self.pathway_genes[e.tgt].clone());
            weights.push(e.weight);
            
            match &e.source {
                EdgeSource::KEGG { pathway } => {
                    edge_types.push("KEGG".to_string());
                    pathways.push(pathway.clone());
                    cascade_ids.push(-1i64);
                }
                EdgeSource::DoRothEA { cascade_id } => {
                    edge_types.push("DoRothEA".to_string());
                    pathways.push("".to_string());
                    cascade_ids.push(cascade_id.map(|id| id as i64).unwrap_or(-1));
                }
            }
        }
        
        dict.set_item("source", sources)?;
        dict.set_item("target", targets)?;
        dict.set_item("weight", weights)?;
        dict.set_item("edge_type", edge_types)?;
        dict.set_item("pathway", pathways)?;
        dict.set_item("cascade_id", cascade_ids)?;
        
        Ok(dict.into())
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SelectiveIntegratedGraph(KEGG='{}', cascade={}, via='{}', genes={}, edges={})",
            self.kegg_name, self.cascade_id, self.connection_tf,
            self.n_genes, self.n_total_edges
        )
    }
}

// Internal builder (not exposed to Python)
impl SelectiveIntegratedGraph {
    fn build_internal(
        adata_genes: Vec<String>,
        kegg_pathway: KEGGPathway,
        cascade_path: CascadePath,
        doro_sources: Vec<String>,
        doro_targets: Vec<String>,
        doro_weights: Vec<f64>,
        beta_threshold: f64,
        connection_tf: String,
    ) -> Self {
        // DoRothEA weight lookup
        let doro_weight: HashMap<(String, String), f64> = doro_sources.iter()
            .zip(doro_targets.iter())
            .zip(doro_weights.iter())
            .map(|((s, t), &w)| ((s.clone(), t.clone()), w))
            .collect();
        
        // Gene mapping
        let gene_to_global: HashMap<String, usize> = adata_genes.iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();
        
        let mut genes_set: HashSet<String> = HashSet::new();
        
        // KEGG genes
        for g in kegg_pathway.sources.iter().chain(kegg_pathway.targets.iter()) {
            if gene_to_global.contains_key(g) {
                genes_set.insert(g.clone());
            }
        }
        
        // Cascade genes (모든 TF 포함)
        for tf in &cascade_path.tfs {
            if gene_to_global.contains_key(tf) {
                genes_set.insert(tf.clone());
            }
        }
        
        let mut pathway_genes: Vec<String> = genes_set.into_iter().collect();
        pathway_genes.sort();
        
        let gene_to_local: HashMap<String, usize> = pathway_genes.iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();
        
        let gene_global_indices: Vec<usize> = pathway_genes.iter()
            .map(|g| *gene_to_global.get(g).unwrap())
            .collect();
        
        // Edge 수집
        let mut edges: Vec<Edge> = Vec::new();
        let mut n_kegg = 0;
        let mut n_doro = 0;
        
        // KEGG edges
        let kegg_weights = kegg_pathway.compute_base_weights_internal();
        for i in 0..kegg_pathway.sources.len() {
            if let (Some(&src), Some(&tgt)) = (
                gene_to_local.get(&kegg_pathway.sources[i]),
                gene_to_local.get(&kegg_pathway.targets[i]),
            ) {
                edges.push(Edge {
                    src, tgt,
                    weight: kegg_weights[i],
                    source: EdgeSource::KEGG { pathway: kegg_pathway.name.clone() },
                });
                n_kegg += 1;
            }
        }
        
        // DoRothEA cascade edges (beta >= threshold만)
        for i in 0..cascade_path.tfs.len().saturating_sub(1) {
            let edge_beta = cascade_path.betas.get(i).copied().unwrap_or(0.0);
            if edge_beta < beta_threshold { continue; }
            
            let src_tf = &cascade_path.tfs[i];
            let tgt_tf = &cascade_path.tfs[i + 1];
            
            let w = doro_weight
                .get(&(src_tf.clone(), tgt_tf.clone()))
                .copied()
                .unwrap_or(edge_beta);
            
            if let (Some(&src), Some(&tgt)) = (
                gene_to_local.get(src_tf),
                gene_to_local.get(tgt_tf),
            ) {
                edges.push(Edge {
                    src, tgt,
                    weight: w,
                    source: EdgeSource::DoRothEA { cascade_id: Some(cascade_path.path_id) },
                });
                n_doro += 1;
            }
        }
        
        Self {
            pathway_genes,
            gene_global_indices,
            edges,
            calc: BetaCalculator,
            kegg_name: kegg_pathway.name.clone(),
            cascade_id: cascade_path.path_id,
            connection_tf,
            n_genes: gene_to_local.len(),
            n_kegg_edges: n_kegg,
            n_dorothea_edges: n_doro,
            n_total_edges: n_kegg + n_doro,
        }
    }
}

// ============================================================
// 7. 배치 함수
// ============================================================

#[pyfunction]
#[pyo3(signature = (adata_genes, kegg_pathways, cascade_paths, doro_sources, doro_targets, doro_weights, beta_threshold = 0.0, verbose = true))]
pub fn build_all_integrated_graphs(
    adata_genes: Vec<String>,
    kegg_pathways: Vec<KEGGPathway>,
    cascade_paths: Vec<CascadePath>,
    doro_sources: Vec<String>,
    doro_targets: Vec<String>,
    doro_weights: Vec<f64>,
    beta_threshold: f64,
    verbose: bool,
) -> Vec<SelectiveIntegratedGraph> {
    
    if verbose {
        println!("============================================================");
        println!("Building all (KEGG, Cascade) pairs");
        println!("  KEGG: {}, Cascades: {}", kegg_pathways.len(), cascade_paths.len());
        println!("  Pairs to check: {}", kegg_pathways.len() * cascade_paths.len());
        println!("  beta_threshold: {:.2}", beta_threshold);
        println!("============================================================");
    }
    
    let mut results: Vec<SelectiveIntegratedGraph> = Vec::new();
    
    for kegg in &kegg_pathways {
        for cascade in &cascade_paths {
            if let Some(graph) = SelectiveIntegratedGraph::try_new(
                adata_genes.clone(),
                kegg.clone(),
                cascade.clone(),
                doro_sources.clone(),
                doro_targets.clone(),
                doro_weights.clone(),
                beta_threshold,
            ) {
                if verbose {
                    println!("  ✓ {} + cascade_{} via {}", 
                        graph.kegg_name, graph.cascade_id, graph.connection_tf);
                }
                results.push(graph);
            }
        }
    }
    
    if verbose {
        println!("------------------------------------------------------------");
        println!("  Connected: {}/{}", results.len(), kegg_pathways.len() * cascade_paths.len());
        println!("============================================================");
    }
    
    results
}

// ============================================================
// Batch sparse norm functions (CSR 변환 1회)
// ============================================================


// selective_integrated.rs — #[pyfunction] 영역에 추가
#[pyfunction]
pub fn compute_all_kegg_norms_cluster_mean(
    py: Python<'_>,
    sparse_matrix: &Bound<'_, PyAny>,
    adata_genes: Vec<String>,
    pathways: Vec<PyRef<KEGGPathway>>,
    cluster_indices: Vec<usize>,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;

    let csr       = csr_to_rust(sparse_matrix)?;
    let n_cluster = cluster_indices.len();
    let n_genes   = adata_genes.len();

    let gene_to_global: HashMap<String, usize> = adata_genes
        .iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();

    // ── cluster mean α (전체 유전자) 한 번만 계산 ───────────
    let mut gene_sum    = vec![0.0f64; n_genes];
    let mut alpha_g_sum = 0.0f64;

    for &cell in &cluster_indices {
        let row = csr.row(cell);
        alpha_g_sum += row.values().iter().sum::<f64>();
        for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
            if col < n_genes { gene_sum[col] += val; }
        }
    }

    let n_c      = n_cluster as f64;
    let alpha_g  = alpha_g_sum / n_c;
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

    // ── pathway별 G2(mean_α) ────────────────────────────────
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

/// 모든 KEGGPathway의 norm을 한번의 CSR 변환으로 계산
/// 세포 하나하나 계산하여 heavy.
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
        .iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();

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
            .iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();

        let global_indices: Vec<usize> = genes
            .iter().map(|g| *gene_to_global.get(g).unwrap()).collect();

        let global_to_local: HashMap<usize, usize> = global_indices.iter()
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
            (0..n_cells).into_par_iter().map(|cell| {
                let mut alpha = vec![0.0f64; n_local];

                let row = csr.row(cell);
                // [수정] sparse row 전체 합 = 전체 유전자 발현 총합
                let alpha_g: f64 = row.values().iter().sum();
                for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                    if let Some(&local) = global_to_local.get(&col) {
                        alpha[local] = val;
                    }
                }

                if alpha_g < 1e-10 { return 0.0; }

                let alpha_g_sq = alpha_g * alpha_g;
                let norm_sq: f64 = edges.iter().map(|&(i, j, w)| {
                    let beta = calc.compute(alpha[i], alpha[j], w);
                    (alpha[i] * alpha[j] / alpha_g_sq) * beta * beta
                }).sum();

                norm_sq.sqrt()
            }).collect()
        });

        result_dict.set_item(
            &pathway.name,
            PyArray1::from_vec_bound(py, norms),
        )?;
    }

    Ok(result_dict.into())
}

/// 모든 SelectiveIntegratedGraph의 norm을 한번의 CSR 변환으로 계산
#[pyfunction]
pub fn compute_all_integrated_norms_sparse(
    py: Python<'_>,
    sparse_matrix: &Bound<'_, PyAny>,
    graphs: Vec<PyRef<SelectiveIntegratedGraph>>,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;

    let csr = csr_to_rust(sparse_matrix)?;
    let n_cells = csr.nrows();

    let result_dict = PyDict::new_bound(py);

    for graph in graphs.iter() {
        let edges = &graph.edges;
        let calc = graph.calc;

        let global_to_local: HashMap<usize, usize> = graph.gene_global_indices.iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        let n_local = graph.gene_global_indices.len();
        let col_name = format!("{}_cascade{}", graph.kegg_name, graph.cascade_id);

        let norms: Vec<f64> = py.allow_threads(|| {
            (0..n_cells).into_par_iter().map(|cell| {
                let mut alpha = vec![0.0f64; n_local];

                let row = csr.row(cell);
                // [수정] sparse row 전체 합 = 전체 유전자 발현 총합
                let alpha_g: f64 = row.values().iter().sum();
                for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                    if let Some(&local) = global_to_local.get(&col) {
                        alpha[local] = val;
                    }
                }

                if alpha_g < 1e-10 { return 0.0; }

                let alpha_g_sq = alpha_g * alpha_g;
                let norm_sq: f64 = edges.iter().map(|e| {
                    let beta = calc.compute(alpha[e.src], alpha[e.tgt], e.weight);
                    (alpha[e.src] * alpha[e.tgt] / alpha_g_sq) * beta * beta
                }).sum();

                norm_sq.sqrt()
            }).collect()
        });

        result_dict.set_item(col_name, PyArray1::from_vec_bound(py, norms))?;
    }

    Ok(result_dict.into())
}
