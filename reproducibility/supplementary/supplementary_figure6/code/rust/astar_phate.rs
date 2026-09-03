// src/astar_phate.rs
// ============================================================
// Parallel PHATE A* path search in PHATE coordinate space
// ============================================================
//
// Python prepares the PHATE coordinates, n_genes values, candidate
// start and goal cells, neighborhood radius, gene-weight parameter
// and iteration limit. It calls single Rust entry point:
//     cwg_rust.astar_all_pairs(
//         coords,        ← adata.obsm["X_phate"]  (numpy, zero-copy)
//         gene_values,   ← adata.obs[gene_col].values (numpy, zero-copy)
//         low_indices,   ← Python list[int]
//         high_indices,  ← Python list[int]
//         delta,
//         gene_weight,
//         max_iter,
//     ) -> list[list[int]]
//
// Rust converts the input arrays to contiguous f32 buffers, builds
// an internal KDTree and searches all candidate endpoint pairs in
// parallel with rayon while the Python GIL is released.
//
// Each search uses accumulated L2 arc length as g and
//
//     h = L2(current, goal)
//         + gene_weight * abs(gene_values[goal] - gene_values[current]).
//
// ============================================================

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;
use rand::Rng;


// ============================================================
// 1. A* min-heap entry
//    BinaryHeap is a max-heap; neg_f = -f_score makes it operate as a min-heap.
// ============================================================


#[derive(PartialEq)]
struct EntryL2 {
    neg_f:  f64,
    g_bits: u32,   // f32::to_bits(g_score)
    idx:    u32,
}
 
impl Eq for EntryL2 {}
 
impl PartialOrd for EntryL2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
 
impl Ord for EntryL2 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.neg_f
            .partial_cmp(&other.neg_f)
            .unwrap_or(Ordering::Equal)
    }
}
 
// ============================================================
// 2. Internal KDTree
//
//    Rationale: rayon threads cannot access Python objects such as scipy KDTree.
//    A Rust-native KDTree is required to call query_ball without the GIL.
//
//    Properties:
//    - PHATE has 2-3 dimensions, yielding shallow trees and effective pruning.
//    - Nodes use arena Vec<Node> indices and avoid per-node heap allocation.
//    - Coordinates use a flat Vec<f32>, reducing storage per copied value relative to f64.
//    - Construction uses select_nth_unstable for O(n log n) median partitioning.
// ============================================================

enum Node {
    Leaf(u32),
    Internal {
        dim:   u8,    // Split dimension; u8 is sufficient for 2-3 dimensions.
        val:   f32,   // Split coordinate value.
        left:  u32,   // Index into nodes[left].
        right: u32,   // Index into nodes[right].
    },
}

struct KDTree {
    ndim:   usize,
    coords: Vec<f32>,   // coords[cell * ndim + dim]
    nodes:  Vec<Node>,
    root:   u32,
}

impl KDTree {
    fn build(coords_f32: Vec<f32>, ndim: usize) -> Self {
        let n = coords_f32.len() / ndim;
        let mut nodes: Vec<Node> = Vec::with_capacity(2 * n);
        let mut idx: Vec<u32> = (0u32..n as u32).collect();
        let root = Self::build_rec(&coords_f32, ndim, &mut idx, 0, n, &mut nodes);
        KDTree { ndim, coords: coords_f32, nodes, root }
    }

    fn build_rec(
        coords: &[f32],
        ndim:   usize,
        idx:    &mut [u32],
        start:  usize,
        end:    usize,
        nodes:  &mut Vec<Node>,
    ) -> u32 {
        let count = end - start;

        if count == 1 {
            let ni = nodes.len() as u32;
            nodes.push(Node::Leaf(idx[start]));
            return ni;
        }

        // Select the dimension with the largest spread.
        let dim = (0..ndim).max_by(|&a, &b| {
            let spread = |d: usize| {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &ci in &idx[start..end] {
                    let v = coords[ci as usize * ndim + d];
                    if v < lo { lo = v; }
                    if v > hi { hi = v; }
                }
                hi - lo
            };
            spread(a).partial_cmp(&spread(b)).unwrap_or(Ordering::Equal)
        }).unwrap_or(0);

        // Partition at the median using an O(n) partial sort.
        let mid    = start + count / 2;
        let sub    = &mut idx[start..end];
        let mid_rel = count / 2;
        sub.select_nth_unstable_by(mid_rel, |&a, &b| {
            coords[a as usize * ndim + dim]
                .partial_cmp(&coords[b as usize * ndim + dim])
                .unwrap_or(Ordering::Equal)
        });
        let split_val = coords[idx[mid] as usize * ndim + dim];

        // Push a placeholder first, then replace it after recursively building the children.
        let ni = nodes.len() as u32;
        nodes.push(Node::Leaf(0));

        let left  = Self::build_rec(coords, ndim, idx, start, mid, nodes);
        let right = Self::build_rec(coords, ndim, idx, mid,   end, nodes);

        nodes[ni as usize] = Node::Internal {
            dim: dim as u8,
            val: split_val,
            left,
            right,
        };
        ni
    }

    /// Return indices of cells within radius of cell_idx
    #[inline]
    fn query_ball(&self, cell_idx: usize, r: f32) -> Vec<u32> {
        let off = cell_idx * self.ndim;
        let center = &self.coords[off..off + self.ndim];
        let mut out = Vec::new();
        self.query_rec(self.root, center, r, &mut out);
        out
    }

    fn query_rec(&self, ni: u32, center: &[f32], r: f32, out: &mut Vec<u32>) {
        match &self.nodes[ni as usize] {
            Node::Leaf(ci) => {
                if self.l2(center, *ci as usize) <= r {
                    out.push(*ci);
                }
            }
            Node::Internal { dim, val, left, right } => {
                let d = center[*dim as usize] - val;
                // Search the center side first and the opposite side when the split-plane distance is ≤ r.
                if d <= 0.0 {
                    self.query_rec(*left,  center, r, out);
                    if -d <= r { self.query_rec(*right, center, r, out); }
                } else {
                    self.query_rec(*right, center, r, out);
                    if  d <= r { self.query_rec(*left,  center, r, out); }
                }
            }
        }
    }

    #[inline]
    fn l2(&self, center: &[f32], ci: usize) -> f32 {
        let off = ci * self.ndim;
        center.iter()
            .zip(&self.coords[off..off + self.ndim])
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }

    #[inline]
    fn coord_slice(&self, ci: usize) -> &[f32] {
        let off = ci * self.ndim;
        &self.coords[off..off + self.ndim]
    }
}


// =============================================================================
// 3. Weighted heuristic
//
//    h(current) = ||coords[current] - coords[goal]||_2
//                 + gene_weight * |gene_values[goal] - gene_values[current]|
//
// The first term is a spatial distance in PHATE coordinates. The second term
// biases the search toward cells with gene-complexity values closer to the
// goal. Because g accumulates spatial arc length only, the added gene term is
// not guaranteed to be admissible with respect to g; the routine should be
// interpreted as a guided A* traversal rather than a guaranteed geodesic solver.
// ==============================================================================

#[inline]
fn heuristic_l2(
    tree:        &KDTree,
    gene_vals:   &[f32],
    gene_weight: f32,
    current:     usize,
    goal:        usize,
) -> f32 {
    let cur_c  = tree.coord_slice(current);
    let goal_c = tree.coord_slice(goal);
 
    // L2 spatial distance in the same units as g.
    let spatial: f32 = cur_c.iter()
        .zip(goal_c.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
 
    let gene_dist = (gene_vals[goal] - gene_vals[current]).abs();
 
    spatial + gene_weight * gene_dist
}

// ============================================================
// 4. Path reconstruction with generation-tagged parent pointers
//
//    Generation-array technique:
//      Record the current search generation at each node instead of resetting
//      the entire came_from array to u32::MAX for every A* search.
//      During backtracking, a generation mismatch acts as a safety boundary;
//      came_from[node] == u32::MAX marks the initialized start node.
//      This gives O(path_len) reconstruction and avoids clearing every parent slot.
// ============================================================


#[inline]
fn reconstruct_path(
    came_from:  &[u32],
    generation: &[u32],
    gen_id:     u32,
    goal:       usize,
) -> Vec<usize> {
    let mut path = Vec::new();
    let mut cur  = goal;

    loop {
        path.push(cur);
        // Check whether the parent was set during the current search.
        if generation[cur] != gen_id {
            break;
        }
        let parent = came_from[cur];
        if parent == u32::MAX {
            break; // Reached the start node.
        }
        cur = parent as usize;
    }

    path.reverse();
    path
}



// ============================================================
// 5. Single-pair guided A* search
//
// g is accumulated L2 arc length in PHATE space. The search is deterministic
// for fixed inputs and contains no stochastic noise term. max_iter limits the
// number of finalized, non-stale heap entries.
// ============================================================
 
fn astar_single(
    tree:        &KDTree,
    gene_vals:   &[f32],
    gene_weight: f32,
    delta:       f32,
    start:       usize,
    goal:        usize,
    g_score:     &mut Vec<f32>,
    came_from:   &mut Vec<u32>,
    visited:     &mut Vec<bool>,
    generation:  &mut Vec<u32>,
    gen_id:      u32,
    max_iter:    u32,
) -> Option<Vec<usize>> {
 
    g_score[start]    = 0.0;
    came_from[start]  = u32::MAX;
    visited[start]    = false;
    generation[start] = gen_id;
 
    let start_h = heuristic_l2(tree, gene_vals, gene_weight, start, goal);
 
    let mut heap: BinaryHeap<EntryL2> = BinaryHeap::new();
    heap.push(EntryL2 {
        neg_f:  -(start_h as f64),
        g_bits: 0f32.to_bits(),
        idx:    start as u32,
    });
 
    let mut iter_count = 0u32;
 
    while let Some(entry) = heap.pop() {
        let current = entry.idx as usize;
        let g_cur   = f32::from_bits(entry.g_bits);
 
        if current == goal {
            return Some(reconstruct_path(came_from, generation, gen_id, goal));
        }
 
        // Skip stale entries before increasing the counter.
        if generation[current] == gen_id && visited[current] {
            continue;
        }
        if generation[current] == gen_id && g_cur > g_score[current] {
            continue;
        }
 
        
        // Count finalized nodes after excluding stale heap entries.
        iter_count += 1;
        if iter_count > max_iter {
            return None;
        }
 
        visited[current]    = true;
        generation[current] = gen_id;
 
        let neighbors = tree.query_ball(current, delta);
 
        for nb_u32 in neighbors {
            let nb = nb_u32 as usize;
 
            if generation[nb] == gen_id && visited[nb] {
                continue;
            }
 
            // g accumulates L2 arc length rather than hop count.
            let arc         = tree.l2(tree.coord_slice(current), nb);
            let tentative_g = g_cur + arc;
 
            let old_g = if generation[nb] == gen_id {
                g_score[nb]
            } else {
                f32::INFINITY
            };
 
            if tentative_g < old_g {
                g_score[nb]    = tentative_g;
                came_from[nb]  = current as u32;
                generation[nb] = gen_id;
                visited[nb]    = false;
 
                let h = heuristic_l2(tree, gene_vals, gene_weight, nb, goal);
                let f = tentative_g + h;
 
                heap.push(EntryL2 {
                    neg_f:  -(f as f64),
                    g_bits: tentative_g.to_bits(),
                    idx:    nb_u32,
                });
            }
        }
    }
 
    None
}
// ============================================================
// 6. Python-exposed all-pairs entry point
//
// Input arrays are first viewed through PyReadonlyArray and then copied and
// converted to owned f32 buffers. The endpoint Cartesian product is processed
// with rayon. map_init allocates one reusable search workspace per worker.
// Callers must supply a non-empty two-dimensional coordinate array, one gene
// value per cell and valid endpoint indices; these preconditions are enforced
// by the Python workflow rather than checked exhaustively in this function.
// ============================================================
 
#[pyfunction]
#[pyo3(signature = (
    coords,
    gene_values,
    low_indices,
    high_indices,
    delta,
    gene_weight,
    max_iter = None,
))]
pub fn astar_all_pairs(
    py:          Python<'_>,
    coords:      PyReadonlyArray2<f64>,
    gene_values: PyReadonlyArray1<f64>,
    low_indices:  Vec<usize>,
    high_indices: Vec<usize>,
    delta:        f64,
    gene_weight:  f64,
    max_iter:     Option<u64>,
) -> PyResult<PyObject> {
 
    let coords_arr = coords.as_array();
    let gene_arr   = gene_values.as_array();
 
    let n_cells = coords_arr.nrows();
    let ndim    = coords_arr.ncols();
 
    let coords_f32: Vec<f32> = coords_arr.iter().map(|&v| v as f32).collect();
    let gene_f32:   Vec<f32> = gene_arr.iter().map(|&v| v as f32).collect();
 
    let delta_f32       = delta       as f32;
    let gene_weight_f32 = gene_weight as f32;
 
    // Use the supplied per-search limit or default to three times the cell count.
    let max_iter_u32: u32 = match max_iter {
        Some(v) => v.min(u32::MAX as u64) as u32,
        None    => (n_cells as u64 * 3).min(u32::MAX as u64) as u32,
    };
    // ── Build the KDTree once and share it through Arc ── 
    // Build with the GIL released, independently of the Python scipy KDTree.
    // The owned f32 coordinate buffer duplicates input data but can be shared safely.
    let tree = py.allow_threads(|| {
        Arc::new(KDTree::build(coords_f32, ndim))
    });
    let gene_vals = Arc::new(gene_f32);
 
    let pairs: Vec<(usize, usize)> = low_indices.iter()
        .flat_map(|&s| high_indices.iter().map(move |&e| (s, e)))
        .collect();
 
    let n_pairs = pairs.len();
    
    // ── Parallel A* with rayon ─────────────────────────────────
    //
    // map_init runs its closure once per worker to initialize reusable buffers.
    // This avoids reallocating search vectors for every endpoint pair.
    //
    // Generation technique:
    //   Increment gen_id for each endpoint-pair search.
    //   A node with generation[node] != gen_id is considered unvisited in the
    //   current search, providing O(1) logical initialization.
    //   Run Vec::fill only when gen_id overflows and wraps as u32.
    //
    // Release the GIL with py.allow_threads; rayon threads do not access Python objects.
    let results: Vec<Option<Vec<usize>>> = py.allow_threads(|| {
        pairs.into_par_iter().map_init(
            || (
                vec![f32::INFINITY; n_cells],
                vec![u32::MAX;      n_cells],
                vec![false;         n_cells],
                vec![0u32;          n_cells],
                0u32,
            ),
            |(g_score, came_from, visited, generation, gen_id), (start, goal)| {
 
                *gen_id = gen_id.wrapping_add(1);
                if *gen_id == 0 {
                    g_score.fill(f32::INFINITY);
                    came_from.fill(u32::MAX);
                    visited.fill(false);
                    generation.fill(0);
                    *gen_id = 1;
                }
 
                astar_single(
                    &tree, &gene_vals,
                    gene_weight_f32, delta_f32,
                    start, goal,
                    g_score, came_from, visited, generation,
                    *gen_id, max_iter_u32,
                )
            },
        ).collect()
    });
 
    let found: Vec<Vec<usize>> = results.into_iter().flatten().collect();
 
    println!(
        "[astar_all_pairs_v2] {}/{} pairs → {} paths found  (max_iter={})",
        found.len(), n_pairs, found.len(), max_iter_u32
    );
 
    let py_outer = PyList::empty_bound(py);
    for path in found {
        let py_inner = PyList::new_bound(py, &path);
        py_outer.append(py_inner)?;
    }
    Ok(py_outer.into())
}
