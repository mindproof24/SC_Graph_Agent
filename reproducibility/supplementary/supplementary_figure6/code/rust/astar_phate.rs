// src/astar_phate.rs
// ============================================================
// PHATE A* 병렬 경로 탐색 — Python 구조 보존, 무거운 부분만 Rust
// ============================================================
//
// 설계 원칙:
//   PhateGenePathFinder 클래스의 Python 구조는 100% 유지.
//   Python이 호출하는 단 하나의 진입점:
//
//     cwg_rust.astar_all_pairs(
//         coords,        ← adata.obsm["X_phate"]  (numpy, zero-copy)
//         gene_values,   ← adata.obs[gene_col].values (numpy, zero-copy)
//         low_indices,   ← Python list[int]
//         high_indices,  ← Python list[int]
//         delta,
//         gene_weight,
//         noise_scale,   ← delta * 1.5  (Python heuristic과 동일)
//     ) -> list[list[int]]
//
// Rust가 하는 일:
//   1. coords / gene_values numpy 배열을 zero-copy 슬라이스로 참조
//   2. 내부용 KDTree 빌드 (Arc로 rayon 스레드 간 공유, read-only)
//   3. 8100쌍 rayon par_iter — GIL 완전 해제 상태에서 병렬 A*
//      각 A*: BinaryHeap(min-heap) + Vec<f32> g_score + generation 배열
//
// Python이 그대로 유지하는 것:
//   - PhateGenePathFinder.__init__ / get_neighbors / heuristic
//   - reconstruct_path / get_optimized_params / select_candidates
//   - trajectory_find 시각화 로직
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
// 1. A* min-heap 엔트리
//    BinaryHeap은 max-heap → neg_f = -f_score 로 min-heap 동작
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
// 2. 내부 KDTree
//
//    이유: rayon 스레드에서 Python 객체(scipy KDTree)에 접근 불가.
//          GIL 없이 query_ball을 호출하려면 Rust 자체 KDTree 필요.
//
//    특성:
//    - PHATE는 2~3차원 → 트리 깊이가 낮아 pruning 매우 효과적
//    - 노드: arena(Vec<Node>) 인덱스 참조 → 힙 재귀 없음
//    - 좌표: flat Vec<f32> (f64→f32: 메모리 절반, SIMD 연산 친화)
//    - 빌드: select_nth_unstable 로 O(n log n) 중앙값 분할
// ============================================================

enum Node {
    Leaf(u32),
    Internal {
        dim:   u8,    // 분할 차원 (2~3이므로 u8 충분)
        val:   f32,   // 분할 좌표값
        left:  u32,   // nodes[left] 인덱스
        right: u32,   // nodes[right] 인덱스
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

        // 분산이 가장 큰 차원 선택
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

        // 중앙값으로 분할 (O(n), partial sort)
        let mid    = start + count / 2;
        let sub    = &mut idx[start..end];
        let mid_rel = count / 2;
        sub.select_nth_unstable_by(mid_rel, |&a, &b| {
            coords[a as usize * ndim + dim]
                .partial_cmp(&coords[b as usize * ndim + dim])
                .unwrap_or(Ordering::Equal)
        });
        let split_val = coords[idx[mid] as usize * ndim + dim];

        // placeholder 먼저 push → 자식 재귀 후 실제 값으로 교체
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

    /// cell_idx 기준 반경 r 이내 이웃 세포 인덱스 반환
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
                // center 쪽을 먼저 탐색, 분할면 거리 ≤ r이면 반대쪽도 탐색
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


// ============================================================
// 3. heuristic — Python heuristic() 수식 인라인 재현
//
//    Python 원본:
//      dist_spatial = np.linalg.norm(coords[cur] - coords[goal])
//      dist_gene    = abs(gene[goal] - gene[cur])
//      total        = dist_spatial + gene_weight * dist_gene
//      noise        = Uniform(1 - delta*1.5, 1 + delta*1.5)
//      return total * noise          ← find_shortest_path에서 / delta
//
//    Rust에서는 / delta까지 포함해 f_score 계산에 바로 사용.
// ============================================================


// ── v2: ng와 단위 일치
// admissibility: L2 triangle inequality → h(n) ≤ h*(n) 보장
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
 
    // L2 공간 거리 (g와 동일 단위)
    let spatial: f32 = cur_c.iter()
        .zip(goal_c.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
 
    let gene_dist = (gene_vals[goal] - gene_vals[current]).abs();
 
    spatial + gene_weight * gene_dist
}

// ============================================================
// 4. came_from 역추적 — Python reconstruct_path() 재현
//
//    generation 배열 기법:
//      매 A*마다 came_from 전체를 u32::MAX로 초기화하는 대신,
//      "이번 탐색 세대 번호"를 노드마다 기록.
//      역추적 시 generation[node] != gen_id 이면 시작점으로 간주.
//      → O(path_len) 역추적, O(방문 수) 초기화 비용 절감.
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
        // 이번 탐색에서 설정된 부모인지 확인
        if generation[cur] != gen_id {
            break;
        }
        let parent = came_from[cur];
        if parent == u32::MAX {
            break; // 시작점 도달
        }
        cur = parent as usize;
    }

    path.reverse();
    path
}


// ============================================================
// 6. astar_single_v2 — L2 g, L2 h, noise 없음, iter_count 위치 B
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
 
        // stale 체크 — 카운트 전에 skip
        if generation[current] == gen_id && visited[current] {
            continue;
        }
        if generation[current] == gen_id && g_cur > g_score[current] {
            continue;
        }
 
        // 노드 확정 시점에 카운트 (위치 B — stale pop 미포함)
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
 
            // g: L2 arc length 누적 (hop count 아님)
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
// 6. Python 노출 함수 — 단 하나의 진입점
//
//    PhateGenePathFinder에 아래 메서드를 추가:
//
//      def find_all_pairs(self, low_indices, high_indices, delta):
//          from cwg_rust import astar_all_pairs
//          return astar_all_pairs(
//              self.coords,           # numpy (n_cells, ndim)
//              self.gene_values,      # numpy (n_cells,)
//              low_indices,
//              high_indices,
//              delta,
//              self.gene_weight,
//              delta * 1.5,           # noise_scale
//          )
// ============================================================


// ============================================================
// 8. astar_all_pairs_v2 — v2 Python 노출 함수
//    변경점: noise_scale 없음, max_iter 추가, astar_single_v2 호출
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
 
    // max_iter: None이면 n_cells * 3 (Python에서 명시 권장)
    let max_iter_u32: u32 = match max_iter {
        Some(v) => v.min(u32::MAX as u64) as u32,
        None    => (n_cells as u64 * 3).min(u32::MAX as u64) as u32,
    };
    // ── KDTree 빌드 (1회, Arc 공유) ───────────────────────────
    // GIL 해제 상태에서 빌드: Python 쪽 scipy KDTree와 독립적으로 존재.
    // 메모리 이중 사용이지만 병렬화 이득이 압도적으로 큼.
    let tree = py.allow_threads(|| {
        Arc::new(KDTree::build(coords_f32, ndim))
    });
    let gene_vals = Arc::new(gene_f32);
 
    let pairs: Vec<(usize, usize)> = low_indices.iter()
        .flat_map(|&s| high_indices.iter().map(move |&e| (s, e)))
        .collect();
 
    let n_pairs = pairs.len();
    
    // ── rayon 병렬 A* ─────────────────────────────────────────
    //
    // map_init: 스레드당 1회 클로저를 실행해 재사용 버퍼 초기화.
    //   → 8100번 A*마다 Vec 재할당 없음.
    //
    // generation 기법:
    //   각 A* 호출마다 gen_id를 1씩 증가.
    //   노드에 기록된 generation[node]와 gen_id가 다르면
    //   "이번 탐색에서 미방문"으로 간주 → O(1) 논리적 초기화.
    //   gen_id overflow(u32 wrap) 시에만 실제 Vec::fill 수행.
    //
    // GIL 해제 (py.allow_threads):
    //   rayon 스레드는 Python 객체에 접근하지 않으므로 안전.
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
