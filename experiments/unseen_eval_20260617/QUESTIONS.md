# Evaluation Questions

This file records the exact prompts and answer labels used for the five UNSEEN 2-vs-2 tasks in this snapshot.

Note: the original task question text still contained the stale line `get_kg_context (marker / mechanism)`.
The actual evaluation harness was the NO_KG harness and exposed only these nine tools to the model:
`run_astar_pipeline`, `get_astar_graph_summary`, `get_astar_cellular_info`, `get_cluster_rl_map`,
`get_cluster_kegg_edges`, `get_cell_kegg_edges`, `execute_pipeline_code`,
`get_expressed_dorothea_edges`, and `custom_pathway_calc`.
The result trajectories in this snapshot contain zero `get_kg_context` calls.

## T Cell Tasks

### imm_atlas_TT_p01_CD8-positive_alpha-b_vs_effector_memory_CD8-_r000_anon

Cell types:

- A: CD8-positive, alpha-beta memory T cell, CD45RO-positive
- B: effector memory CD8-positive, alpha-beta T cell, terminally differentiated

Answer labels, in cell order:

```json
[
  "effector memory CD8-positive, alpha-beta T cell, terminally differentiated",
  "effector memory CD8-positive, alpha-beta T cell, terminally differentiated",
  "CD8-positive, alpha-beta memory T cell, CD45RO-positive",
  "CD8-positive, alpha-beta memory T cell, CD45RO-positive"
]
```

Prompt:

```text
Sample: imm_atlas_TT_p01_CD8-positive_alpha-b_vs_effector_memory_CD8-_r000_anon  (a 4-cell subset; obs has no leiden / cell_type / tissue)
Tissues (per cell, in order): ['blood', 'blood', 'bone marrow', 'mesenteric lymph node']
Cells (in order): [40424, 63059, 133324, 57310]

Two of these 4 cells are 'CD8-positive, alpha-beta memory T cell, CD45RO-positive', the other two are 'effector memory CD8-positive, alpha-beta T cell, terminally differentiated'.
Identify each cell's celltype.

Available tools (operate on this 4-cell sample):
  - custom_pathway_calc  (KG-free; design marker edges and score cells)
  - get_astar_cellular_info, get_cell_kegg_edges (per-cell evidence)
  - get_kg_context (marker / mechanism)
  - execute_pipeline_code (custom analysis on adata)
  Note: cluster-level tools and adata.obs metadata are not available here.
  Gene symbols are UPPERCASE human (e.g. CD3D, CD8A, CD4, FOXP3, TRDC, GZMB, not Cd3d/Cd8a).
Cite specific marker genes or pathways in your reason.
Time limit: 6 min  |  Tool call limit: 10
Answer format: {"labels": ["<celltype string>" for each cell in order], "reason": "<=200 chars"}
```

### imm_atlas_TT_p01_CD4-positive_helper__vs_effector_memory_CD4-_r000_anon

Cell types:

- A: CD4-positive helper T cell
- B: effector memory CD4-positive, alpha-beta T cell

Answer labels, in cell order:

```json
[
  "CD4-positive helper T cell",
  "effector memory CD4-positive, alpha-beta T cell",
  "effector memory CD4-positive, alpha-beta T cell",
  "CD4-positive helper T cell"
]
```

Prompt:

```text
Sample: imm_atlas_TT_p01_CD4-positive_helper__vs_effector_memory_CD4-_r000_anon  (a 4-cell subset; obs has no leiden / cell_type / tissue)
Tissues (per cell, in order): ['lamina propria', 'blood', 'blood', 'jejunal epithelium']
Cells (in order): [196244, 40498, 37402, 187032]

Two of these 4 cells are 'CD4-positive helper T cell', the other two are 'effector memory CD4-positive, alpha-beta T cell'.
Identify each cell's celltype.

Available tools (operate on this 4-cell sample):
  - custom_pathway_calc  (KG-free; design marker edges and score cells)
  - get_astar_cellular_info, get_cell_kegg_edges (per-cell evidence)
  - get_kg_context (marker / mechanism)
  - execute_pipeline_code (custom analysis on adata)
  Note: cluster-level tools and adata.obs metadata are not available here.
  Gene symbols are UPPERCASE human (e.g. CD3D, CD8A, CD4, FOXP3, TRDC, GZMB, not Cd3d/Cd8a).
Cite specific marker genes or pathways in your reason.
Time limit: 6 min  |  Tool call limit: 10
Answer format: {"labels": ["<celltype string>" for each cell in order], "reason": "<=200 chars"}
```

### imm_atlas_TT_p01_gamma-delta_T_cell_vs_regulatory_T_cell_r000_anon

Cell types:

- A: gamma-delta T cell
- B: regulatory T cell

Answer labels, in cell order:

```json
[
  "gamma-delta T cell",
  "regulatory T cell",
  "gamma-delta T cell",
  "regulatory T cell"
]
```

Prompt:

```text
Sample: imm_atlas_TT_p01_gamma-delta_T_cell_vs_regulatory_T_cell_r000_anon  (a 4-cell subset; obs has no leiden / cell_type / tissue)
Tissues (per cell, in order): ['jejunal epithelium', 'mesenteric lymph node', 'jejunal epithelium', 'mesenteric lymph node']
Cells (in order): [310320, 52913, 229551, 9856]

Two of these 4 cells are 'gamma-delta T cell', the other two are 'regulatory T cell'.
Identify each cell's celltype.

Available tools (operate on this 4-cell sample):
  - custom_pathway_calc  (KG-free; design marker edges and score cells)
  - get_astar_cellular_info, get_cell_kegg_edges (per-cell evidence)
  - get_kg_context (marker / mechanism)
  - execute_pipeline_code (custom analysis on adata)
  Note: cluster-level tools and adata.obs metadata are not available here.
  Gene symbols are UPPERCASE human (e.g. CD3D, CD8A, CD4, FOXP3, TRDC, GZMB, not Cd3d/Cd8a).
Cite specific marker genes or pathways in your reason.
Time limit: 6 min  |  Tool call limit: 10
Answer format: {"labels": ["<celltype string>" for each cell in order], "reason": "<=200 chars"}
```

## B Cell Tasks

### imm_atlas_BB_p01_c15c27_cell_r001_anon

Cell types:

- A: memory B cell
- B: naive B cell

Answer labels, in cell order:

```json
[
  "naive B cell",
  "memory B cell",
  "naive B cell",
  "memory B cell"
]
```

Prompt:

```text
Sample: imm_atlas_BB_p01_c15c27_cell_r001_anon  (a 4-cell subset; obs has no leiden / cell_type / tissue)
Tissue: spleen
Cells (in order): [84275, 41908, 79793, 48424]

Two of these 4 cells are 'memory B cell', the other two are 'naive B cell'.
Identify each cell's celltype.

Available tools (operate on this 4-cell sample):
  - custom_pathway_calc  (KG-free; design marker edges and score cells)
  - get_astar_cellular_info, get_cell_kegg_edges (per-cell evidence)
  - get_kg_context (marker / mechanism)
  - execute_pipeline_code (custom analysis on adata)
  Note: cluster-level tools and adata.obs metadata are not available here.
  Gene symbols are UPPERCASE human (e.g. CD3D, CD8A, CD4, FOXP3, TRDC, GZMB, not Cd3d/Cd8a).
Cite specific marker genes or pathways in your reason.
Time limit: 6 min  |  Tool call limit: 10
Answer format: {"labels": ["<celltype string>" for each cell in order], "reason": "<=200 chars"}
```

### imm_atlas_BB_p01_c24c48_cell_r001_anon

Cell types:

- A: memory B cell
- B: plasmablast

Answer labels, in cell order:

```json
[
  "memory B cell",
  "plasmablast",
  "memory B cell",
  "plasmablast"
]
```

Prompt:

```text
Sample: imm_atlas_BB_p01_c24c48_cell_r001_anon  (a 4-cell subset; obs has no leiden / cell_type / tissue)
Tissue: thoracic lymph node
Cells (in order): [280613, 290733, 284437, 328865]

Two of these 4 cells are 'memory B cell', the other two are 'plasmablast'.
Identify each cell's celltype.

Available tools (operate on this 4-cell sample):
  - custom_pathway_calc  (KG-free; design marker edges and score cells)
  - get_astar_cellular_info, get_cell_kegg_edges (per-cell evidence)
  - get_kg_context (marker / mechanism)
  - execute_pipeline_code (custom analysis on adata)
  Note: cluster-level tools and adata.obs metadata are not available here.
  Gene symbols are UPPERCASE human (e.g. CD3D, CD8A, CD4, FOXP3, TRDC, GZMB, not Cd3d/Cd8a).
Cite specific marker genes or pathways in your reason.
Time limit: 6 min  |  Tool call limit: 10
Answer format: {"labels": ["<celltype string>" for each cell in order], "reason": "<=200 chars"}
```
