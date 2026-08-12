# scGraphAgent

**Graph-informed, tool-using analysis of single-cell RNA-seq data**

scGraphAgent connects an open-weight language model to executable AnnData
analysis and two graph-based biological tools. The pathway tool evaluates
KEGG or user-defined gene-edge sets and exposes the edges contributing to each
score. The regulatory tool uses A* cell-state paths to prioritize DoRothEA
transcription factor (TF)-target relationships by path support and
expression-weighted activity. Together, these tools allow pathway and
regulatory evidence to be inspected and refined during a multi-step analysis
rather than used only as descriptive background knowledge.

The released model was derived from Qwen3.5-27B and trained on tool-using
single-cell analysis trajectories. Both training rounds used OpenRLHF 0.10.3
with the `reinforce_baseline` advantage estimator, corresponding to the
baseline variant of REINFORCE++.

## What this repository provides

- an AnnData-aware analysis server built with FastMCP;
- KEGG and customized pathway scoring with edge-level output;
- A* path search and TF-target edge prioritization;
- Rust/PyO3 kernels for repeated graph calculations;
- an Ollama-based interactive agent with optional researcher guidance; and
- figure-specific benchmark, source-data and reproduction workflows.

The framework is intended for interactive, quantitatively grounded analysis.
It is not a fully autonomous cell-annotation system, and biological conclusions
should be evaluated against the underlying data and appropriate external
references.

## Data and model availability

The trained `scGraphAgent_qwen3.5-27B_Q4_K_M.gguf` model, the processed
`cardio_perturb_phate.h5ad` object and machine-readable figure source data are
available from Zenodo:

<https://doi.org/10.5281/zenodo.21759232>

Inputs, scripts and instructions for reproducing the figures are provided
under [`reproducibility/`](reproducibility/).

## Repository structure

```text
agent/             interactive Ollama agent and tool-calling loop
server/            FastMCP AnnData server and graph-analysis tools
rust/cwg_rust/     PyO3/Rust graph and A* kernels
ollama/            model templates and model-creation helpers
scripts/           data registration and analysis utilities
reproducibility/   main and supplementary figure workflows
vendor/keggx/      modified KGML parser with upstream attribution
```

## Installation

The runtime requires Python, a Rust toolchain and
[Ollama](https://ollama.com/). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
python -m pip install -r requirements.txt

cd rust/cwg_rust
maturin develop --release
cd ../..
```

Verify that the compiled extension is available:

```bash
python - <<'PY'
import cwg_rust
print(cwg_rust)
PY
```

## Configure the model

Download the GGUF and place it in `ollama/models/`, or create a symbolic link:

```bash
mkdir -p ollama/models
ln -s /path/to/scGraphAgent_qwen3.5-27B_Q4_K_M.gguf ollama/models/
```

Create an Ollama model. A 16k context is the recommended starting point for a
24GB GPU or adjust it according to available memory of your environment.

```bash
bash ollama/create_model.sh 16k scgraphagent-qwen3.5-27b-16k
```

Available templates are provided for 4k, 8k, 16k and 32k contexts.

## Register an AnnData object

The default data directory is `./data`. Register an H5AD under a sample ID:

```bash
bash scripts/register_adata.sh my_sample /path/to/my_sample.h5ad --force
```

Alternative data and KEGG locations can be configured before starting the
server:

```bash
export MCP_DATA_DIR=/path/to/data
export KEGG_DIR=/path/to/KEGG_Graph_processing
```

The server assumes that each AnnData object has undergone preprocessing
appropriate to its source dataset. Tool-specific metadata requirements are
reported by the server when a requested analysis needs an observation column
or embedding that is not available.

## Run an interactive analysis

Start the analysis server in the first terminal:

```bash
bash server/start_mcp_server.sh
```

Start the agent in a second terminal:

```bash
SHOW_THINKING=1 \
OLLAMA_MODEL=scgraphagent-qwen3.5-27b-16k \
NUM_CTX=16384 \
MAX_TOOL_TURNS=30 \
TOOL_RESPONSE_MAX_CHARS=12000 \
bash agent/run_interactive.sh my_sample
```

During a model or tool turn, press `Ctrl-G` to queue a guiding message at the
next safe point. For example:

```text
[Guiding Message] : This dataset is mouse. Use organism="mouse".
```

Guidance becomes part of the analysis trajectory and should therefore be
retained when reporting or auditing an interactive case study.

## Reproducing the analyses

The [`reproducibility/`](reproducibility/) directory separates the workflows
used for the main figures and Supplementary Information. Each figure directory
identifies its inputs, calculation scripts and reported outputs. Large external
objects are referenced through the Zenodo record or their original public data
source rather than duplicated in Git.

For exact model and data provenance, use the archived Zenodo record together
with the repository commit or release associated with the manuscript version.

## License and third-party attribution

Project source code is distributed under the MIT License; see
[`LICENSE`](LICENSE). Third-party components retain their original licenses.
Attribution and license information for the modified keggx derivative is
provided in [`vendor/keggx/NOTICE`](vendor/keggx/NOTICE) and
[`vendor/keggx/LICENSE`](vendor/keggx/LICENSE). KEGG data and KGML records are
not relicensed by this repository and remain subject to the applicable KEGG
terms of use.
not relicensed by this repository and remain subject to the applicable KEGG
terms of use.
