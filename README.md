# scGraphAgent

**Graph-informed, tool-using analysis of single-cell RNA-seq data**

scGraphAgent connects an open-weight language model to executable AnnData
analysis and two principal types of graph-based biological tools. The pathway tool evaluates
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

  ### Installation

  ### Requirements

  The tested installation requires:

  - Git;
  - Python 3.11 or later;
  - a Rust toolchain installed through `rustup`;
  - Ollama with Qwen3.5 renderer and tool-call support; and
  - approximately 35 GB of free disk space when installing the released GGUF
    model.

  The pinned Python environment includes `anndata==0.12.10` and should be
  installed with Python 3.11 or later. The released model is approximately
  17 GB, and Ollama creates a separate model blob during registration. We
  recommend at least 35 GB of free space for installation and temporary files.

  A GPU with at least 24 GB of memory is recommended for the Q4_K_M model with
  a 16k context. CPU execution is possible but is generally too slow for
  interactive analysis.

  ### Clone the repository

  ```bash
  git clone https://github.com/mindproof24/SC_Graph_Agent.git
  cd SC_Graph_Agent

  The following commands must be run from the repository root because
  requirements.txt installs the vendored keggx package through the relative
  path -e ./vendor/keggx.

  ### Create the Python environment

  Use python3.11 explicitly on systems where python is unavailable or refers
  to an older installation.

  python3.11 --version
  python3.11 -m venv .venv
  source .venv/bin/activate

  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  After activation, python refers to the interpreter inside .venv.

  ### Install Rust

  If Rust is not already installed, install it through rustup:

  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  cargo --version

  The Cargo environment must be loaded in the current shell before compiling
  the extension. New terminal sessions may also require:

  source "$HOME/.cargo/env"

  ### Build the Rust extension

  From the repository root, with the Python environment activated:

  maturin develop --release --manifest-path rust/cwg_rust/Cargo.toml

  Verify the installation:

  python - <<'PY'
  import anndata
  import cwg_rust

  print("anndata:", anndata.__version__)
  print("cwg_rust:", cwg_rust.__file__)
  PY

  ### Install and verify Ollama

  Install an Ollama release that supports the Qwen3.5 renderer and tool-call
  format. The released model package was tested with Ollama 0.24.0.

  ollama --version

  On Linux with an NVIDIA GPU, the installed driver must be compatible with the
  Ollama CUDA backend. An incompatible driver may cause Ollama to run the model
  on the CPU without terminating the request. A successful response therefore
  does not by itself confirm GPU execution.

  After creating and running the model, check the active processor in another
  terminal:

  ollama ps
  nvidia-smi

  The PROCESSOR column reported by ollama ps should indicate GPU use.

  | Disk | **~35 GB free during setup** | `ollama create` copies the 16 GiB GGUF into
  `~/.ollama/models/blobs`, so the file exists twice until you delete the
  download. Extra context variants (4k/8k/16k/32k) reuse the same blob and cost
  nothing. Relocate the store with `OLLAMA_MODELS` if space is tight. |
  | GPU | **~23 GiB VRAM at 16k context** | measured peak 22.9 GiB
  (weights 15.3 + KV cache 4.7 + compute graph 1.1). A 24 GB card is close to
  full; use `ctx8k` if it does not fit. 32k needs roughly 27 GiB. |
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
