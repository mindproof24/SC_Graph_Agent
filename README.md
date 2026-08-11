# scMCP Interactive Runtime

Runtime package for interactive AnnData analysis with:

- FastMCP AnnData tool server

- PyO3/Rust kernels for A*/KEGG/custom-pathway scoring
- Ollama Qwen3.5 GRPO step9 interactive command-line agent
- PyO3/Rust kernels (`cwg_rust`) for A*/KEGG/custom-pathway scoring
- Ollama Qwen3.5 reinforcement-trained interactive command-line agent


Large assets are intentionally not tracked:

- `scGraphAgent_qwen3.5-27B_Q4_K_M.gguf`
- `.h5ad` datasets
- KEGG/DoRothEA data caches



KEGG KGML parsing uses a vendored and substantially modified derivative of
[keggx v0.1.0](https://github.com/iamjli/keggx), originally developed by Johnny
Li and distributed under the MIT License. The vendored parser adapts KGML
relation processing, official gene-symbol mapping, group-node expansion, and
server integration for the SC_Graph_Agent workflow. Attribution and license
details are provided in `vendor/keggx/NOTICE` and `vendor/keggx/LICENSE`.




## Layout

```text
server/      FastMCP server and tool implementation
agent/       Ollama + MCP interactive agent
scripts/     data registration helpers
rust/        PyO3 cwg_rust crate
ollama/      Modelfile templates and model creation helper
docs/        notes and troubleshooting
```

## Install On A New Machine

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin
pip install -r requirements.txt

cd rust/cwg_rust
maturin develop --release
cd ../..
```

Verify Rust module:

```bash
python - <<'PY'
import cwg_rust
print(cwg_rust)
PY
```

## Data Directory

Default data location is `./data`.

Expected examples:

```text
data/<sampleid>.h5ad
data/KEGG_Graph_processing/*.kgml
data/KEGG_Graph_processing/mmu/*.kgml
data/dorothea_ABC_human.parquet
data/dorothea_ABC_mouse.parquet
```

You can override paths:

```bash
export MCP_DATA_DIR=/path/to/data
export KEGG_DIR=/path/to/data/KEGG_Graph_processing
```

## Ollama Model

Put or symlink the GGUF here:

```bash
mkdir -p ollama/models
ln -s /path/to/scGraphAgent_qwen3.5-27B_Q4_K_M.gguf ollama/models/
```

Create an Ollama model. For 24GB GPUs, start with 16k; use 8k if OOM.

```bash
bash ollama/create_model.sh 16k scgraphagent-qwen3.5-27b-16k
```

## Run

Terminal 1:

```bash
bash server/start_mcp_server.sh
```

Register data if needed:

```bash
bash scripts/register_adata.sh my_sample /path/to/my_sample.h5ad --force
```

Terminal 2:

```bash
SHOW_THINKING=1 \
OLLAMA_MODEL=scgraphagent-qwen3.5-27b-16k \
NUM_CTX=16384 \
MAX_TOOL_TURNS=30 \
TOOL_RESPONSE_MAX_CHARS=12000 \
bash agent/run_interactive.sh my_sample
```

During a model/tool turn, press `Ctrl-G` to queue a guiding message at the next safe point.

```text
[Guiding Message] : This dataset is mouse. Use organism="mouse".
```

## License

SC_Graph_Agent source code is distributed under the MIT License; see
[`LICENSE`](LICENSE). Third-party components retain their original licenses.
The vendored keggx derivative is covered by the attribution and MIT license in
`vendor/keggx/NOTICE` and `vendor/keggx/LICENSE`. KEGG data and KGML files are
not relicensed by this repository and remain subject to the applicable KEGG
terms of use.
