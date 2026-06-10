# scMCP Interactive Runtime

Runtime package for interactive AnnData analysis with:

- FastMCP AnnData tool server
- PyO3/Rust kernels (`cwg_rust`) for A*/KEGG/custom-pathway scoring
- Ollama Qwen3.5 GRPO step9 interactive command-line agent

Large assets are intentionally not tracked:

- `qwen35-grpo-step9-Q4_K_M.gguf`
- `.h5ad` datasets
- KEGG/DoRothEA data caches

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
ln -s /path/to/qwen35-grpo-step9-Q4_K_M.gguf ollama/models/
```

Create an Ollama model. For 24GB GPUs, start with 16k; use 8k if OOM.

```bash
bash ollama/create_model.sh 16k qwen35-grpo-step9-16k
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
OLLAMA_MODEL=qwen35-grpo-step9-16k \
NUM_CTX=16384 \
MAX_TOOL_TURNS=30 \
TOOL_RESPONSE_MAX_CHARS=12000 \
bash agent/run_interactive.sh my_sample
```

During a turn, press `Ctrl-C` once to inject a direction message without killing the agent.

```text
[Directing Message] : This dataset is mouse. Use organism="mouse".
```
