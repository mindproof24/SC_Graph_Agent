# SC Graph Agent Qwen3.5-27B GGUF

This directory is the companion package for the quantized model used by
SC Graph Agent.

## Model artifact

| Field | Value |
|---|---|
| File | `scGraphAgent_qwen3.5-27B_Q4_K_M.gguf` |
| Architecture | Qwen3.5-27B dense |
| Upstream model | `Qwen/Qwen3.5-27B` |
| Training source identifier | `unsloth/Qwen3.5-27B` (mirror of the upstream model) |
| Modification | REINFORCE++-baseline training with QLoRA adapters for scRNA-seq tool use |
| Released checkpoint | Training schedule step 9 |
| Quantization | GGUF `Q4_K_M` |
| Size | 17,157,098,528 bytes |
| SHA-256 | `3925248cd21f6190e8f9fc20867e394237b29d552e3432ae5aba384c0bed741f` |
| Intended interface | Text-based Ollama tool use with the SC Graph Agent server |
| License | Apache License 2.0; see `LICENSE` and `NOTICE` |

The exact upstream Git revision was not retained in the archived training
manifest. The model name, training environment and update procedure are
recorded, but this limitation should remain disclosed in the Zenodo metadata.

**Both training rounds used OpenRLHF 0.10.3 with the `reinforce_baseline`
advantage estimator, corresponding to the baseline variant of REINFORCE++.**

## Required files

Keep these files in one directory after downloading:

```text
scGraphAgent_qwen3.5-27B_Q4_K_M.gguf
Modelfile
create_model.sh
verify_model.sh
SHA256SUMS
LICENSE
NOTICE
MODEL_PROVENANCE.json
README.md
```

The SC Graph Agent source code, server, interactive prompt and Python/Rust
dependencies are maintained separately at:

```text
https://github.com/mindproof24/SC_Graph_Agent
```

Pin the repository release or commit stated in the Zenodo record. The GGUF
alone does not provide AnnData access or graph-analysis tools.

## Install and verify

Install an Ollama release with Qwen3.5 renderer/parser support. The archived
deployment was created and tested with Ollama 0.24.0.

```bash
chmod +x verify_model.sh create_model.sh
./verify_model.sh
./create_model.sh scgraphagent-qwen3.5-27b
ollama show scgraphagent-qwen3.5-27b
```

Run a basic model check:

```bash
ollama run scgraphagent-qwen3.5-27b "Return only: model ready"
```

For the full agent, clone the source repository and follow its root README.
A typical interactive run uses:

```bash
SHOW_THINKING=1 \
OLLAMA_MODEL=scgraphagent-qwen3.5-27b \
NUM_CTX=16384 \
MAX_TOOL_TURNS=30 \
MCP_URL=http://127.0.0.1:8005/mcp \
bash agent/run_interactive.sh <sampleid>
```

The server must be running and `<sampleid>.h5ad` must be registered separately.

## Context and hardware

The supplied `Modelfile` defaults to a 16,384-token context. A 32,768-token
context was also evaluated, but it requires more KV-cache memory. Reduce
`num_ctx` to 8,192 if the model does not fit. The archived Q4_K_M model is
approximately 16 GiB before runtime memory and KV-cache allocation.

## Scope and limitations

- This is a research artifact for graph-guided scRNA-seq analysis, not a
  clinical model or an autonomous annotation authority.
- The deposited GGUF is quantized and may not reproduce the full-precision
  adapter output bit-for-bit.
- Generation is stochastic unless a seed and all inference settings are fixed.
- The model requires the accompanying SC Graph Agent server to reproduce tool
  calls and AnnData analysis.
- The released artifact is used through its text interface; multimodal behavior
  was not evaluated in this study.

## Attribution

This model is a modified and quantized derivative of Qwen3.5-27B, developed by
the Qwen Team and released under the Apache License 2.0. SC Graph Agent-specific
The model updates were produced by the SC Graph Agent authors in 2026 using
QLoRA adapters.

Upstream model card:

```text
https://huggingface.co/Qwen/Qwen3.5-27B
```
