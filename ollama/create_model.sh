#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CTX="${1:-8k}"
MODEL="${2:-scgraphagent-qwen3.5-27b-$CTX}"
MODELS_DIR="$ROOT/ollama/models"
GGUF="$MODELS_DIR/scGraphAgent_qwen3.5-27B_Q4_K_M.gguf"
MODELFILE="$ROOT/ollama/Modelfile.scgraphagent.ctx$CTX"

if [ ! -f "$GGUF" ]; then
  echo "[ollama] missing GGUF: $GGUF" >&2
  echo "[ollama] put or symlink scGraphAgent_qwen3.5-27B_Q4_K_M.gguf into $MODELS_DIR" >&2
  exit 1
fi
if [ ! -f "$MODELFILE" ]; then
  echo "[ollama] missing Modelfile: $MODELFILE" >&2
  echo "[ollama] ctx must be one of: 4k, 8k, 16k, 32k" >&2
  exit 1
fi

cd "$ROOT/ollama"
exec ollama create "$MODEL" -f "$MODELFILE"
