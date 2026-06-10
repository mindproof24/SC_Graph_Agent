#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CTX="${1:-8k}"
MODEL="${2:-qwen35-grpo-step9-$CTX}"
MODELS_DIR="$ROOT/ollama/models"
GGUF="$MODELS_DIR/qwen35-grpo-step9-Q4_K_M.gguf"
MODELFILE="$ROOT/ollama/Modelfile.qwen35-grpo-step9.ctx$CTX"

if [ ! -f "$GGUF" ]; then
  echo "[ollama] missing GGUF: $GGUF" >&2
  echo "[ollama] put or symlink qwen35-grpo-step9-Q4_K_M.gguf into $MODELS_DIR" >&2
  exit 1
fi
if [ ! -f "$MODELFILE" ]; then
  echo "[ollama] missing Modelfile: $MODELFILE" >&2
  echo "[ollama] ctx must be one of: 4k, 8k, 16k" >&2
  exit 1
fi

cd "$ROOT/ollama"
exec ollama create "$MODEL" -f "$MODELFILE"
