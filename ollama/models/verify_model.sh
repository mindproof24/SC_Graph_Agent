#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

test -f scGraphAgent_qwen3.5-27B_Q4_K_M.gguf || {
  echo "Missing scGraphAgent_qwen3.5-27B_Q4_K_M.gguf" >&2
  exit 1
}

sha256sum --check SHA256SUMS

command -v ollama >/dev/null || {
  echo "Ollama is not installed or is not on PATH." >&2
  exit 1
}

echo "Model artifact and Ollama executable found."
