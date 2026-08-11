#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${1:-scgraphagent-qwen3.5-27b}"

cd "$ROOT"
./verify_model.sh
exec ollama create "$MODEL_NAME" -f Modelfile
