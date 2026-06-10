#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${OLLAMA_MODEL:-qwen35-grpo-step9-8k}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MCP_URL="${MCP_URL:-http://localhost:8005/mcp}"
NUM_CTX="${NUM_CTX:-8192}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-30}"
TOOL_RESPONSE_MAX_CHARS="${TOOL_RESPONSE_MAX_CHARS:-20000}"
SHOW_THINKING="${SHOW_THINKING:-0}"
ALLOW_SAMPLE_SWITCH="${ALLOW_SAMPLE_SWITCH:-0}"

SAMPLEID="${1:-${SAMPLEID:-}}"
if [ -z "$SAMPLEID" ]; then
  echo "Usage: $0 <sampleid>" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/agent:$ROOT/server:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [ "$SHOW_THINKING" = "1" ] || [ "$SHOW_THINKING" = "true" ]; then
  EXTRA_ARGS+=(--show-thinking)
fi
if [ "$ALLOW_SAMPLE_SWITCH" = "1" ] || [ "$ALLOW_SAMPLE_SWITCH" = "true" ]; then
  EXTRA_ARGS+=(--allow-sample-switch)
fi

exec "$PYTHON_BIN" "$ROOT/agent/interactive_adata_agent.py" \
  --model "$MODEL" \
  --ollama-url "$OLLAMA_URL" \
  --sampleid "$SAMPLEID" \
  --mcp-url "$MCP_URL" \
  --num-ctx "$NUM_CTX" \
  --max-tool-turns "$MAX_TOOL_TURNS" \
  --tool-response-max-chars "$TOOL_RESPONSE_MAX_CHARS" \
  "${EXTRA_ARGS[@]}"
