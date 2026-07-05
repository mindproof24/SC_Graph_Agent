#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${OLLAMA_MODEL:-qwen35-grpo-step9-8k}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MCP_URL="${MCP_URL:-http://localhost:8005/mcp}"
NUM_CTX="${NUM_CTX:-8192}"
OLLAMA_SEED="${OLLAMA_SEED:-}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-30}"
TOOL_RESPONSE_MAX_CHARS="${TOOL_RESPONSE_MAX_CHARS:-20000}"
SHOW_THINKING="${SHOW_THINKING:-0}"
ALLOW_SAMPLE_SWITCH="${ALLOW_SAMPLE_SWITCH:-0}"
ANALYSIS_LOG="${ANALYSIS_LOG:-}"
ANALYSIS_LOG_DIR="${ANALYSIS_LOG_DIR:-}"
NO_ANALYSIS_LOG="${NO_ANALYSIS_LOG:-0}"
STARTUP_MESSAGE="${STARTUP_MESSAGE:-}"
STARTUP_MESSAGE_FILE="${STARTUP_MESSAGE_FILE:-}"

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
if [ -n "$ANALYSIS_LOG" ]; then
  EXTRA_ARGS+=(--log "$ANALYSIS_LOG")
fi
if [ -n "$OLLAMA_SEED" ]; then
  EXTRA_ARGS+=(--seed "$OLLAMA_SEED")
fi
if [ -n "$ANALYSIS_LOG_DIR" ]; then
  EXTRA_ARGS+=(--log-dir "$ANALYSIS_LOG_DIR")
fi
if [ "$NO_ANALYSIS_LOG" = "1" ] || [ "$NO_ANALYSIS_LOG" = "true" ]; then
  EXTRA_ARGS+=(--no-log)
fi
if [ -n "$STARTUP_MESSAGE" ]; then
  EXTRA_ARGS+=(--startup-message "$STARTUP_MESSAGE")
fi
if [ -n "$STARTUP_MESSAGE_FILE" ]; then
  EXTRA_ARGS+=(--startup-message-file "$STARTUP_MESSAGE_FILE")
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
