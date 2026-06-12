#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MCP_PORT="${MCP_PORT:-8005}"
MCP_DATA_DIR="${MCP_DATA_DIR:-$ROOT/data}"

export MCP_PORT MCP_DATA_DIR
export PYTHONPATH="$ROOT/server:${PYTHONPATH:-}"

echo "[mcp] root    : $ROOT"
echo "[mcp] python  : $PYTHON_BIN"
echo "[mcp] url     : http://127.0.0.1:$MCP_PORT/mcp"
echo "[mcp] data    : $MCP_DATA_DIR"
if [[ -n "${KEGG_DIR:-}" ]]; then
  export KEGG_DIR
  echo "[mcp] kegg    : $KEGG_DIR"
else
  echo "[mcp] kegg    : packaged keggx data or \$MCP_DATA_DIR/KEGG_Graph_processing"
fi
exec "$PYTHON_BIN" "$ROOT/server/sc_graph_mcp_server.py"
