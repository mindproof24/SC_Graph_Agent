#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MCP_DATA_DIR="${MCP_DATA_DIR:-$ROOT/data}"
MCP_URL="${MCP_URL:-http://localhost:8005/mcp}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <sampleid> </path/to/file.h5ad> [--copy] [--force] [--no-verify]" >&2
  exit 2
fi

SAMPLEID="$1"
H5AD="$2"
shift 2

exec "$PYTHON_BIN" "$ROOT/scripts/register_adata.py" \
  --sampleid "$SAMPLEID" \
  --h5ad "$H5AD" \
  --data-dir "$MCP_DATA_DIR" \
  --server-py "$ROOT/server/sc_graph_mcp_server.py" \
  --mcp-url "$MCP_URL" \
  "$@"
