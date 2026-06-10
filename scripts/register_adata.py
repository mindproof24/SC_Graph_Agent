#!/usr/bin/env python3
"""Register a user .h5ad so the MCP AnnData server can serve it under a sampleid.

How the MCP server (sc_graph_mcp_server.py) finds data:
    It lazily loads `<DATA_DIR>/<sampleid>.h5ad` on the first tool call that needs
    that sampleid (AnnData backend). So "registering" = placing the user's file at
    `<DATA_DIR>/<sampleid>.h5ad`, then (optionally) verifying the server can load it.

Typical use (run on the SAME host as the MCP server, so DATA_DIR is writable):
    python register_adata.py --h5ad /path/to/my_data.h5ad --sampleid my_sample
    python register_adata.py --sampleid my_sample --verify-only   # just check it loads

DATA_DIR resolution order: --data-dir > $MCP_DATA_DIR > DATA_DIR parsed from --server-py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_SERVER_PY = os.getenv(
    "MCP_SERVER_PY",
    str(Path(__file__).resolve().parents[1] / "server" / "sc_graph_mcp_server.py"),
)


def resolve_data_dir(cli_dir: str | None, server_py: str) -> Path:
    if cli_dir:
        return Path(cli_dir).expanduser()
    if os.getenv("MCP_DATA_DIR"):
        return Path(os.environ["MCP_DATA_DIR"]).expanduser()
    # parse `DATA_DIR = "..."` from the server source (no heavy import)
    try:
        txt = Path(server_py).read_text()
        m = re.search(r'^\s*DATA_DIR\s*=\s*["\']([^"\']+)["\']', txt, re.MULTILINE)
        if m:
            return Path(m.group(1))
    except Exception:
        pass
    raise SystemExit("Could not resolve DATA_DIR. Pass --data-dir or set MCP_DATA_DIR.")


# ── minimal MCP streamable-http client (verify only) ────────────────────────
class MCPClient:
    def __init__(self, url: str, timeout: int = 300):
        self.url, self.timeout, self.session_id = url, timeout, ""

    def _req(self, payload: dict) -> str:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json, text/event-stream",
                     **({"mcp-session-id": self.session_id} if self.session_id else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if not self.session_id:
                self.session_id = r.headers.get("mcp-session-id", "")
            return r.read().decode("utf-8", "replace")

    def initialize(self):
        self._req({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                   "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                              "clientInfo": {"name": "register_adata", "version": "0.1"}}})
        self._req({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def call(self, name: str, arguments: dict) -> str:
        txt = self._req({"jsonrpc": "2.0", "id": int(time.time() * 1000) % 1_000_000,
                         "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        for line in txt.splitlines():
            if line.startswith("data:"):
                d = json.loads(line[5:].strip())
                return d.get("result", {}).get("content", [{}])[0].get("text", txt[:500])
        return txt[:500]


def verify(mcp_url: str, sampleid: str) -> bool:
    print(f"[verify] MCP {mcp_url} → load '{sampleid}'")
    cli = MCPClient(mcp_url)
    cli.initialize()
    code = ("import json as _j; print(_j.dumps({'shape': list(adata.shape), "
            "'obs_cols': list(adata.obs.columns)[:10], 'n_vars': int(adata.n_vars)}))")
    out = cli.call("execute_pipeline_code", {"sampleid": sampleid, "code": code})
    print(f"[verify] {out[:400]}")
    return '"shape"' in out or '"success": true' in out or '"success":true' in out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampleid", required=True, help="sampleid the model will refer to")
    p.add_argument("--h5ad", help="path to user's .h5ad (omit with --verify-only)")
    p.add_argument("--data-dir", default=None, help="MCP server DATA_DIR (override)")
    p.add_argument("--server-py", default=DEFAULT_SERVER_PY, help="server .py to parse DATA_DIR from")
    p.add_argument("--mcp-url", default=os.getenv("MCP_URL", "http://localhost:8005/mcp"))
    p.add_argument("--copy", action="store_true", help="copy file instead of symlink")
    p.add_argument("--force", action="store_true", help="overwrite existing registration")
    p.add_argument("--no-verify", action="store_true", help="skip MCP load check")
    p.add_argument("--verify-only", action="store_true", help="skip placement, only verify load")
    args = p.parse_args()

    if args.verify_only:
        ok = verify(args.mcp_url, args.sampleid)
        print("[done]", "OK ✅" if ok else "FAILED ⚠️")
        sys.exit(0 if ok else 1)

    if not args.h5ad:
        raise SystemExit("--h5ad is required (unless --verify-only).")
    src = Path(args.h5ad).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"h5ad not found: {src}")

    data_dir = resolve_data_dir(args.data_dir, args.server_py)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / f"{args.sampleid}.h5ad"
    print(f"[register] sampleid={args.sampleid}")
    print(f"[register] src ={src}")
    print(f"[register] dest={dest}")

    if dest.exists() or dest.is_symlink():
        if not args.force:
            raise SystemExit(f"{dest} already exists. Use --force to overwrite.")
        dest.unlink()

    if args.copy:
        shutil.copy2(src, dest)
        print("[register] copied.")
    else:
        dest.symlink_to(src)
        print("[register] symlinked.")

    if args.no_verify:
        print("[done] placed (verify skipped).")
        return
    ok = verify(args.mcp_url, args.sampleid)
    print("[done]", "registered + load OK ✅" if ok else "placed but load FAILED ⚠️ (check MCP host/port & DATA_DIR match)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
