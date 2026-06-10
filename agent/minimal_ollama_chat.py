#!/usr/bin/env python3
"""Minimal Ollama smoke test without MCP tools."""

import argparse
import json
import urllib.request


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen35-grpo-step9")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--num-ctx", type=int, default=32000)
    p.add_argument("prompt", nargs="?", default="What is a CD8 memory T cell? Answer in one sentence.")
    args = p.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
        "options": {"num_ctx": args.num_ctx, "temperature": 0.2},
    }
    out = post_json(f"{args.ollama_url.rstrip('/')}/api/chat", payload)
    print(out.get("message", {}).get("content", ""))


if __name__ == "__main__":
    main()

