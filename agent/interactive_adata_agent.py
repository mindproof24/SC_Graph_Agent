#!/usr/bin/env python3
"""Interactive Ollama + MCP AnnData agent.

Ollama does not execute Python. This script:
1. sends chat messages and tool schemas to Ollama,
2. parses model-emitted tool calls,
3. calls the MCP server,
4. appends tool responses back into the conversation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import select
import sys
import termios
import threading
import time
import tty
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from prompt_toolkit import prompt as _prompt_toolkit_prompt
except Exception:
    _prompt_toolkit_prompt = None

try:
    import readline  # noqa: F401
except Exception:
    readline = None


ROOTS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[1] / "server",
]
for root in ROOTS:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

try:
    from interactive_prompt import SYSTEM_PROMPT_INTERACTIVE
    from tools_no_kg import TOOLS
except Exception as exc:  # pragma: no cover - user-facing startup error
    raise SystemExit(f"Could not import interactive prompt or tools_no_kg.py: {exc}")


NO_SAMPLEID_TOOLS = {
    "get_kg_context",
    "resolve_query_to_context_set",
    "score_context_subgraph",
    "synthesize_context_kg_paths",
}


def safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "sample"


class SessionLogger:
    def __init__(self, path: Path, sampleid: str):
        self.path = path
        self.sampleid = sampleid
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "sampleid": self.sampleid,
            "event": event,
            **payload,
        }
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class GuidingKeyWatcher:
    """Watch Ctrl-G during model/tool turns and request guidance at safe points."""

    def __init__(self):
        self.triggered = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._old_attrs = None
        self.enabled = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self._fd = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None:
            return
        self._stop.clear()
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._watch, args=(self._fd,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None
        if self._old_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None
        self._drain_pending_input()

    def _watch(self, fd: int) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([fd], [], [], 0.1)
            if not readable:
                continue
            ch = os.read(fd, 1)
            if ch == b"\x07":  # Ctrl-G / BEL
                self.triggered.set()

    def _drain_pending_input(self) -> None:
        if self._fd is None:
            return
        try:
            while select.select([self._fd], [], [], 0)[0]:
                os.read(self._fd, 1)
        except OSError:
            pass

    def pop_triggered(self) -> bool:
        if not self.triggered.is_set():
            return False
        self.triggered.clear()
        return True

    def prompt_guidance(self, messages: list[dict], logger: "SessionLogger | None") -> bool:
        self.stop()
        try:
            return inject_guiding_message(messages, logger)
        finally:
            self.start()


def post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


class MCPClient:
    def __init__(self, url: str, timeout: int = 600):
        self.url = url
        self.timeout = timeout
        self.session_id = ""

    def initialize(self) -> None:
        init_payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ollama-interactive-agent", "version": "0.1"},
            },
            "id": 0,
        }
        data = json.dumps(init_payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self.session_id = resp.headers.get("mcp-session-id", "")
            resp.read()
        self._post_raw({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _post_raw(self, payload: dict) -> str:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": self.session_id,
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def call(self, name: str, arguments: dict) -> str:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
            "id": int(time.time() * 1000) % 1_000_000_000,
        }
        text = self._post_raw(payload)
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = json.loads(line[len("data:") :].strip())
            return data["result"]["content"][0]["text"]
        return json.dumps({"success": False, "error": f"no SSE data: {text[:500]}"})


def coerce_param(value: str) -> Any:
    s = value.strip()
    if s in {"true", "True"}:
        return True
    if s in {"false", "False"}:
        return False
    if s in {"null", "None"}:
        return None
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        pass
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except Exception:
            pass
    return s


def parse_xml_tool_calls(content: str) -> list[dict]:
    calls: list[dict] = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content or "", re.DOTALL):
        body = m.group(1)
        fm = re.search(r"<function=([^>\s]+)\s*>", body)
        if fm:
            args = {}
            for pm in re.finditer(r"<parameter=([^>]+?)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                args[pm.group(1).strip()] = coerce_param(pm.group(2))
            calls.append({"name": fm.group(1).strip(), "arguments": args})
            continue
        jm = re.search(r"(\{.*\})", body, re.DOTALL)
        if jm:
            try:
                obj = json.loads(jm.group(1))
                calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", {})})
            except Exception:
                pass
    return [c for c in calls if c.get("name")]


def extract_tool_calls(message: dict) -> list[dict]:
    native = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        args = fn.get("arguments") or tc.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            native.append({"name": name, "arguments": args})
    return native or parse_xml_tool_calls(message.get("content", ""))


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head
    return text[:head] + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-tail:]


def ollama_chat(
    ollama_url: str,
    model: str,
    messages: list[dict],
    num_ctx: int,
    temperature: float,
    seed: int | None = None,
) -> dict:
    options = {"num_ctx": num_ctx, "temperature": temperature, "top_p": 0.9}
    if seed is not None:
        options["seed"] = int(seed)
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "options": options,
    }
    return post_json(f"{ollama_url.rstrip('/')}/api/chat", payload)


def extract_thinking(out: dict, msg: dict) -> tuple[str, str]:
    chunks = []
    for obj in (msg, out):
        for key in ("thinking", "thought", "reasoning", "reasoning_content"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                chunks.append(val.strip())

    content = msg.get("content", "") or ""
    think_blocks = re.findall(r"<think>\s*(.*?)\s*</think>", content, re.DOTALL)
    chunks.extend(t.strip() for t in think_blocks if t.strip())
    visible = re.sub(r"<think>\s*.*?\s*</think>", "", content, flags=re.DOTALL).strip()
    return "\n\n".join(chunks), visible


def append_tool_response(messages: list[dict], tool_name: str, content: str) -> None:
    # Ollama accepts role=tool for models with tool support. The XML fallback text
    # helps Qwen-style renderers that expect explicit tool_response blocks.
    messages.append({
        "role": "tool",
        "content": f"<tool_response>\n{content}\n</tool_response>",
        "name": tool_name,
    })


def read_user_line(prompt_text: str) -> str:
    if _prompt_toolkit_prompt is not None and getattr(sys.stdin, "isatty", lambda: False)():
        return _prompt_toolkit_prompt(prompt_text)
    return input(prompt_text)


def read_startup_message(args: argparse.Namespace) -> str:
    if args.startup_message_file:
        return Path(args.startup_message_file).expanduser().read_text(encoding="utf-8").strip()
    return args.startup_message.strip()


def inject_guiding_message(messages: list[dict], logger: SessionLogger | None) -> bool:
    print()
    directive = read_user_line("[Guiding Message] : ").strip()
    if not directive:
        print("[agent] empty guiding message; continuing.")
        return False
    injected = f"[Guiding Message] : {directive}"
    print(injected)
    messages.append({"role": "user", "content": injected})
    if logger:
        logger.write("guiding_message", content=directive)
        logger.write("ctrl_g_guiding_message", content=directive, injected=injected)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "scgraphagent-qwen3.5-27b"))
    p.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--mcp-url", default=os.getenv("MCP_URL", "http://localhost:8005/mcp"))
    p.add_argument("--sampleid", required=True)
    p.add_argument("--num-ctx", type=int, default=int(os.getenv("NUM_CTX", "32000")))
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "1.0")))
    _seed_env = os.getenv("OLLAMA_SEED", "").strip()
    p.add_argument("--seed", type=int, default=(int(_seed_env) if _seed_env else None))
    p.add_argument("--max-tool-turns", type=int, default=30)
    p.add_argument("--tool-response-max-chars", type=int, default=20000)
    p.add_argument("--show-thinking", action="store_true")
    p.add_argument("--allow-sample-switch", action="store_true")
    p.add_argument("--log", default=os.getenv("ANALYSIS_LOG", ""))
    p.add_argument("--log-dir", default=os.getenv("ANALYSIS_LOG_DIR", str(Path(__file__).resolve().parents[1] / "analysis_logs")))
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--startup-message", default=os.getenv("STARTUP_MESSAGE", ""))
    p.add_argument("--startup-message-file", default=os.getenv("STARTUP_MESSAGE_FILE", ""))
    args = p.parse_args()
    startup_message = read_startup_message(args)

    logger = None
    if not args.no_log:
        if args.log:
            log_path = Path(args.log).expanduser()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_part = safe_filename_part(args.sampleid)
            log_path = Path(args.log_dir).expanduser() / f"{sample_part}_{stamp}.jsonl"
        logger = SessionLogger(log_path, args.sampleid)
        logger.write(
            "session_start",
            model=args.model,
            ollama_url=args.ollama_url,
            mcp_url=args.mcp_url,
            num_ctx=args.num_ctx,
            max_tool_turns=args.max_tool_turns,
            ollama_seed=args.seed,
        )
        if startup_message:
            logger.write("startup_message", content=startup_message)

    mcp = MCPClient(args.mcp_url)
    mcp.initialize()
    print(f"[mcp] connected: {args.mcp_url}")
    print(f"[mcp] reset sample: {args.sampleid}")
    reset_result = mcp.call("reset_pipeline_namespace", {"sampleid": args.sampleid})
    print(reset_result[:1000])
    if logger:
        logger.write("mcp_reset", result=truncate_text(reset_result, args.tool_response_max_chars))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_INTERACTIVE},
        {
            "role": "user",
            "content": (
                f"Sample: {args.sampleid}\n"
                "You are now in an interactive analysis session. Inspect `adata` "
                "with tools when needed. Keep answers concise and cite concrete evidence."
            ),
        },
    ]
    if logger:
        logger.write("system_prompt", content=SYSTEM_PROMPT_INTERACTIVE)
        logger.write("tool_schema_summary", tools=[t["function"]["name"] for t in TOOLS])
        logger.write("session_context", content=messages[1]["content"])
    if startup_message:
        print("\n[startup message]\n" + startup_message + "\n")
        messages.append({
            "role": "user",
            "content": f"[Startup Message] : {startup_message}",
        })

    if logger:
        print(f"[log] writing session: {logger.path}")
    print("\nType questions. Use 'exit' or 'quit' to stop. During tool/model turns, press Ctrl-G to queue a guiding message.\n")
    try:
        while True:
            user_text = read_user_line("Request> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                if logger:
                    logger.write("session_end", reason=user_text.lower())
                break
            if not user_text:
                continue
            messages.append({"role": "user", "content": user_text})
            if logger:
                logger.write("user", content=user_text)
                logger.write("question", content=user_text)

            with GuidingKeyWatcher() as guide:
                for _ in range(args.max_tool_turns):
                    try:
                        if guide.pop_triggered() and guide.prompt_guidance(messages, logger):
                            continue
                        out = ollama_chat(args.ollama_url, args.model, messages, args.num_ctx, args.temperature, seed=args.seed)
                        msg = out.get("message", {})
                        thinking, content = extract_thinking(out, msg)
                        if args.show_thinking and thinking:
                            print("\nthinking>\n" + thinking.strip() + "\n")
                        if thinking and logger:
                            logger.write("thinking", content=thinking.strip())
                        if content.strip():
                            print("\nassistant>\n" + content.strip() + "\n")
                            if logger:
                                logger.write("assistant", content=content.strip())
                        messages.append(msg)

                        if guide.pop_triggered() and guide.prompt_guidance(messages, logger):
                            continue

                        calls = extract_tool_calls(msg)
                        if not calls:
                            break

                        for call in calls:
                            name = call["name"]
                            arguments = call.get("arguments") or {}
                            if isinstance(arguments, dict) and name not in NO_SAMPLEID_TOOLS:
                                old_sampleid = arguments.get("sampleid")
                                arguments["sampleid"] = args.sampleid
                                if args.allow_sample_switch and old_sampleid:
                                    arguments["sampleid"] = old_sampleid
                                elif old_sampleid and old_sampleid != args.sampleid:
                                    print(f"[agent] overriding sampleid: {old_sampleid} -> {args.sampleid}")
                            print(f"[tool_call] {name}({json.dumps(arguments, ensure_ascii=False)})")
                            if logger:
                                logger.write("tool_call", name=name, arguments=arguments)
                            result = mcp.call(name, arguments)
                            result_short = truncate_text(result, args.tool_response_max_chars)
                            print(f"[tool_response] {result_short}\n")
                            if logger:
                                logger.write("tool_response", name=name, content=result_short)
                            append_tool_response(messages, name, result_short)
                            if guide.pop_triggered() and guide.prompt_guidance(messages, logger):
                                break
                    except KeyboardInterrupt:
                        print("[agent] interrupted; returning to prompt.")
                        if logger:
                            logger.write("interrupt")
                        break
                    except Exception as exc:
                        msg = f"{type(exc).__name__}: {exc}"
                        print(f"[agent] error; returning to prompt.\n{msg}")
                        if logger:
                            logger.write("error", error=msg)
                        messages.append({
                            "role": "user",
                            "content": (
                                "[Agent Runtime Error] The previous model/tool turn failed: "
                                f"{msg}. Continue with corrected tool-call syntax."
                            ),
                        })
                        break
                else:
                    print("[agent] max tool turns reached; ask a follow-up or request a final summary.")
                    if logger:
                        logger.write("max_tool_turns_reached")
    finally:
        if logger:
            logger.close()


if __name__ == "__main__":
    main()
