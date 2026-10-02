#!/usr/bin/env python3
"""MCP stdio transport for Claude Code, Claude Desktop, Cursor, and other MCP hosts.

Speaks newline-delimited JSON-RPC 2.0 as required by the MCP stdio transport.
Legacy ``Content-Length`` framed messages are also accepted and answered in the
same framing. stdout carries only protocol messages: anything a tool prints is
redirected to stderr.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("prithvi-wxc-mcp-stdio")

# Keep the real stdout for protocol frames before importing tool modules that may print.
PROTOCOL_OUT: BinaryIO = sys.stdout.buffer
sys.stdout = sys.stderr

from mcp_downscaling_tools import MERRA2Analyzer, process_tool_call  # noqa: E402

CONFIG_PATH = CURRENT_DIR / "mcp-merra2-config.json"
SERVER_NAME = "prithvi-wxc-downscaling"
SERVER_VERSION = "2.0.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = (
    "Prithvi WxC downscaling: download MERRA-2/NARR/PRISM/elevation/weights, build configs, "
    "train, run inference, and analyze NetCDF outputs on THIS machine. On a new machine call "
    "check_environment first and follow its next_steps. Always pass configs through "
    "create_custom_yaml so data paths are localized. Long steps run as background jobs: "
    "poll get_job_status. Every job writes a run manifest (get_run_manifest) that replay_run "
    "can re-execute elsewhere."
)


def load_tools() -> list[dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    tools = config.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"Invalid tools list in {CONFIG_PATH}")
    return tools


TOOLS = load_tools()
_ANALYZER: MERRA2Analyzer | None = None


def analyzer() -> MERRA2Analyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = MERRA2Analyzer()
    return _ANALYZER


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            return process_tool_call(tool_name, tool_args, analyzer())
    except Exception as exc:
        logger.error("Error executing tool %s: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": str(exc)})


def _is_error(raw: str) -> bool:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(parsed, dict) and ("error" in parsed or parsed.get("status") == "error")


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method", "")
    req_id = message.get("id")
    params = message.get("params") or {}

    if req_id is None:  # notification (e.g. notifications/initialized)
        return None

    if method == "initialize":
        requested = params.get("protocolVersion")
        result = {
            "protocolVersion": requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        raw = execute_tool(params.get("name", ""), params.get("arguments") or {})
        result = {"content": [{"type": "text", "text": raw}], "isError": _is_error(raw)}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


class PrefixedReader:
    """stdin with bytes already read by the setup stand-in put back in front."""

    def __init__(self, prefix: bytes, stream: BinaryIO):
        self._buf, self._stream = prefix, stream

    def readline(self) -> bytes:
        if not self._buf:
            return self._stream.readline()
        newline = self._buf.find(b"\n")
        if newline >= 0:
            line, self._buf = self._buf[: newline + 1], self._buf[newline + 1 :]
            return line
        line, self._buf = self._buf, b""
        return line + self._stream.readline()

    def read(self, size: int) -> bytes:
        head, self._buf = self._buf[:size], self._buf[size:]
        return head if len(head) == size else head + self._stream.read(size - len(head))


def read_message(stream: BinaryIO) -> tuple[dict[str, Any] | list | None, bool] | None:
    """Return (message, framed) or None at EOF. framed=True for Content-Length input."""
    while True:
        line = stream.readline()
        if not line:
            return None
        text = line.decode("utf-8").strip()
        if not text:
            continue
        if text.lower().startswith("content-length:"):
            length = int(text.split(":", 1)[1].strip())
            while stream.readline().strip():  # remaining headers up to the blank line
                pass
            return json.loads(stream.read(length).decode("utf-8")), True
        return json.loads(text), False


def write_message(payload: Any, framed: bool) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if framed:
        PROTOCOL_OUT.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
    else:
        PROTOCOL_OUT.write(data + b"\n")
    PROTOCOL_OUT.flush()


def main() -> None:
    logger.info("Starting %s %s (stdio)", SERVER_NAME, SERVER_VERSION)
    stdin = sys.stdin.buffer
    # Handed over from mcp/setup_status_server.py after a first-start install:
    # take back the input it already read, and have the host reload the tools.
    pending = os.environ.pop("PRITHVI_MCP_PENDING", "")
    if pending:
        stdin = PrefixedReader(base64.b64decode(pending), stdin)
    if os.environ.pop("PRITHVI_MCP_RESUMED", "") == "1":
        write_message({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}, False)
    while True:
        try:
            read = read_message(stdin)
        except (json.JSONDecodeError, ValueError) as exc:
            write_message({"jsonrpc": "2.0", "id": None,
                           "error": {"code": -32700, "message": f"Parse error: {exc}"}}, False)
            continue
        if read is None:
            break
        message, framed = read
        if isinstance(message, list):  # JSON-RPC batch
            responses = [r for r in (handle(m) for m in message) if r is not None]
            if responses:
                write_message(responses, framed)
            continue
        response = handle(message)
        if response is not None:
            write_message(response, framed)


if __name__ == "__main__":
    main()
