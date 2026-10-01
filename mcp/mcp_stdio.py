#!/usr/bin/env python3
"""MCP stdio transport for Cursor, Claude Desktop, and other MCP hosts.

Logs go to stderr only. stdout is reserved for JSON-RPC framing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from mcp_downscaling_tools import MERRA2Analyzer, process_tool_call  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("prithvi-wxc-mcp-stdio")

CONFIG_PATH = CURRENT_DIR / "mcp-merra2-config.json"
SERVER_NAME = "Prithvi-WxC-Downscaling MCP"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"


def load_tools() -> list[dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    tools = config.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"Invalid tools list in {CONFIG_PATH}")
    return tools


TOOLS = load_tools()
ANALYZER = MERRA2Analyzer()


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    try:
        return process_tool_call(tool_name, tool_args, ANALYZER)
    except Exception as exc:
        logger.error("Error executing tool %s: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": str(exc)})


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        decoded = line.decode("utf-8")
        if decoded in ("\r\n", "\n"):
            break
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def write_message(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method", "")
    req_id = message.get("id")
    params = message.get("params") or {}

    if req_id is None:
        return None

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Downscale MERRA-2/NARR/ERA5/CORDEX weather data with PrithviWxC "
                "and analyze NetCDF outputs. Use list_available_configs before "
                "creating YAML. Use run_training_pipeline to train."
            ),
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        raw = execute_tool(name, arguments)
        is_error = False
        try:
            parsed = json.loads(raw)
            is_error = isinstance(parsed, dict) and "error" in parsed
        except (TypeError, json.JSONDecodeError):
            is_error = False
        result = {
            "content": [{"type": "text", "text": raw}],
            "isError": is_error,
        }
    elif method == "ping":
        result = {}
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def main() -> None:
    logger.info("Starting %s stdio transport", SERVER_NAME)
    while True:
        try:
            message = read_message()
        except Exception as exc:
            logger.error("Failed to read MCP message: %s", exc, exc_info=True)
            continue
        if message is None:
            break
        response = handle(message)
        if response is not None:
            write_message(response)


if __name__ == "__main__":
    main()
