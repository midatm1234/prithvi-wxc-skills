#!/usr/bin/env python3
"""Stand-in MCP server used until the full Prithvi server can start.

bin/prithvi-mcp runs this, with any Python 3, when the full server's packages
are not installed yet or setup failed. It answers the MCP handshake at once, so
the host never times out, and offers one tool, prithvi_setup_status, that says
what is happening and how to fix it. While a background install runs, it
watches for completion, then replaces itself with the full server in the same
session and tells the host to refresh its tool list.

Standard library only: this must run on old system Pythons (macOS ships 3.9).
"""

import base64
import json
import os
import select

STATE = os.environ.get("PRITHVI_SETUP_STATE", "error")  # "installing" or "error"
MESSAGE = os.environ.get("PRITHVI_SETUP_MESSAGE", "")
LOG = os.environ.get("PRITHVI_SETUP_LOG", "")
FAILED = os.environ.get("PRITHVI_SETUP_FAILED", "")
STAMP = os.environ.get("PRITHVI_SETUP_STAMP", "")
WANT = os.environ.get("PRITHVI_SETUP_WANT", "")
VENV_PYTHON = os.environ.get("PRITHVI_SETUP_VENV_PYTHON", "")
SERVER = os.environ.get("PRITHVI_SETUP_SERVER", "")
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
POLL_SECONDS = 5.0

TOOL = {
    "name": "prithvi_setup_status",
    "description": (
        "Explains why the Prithvi WxC downscaling tools are not available yet "
        "(first-start install in progress, or a setup problem) and how to fix it. "
        "Call it and relay the answer to the user."
    ),
    "inputSchema": {"type": "object", "properties": {}},
}


def log_tail(lines=15):
    try:
        with open(LOG, "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def install_ready():
    try:
        with open(STAMP) as fh:
            stamped = fh.read().strip()
    except OSError:
        return False
    return stamped == WANT and os.access(VENV_PYTHON, os.X_OK)


def install_failed():
    return bool(FAILED) and os.path.exists(FAILED)


def status():
    """Return (text, is_error) describing the current setup state."""
    if STATE == "installing" and not install_failed():
        if install_ready():
            return ("Installation finished. If the Prithvi tools have not appeared, run /mcp in "
                    "Claude Code and reconnect prithvi-wxc-downscaling.", False)
        return ("The Prithvi WxC MCP server is installing its Python packages (first start, usually "
                "1-3 minutes). Its tools appear automatically when the install finishes; no action is "
                "needed. Progress log: %s\n\nLatest lines:\n%s" % (LOG, log_tail()), True)
    if STATE == "installing":
        return ("Installing the Prithvi WxC MCP server's Python packages failed. Log: %s\n\n"
                "Last lines:\n%s\n\nFix the cause shown above, then run /mcp in Claude Code and "
                "reconnect prithvi-wxc-downscaling to retry." % (LOG, log_tail(25)), True)
    tail = log_tail()
    return (MESSAGE + ("\n\nLast lines of %s:\n%s" % (LOG, tail) if tail else ""), True)


def respond(message, state):
    method = message.get("method", "")
    msg_id = message.get("id")
    params = message.get("params") or {}
    if msg_id is None:  # notification
        return None
    if method == "initialize":
        state["initialized"] = True
        requested = params.get("protocolVersion")
        result = {
            "protocolVersion": requested if requested in PROTOCOLS else PROTOCOLS[0],
            "capabilities": {"tools": {"listChanged": True}},
            "serverInfo": {"name": "prithvi-wxc-downscaling", "version": "setup"},
            "instructions": ("The Prithvi WxC tools are not available yet. Call prithvi_setup_status "
                             "and tell the user what it says."),
        }
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        text, is_error = status()
        name = params.get("name", "")
        if name != TOOL["name"]:
            text, is_error = "The tool %s is not available yet. %s" % (name, text), True
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": "Method not found: %s" % method}}
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def send(payload):
    os.write(1, (json.dumps(payload) + "\n").encode("utf-8"))


def hand_over(pending, initialized):
    """Replace this process with the full server, passing on unread input."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PRITHVI_SETUP_")}
    env["PRITHVI_MCP_PENDING"] = base64.b64encode(pending).decode("ascii")
    if initialized:
        env["PRITHVI_MCP_RESUMED"] = "1"
    os.execve(VENV_PYTHON, [VENV_PYTHON, SERVER], env)


def main():
    state = {"initialized": False}
    buffer = b""
    watching = STATE == "installing"
    while True:
        if watching and install_ready() and not install_failed():
            hand_over(buffer, state["initialized"])
        readable, _, _ = select.select([0], [], [], POLL_SECONDS if watching else None)
        if not readable:
            continue
        chunk = os.read(0, 65536)
        if not chunk:
            return
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                message = json.loads(line.decode("utf-8"))
            except ValueError:
                send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            batch = message if isinstance(message, list) else [message]
            replies = [r for r in (respond(m, state) for m in batch) if r is not None]
            if replies:
                send(replies if isinstance(message, list) else replies[0])


if __name__ == "__main__":
    main()
